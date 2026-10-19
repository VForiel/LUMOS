"""FastAPI adapter of the service layer: the HTTP backend of the web interface (steps P14.1-P14.3, ADR-0018).

Every route translates an HTTP request into one call of :class:`~lumos.service.Service` and its answer (a
Pydantic model) into JSON; a :class:`~lumos.service.ServiceError` becomes a JSON error with the same message,
hint and code, and an HTTP status (``invalid`` 400, ``unauthorized`` 401, ``forbidden`` 403, ``not_found``
404, ``failed`` 422, ``quota`` and ``rate_limited`` 429). Each browser owns a session of the service behind an
HttpOnly cookie, so that its contexts and runs are its own. Runs and journals are followed over the WebSocket
``/api/ws``. Each route needs a role (viewer, simulator, operator; ADR-0019).
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
import re
import secrets
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import HTTPConnection
from starlette.routing import Route

import lumos
from lumos.service import Service, ServiceError
from lumos.service.models import (
    ActionResult,
    ArchiveInfo,
    ConnectionInfo,
    ContextSummary,
    DeviceModel,
    DocHit,
    ElementSummary,
    ElementTypeInfo,
    FocusInfo,
    Issue,
    JobInfo,
    Page,
    PortFile,
    PortSummary,
    PresetInfo,
)
from lumos.web.assistant import PATH, McpGateway
from lumos.web.auth import (
    ROLES,
    User,
    dummy_verify,
    load_users,
    role_allows,
    verify_password,
    verify_ssh_signature,
)
from lumos.web.config import LOOPBACK_HOSTS, WebConfig
from lumos.web.security import AuditLog, RateLimiter, client_address, is_secure, retry_after
from lumos.web.sessions import COOKIE, WebSession, WebSessions
from lumos.web.units import unit_choices

__all__ = ["create_app"]

logger = logging.getLogger("lumos.web")

#: HTTP status of each code of :class:`ServiceError`.
STATUS = {
    "invalid": 400,
    "unauthorized": 401,
    "forbidden": 403,
    "not_found": 404,
    "failed": 422,
    "quota": 429,
    "rate_limited": 429,
}

#: Directory of the built frontend (``web/`` built by Vite).
STATIC = Path(__file__).resolve().parent / "static"

_FALLBACK = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>LUMOS</title>
<meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body style="font-family: system-ui, sans-serif; max-width: 40rem; margin: 4rem auto; padding: 0 1rem">
<h1>LUMOS</h1>
<p>The API is running, but the web interface has not been built.</p>
<p>Build it once with <code>npm ci &amp;&amp; npm run build</code> in the <code>web/</code> directory of the
repository, or install a released version of <code>lumos-maxima[web]</code>.</p>
<p>API: <a href="/api/info">/api/info</a></p>
</body></html>
"""

_LOGIN_FAILED = "wrong user name or password"


class RateLimited(ServiceError):
    """Too many requests; ``retry`` seconds to wait."""

    def __init__(self, message: str, retry: float) -> None:
        super().__init__(
            message, hint=f"wait {retry_after(retry)} s before trying again", code="rate_limited"
        )
        self.retry = retry


# --- Request bodies -----------------------------------------------------------------------------------------


class FocusRequest(BaseModel):
    """The context to bring to the foreground (``null``: none)."""

    context: str | None


class NewContext(BaseModel):
    """Body of ``POST /api/contexts``."""

    name: str = ""
    mode: str = "simulation"


class NewElement(BaseModel):
    """Body of ``POST /api/contexts/{id}/elements``."""

    type_id: str
    name: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class Connection(BaseModel):
    """Body of ``POST /api/contexts/{id}/connections``."""

    source: str
    target: str


class ParamChange(BaseModel):
    """Body of ``PATCH /api/contexts/{id}/params``."""

    path: str
    value: Any


class Observation(BaseModel):
    """Body of ``PUT /api/contexts/{id}/observation``."""

    minimum: Any
    maximum: Any
    n_wavelengths: int = 1


class RunRequest(BaseModel):
    """Body of ``POST /api/contexts/{id}/runs``; no output runs the default outputs of the context."""

    outputs: list[str] = Field(default_factory=list)
    archive: bool = False


class PasswordLogin(BaseModel):
    """Body of ``POST /api/auth/login``; ``bearer`` asks for the token in the answer (scripts)."""

    username: str
    password: str
    bearer: bool = False


class ChallengeRequest(BaseModel):
    """Body of ``POST /api/auth/challenge``."""

    username: str


class SshLogin(BaseModel):
    """Body of ``POST /api/auth/ssh``: the challenge signed by ``ssh-keygen -Y sign -n lumos``."""

    username: str
    challenge: str
    signature: str
    bearer: bool = False


class ContextChange(BaseModel):
    """Body of ``PATCH /api/contexts/{id}``."""

    name: str | None = None
    description: str | None = None
    seed: int | None = None


class FreeChange(BaseModel):
    """Body of ``PUT /api/contexts/{id}/free``: a parameter freed with a prior, or fixed again."""

    path: str
    free: bool = True
    prior: dict[str, Any] | None = None


class FeedbackChange(BaseModel):
    """Body of ``POST /api/contexts/{id}/feedbacks``."""

    source: str
    target: str
    mode: str = "set"


class LoopRequest(BaseModel):
    """Body of ``POST /api/contexts/{id}/loops``."""

    steps: int
    period: Any
    outputs: list[str] = Field(default_factory=list)


class StudyRequest(BaseModel):
    """Body of ``POST /api/contexts/{id}/studies`` (see :meth:`lumos.service.Service.run_study`)."""

    kind: str
    outputs: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] | None = None
    mode: str = "grid"
    n: int | None = None
    noise: str | None = None


class Rename(BaseModel):
    """Body of ``POST /api/contexts/{id}/elements/{path}/rename``."""

    name: str


class Grouping(BaseModel):
    """Body of ``POST /api/contexts/{id}/groups``."""

    elements: list[str]
    name: str


class Selection(BaseModel):
    """Body of ``POST /api/contexts/{id}/copy``."""

    elements: list[str]


class Paste(BaseModel):
    """Body of ``POST /api/contexts/{id}/paste``: a fragment from ``/copy``."""

    fragment: dict[str, Any]


class Discovery(BaseModel):
    """Body of ``POST /api/devices/discover``."""

    tcp: list[str] = Field(default_factory=list)
    serial: bool = False
    visa: bool = False


class Association(BaseModel):
    """Body of ``POST /api/contexts/{id}/devices/associate``: element path → device URL (optional)."""

    assignments: dict[str, str] | None = None


class ActionRequest(BaseModel):
    """Body of ``POST /api/contexts/{id}/actions``; ``confirmed`` once the user confirmed in a dialog."""

    element: str
    action: str
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    confirmed: bool = False


# --- Helpers ------------------------------------------------------------------------------------------------


def _error(
    status: int, message: str, hint: str, code: str, headers: dict[str, str] | None = None
) -> JSONResponse:
    return JSONResponse({"error": message, "hint": hint, "code": code}, status_code=status, headers=headers)


def _filename(name: str, suffix: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_.") or "context"
    return f"{stem}{suffix}"


def _attachment(content: str | bytes, filename: str, media_type: str) -> Response:
    return Response(
        content, media_type=media_type, headers={"content-disposition": f'attachment; filename="{filename}"'}
    )


def _signals() -> dict[str, dict[str, list[str]]]:
    """Signal classes with their parent signal classes (for the typed connections of the editor)."""
    from lumos.core.signal import Signal

    found: dict[str, dict[str, list[str]]] = {}
    stack: list[type] = [Signal]
    while stack:
        cls = stack.pop()
        parents = [c.__name__ for c in cls.__mro__[1:] if isinstance(c, type) and issubclass(c, Signal)]
        found[cls.__name__] = {"parents": parents}
        stack.extend(cls.__subclasses__())
    return found


def _units_of(types: list[dict[str, Any]]) -> set[str]:
    """Units of the parameters, states and quantity settings of the element types."""
    found: set[str] = set()
    for item in types:
        properties = item["schema"]["properties"]
        for group in ("params", "settings", "state"):
            for field in properties[group]["properties"].values():
                if "x-lumos-unit" in field:
                    found.add(field["x-lumos-unit"])
                default = field.get("default")
                if isinstance(default, dict) and isinstance(default.get("unit"), str):
                    found.add(default["unit"])
    return found


def _bounds_in_unit(schema: dict[str, Any]) -> dict[str, Any]:
    """Copy a schema, adding the bounds of each parameter in its unit (``x-lumos-bounds-value``)."""
    import numpy as np
    from astropy import units as u

    result = copy.deepcopy(schema)
    for group in ("params", "state"):
        for field in result["properties"][group]["properties"].values():
            if "x-lumos-unit" not in field:
                continue
            unit = u.Unit(field["x-lumos-unit"])
            values: list[float | None] = []
            for bound in field.get("x-lumos-bounds") or [None, None]:
                if bound is None:
                    values.append(None)
                    continue
                quantity = u.Quantity(float(bound["value"]), bound["unit"])
                values.append(float(np.asarray(quantity.to_value(unit, equivalencies=u.spectral()))))
            field["x-lumos-bounds-value"] = values
    return result


def _is_loopback(host: str | None) -> bool:
    return (host or "").lower() in LOOPBACK_HOSTS


def _allowed_origin(websocket: WebSocket, config: WebConfig) -> bool:
    origin = websocket.headers.get("origin")
    if origin is None:
        return True  # not a browser: the cookie or token is the credential
    if origin in config.allowed_origins:
        return True
    parts = urlsplit(origin)
    if config.profile == "local":
        return _is_loopback(parts.hostname)
    return parts.netloc == websocket.headers.get("host")


def _bearer(connection: Request | WebSocket) -> str | None:
    header = connection.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


class _HttpHostCheck(TrustedHostMiddleware):
    """Refuse HTTP requests addressed to an unknown host name (DNS rebinding).

    WebSockets are checked on their ``Origin`` instead (:func:`_allowed_origin`), which a rebound page cannot
    forge.
    """

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "websocket":
            await self.app(scope, receive, send)
            return
        await super().__call__(scope, receive, send)


class _Users:
    """Users of the instance, read again when their file changes."""

    def __init__(self, path: str | None) -> None:
        self.path = Path(path) if path else None
        self._mtime: float | None = None
        self._users: dict[str, User] = {}
        self.reload()

    def reload(self) -> None:
        if self.path is None:
            return
        mtime = self.path.stat().st_mtime
        if mtime != self._mtime:
            self._users = load_users(self.path)
            self._mtime = mtime

    def get(self, name: str) -> User | None:
        with contextlib.suppress(OSError):
            self.reload()
        return self._users.get(name)

    @property
    def enabled(self) -> bool:
        return self.path is not None


class _Challenges:
    """SSH challenges: single-use, bound to a user name, valid for a few seconds."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._open: dict[str, tuple[str, float]] = {}

    def issue(self, username: str) -> str:
        now = time.monotonic()
        if len(self._open) > 10_000:
            self._open = {c: v for c, v in self._open.items() if v[1] > now}
        challenge = f"lumos-login:{secrets.token_hex(24)}"
        self._open[challenge] = (username, now + self.seconds)
        return challenge

    def consume(self, challenge: str, username: str) -> bool:
        """Whether ``challenge`` was issued to ``username`` and is still valid; it can never be used again."""
        entry = self._open.pop(challenge, None)
        return entry is not None and entry[0] == username and entry[1] > time.monotonic()


# --- Application --------------------------------------------------------------------------------------------


def create_app(config: WebConfig | None = None, *, service: Service | None = None) -> FastAPI:
    """Create the web application of LUMOS.

    Parameters
    ----------
    config
        Settings of the instance (profile, hosts, users, quotas of the service...).
    service
        Service layer to use; by default a new one built from ``config.service``.

    Returns
    -------
    FastAPI
        The ASGI application, to serve with ``uvicorn`` (``lumos web`` does it).

    Raises
    ------
    lumos.web.config.ConfigError
        The configuration is unsafe (see :meth:`WebConfig.check`).
    """
    config = config or WebConfig()
    config.check()
    service = service or Service(config.service)
    sessions = WebSessions(
        service,
        idle_seconds=config.session_idle_minutes * 60,
        lifetime_seconds=config.session_max_hours * 3600,
    )
    users = _Users(config.users_file)
    challenges = _Challenges(config.challenge_seconds)
    audit = AuditLog(config.audit_log)
    requests_limit = RateLimiter(config.requests_per_minute)
    logins_limit = RateLimiter(config.login_attempts_per_minute)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with gateway.running():
            yield
        service.shutdown()

    app = FastAPI(
        title="LUMOS",
        version=lumos.__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.config = config
    app.state.service = service
    app.state.sessions = sessions
    app.state.audit = audit

    hosts = list(config.allowed_hosts)
    if config.profile == "local":
        hosts += [h.strip("[]") for h in LOOPBACK_HOSTS]
    elif not hosts:
        hosts = ["*"]
    app.add_middleware(_HttpHostCheck, allowed_hosts=hosts)

    def address(connection: Request | WebSocket) -> str:
        return client_address(connection, trust_proxy=config.trust_proxy)

    def secure(connection: HTTPConnection) -> bool:
        return is_secure(connection, trust_proxy=config.trust_proxy)

    def role_of(session: WebSession) -> str | None:
        return session.role if session.user is not None else config.anonymous_role

    # the MCP server of each session, at /mcp, before every other route (P14.9)
    gateway = McpGateway(
        service=service,
        sessions=sessions,
        role_of=role_of,
        anonymous_role=config.anonymous_role,
        enabled=config.assistant,
        base_url=lambda connection: f"{'https' if secure(connection) else 'http'}://{connection.url.netloc}",
    )
    methods = ["GET", "POST", "DELETE"]
    app.router.routes.insert(0, Route(PATH, endpoint=gateway, methods=methods))
    # connector links: /mcp/<secret>
    app.router.routes.insert(1, Route(f"{PATH}/{{link}}", endpoint=gateway, methods=methods))

    # --- Sessions, limits and errors -------------------------------------------------------------------

    @app.middleware("http")
    async def attach_session(request: Request, call_next: Any) -> Response:
        if not request.url.path.startswith("/api/") or request.url.path == "/api/health":
            return await call_next(request)
        wait = requests_limit.hit(address(request))
        if wait:
            audit.record("rate_limited", "failure", ip=address(request), path=request.url.path)
            return _error(
                429,
                "too many requests",
                f"wait {retry_after(wait)} s",
                "rate_limited",
                headers={"retry-after": retry_after(wait)},
            )
        bearer = _bearer(request)
        session = sessions.get(bearer or request.cookies.get(COOKIE))
        fresh = session is None and bearer is None
        if session is None:
            if config.max_sessions is not None and len(sessions) >= config.max_sessions:
                return _error(
                    429, "the instance has too many visitors", "try again in a few minutes", "quota"
                )
            session = sessions.open()
        request.state.web_session = session
        request.state.set_cookie = session.token if fresh else None
        response = await call_next(request)
        token = request.state.set_cookie
        if token is not None and bearer is None:
            response.set_cookie(
                COOKIE, token, httponly=True, samesite="strict", secure=secure(request), path="/"
            )
        return response

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, error: ServiceError) -> JSONResponse:
        headers = {"retry-after": retry_after(error.retry)} if isinstance(error, RateLimited) else None
        return _error(STATUS.get(error.code, 400), error.message, error.hint, error.code, headers)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
        problems = []
        for item in error.errors():
            where = ".".join(str(part) for part in item.get("loc", ()) if part != "body")
            problems.append(f"{where}: {item.get('msg', 'invalid')}" if where else str(item.get("msg")))
        return _error(
            400, "; ".join(problems) or "invalid request", "check the body of the request", "invalid"
        )

    def needs(role: str) -> Callable[[Request], None]:
        """Dependency refusing the request unless the session has ``role`` (or a higher one)."""

        def check(request: Request) -> None:
            session: WebSession = request.state.web_session
            current = role_of(session)
            if current is None:
                raise ServiceError(
                    "this instance requires to log in",
                    hint="log in with a password or an SSH key (lumos login)",
                    code="unauthorized",
                )
            if not role_allows(current, role):
                audit.record(
                    "forbidden",
                    "failure",
                    ip=address(request),
                    user=session.user,
                    role=current,
                    path=request.url.path,
                )
                who = "log in as a user with this role" if session.user is None else "ask for this role"
                raise ServiceError(
                    f"the role {current} cannot do this",
                    hint=f"this needs the role {role}: {who}",
                    code="forbidden",
                )

        return check

    viewer, simulator, operator = (Depends(needs(role)) for role in ROLES)

    def require_tls(request: Request, what: str) -> None:
        if not config.local_only and not secure(request):
            raise ServiceError(
                f"{what} needs HTTPS on this instance",
                hint="open the instance with https://",
                code="forbidden",
            )

    def sid(request: Request) -> str:
        session: WebSession = request.state.web_session
        return session.service_session

    # --- Information and discovery ---------------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": lumos.__version__, "sessions": len(sessions)}

    @app.get("/api/info")
    def info(request: Request) -> dict[str, Any]:
        session: WebSession = request.state.web_session
        return {
            "name": "LUMOS",
            "version": lumos.__version__,
            "profile": config.profile,
            "operation": service.config.operation_enabled,
            "languages": ["en", "fr"],
            "user": session.user,
            "role": role_of(session),
            "auth": {
                "enabled": users.enabled,
                "required": config.anonymous_role is None,
                "methods": ["password", "ssh"] if users.enabled else [],
            },
            "assistant": config.assistant,
        }

    @app.get("/api/element-types", dependencies=[viewer])
    def element_types(
        query: str = "", page: int = 1, page_size: int = Query(50, ge=1, le=500)
    ) -> Page[ElementTypeInfo]:
        return service.list_element_types(query=query, page=page, page_size=page_size)

    @app.get("/api/element-types/{type_id}/schema", dependencies=[viewer])
    def element_schema(type_id: str) -> dict[str, Any]:
        return service.get_element_schema(type_id)

    palette_cache: dict[str, Any] = {}

    @app.get("/api/palette", dependencies=[viewer])
    def palette() -> dict[str, Any]:
        if not palette_cache:
            listing = service.list_element_types(page_size=10_000)
            palette_cache["types"] = [
                {
                    **item.model_dump(),
                    "category": item.type_id.split(".")[1] if item.type_id.count(".") >= 2 else "other",
                    "schema": _bounds_in_unit(service.get_element_schema(item.type_id)),
                }
                for item in listing.items
            ]
            palette_cache["signals"] = _signals()
            palette_cache["units"] = unit_choices(_units_of(palette_cache["types"]))
        return palette_cache

    @app.get("/api/docs/search", dependencies=[viewer])
    def docs_search(q: str, limit: int = Query(5, ge=1, le=50)) -> list[DocHit]:
        return service.search_docs(q, limit=limit)

    # --- Authentication --------------------------------------------------------------------------------

    def limit_logins(request: Request, username: str) -> None:
        waits = [logins_limit.hit(f"ip:{address(request)}"), logins_limit.hit(f"user:{username}")]
        if max(waits):
            audit.record("login", "rate_limited", ip=address(request), user=username)
            raise RateLimited("too many login attempts", max(waits))

    def logged_in(request: Request, user: User, method: str, bearer: bool) -> dict[str, Any]:
        session = sessions.login(request.state.web_session, user.name, user.role)
        request.state.web_session = session
        request.state.set_cookie = session.token
        audit.record("login", "success", ip=address(request), user=user.name, role=user.role, method=method)
        answer: dict[str, Any] = {"user": user.name, "role": user.role}
        if bearer:
            answer["token"] = session.token
        return answer

    @app.get("/api/auth/me")
    def me(request: Request) -> dict[str, Any]:
        session: WebSession = request.state.web_session
        return {"user": session.user, "role": role_of(session)}

    @app.post("/api/auth/login")
    def password_login(request: Request, body: PasswordLogin) -> dict[str, Any]:
        require_tls(request, "logging in")
        limit_logins(request, body.username)
        user = users.get(body.username)
        if user is None or user.password is None:
            dummy_verify(body.password)
        elif verify_password(user.password, body.password):
            return logged_in(request, user, "password", body.bearer)
        audit.record("login", "failure", ip=address(request), user=body.username, method="password")
        raise ServiceError(_LOGIN_FAILED, hint="check the user name and the password", code="unauthorized")

    @app.post("/api/auth/challenge")
    def ssh_challenge(request: Request, body: ChallengeRequest) -> dict[str, Any]:
        require_tls(request, "logging in")
        limit_logins(request, body.username)
        audit.record("challenge", "success", ip=address(request), user=body.username)
        return {
            "challenge": challenges.issue(body.username),
            "namespace": "lumos",
            "expires_in": config.challenge_seconds,
        }

    @app.post("/api/auth/ssh")
    def ssh_login(request: Request, body: SshLogin) -> dict[str, Any]:
        require_tls(request, "logging in")
        limit_logins(request, body.username)
        valid = challenges.consume(body.challenge, body.username)
        user = users.get(body.username)
        if valid and user is not None:
            message = body.challenge.encode()
            if any(verify_ssh_signature(key, message, body.signature) for key in user.ssh_keys):
                return logged_in(request, user, "ssh", body.bearer)
        reason = "unknown, expired or used challenge" if not valid else "signature refused"
        audit.record("login", "failure", ip=address(request), user=body.username, method="ssh", reason=reason)
        raise ServiceError(
            "SSH login refused",
            hint="ask for a new challenge and sign it with a key registered for this user",
            code="unauthorized",
        )

    @app.post("/api/auth/logout", status_code=204)
    def logout(request: Request) -> Response:
        session: WebSession = request.state.web_session
        audit.record("logout", "success", ip=address(request), user=session.user)
        sessions.close(session.token)
        request.state.set_cookie = None
        response = Response(status_code=204)
        response.delete_cookie(COOKIE, path="/")
        return response

    # --- Connector links (P14.15) ---------------------------------------------------------------------

    def link_url(request: Request, link: str | None) -> str | None:
        if link is None:
            return None
        scheme = "https" if secure(request) else "http"
        return f"{scheme}://{request.url.netloc}{PATH}/{link}"

    def assistant_on() -> None:
        if not config.assistant:
            raise ServiceError(
                "the assistant is turned off on this instance", hint="ask its administrator", code="not_found"
            )

    @app.get("/api/assistant/link", dependencies=[simulator])
    def get_link(request: Request) -> dict[str, str | None]:
        assistant_on()
        session: WebSession = request.state.web_session
        return {"url": link_url(request, session.link)}

    @app.post("/api/assistant/link", dependencies=[simulator])
    def create_link(request: Request) -> dict[str, str | None]:
        """Create the connector link of the session: its MCP server at a secret URL, for Claude or ChatGPT."""
        assistant_on()
        session: WebSession = request.state.web_session
        link = sessions.create_link(session)
        audit.record("connector-link", "success", ip=address(request), user=session.user)
        return {"url": link_url(request, link)}

    @app.delete("/api/assistant/link", status_code=204, dependencies=[simulator])
    def revoke_link(request: Request) -> Response:
        session: WebSession = request.state.web_session
        sessions.revoke_link(session)
        return Response(status_code=204)

    # --- Focus (P14.18) --------------------------------------------------------------------------------

    @app.get("/api/focus", dependencies=[viewer])
    def get_focus(request: Request) -> FocusInfo:
        """Return the context in the foreground of the session (open here, used by the assistant)."""
        return service.get_focus(sid(request))

    @app.put("/api/focus", dependencies=[viewer])
    def put_focus(request: Request, body: FocusRequest) -> FocusInfo:
        return service.focus(sid(request), body.context)

    # --- Contexts --------------------------------------------------------------------------------------

    @app.get("/api/contexts", dependencies=[viewer])
    def contexts(request: Request) -> list[ContextSummary]:
        return service.list_contexts(sid(request))

    @app.post("/api/contexts", status_code=201, dependencies=[simulator])
    def create_context(request: Request, body: NewContext | None = None) -> ContextSummary:
        body = body or NewContext()
        return service.create_context(sid(request), name=body.name, mode=body.mode)

    @app.post("/api/contexts/import", status_code=201, dependencies=[viewer])
    async def import_context(request: Request) -> ContextSummary:
        text = (await request.body()).decode("utf-8", errors="replace")
        return await asyncio.to_thread(service.import_lumos, sid(request), text)

    @app.get("/api/contexts/{cid}", dependencies=[viewer])
    def describe(request: Request, cid: str) -> ContextSummary:
        return service.describe_context(sid(request), cid)

    @app.delete("/api/contexts/{cid}", status_code=204, dependencies=[viewer])
    def close_context(request: Request, cid: str) -> Response:
        service.close_context(sid(request), cid)
        return Response(status_code=204)

    @app.get("/api/presets", dependencies=[viewer])
    def presets() -> list[PresetInfo]:
        return service.list_presets()

    @app.post("/api/presets/{name}/load", status_code=201, dependencies=[viewer])
    def load_preset(request: Request, name: str) -> ContextSummary:
        return service.load_preset(sid(request), name)

    @app.get("/api/contexts/{cid}/export.lumos", dependencies=[viewer])
    def export_lumos(request: Request, cid: str) -> Response:
        name = service.describe_context(sid(request), cid).name
        text = service.export_lumos(sid(request), cid)
        return _attachment(text, _filename(name, ".lumos"), "application/json")

    @app.get("/api/contexts/{cid}/export.py", dependencies=[viewer])
    def export_python(request: Request, cid: str) -> Response:
        name = service.describe_context(sid(request), cid).name
        text = service.export_python(sid(request), cid)
        return _attachment(text, _filename(name, ".py"), "text/x-python")

    # --- Editing ---------------------------------------------------------------------------------------

    @app.post("/api/contexts/{cid}/elements", status_code=201, dependencies=[simulator])
    def add_element(request: Request, cid: str, body: NewElement) -> ElementSummary:
        return service.add_element(sid(request), cid, body.type_id, name=body.name, params=body.params)

    @app.delete("/api/contexts/{cid}/elements/{path:path}", status_code=204, dependencies=[simulator])
    def remove_element(request: Request, cid: str, path: str) -> Response:
        service.remove_element(sid(request), cid, path)
        return Response(status_code=204)

    @app.post("/api/contexts/{cid}/connections", status_code=201, dependencies=[simulator])
    def connect(request: Request, cid: str, body: Connection) -> ConnectionInfo:
        return service.connect(sid(request), cid, body.source, body.target)

    @app.delete("/api/contexts/{cid}/connections", status_code=204, dependencies=[simulator])
    def disconnect(request: Request, cid: str, source: str, target: str) -> Response:
        service.disconnect(sid(request), cid, source, target)
        return Response(status_code=204)

    @app.patch("/api/contexts/{cid}/params", dependencies=[simulator])
    def set_param(request: Request, cid: str, body: ParamChange) -> ElementSummary:
        return service.set_param(sid(request), cid, body.path, body.value)

    @app.put("/api/contexts/{cid}/observation", dependencies=[simulator])
    def set_observation(request: Request, cid: str, body: Observation) -> ContextSummary:
        return service.set_observation(
            sid(request), cid, minimum=body.minimum, maximum=body.maximum, n_wavelengths=body.n_wavelengths
        )

    @app.patch("/api/contexts/{cid}", dependencies=[simulator])
    def change_context(request: Request, cid: str, body: ContextChange) -> ContextSummary:
        return service.set_context(
            sid(request), cid, name=body.name, description=body.description, seed=body.seed
        )

    @app.put("/api/contexts/{cid}/free", dependencies=[simulator])
    def set_free(request: Request, cid: str, body: FreeChange) -> ContextSummary:
        service.set_free(sid(request), cid, body.path, body.free, body.prior)
        return service.describe_context(sid(request), cid)

    @app.put("/api/contexts/{cid}/ui", dependencies=[simulator])
    def update_ui(request: Request, cid: str, body: dict[str, Any], record: bool = True) -> ContextSummary:
        return service.update_ui(sid(request), cid, body, record=record)

    @app.post("/api/contexts/{cid}/undo", dependencies=[simulator])
    def undo(request: Request, cid: str) -> ContextSummary:
        return service.undo(sid(request), cid)

    @app.post("/api/contexts/{cid}/redo", dependencies=[simulator])
    def redo(request: Request, cid: str) -> ContextSummary:
        return service.redo(sid(request), cid)

    @app.post("/api/contexts/{cid}/elements/{path:path}/rename", dependencies=[simulator])
    def rename(request: Request, cid: str, path: str, body: Rename) -> ContextSummary:
        return service.rename_element(sid(request), cid, path, body.name)

    @app.get("/api/contexts/{cid}/elements/{path:path}/export.lumos", dependencies=[viewer])
    def export_element(request: Request, cid: str, path: str) -> Response:
        text = service.export_element(sid(request), cid, path)
        return _attachment(text, _filename(path.rpartition(".")[2], ".lumos"), "application/json")

    @app.post("/api/contexts/{cid}/groups", dependencies=[simulator])
    def group(request: Request, cid: str, body: Grouping) -> ContextSummary:
        return service.group_elements(sid(request), cid, body.elements, name=body.name)

    @app.post("/api/contexts/{cid}/groups/{path:path}/ungroup", dependencies=[simulator])
    def ungroup(request: Request, cid: str, path: str) -> ContextSummary:
        return service.ungroup(sid(request), cid, path)

    @app.post("/api/contexts/{cid}/copy", dependencies=[viewer])
    def copy_elements(request: Request, cid: str, body: Selection) -> dict[str, Any]:
        return service.copy_elements(sid(request), cid, body.elements)

    @app.post("/api/contexts/{cid}/paste", dependencies=[simulator])
    def paste(request: Request, cid: str, body: Paste) -> dict[str, list[str]]:
        return {"ids": service.paste_elements(sid(request), cid, body.fragment)}

    @app.post("/api/contexts/{cid}/blocks", status_code=201, dependencies=[simulator])
    async def insert_block(request: Request, cid: str) -> ElementSummary:
        text = (await request.body()).decode("utf-8", errors="replace")
        return await asyncio.to_thread(service.import_element, sid(request), cid, text)

    # --- Verification ----------------------------------------------------------------------------------

    @app.get("/api/contexts/{cid}/validate", dependencies=[viewer])
    def validate(request: Request, cid: str) -> list[Issue]:
        return service.validate_context(sid(request), cid)

    @app.get("/api/contexts/{cid}/sampling/{element:path}", dependencies=[simulator])
    def sampling(request: Request, cid: str, element: str) -> dict[str, str]:
        return service.explain_sampling(sid(request), cid, element)

    # --- Runs and inspection ---------------------------------------------------------------------------

    @app.post("/api/contexts/{cid}/runs", status_code=202, dependencies=[simulator])
    def run(request: Request, cid: str, body: RunRequest | None = None) -> JobInfo:
        body = body or RunRequest()
        outputs = body.outputs or service.default_outputs(sid(request), cid)
        if not outputs:
            raise ServiceError(
                "nothing to run", hint="add elements with outputs, or name the outputs to compute"
            )
        return service.run(sid(request), cid, outputs, archive=body.archive)

    @app.post("/api/contexts/{cid}/studies", status_code=202, dependencies=[simulator])
    def study(request: Request, cid: str, body: StudyRequest) -> JobInfo:
        return service.run_study(
            sid(request),
            cid,
            kind=body.kind,
            outputs=body.outputs or None,
            parameters=body.parameters,
            mode=body.mode,
            n=body.n,
            noise=body.noise,
        )

    @app.post("/api/contexts/{cid}/feedbacks", status_code=201, dependencies=[simulator])
    def add_feedback(request: Request, cid: str, body: FeedbackChange) -> ContextSummary:
        return service.add_feedback(sid(request), cid, body.source, body.target, body.mode)

    @app.delete("/api/contexts/{cid}/feedbacks", dependencies=[simulator])
    def remove_feedback(request: Request, cid: str, source: str, target: str) -> ContextSummary:
        return service.remove_feedback(sid(request), cid, source, target)

    @app.post("/api/contexts/{cid}/loops", status_code=202, dependencies=[simulator])
    def loop(request: Request, cid: str, body: LoopRequest) -> JobInfo:
        return service.run_loop(
            sid(request), cid, steps=body.steps, period=body.period, outputs=body.outputs or None
        )

    @app.get("/api/jobs/{job}/study", dependencies=[viewer])
    def study_points(request: Request, job: str) -> dict[str, Any]:
        return service.get_study(sid(request), job).model_dump(
            mode="json", exclude={"points": {"__all__": {"outputs": {"__all__": {"figure_png"}}}}}
        )

    @app.get("/api/jobs/{job}/study.csv", dependencies=[viewer])
    def study_table(request: Request, job: str) -> Response:
        return _attachment(
            service.study_csv(sid(request), job), f"study-{job}.csv", "text/csv; charset=utf-8"
        )

    @app.get("/api/jobs", dependencies=[viewer])
    def jobs(request: Request) -> list[JobInfo]:
        return service.list_jobs(sid(request))

    @app.get("/api/jobs/{job}", dependencies=[viewer])
    def job(request: Request, job: str) -> JobInfo:
        return service.get_job(sid(request), job)

    def send(file: PortFile) -> Response:
        return _attachment(file.content, file.filename, file.media_type)

    @app.post("/api/jobs/{job}/cancel", dependencies=[simulator])
    def cancel(request: Request, job: str) -> JobInfo:
        return service.cancel_job(sid(request), job)

    @app.get("/api/jobs/{job}/results/{port:path}/download", dependencies=[viewer])
    def result_download(request: Request, job: str, port: str, format: str = "png") -> Response:
        return send(service.result_file(sid(request), job, port, format))

    @app.get("/api/jobs/{job}/results/{port:path}/figure.png", dependencies=[viewer])
    def result_figure(request: Request, job: str, port: str) -> Response:
        return Response(service.result_file(sid(request), job, port, "png").content, media_type="image/png")

    @app.get("/api/contexts/{cid}/ports/{port:path}/download", dependencies=[simulator])
    def port_download(request: Request, cid: str, port: str, format: str = "png") -> Response:
        return send(service.port_file(sid(request), cid, port, format))

    @app.get("/api/jobs/{job}/results/{port:path}", dependencies=[viewer])
    def result(request: Request, job: str, port: str) -> dict[str, Any]:
        summary = service.get_result(sid(request), job, port)
        return summary.model_dump(mode="json", exclude={"figure_png"})

    @app.get("/api/contexts/{cid}/ports/{port:path}/figure.png", dependencies=[simulator])
    def port_figure(request: Request, cid: str, port: str) -> Response:
        summary: PortSummary = service.inspect_port(sid(request), cid, port, figure=True)
        if summary.figure_png is None:
            raise ServiceError(
                f"{port} carries a {summary.kind}, which has no figure", hint="read its summary instead"
            )
        return Response(summary.figure_png, media_type="image/png")

    @app.get("/api/contexts/{cid}/ports/{port:path}", dependencies=[simulator])
    def inspect(request: Request, cid: str, port: str) -> dict[str, Any]:
        summary = service.inspect_port(sid(request), cid, port, figure=False)
        return summary.model_dump(mode="json", exclude={"figure_png"})

    @app.get("/api/contexts/{cid}/journal", dependencies=[viewer])
    def journal(
        request: Request, cid: str, kind: str | None = None, limit: int = Query(50, ge=1, le=10_000)
    ) -> list[dict[str, Any]]:
        return service.journal(sid(request), cid, kind=kind, limit=limit)

    # --- Catalogues and archives -----------------------------------------------------------------------

    @app.get("/api/catalogs/resolve", dependencies=[simulator])
    def resolve(name: str) -> dict[str, Any]:
        return service.resolve_object(name)

    @app.get("/api/archives", dependencies=[viewer])
    def archives(limit: int = Query(50, ge=1, le=1000)) -> list[ArchiveInfo]:
        return service.list_archives(limit=limit)

    @app.get("/api/archives/download", dependencies=[viewer])
    def archive_download(path: str) -> Response:
        return _attachment(service.archive_zip(path), f"{Path(path).name}.zip", "application/zip")

    @app.get("/api/archives/metadata", dependencies=[viewer])
    def archive_metadata(path: str) -> dict[str, Any]:
        return service.read_archive_metadata(path)

    # --- Operation -------------------------------------------------------------------------------------

    @app.post("/api/devices/discover", dependencies=[operator])
    def discover(request: Request, body: Discovery) -> list[DeviceModel]:
        require_tls(request, "the operation of equipment")
        session: WebSession = request.state.web_session
        found = service.discover_devices(tcp=body.tcp, serial=body.serial, visa=body.visa)
        audit.record("discover", "success", ip=address(request), user=session.user, devices=len(found))
        return found

    @app.post("/api/contexts/{cid}/devices/associate", dependencies=[operator])
    def associate(request: Request, cid: str, body: Association | None = None) -> dict[str, DeviceModel]:
        require_tls(request, "the operation of equipment")
        body = body or Association()
        return service.associate_devices(sid(request), cid, assignments=body.assignments)

    @app.get("/api/contexts/{cid}/devices", dependencies=[viewer])
    def devices(request: Request, cid: str) -> list[dict[str, Any]]:
        return service.list_devices(sid(request), cid)

    @app.get("/api/contexts/{cid}/devices/{element:path}/status", dependencies=[viewer])
    def device_status(request: Request, cid: str, element: str) -> dict[str, Any]:
        return service.device_status(sid(request), cid, element)

    @app.get("/api/contexts/{cid}/elements/{element:path}/confirmation", dependencies=[viewer])
    def confirmation(request: Request, cid: str, element: str) -> dict[str, Any]:
        return {"device": service.confirmation_needed(sid(request), cid, element)}

    @app.post("/api/contexts/{cid}/actions", dependencies=[simulator])
    def action(request: Request, cid: str, body: ActionRequest) -> ActionResult:
        session: WebSession = request.state.web_session
        device = service.confirmation_needed(sid(request), cid, body.element)
        if device is not None:
            needs("operator")(request)
            require_tls(request, "the operation of equipment")
        try:
            result = service.execute_action(
                sid(request),
                cid,
                body.element,
                body.action,
                args=body.args,
                kwargs=body.kwargs,
                confirmed=body.confirmed,
            )
        except ServiceError as error:
            if device is not None:
                audit.record(
                    "action",
                    "failure",
                    ip=address(request),
                    user=session.user,
                    element=body.element,
                    action=body.action,
                    device=device,
                    reason=error.message,
                )
            raise
        if device is not None:
            audit.record(
                "action",
                "success",
                ip=address(request),
                user=session.user,
                element=body.element,
                action=body.action,
                device=device,
            )
        return result

    @app.post("/api/contexts/{cid}/stop", status_code=204, dependencies=[viewer])
    def stop(request: Request, cid: str) -> Response:
        session: WebSession = request.state.web_session
        service.stop_all(sid(request), cid)
        audit.record("stop", "success", ip=address(request), user=session.user, context=cid)
        return Response(status_code=204)

    # --- WebSocket -------------------------------------------------------------------------------------

    @app.websocket("/api/ws")
    async def events(websocket: WebSocket) -> None:
        if not _allowed_origin(websocket, config):
            await websocket.close(code=1008, reason="origin not allowed")
            return
        session = sessions.get(_bearer(websocket) or websocket.cookies.get(COOKIE))
        if session is None or role_of(session) is None:
            await websocket.close(code=1008, reason="no session: call the API first, or log in")
            return
        await websocket.accept()
        sent_jobs: dict[str, dict[str, Any]] = {}
        journal_seen: dict[str, int] = {}
        revisions: dict[str, int] = {}  # changes of the contexts, sent when they grow (P14.18)
        focus_count: int | None = None
        try:
            for summary in service.list_contexts(session.service_session):
                journal_seen[summary.id] = len(service.journal_since(session.service_session, summary.id))
            while True:
                try:
                    state = service.list_jobs(session.service_session)
                    summaries = service.list_contexts(session.service_session)
                    focus = service.get_focus(session.service_session)
                except ServiceError:
                    await websocket.close(code=1008, reason="session expired")
                    return
                contexts_ = [c.id for c in summaries]
                for summary in summaries:
                    if revisions.get(summary.id) != summary.revision:
                        revisions[summary.id] = summary.revision
                        await websocket.send_json(
                            {"type": "context", "context": summary.id, "revision": summary.revision}
                        )
                if focus.count != focus_count:
                    focus_count = focus.count
                    await websocket.send_json({"type": "focus", **focus.model_dump(mode="json")})
                for cid in contexts_:
                    start = journal_seen.get(cid, 0)
                    try:
                        entries = service.journal_since(session.service_session, cid, start)
                    except ServiceError:
                        continue  # closed meanwhile
                    for entry in entries:
                        await websocket.send_json({"type": "journal", "context": cid, "entry": entry})
                    journal_seen[cid] = start + len(entries)
                for job_info in state:
                    data = job_info.model_dump(mode="json")
                    if sent_jobs.get(job_info.id) != data:
                        sent_jobs[job_info.id] = data
                        await websocket.send_json({"type": "job", "job": data})
                session.last_seen = time.monotonic()  # an open interface keeps its session alive
                await asyncio.sleep(0.2)
        except (WebSocketDisconnect, RuntimeError):
            return

    # --- View links (P14.18) ---------------------------------------------------------------------------

    @app.get("/view/{secret}", include_in_schema=False)
    def view(request: Request, secret: str, context: str = "") -> Response:
        """Attach this browser to the session of a view link (from an assistant), then open its context."""
        session = sessions.by_view(secret)
        if session is None:
            return _error(
                404,
                "unknown or expired view link",
                "ask the assistant for a new link (get_view_link)",
                "not_found",
            )
        audit.record("view-link", "success", ip=address(request), user=session.user)
        target = f"/?context={context}" if re.fullmatch(r"[0-9a-f]{1,32}", context) else "/"
        response = Response(status_code=303, headers={"location": target, "cache-control": "no-store"})
        response.set_cookie(
            COOKIE, session.token, httponly=True, samesite="strict", secure=secure(request), path="/"
        )
        return response

    # --- Interface -------------------------------------------------------------------------------------

    @app.get("/api/{rest:path}", include_in_schema=False)
    def unknown_api(rest: str) -> JSONResponse:
        return _error(404, f"no API route /api/{rest}", "see /api/docs for the routes", "not_found")

    @app.get("/{rest:path}", include_in_schema=False)
    def interface(rest: str) -> Response:
        index = STATIC / "index.html"
        if not index.is_file():
            return HTMLResponse(_FALLBACK)
        candidate = (STATIC / rest).resolve()
        if rest and candidate.is_file() and STATIC in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(index, headers={"cache-control": "no-cache"})

    return app
