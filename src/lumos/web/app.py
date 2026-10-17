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
    Issue,
    JobInfo,
    Page,
    PortSummary,
    PresetInfo,
)
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

    def secure(connection: Request | WebSocket) -> bool:
        return is_secure(connection, trust_proxy=config.trust_proxy)

    def role_of(session: WebSession) -> str | None:
        return session.role if session.user is not None else config.anonymous_role

    # --- Sessions, limits and errors -------------------------------------------------------------------

    @app.middleware("http")
    async def attach_session(request: Request, call_next: Any) -> Response:
        if not request.url.path.startswith("/api/"):
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
                    "schema": service.get_element_schema(item.type_id),
                }
                for item in listing.items
            ]
            palette_cache["signals"] = _signals()
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

    @app.get("/api/jobs", dependencies=[viewer])
    def jobs(request: Request) -> list[JobInfo]:
        return service.list_jobs(sid(request))

    @app.get("/api/jobs/{job}", dependencies=[viewer])
    def job(request: Request, job: str) -> JobInfo:
        return service.get_job(sid(request), job)

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
        try:
            for summary in service.list_contexts(session.service_session):
                journal_seen[summary.id] = len(service.journal_since(session.service_session, summary.id))
            while True:
                try:
                    state = service.list_jobs(session.service_session)
                    contexts_ = [c.id for c in service.list_contexts(session.service_session)]
                except ServiceError:
                    await websocket.close(code=1008, reason="session expired")
                    return
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
