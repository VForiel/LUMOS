"""FastAPI adapter of the service layer: the HTTP backend of the web interface (step P14.1, ADR-0018).

Every route translates an HTTP request into one call of :class:`~lumos.service.Service` and its answer (a
Pydantic model) into JSON; a :class:`~lumos.service.ServiceError` becomes a JSON error with the same message,
hint and code, and an HTTP status (``invalid`` 400, ``forbidden`` 403, ``not_found`` 404, ``failed`` 422,
``quota`` 429). Each browser owns a session of the service behind an HttpOnly cookie, so that its contexts and
runs are its own. Runs and journals are followed over the WebSocket ``/api/ws``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

import lumos
from lumos.service import Service, ServiceError
from lumos.service.models import (
    ArchiveInfo,
    ConnectionInfo,
    ContextSummary,
    DocHit,
    ElementSummary,
    ElementTypeInfo,
    Issue,
    JobInfo,
    Page,
    PortSummary,
    PresetInfo,
)
from lumos.web.config import LOOPBACK_HOSTS, WebConfig
from lumos.web.sessions import COOKIE, WebSession, WebSessions

__all__ = ["create_app"]

logger = logging.getLogger("lumos.web")

#: HTTP status of each code of :class:`ServiceError`.
STATUS = {"invalid": 400, "forbidden": 403, "not_found": 404, "failed": 422, "quota": 429}

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


# --- Helpers ------------------------------------------------------------------------------------------------


def _error(status: int, message: str, hint: str, code: str) -> JSONResponse:
    return JSONResponse({"error": message, "hint": hint, "code": code}, status_code=status)


def _filename(name: str, suffix: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_.") or "context"
    return f"{stem}{suffix}"


def _attachment(text: str, filename: str, media_type: str) -> Response:
    return Response(
        text, media_type=media_type, headers={"content-disposition": f'attachment; filename="{filename}"'}
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
        return True  # not a browser: the cookie is the credential
    if origin in config.allowed_origins:
        return True
    parts = urlsplit(origin)
    if config.profile == "local":
        return _is_loopback(parts.hostname)
    return parts.netloc == websocket.headers.get("host")


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


# --- Application --------------------------------------------------------------------------------------------


def create_app(config: WebConfig | None = None, *, service: Service | None = None) -> FastAPI:
    """Create the web application of LUMOS.

    Parameters
    ----------
    config
        Settings of the instance (profile, hosts, quotas of the service...).
    service
        Service layer to use; by default a new one built from ``config.service``.

    Returns
    -------
    FastAPI
        The ASGI application, to serve with ``uvicorn`` (``lumos web`` does it).
    """
    config = config or WebConfig()
    service = service or Service(config.service)
    sessions = WebSessions(service, idle_seconds=config.session_idle_minutes * 60)

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

    hosts = list(config.allowed_hosts)
    if config.profile == "local":
        hosts += [h.strip("[]") for h in LOOPBACK_HOSTS]
    elif not hosts:
        hosts = ["*"]
    app.add_middleware(_HttpHostCheck, allowed_hosts=hosts)

    # --- Sessions and errors ---------------------------------------------------------------------------

    @app.middleware("http")
    async def attach_session(request: Request, call_next: Any) -> Response:
        if not request.url.path.startswith("/api/"):
            return await call_next(request)
        session = sessions.get(request.cookies.get(COOKIE))
        fresh = session is None
        if session is None:
            session = sessions.open()
        request.state.web_session = session
        response = await call_next(request)
        if fresh:
            response.set_cookie(
                COOKIE,
                session.token,
                httponly=True,
                samesite="strict",
                secure=request.url.scheme == "https",
                path="/",
            )
        return response

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, error: ServiceError) -> JSONResponse:
        return _error(STATUS.get(error.code, 400), error.message, error.hint, error.code)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
        problems = []
        for item in error.errors():
            where = ".".join(str(part) for part in item.get("loc", ()) if part != "body")
            problems.append(f"{where}: {item.get('msg', 'invalid')}" if where else str(item.get("msg")))
        return _error(
            400, "; ".join(problems) or "invalid request", "check the body of the request", "invalid"
        )

    def sid(request: Request) -> str:
        session: WebSession = request.state.web_session
        return session.service_session

    # --- Information and discovery ---------------------------------------------------------------------

    @app.get("/api/info")
    def info(request: Request) -> dict[str, Any]:
        return {
            "name": "LUMOS",
            "version": lumos.__version__,
            "profile": config.profile,
            "operation": service.config.operation_enabled,
            "languages": ["en", "fr"],
        }

    @app.get("/api/element-types")
    def element_types(
        query: str = "", page: int = 1, page_size: int = Query(50, ge=1, le=500)
    ) -> Page[ElementTypeInfo]:
        return service.list_element_types(query=query, page=page, page_size=page_size)

    @app.get("/api/element-types/{type_id}/schema")
    def element_schema(type_id: str) -> dict[str, Any]:
        return service.get_element_schema(type_id)

    palette_cache: dict[str, Any] = {}

    @app.get("/api/palette")
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

    @app.get("/api/docs/search")
    def docs_search(q: str, limit: int = Query(5, ge=1, le=50)) -> list[DocHit]:
        return service.search_docs(q, limit=limit)

    # --- Contexts --------------------------------------------------------------------------------------

    @app.get("/api/contexts")
    def contexts(request: Request) -> list[ContextSummary]:
        return service.list_contexts(sid(request))

    @app.post("/api/contexts", status_code=201)
    def create_context(request: Request, body: NewContext | None = None) -> ContextSummary:
        body = body or NewContext()
        return service.create_context(sid(request), name=body.name, mode=body.mode)

    @app.post("/api/contexts/import", status_code=201)
    async def import_context(request: Request) -> ContextSummary:
        text = (await request.body()).decode("utf-8", errors="replace")
        return await asyncio.to_thread(service.import_lumos, sid(request), text)

    @app.get("/api/contexts/{cid}")
    def describe(request: Request, cid: str) -> ContextSummary:
        return service.describe_context(sid(request), cid)

    @app.delete("/api/contexts/{cid}", status_code=204)
    def close_context(request: Request, cid: str) -> Response:
        service.close_context(sid(request), cid)
        return Response(status_code=204)

    @app.get("/api/presets")
    def presets() -> list[PresetInfo]:
        return service.list_presets()

    @app.post("/api/presets/{name}/load", status_code=201)
    def load_preset(request: Request, name: str) -> ContextSummary:
        return service.load_preset(sid(request), name)

    @app.get("/api/contexts/{cid}/export.lumos")
    def export_lumos(request: Request, cid: str) -> Response:
        name = service.describe_context(sid(request), cid).name
        text = service.export_lumos(sid(request), cid)
        return _attachment(text, _filename(name, ".lumos"), "application/json")

    @app.get("/api/contexts/{cid}/export.py")
    def export_python(request: Request, cid: str) -> Response:
        name = service.describe_context(sid(request), cid).name
        text = service.export_python(sid(request), cid)
        return _attachment(text, _filename(name, ".py"), "text/x-python")

    # --- Editing ---------------------------------------------------------------------------------------

    @app.post("/api/contexts/{cid}/elements", status_code=201)
    def add_element(request: Request, cid: str, body: NewElement) -> ElementSummary:
        return service.add_element(sid(request), cid, body.type_id, name=body.name, params=body.params)

    @app.delete("/api/contexts/{cid}/elements/{path:path}", status_code=204)
    def remove_element(request: Request, cid: str, path: str) -> Response:
        service.remove_element(sid(request), cid, path)
        return Response(status_code=204)

    @app.post("/api/contexts/{cid}/connections", status_code=201)
    def connect(request: Request, cid: str, body: Connection) -> ConnectionInfo:
        return service.connect(sid(request), cid, body.source, body.target)

    @app.delete("/api/contexts/{cid}/connections", status_code=204)
    def disconnect(request: Request, cid: str, source: str, target: str) -> Response:
        service.disconnect(sid(request), cid, source, target)
        return Response(status_code=204)

    @app.patch("/api/contexts/{cid}/params")
    def set_param(request: Request, cid: str, body: ParamChange) -> ElementSummary:
        return service.set_param(sid(request), cid, body.path, body.value)

    @app.put("/api/contexts/{cid}/observation")
    def set_observation(request: Request, cid: str, body: Observation) -> ContextSummary:
        return service.set_observation(
            sid(request), cid, minimum=body.minimum, maximum=body.maximum, n_wavelengths=body.n_wavelengths
        )

    # --- Verification ----------------------------------------------------------------------------------

    @app.get("/api/contexts/{cid}/validate")
    def validate(request: Request, cid: str) -> list[Issue]:
        return service.validate_context(sid(request), cid)

    @app.get("/api/contexts/{cid}/sampling/{element:path}")
    def sampling(request: Request, cid: str, element: str) -> dict[str, str]:
        return service.explain_sampling(sid(request), cid, element)

    # --- Runs and inspection ---------------------------------------------------------------------------

    @app.post("/api/contexts/{cid}/runs", status_code=202)
    def run(request: Request, cid: str, body: RunRequest | None = None) -> JobInfo:
        body = body or RunRequest()
        outputs = body.outputs or service.default_outputs(sid(request), cid)
        if not outputs:
            raise ServiceError(
                "nothing to run", hint="add elements with outputs, or name the outputs to compute"
            )
        return service.run(sid(request), cid, outputs, archive=body.archive)

    @app.get("/api/jobs")
    def jobs(request: Request) -> list[JobInfo]:
        return service.list_jobs(sid(request))

    @app.get("/api/jobs/{job}")
    def job(request: Request, job: str) -> JobInfo:
        return service.get_job(sid(request), job)

    @app.get("/api/jobs/{job}/results/{port:path}")
    def result(request: Request, job: str, port: str) -> dict[str, Any]:
        summary = service.get_result(sid(request), job, port)
        return summary.model_dump(mode="json", exclude={"figure_png"})

    @app.get("/api/contexts/{cid}/ports/{port:path}/figure.png")
    def port_figure(request: Request, cid: str, port: str) -> Response:
        summary: PortSummary = service.inspect_port(sid(request), cid, port, figure=True)
        if summary.figure_png is None:
            raise ServiceError(
                f"{port} carries a {summary.kind}, which has no figure", hint="read its summary instead"
            )
        return Response(summary.figure_png, media_type="image/png")

    @app.get("/api/contexts/{cid}/ports/{port:path}")
    def inspect(request: Request, cid: str, port: str) -> dict[str, Any]:
        summary = service.inspect_port(sid(request), cid, port, figure=False)
        return summary.model_dump(mode="json", exclude={"figure_png"})

    @app.get("/api/contexts/{cid}/journal")
    def journal(
        request: Request, cid: str, kind: str | None = None, limit: int = Query(50, ge=1, le=10_000)
    ) -> list[dict[str, Any]]:
        return service.journal(sid(request), cid, kind=kind, limit=limit)

    # --- Catalogues and archives -----------------------------------------------------------------------

    @app.get("/api/catalogs/resolve")
    def resolve(name: str) -> dict[str, Any]:
        return service.resolve_object(name)

    @app.get("/api/archives")
    def archives(limit: int = Query(50, ge=1, le=1000)) -> list[ArchiveInfo]:
        return service.list_archives(limit=limit)

    @app.get("/api/archives/metadata")
    def archive_metadata(path: str) -> dict[str, Any]:
        return service.read_archive_metadata(path)

    # --- WebSocket -------------------------------------------------------------------------------------

    @app.websocket("/api/ws")
    async def events(websocket: WebSocket) -> None:
        if not _allowed_origin(websocket, config):
            await websocket.close(code=1008, reason="origin not allowed")
            return
        session = sessions.get(websocket.cookies.get(COOKIE))
        if session is None:
            await websocket.close(code=1008, reason="no session: call the API first")
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
