"""The MCP server inside the web instance (step P14.9, REQ-MCP-1, REQ-WEB-12, ADR-0019).

``/mcp`` serves the MCP server over streamable HTTP with the authentication of the interface: the cookie or
the bearer token of a session. Each session of the interface gets its own MCP server working in the same
session of the service, so that an assistant (the assistant panel of the browser, or any MCP client given a
token) sees and edits the contexts of the user. Viewers are refused; only operators get the tools that
drive equipment.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable
from typing import Any

import anyio
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse

from lumos.service import Service
from lumos.web.auth import role_allows
from lumos.web.sessions import COOKIE, WebSession, WebSessions

__all__ = ["McpGateway"]

PATH = "/mcp"


def _bearer(connection: HTTPConnection) -> str | None:
    scheme, _, token = connection.headers.get("authorization", "").partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


class McpGateway:
    """ASGI application answering ``/mcp`` with the MCP server of the session of the request.

    Parameters
    ----------
    service, sessions
        Service layer and sessions of the interface.
    role_of
        Role of a session (``None`` when it must log in).
    anonymous_role
        Role of the requests without a session.
    enabled
        Whether the assistant is offered (``WebConfig.assistant``).
    """

    def __init__(
        self,
        *,
        service: Service,
        sessions: WebSessions,
        role_of: Callable[[WebSession], str | None],
        anonymous_role: str | None,
        enabled: bool,
    ) -> None:
        self.service = service
        self.sessions = sessions
        self.role_of = role_of
        self.anonymous_role = anonymous_role
        self.enabled = enabled
        self._servers: dict[tuple[str, bool], Any] = {}
        self._tasks: anyio.abc.TaskGroup | None = None
        self._lock = anyio.Lock()

    @contextlib.asynccontextmanager
    async def running(self) -> AsyncIterator[None]:
        """Keep the MCP servers of the sessions alive (inside the lifespan of the web application)."""
        async with anyio.create_task_group() as tasks:
            self._tasks = tasks
            try:
                yield
            finally:
                tasks.cancel_scope.cancel()
                self._tasks = None
                self._servers.clear()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Authenticate the request and pass it to the MCP server of its session."""
        connection = HTTPConnection(scope)
        if not self.enabled:
            await self._refuse(scope, receive, send, 404, "the assistant is turned off on this instance", "")
            return
        link = scope.get("path_params", {}).get("link")
        if link is not None:
            # a connector link (/mcp/<secret>): the secret is the credential of the session
            session = self.sessions.by_link(link)
            if session is None:
                hint = "create a new connector link in the assistant panel of the interface"
                await self._refuse(scope, receive, send, 404, "unknown or revoked connector link", hint)
                return
            scope = {**scope, "path": PATH, "raw_path": PATH.encode()}
        else:
            session = self.sessions.get(_bearer(connection) or connection.cookies.get(COOKIE))
        role = self.role_of(session) if session is not None else self.anonymous_role
        if role is not None and not role_allows(role, "simulator"):
            hint = "this needs the role simulator"
            await self._refuse(scope, receive, send, 403, f"the role {role} cannot use the assistant", hint)
            return
        if session is None or role is None:
            hint = "open the interface first, or log in (lumos login) and send the token as a bearer"
            await self._refuse(
                scope, receive, send, 401, "the MCP server needs a session of the instance", hint
            )
            return
        server = await self._server(session.service_session, role == "operator")
        await server(scope, receive, send)

    @staticmethod
    async def _refuse(scope: Any, receive: Any, send: Any, status: int, message: str, hint: str) -> None:
        code = {401: "unauthorized", 403: "forbidden", 404: "not_found"}[status]
        response = JSONResponse({"error": message, "hint": hint, "code": code}, status_code=status)
        await response(scope, receive, send)

    async def _server(self, service_session: str, can_operate: bool) -> Any:
        key = (service_session, can_operate)
        async with self._lock:
            if key in self._servers:
                return self._servers[key]
            if self._tasks is None:
                raise RuntimeError("the MCP gateway runs inside the lifespan of the web application")
            from mcp.server.transport_security import TransportSecuritySettings

            from lumos.mcp import create_server

            server = create_server(self.service, session=service_session, can_operate=can_operate)
            # host names and origins are checked by the web application itself
            security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
            app = server.streamable_http_app(streamable_http_path=PATH, transport_security=security)
            ready = anyio.Event()

            async def keep(task_status: Any = anyio.TASK_STATUS_IGNORED) -> None:
                async with app.router.lifespan_context(app):
                    ready.set()
                    task_status.started()
                    await anyio.sleep_forever()

            await self._tasks.start(keep)
            await ready.wait()
            self._servers[key] = app
            return app
