"""Browser sessions of the web instance: an opaque token bound to a session of the service layer.

The token travels in an HttpOnly cookie (browsers) or an ``Authorization: Bearer`` header (scripts,
``lumos login``). It is never the identifier of the service session, and it changes when the user logs in
(no session fixation).
"""

from __future__ import annotations

import contextlib
import secrets
import threading
import time
from dataclasses import dataclass, field

from lumos.service import Service, ServiceError

__all__ = ["COOKIE", "WebSession", "WebSessions"]

#: Name of the session cookie.
COOKIE = "lumos_session"


@dataclass
class WebSession:
    """A browser session.

    Attributes
    ----------
    token
        Secret value of the cookie or bearer token.
    service_session
        Session of the service layer holding the contexts and runs of this browser.
    user, role
        Authenticated user and their role; ``None`` for an anonymous visitor.
    created, last_seen, authenticated
        Monotonic times of creation, of the last request and of the login.
    link
        Secret of the connector link of the session (``/mcp/<link>``), or ``None``.
    """

    token: str
    service_session: str
    user: str | None = None
    role: str | None = None
    created: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    authenticated: float | None = None
    link: str | None = None


class WebSessions:
    """Sessions of the browsers, expiring after an idle time (and a lifetime once authenticated).

    Parameters
    ----------
    service
        Service layer; each browser session owns one of its sessions (origin ``web``).
    idle_seconds
        A session unused for this long is closed with its contexts.
    lifetime_seconds
        An authenticated session is closed this long after the login, even when used.
    """

    def __init__(
        self, service: Service, *, idle_seconds: float, lifetime_seconds: float = float("inf")
    ) -> None:
        self.service = service
        self.idle_seconds = idle_seconds
        self.lifetime_seconds = lifetime_seconds
        self._sessions: dict[str, WebSession] = {}
        self._links: dict[str, str] = {}  # secret of a connector link -> token of its session
        self._lock = threading.Lock()

    def get(self, token: str | None) -> WebSession | None:
        """Return the live session of ``token`` (and mark it as used), or ``None``."""
        self.expire()
        if not token:
            return None
        with self._lock:
            session = self._sessions.get(token)
            if session is not None:
                session.last_seen = time.monotonic()
            return session

    def by_link(self, link: str | None) -> WebSession | None:
        """Return the live session of a connector link (and mark it as used), or ``None``."""
        self.expire()
        if not link:
            return None
        with self._lock:
            token = self._links.get(link)
        return self.get(token)

    def create_link(self, session: WebSession) -> str:
        """Give ``session`` a new connector link (the previous one stops working); return its secret."""
        with self._lock:
            if session.link is not None:
                self._links.pop(session.link, None)
            session.link = secrets.token_urlsafe(32)
            self._links[session.link] = session.token
            return session.link

    def revoke_link(self, session: WebSession) -> None:
        """Remove the connector link of ``session``."""
        with self._lock:
            if session.link is not None:
                self._links.pop(session.link, None)
            session.link = None

    def open(self) -> WebSession:
        """Open a new anonymous session."""
        session = WebSession(
            token=secrets.token_urlsafe(32), service_session=self.service.create_session(origin="web").id
        )
        with self._lock:
            self._sessions[session.token] = session
        return session

    def login(self, session: WebSession, user: str, role: str) -> WebSession:
        """Give ``session`` to ``user``: same contexts, new token (the old one stops working)."""
        renewed = WebSession(
            token=secrets.token_urlsafe(32),
            service_session=session.service_session,
            user=user,
            role=role,
            authenticated=time.monotonic(),
            link=session.link,
        )
        with self._lock:
            self._sessions.pop(session.token, None)
            self._sessions[renewed.token] = renewed
            if renewed.link is not None:
                self._links[renewed.link] = renewed.token
        return renewed

    def close(self, token: str) -> None:
        """Close a session and its service session."""
        with self._lock:
            session = self._sessions.pop(token, None)
            if session is not None and session.link is not None:
                self._links.pop(session.link, None)
        if session is not None:
            with contextlib.suppress(ServiceError):  # already closed by the service
                self.service.close_session(session.service_session)

    def expire(self) -> None:
        """Close the sessions idle for too long, and the authenticated ones past their lifetime."""
        now = time.monotonic()
        with self._lock:
            stale = [
                token
                for token, s in self._sessions.items()
                if s.last_seen < now - self.idle_seconds
                or (s.authenticated is not None and s.authenticated < now - self.lifetime_seconds)
            ]
        for token in stale:
            self.close(token)

    def __len__(self) -> int:
        """Return the number of open sessions."""
        return len(self._sessions)
