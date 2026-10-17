"""Browser sessions of the web instance: an opaque cookie bound to a session of the service layer."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field

from lumos.service import Service

__all__ = ["COOKIE", "WebSession", "WebSessions"]

#: Name of the session cookie.
COOKIE = "lumos_session"


@dataclass
class WebSession:
    """A browser session.

    Attributes
    ----------
    token
        Secret value of the cookie (never the identifier of the service session).
    service_session
        Session of the service layer holding the contexts and runs of this browser.
    created, last_seen
        Monotonic times of creation and of the last request.
    """

    token: str
    service_session: str
    created: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)


class WebSessions:
    """Sessions of the browsers, expiring after an idle time.

    Parameters
    ----------
    service
        Service layer; each browser session owns one of its sessions (origin ``web``).
    idle_seconds
        A session unused for this long is closed with its contexts.
    """

    def __init__(self, service: Service, *, idle_seconds: float) -> None:
        self.service = service
        self.idle_seconds = idle_seconds
        self._sessions: dict[str, WebSession] = {}
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

    def open(self) -> WebSession:
        """Open a new session."""
        session = WebSession(
            token=secrets.token_urlsafe(32), service_session=self.service.create_session(origin="web").id
        )
        with self._lock:
            self._sessions[session.token] = session
        return session

    def close(self, token: str) -> None:
        """Close a session and its service session."""
        with self._lock:
            session = self._sessions.pop(token, None)
        if session is not None:
            self.service.close_session(session.service_session)

    def expire(self) -> None:
        """Close the sessions idle for too long."""
        limit = time.monotonic() - self.idle_seconds
        with self._lock:
            stale = [token for token, s in self._sessions.items() if s.last_seen < limit]
        for token in stale:
            self.close(token)

    def __len__(self) -> int:
        """Return the number of open sessions."""
        return len(self._sessions)
