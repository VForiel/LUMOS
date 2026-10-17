"""Rate limits, audit journal and transport checks of a web instance (step P14.2, REQ-WEB-9, ADR-0019)."""

from __future__ import annotations

import collections
import json
import logging
import math
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from starlette.requests import HTTPConnection

__all__ = ["AuditLog", "RateLimiter", "client_address", "is_secure"]

audit_logger = logging.getLogger("lumos.audit")


class RateLimiter:
    """At most ``limit`` events per key in any window of ``window`` seconds (sliding window).

    Parameters
    ----------
    limit
        Events allowed per window; ``None`` allows everything.
    window
        Length of the window in seconds.
    """

    def __init__(self, limit: int | None, window: float = 60.0) -> None:
        self.limit = limit
        self.window = window
        self._events: dict[str, collections.deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> float:
        """Record an event of ``key``; return 0 if allowed, else the seconds to wait before the next one."""
        if self.limit is None:
            return 0.0
        now = time.monotonic()
        with self._lock:
            if len(self._events) > 50_000:  # forget idle keys
                self._events = {k: q for k, q in self._events.items() if q and q[-1] > now - self.window}
            events = self._events.setdefault(key, collections.deque())
            while events and events[0] <= now - self.window:
                events.popleft()
            if len(events) >= self.limit:
                return max(events[0] + self.window - now, 0.001)
            events.append(now)
            return 0.0


def retry_after(seconds: float) -> str:
    """Value of a ``Retry-After`` header (whole seconds, at least 1)."""
    return str(max(1, math.ceil(seconds)))


class AuditLog:
    """Journal of the security events: one JSON object per line, also sent to the logger ``lumos.audit``.

    Entries never hold passwords, signatures or session tokens.

    Parameters
    ----------
    path
        JSON-lines file, or ``None`` for the logger only.
    """

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self, event: str, outcome: str, *, ip: str | None, user: str | None = None, **details: Any
    ) -> None:
        """Append an event (``login``, ``logout``, ``forbidden``, ``action``...) and its outcome."""
        entry = {
            "time": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "event": event,
            "outcome": outcome,
            "ip": ip,
            "user": user,
            **details,
        }
        line = json.dumps(entry, ensure_ascii=False, default=str)
        audit_logger.info(line)
        if self.path is not None:
            with self._lock, self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")


def client_address(connection: HTTPConnection, *, trust_proxy: bool) -> str:
    """Address of the client: the first ``X-Forwarded-For`` entry behind a trusted proxy, else the peer."""
    if trust_proxy:
        forwarded = connection.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return connection.client.host if connection.client else "unknown"


def is_secure(connection: HTTPConnection, *, trust_proxy: bool) -> bool:
    """Whether the request came over TLS (directly, or through a trusted proxy terminating TLS)."""
    if connection.url.scheme in ("https", "wss"):
        return True
    return trust_proxy and connection.headers.get("x-forwarded-proto", "").lower() in ("https", "wss")
