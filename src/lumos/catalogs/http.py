"""Network layer of the catalogues: TLS, timeouts, user cache and offline mode (step P9.1, REQ-CAT-5, NF-7).

Every request of a catalogue provider goes through :func:`fetch`:

- only ``https`` URLs are accepted and **TLS certificates are always verified**: there is no way to turn
  verification off (NF-7);
- every request has a **timeout** (``settings.timeout`` seconds, or the ``timeout`` argument);
- successful responses are kept in a **user cache** (``platformdirs.user_cache_dir("lumos-maxima")``) for
  ``settings.ttl`` seconds (7 days by default); the cache key is the method, URL, parameters and body;
- in **offline mode** (``settings.offline = True``) only the cache is read, whatever its age, and a missing
  entry raises :class:`OfflineError`.

The actual HTTP calls are made by a **transport**, by default :class:`RequestsTransport` (``requests``).
Tests install a fake one with :func:`use_transport`, so that they never reach the network.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "CatalogError",
    "CatalogSettings",
    "OfflineError",
    "RequestsTransport",
    "Response",
    "Transport",
    "default_transport",
    "fetch",
    "settings",
    "use_transport",
]

logger = logging.getLogger("lumos.catalogs")


class CatalogError(RuntimeError):
    """A catalogue service answered with an error or an unexpected content."""


class OfflineError(CatalogError):
    """Offline mode is on and the request is not in the cache."""


def _default_cache_dir() -> Path:
    configured = os.environ.get("LUMOS_CATALOGS_CACHE")
    if configured:
        return Path(configured)
    from platformdirs import user_cache_dir

    return Path(user_cache_dir("lumos-maxima", appauthor=False)) / "catalogs"


def _default_offline() -> bool:
    return os.environ.get("LUMOS_CATALOGS_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class CatalogSettings:
    """Settings of the catalogue requests.

    The defaults can be set with the environment variables ``LUMOS_CATALOGS_CACHE`` (directory of the cache)
    and ``LUMOS_CATALOGS_OFFLINE`` (``1`` for the offline mode), e.g. to run notebooks on recorded answers.

    Attributes
    ----------
    cache_dir
        Directory of the cache.
    ttl
        Lifetime of cache entries in seconds.
    timeout
        Default timeout of a request in seconds.
    offline
        Read the cache only.
    """

    cache_dir: Path = field(default_factory=_default_cache_dir)
    ttl: float = 7 * 86400.0
    timeout: float = 60.0
    offline: bool = field(default_factory=_default_offline)


#: Settings of the process.
settings = CatalogSettings()


@dataclass
class Response:
    """Body of an HTTP response.

    Attributes
    ----------
    status
        HTTP status code.
    content
        Body, as bytes.
    content_type
        ``Content-Type`` header.
    from_cache
        Whether the response comes from the cache.
    """

    status: int
    content: bytes
    content_type: str = ""
    from_cache: bool = False

    @property
    def text(self) -> str:
        """Body decoded as UTF-8."""
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        """Body decoded as JSON."""
        return json.loads(self.text)


class Transport(Protocol):
    """Object performing HTTP requests."""

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        data: Mapping[str, Any] | None = None,
        timeout: float,
        verify: bool,
    ) -> Response:
        """Perform a request and return its response."""
        ...


class RequestsTransport:
    """Transport based on :mod:`requests` (extra ``[catalogs]``)."""

    def __init__(self) -> None:
        self._session: Any = None

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        data: Mapping[str, Any] | None = None,
        timeout: float,
        verify: bool,
    ) -> Response:
        """Perform the request with a shared session (connection reuse)."""
        if not verify:  # pragma: no cover - fetch() never asks for it
            raise ValueError("TLS verification cannot be disabled")
        try:
            import requests
        except ImportError as error:  # pragma: no cover - optional dependency
            raise CatalogError("the catalogues need requests: pip install lumos-maxima[catalogs]") from error
        if self._session is None:
            self._session = requests.Session()
            self._session.headers["User-Agent"] = "lumos-maxima (https://github.com/VForiel/LUMOS)"
        try:
            reply = self._session.request(method, url, params=params, data=data, timeout=timeout, verify=True)
        except requests.RequestException as error:
            raise CatalogError(f"request to {url} failed: {error}") from error
        return Response(reply.status_code, reply.content, reply.headers.get("Content-Type", ""))


_DEFAULT = RequestsTransport()
_local = threading.local()


def default_transport() -> Transport:
    """Return the transport used when none is installed with :func:`use_transport`."""
    return _DEFAULT


def _transport() -> Transport:
    stack = getattr(_local, "stack", [])
    return stack[-1] if stack else _DEFAULT


@contextmanager
def use_transport(transport: Transport) -> Iterator[Transport]:
    """Use ``transport`` for the requests of the block (tests, recordings)."""
    stack = getattr(_local, "stack", None)
    if stack is None:
        stack = _local.stack = []
    stack.append(transport)
    try:
        yield transport
    finally:
        stack.pop()


def _key(method: str, url: str, params: Mapping[str, Any] | None, data: Mapping[str, Any] | None) -> str:
    payload = json.dumps(
        {"method": method.upper(), "url": url, "params": params or {}, "data": data or {}},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_cache(path: Path, *, max_age: float | None) -> Response | None:
    if not path.is_file():
        return None
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if max_age is not None and time.time() - float(entry.get("fetched_at", 0)) > max_age:
        return None
    content = base64.b64decode(entry["content"])
    return Response(int(entry["status"]), content, entry.get("content_type", ""), from_cache=True)


def _write_cache(path: Path, url: str, params: Any, data: Any, response: Response) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "url": url,
        "params": params or {},
        "data": data or {},
        "fetched_at": time.time(),
        "status": response.status,
        "content_type": response.content_type,
        "content": base64.b64encode(response.content).decode("ascii"),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(entry, default=str), encoding="utf-8")
    temporary.replace(path)


def fetch(
    url: str,
    *,
    method: str = "GET",
    params: Mapping[str, Any] | None = None,
    data: Mapping[str, Any] | None = None,
    timeout: float | None = None,
    cache: bool = True,
    verify: bool = True,
) -> Response:
    """Perform an HTTPS request through the cache.

    Parameters
    ----------
    url
        ``https`` URL.
    method
        ``"GET"`` or ``"POST"``.
    params, data
        Query parameters and form body.
    timeout
        Timeout in seconds (default ``settings.timeout``).
    cache
        Read and write the cache (offline mode always reads it).
    verify
        Must stay ``True``: TLS verification cannot be disabled.

    Raises
    ------
    OfflineError
        Offline mode is on and the request is not cached.
    CatalogError
        The service answered with an HTTP error.
    """
    if verify is not True:
        raise ValueError("TLS verification cannot be disabled in LUMOS (NF-7)")
    if not url.startswith("https://"):
        raise ValueError(f"only https URLs are accepted, got {url!r}")
    seconds = settings.timeout if timeout is None else float(timeout)
    if not seconds > 0:
        raise ValueError(f"the timeout must be positive, got {timeout!r}")
    path = Path(settings.cache_dir) / f"{_key(method, url, params, data)}.json"
    if settings.offline:
        cached = _read_cache(path, max_age=None)
        if cached is None:
            raise OfflineError(f"offline mode: {url} is not in the cache ({settings.cache_dir})")
        return cached
    if cache:
        cached = _read_cache(path, max_age=settings.ttl)
        if cached is not None:
            return cached
    logger.info("catalogue request %s %s", method, url)
    response = _transport().request(
        method.upper(), url, params=params, data=data, timeout=seconds, verify=True
    )
    if not 200 <= response.status < 300:
        excerpt = response.text[:200].strip()
        raise CatalogError(f"{url} answered HTTP {response.status}: {excerpt}")
    if cache:
        _write_cache(path, url, params, data, response)
    return response
