"""Web interface of LUMOS: the FastAPI backend serving the visual editor (phase 14, ADR-0018).

- :func:`create_app`: the ASGI application, a thin adapter of the service layer (:mod:`lumos.service`);
- :class:`WebConfig`: profile, hosts and quotas of an instance.

Serve it with ``lumos web`` (``localhost:8000`` by default). Needs the extra ``web``.
"""

from lumos.web.app import create_app
from lumos.web.config import WebConfig

__all__ = ["WebConfig", "create_app"]
