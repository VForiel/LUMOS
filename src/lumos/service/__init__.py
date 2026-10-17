"""Service layer of LUMOS, shared by the web backend and the MCP server (phase 13, ADR-0015).

- :class:`Service`: sessions, contexts, editing, runs, inspection, catalogues, archives, operation;
- :mod:`~lumos.service.models`: typed messages (Pydantic), quantities written as text (``"0.5 mm"``);
- :class:`ServiceError`: errors with a cause and a suggested fix.
"""

from lumos.service.errors import ServiceError
from lumos.service.models import ServiceConfig
from lumos.service.service import Service

__all__ = ["Service", "ServiceConfig", "ServiceError"]
