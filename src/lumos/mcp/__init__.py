"""MCP server of LUMOS, for AI assistants (phase 13, REQ-MCP-1, ADR-0016).

``lumos mcp`` serves it over stdio; :func:`create_server` builds it for other transports and for tests.
It needs the extra ``mcp`` (``pip install "lumos-maxima[mcp]"``).
"""

from lumos.mcp.server import INSTRUCTIONS, create_server

__all__ = ["INSTRUCTIONS", "create_server"]
