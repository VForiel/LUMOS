"""MCP server of LUMOS, for AI assistants (phase 13, REQ-MCP-1, ADR-0016).

``lumos mcp`` serves it over stdio, ``lumos mcp --transport http`` over streamable HTTP; :func:`create_server`
builds it and :func:`create_http_app` gives its ASGI application, to mount in a web server.
It needs the extra ``mcp`` (``pip install "lumos-maxima[mcp]"``).
"""

from lumos.mcp.server import INSTRUCTIONS, create_http_app, create_server

__all__ = ["INSTRUCTIONS", "create_http_app", "create_server"]
