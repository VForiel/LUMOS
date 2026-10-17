"""Resources of the MCP server: documentation, element schemas, presets, contexts, journals, archives.

Step P13.3 (REQ-MCP-3). Resources are read-only views that a client can attach to the conversation; they
are served from the same service layer as the tools.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lumos.service import ServiceError
from lumos.service import docs as service_docs

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer

    from lumos.mcp.server import LumosTools

__all__ = ["register_resources"]

_JSON = "application/json"


def _json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude={"figure_png"})
    elif isinstance(value, list):
        value = [v.model_dump(mode="json") if hasattr(v, "model_dump") else v for v in value]
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def register_resources(server: MCPServer, state: LumosTools) -> None:
    """Add the resources of LUMOS to ``server``."""
    service = state.service
    session = state.session

    # --- Documentation -------------------------------------------------------------------------------------

    @server.resource("lumos://docs", name="documentation", mime_type=_JSON)
    def documentation() -> str:
        """Index of the documentation pages of LUMOS (guides, physics, element types) with their URIs."""
        return _json(
            [
                {"path": path, "title": title, "uri": f"lumos://docs/{path}"}
                for path, (title, _) in service_docs.pages().items()
            ]
        )

    @server.resource("lumos://docs/{+path}", name="documentation page", mime_type="text/markdown")
    def documentation_page(path: str) -> str:
        """Read a page of the documentation of LUMOS (Markdown); ``path`` comes from lumos://docs."""
        pages = service_docs.pages()
        if path not in pages:
            raise ServiceError(f"no documentation page {path!r}", hint="see lumos://docs", code="not_found")
        return pages[path][1]

    # --- Element types and presets -------------------------------------------------------------------------

    @server.resource("lumos://elements", name="element types", mime_type=_JSON)
    def element_types() -> str:
        """Every element type of LUMOS with its summary and ports."""
        listed = service.list_element_types(page_size=10_000)
        return _json(listed.items)

    @server.resource("lumos://elements/{type_id}", name="element schema", mime_type=_JSON)
    def element_schema(type_id: str) -> str:
        """JSON Schema of an element type: parameters with their units and bounds, ports."""
        return _json(service.get_element_schema(type_id))

    @server.resource("lumos://presets", name="presets", mime_type=_JSON)
    def presets() -> str:
        """Preset contexts (ready-made instruments and benches) with their descriptions and outputs."""
        return _json(service.list_presets())

    @server.resource("lumos://presets/{name}", name="preset", mime_type=_JSON)
    def preset(name: str) -> str:
        """Read the ``.lumos`` document of a preset context."""
        known = [p.name for p in service.list_presets()]
        if name not in known:
            raise ServiceError(
                f"unknown preset {name!r}", hint=f"presets: {', '.join(known)}", code="not_found"
            )
        return (service.presets_dir() / f"{name}.lumos").read_text(encoding="utf-8")

    # --- Contexts and journals -----------------------------------------------------------------------------

    @server.resource("lumos://context/current", name="current context", mime_type=_JSON)
    def current_context() -> str:
        """Describe the current context: elements with their values and units, connections, mode, band."""
        return _json(service.describe_context(session, state.context(None)))

    @server.resource(
        "lumos://context/current/journal", name="journal of the current context", mime_type=_JSON
    )
    def current_journal() -> str:
        """Last entries of the journal of the current context (runs, actions, logs, with their origin)."""
        return _json(service.journal(session, state.context(None), limit=200))

    @server.resource("lumos://contexts/{id}", name="context", mime_type=_JSON)
    def context(id: str) -> str:
        """Describe a context of the conversation, by identifier."""
        return _json(service.describe_context(session, id))

    @server.resource("lumos://contexts/{id}/journal", name="journal", mime_type=_JSON)
    def journal(id: str) -> str:
        """Last entries of the journal of a context (runs, actions, logs, with their origin)."""
        return _json(service.journal(session, id, limit=200))

    # --- Archives ------------------------------------------------------------------------------------------

    @server.resource("lumos://archives", name="archives", mime_type=_JSON)
    def archives() -> str:
        """List the archived runs, most recent first, with the URIs of their metadata."""
        root = service.archive_root().resolve()
        listed = []
        for info in service.list_archives():
            relative = Path(info.path).resolve().relative_to(root).as_posix()
            listed.append({**info.model_dump(mode="json"), "uri": f"lumos://archives/{relative}"})
        return _json(listed)

    @server.resource("lumos://archives/{+path}", name="archive metadata", mime_type=_JSON)
    def archive_metadata(path: str) -> str:
        """Metadata of an archived run: parameters, versions, origin, outputs."""
        return _json(service.read_archive_metadata(str(service.archive_root() / path)))
