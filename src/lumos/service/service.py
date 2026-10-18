"""The service layer of LUMOS (step P13.1, REQ-SVC-1, ADR-0015).

:class:`Service` exposes the high-level operations shared by the web backend and the MCP server: sessions
and contexts, editing of the graph, validation, asynchronous runs with progress, inspection of ports,
catalogues, archives and the operation of equipment. Its inputs and outputs are typed messages
(:mod:`lumos.service.models`) where quantities are written as text (``"0.5 mm"``); its errors
(:class:`ServiceError`) say what went wrong and how to fix it. The adapters only translate between their
protocol and these calls, so that the same request gives the same result in Python, on the web and through an
AI assistant.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import contextvars
import difflib
import json
import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astropy import units as u

import lumos.elements
import lumos.io
from lumos.core.action import actions_of
from lumos.core.context import Context
from lumos.core.driver import DriverError, LimitError
from lumos.core.element import Element
from lumos.core.graph import GraphError, Result
from lumos.core.group import GROUP_TYPE, ElementGroup, group_type_of
from lumos.core.param import Param
from lumos.core.port import PortCompatibilityError
from lumos.core.registry import UnknownTypeError, registry
from lumos.io.lumos_format import entry_of, from_document
from lumos.service import docs, studies
from lumos.service import editing as edits
from lumos.service.errors import ServiceError
from lumos.service.models import (
    ActionResult,
    ArchiveInfo,
    ConnectionInfo,
    ContextSummary,
    DeviceModel,
    DocHit,
    ElementSummary,
    ElementTypeInfo,
    FeedbackInfo,
    GroupInfo,
    Issue,
    JobInfo,
    Page,
    PortFile,
    PortSummary,
    PresetInfo,
    ServiceConfig,
    SessionInfo,
    StudyPoint,
    StudyResult,
)
from lumos.service.summaries import summarize
from lumos.service.values import format_value, parse_quantity, parse_value

__all__ = ["Service", "acting_as"]

logger = logging.getLogger("lumos.service")


#: Formats of the files of a port (``port_file``): figure, PltEdit figure, data, JSON form of the signal.
FILE_FORMATS = ("png", "plt", "npz", "json")


class _Cancelled(Exception):
    """A run stopped at the request of the user."""


@dataclass
class _Job:
    info: JobInfo
    session: str
    result: Result | None = None
    study: StudyResult | None = None
    future: concurrent.futures.Future[Any] | None = None
    started: float = 0.0
    cancelled: bool = False


#: Origin of the actions of the current call when it differs from the origin of the session (see acting_as).
_ACTING_AS: contextvars.ContextVar[str | None] = contextvars.ContextVar("lumos_acting_as", default=None)


@contextlib.contextmanager
def acting_as(origin: str) -> Iterator[None]:
    """Journal the actions of the calls made in this block with ``origin``.

    The web instance uses it for an assistant working in the session of a browser (origin ``"mcp"``).
    """
    token = _ACTING_AS.set(origin)
    try:
        yield
    finally:
        _ACTING_AS.reset(token)


#: Edits kept for undo, per context.
HISTORY_DEPTH = 100


@dataclass
class _History:
    undo: list[dict[str, Any]] = field(default_factory=list)
    redo: list[dict[str, Any]] = field(default_factory=list)
    depth: int = 0  # nesting of edits: only the outermost one is recorded


@dataclass
class _Session:
    info: SessionInfo
    contexts: dict[str, Context] = field(default_factory=dict)
    jobs: dict[str, _Job] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)


#: Working copies of a field held at once by a computation, in the memory estimate of the quotas.
MEMORY_COPIES = 4


def _grid_sides(element: Any) -> list[tuple[str, int]]:
    """Grid sides (in pixels) set by an element: ``n_pixels`` settings and grid settings."""
    sides = []
    for name, declared in type(element).declared().items():
        if declared.kind != "setting":
            continue
        value = getattr(element, name)
        if name == "n_pixels" and isinstance(value, int):
            sides.append((name, value))
        elif hasattr(value, "nx") and hasattr(value, "ny"):
            sides.append((name, max(int(value.nx), int(value.ny))))
    return sides


def _hint_for(error: BaseException) -> str:
    text = str(error)
    if "not connected" in text:
        return "connect an output to this input with connect(source='id.port', target='id.port')"
    if isinstance(error, PortCompatibilityError):
        return "connect ports of compatible signal types (see get_element_schema for the ports of each type)"
    if isinstance(error, LimitError):
        return "stay within the limits of the device (see device_status)"
    if isinstance(error, u.UnitsError):
        return "give a quantity in a unit of the right dimension, e.g. '5 mm'"
    if "band" in text:
        return "set the observation band with set_observation(minimum='1 um', maximum='2 um')"
    if isinstance(error, DriverError):
        return "check the connection and the state of the device (device_status), then retry"
    return "check the parameters of the elements involved (describe_context, validate_context)"


class Service:
    """Sessions, contexts and operations of LUMOS for the web interface and the MCP server.

    Parameters
    ----------
    config
        Settings of the deployment (operation, quotas, directories).
    """

    def __init__(self, config: ServiceConfig | None = None) -> None:
        self.config = config or ServiceConfig()
        self._sessions: dict[str, _Session] = {}
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_workers, thread_name_prefix="lumos-run"
        )
        self._devices: list[Any] = []
        self._histories: dict[str, _History] = {}

    # --- Sessions ------------------------------------------------------------------------------------------

    @staticmethod
    def _origin(state: _Session) -> str:
        return _ACTING_AS.get() or state.info.origin

    def create_session(self, *, origin: str = "python", user: str | None = None) -> SessionInfo:
        """Open a session; its actions are journaled with ``origin`` (``python``, ``web`` or ``mcp``)."""
        if origin not in ("python", "web", "mcp"):
            raise ServiceError(f"unknown origin {origin!r}", hint="use 'python', 'web' or 'mcp'")
        info = SessionInfo(id=uuid.uuid4().hex[:12], origin=origin, user=user)
        self._sessions[info.id] = _Session(info)
        return info

    def close_session(self, session: str) -> None:
        """Close a session, its contexts and the drivers they hold."""
        state = self._session(session)
        for ctx in state.contexts.values():
            for driver in ctx.drivers().values():
                try:
                    driver.disconnect()
                except DriverError:
                    logger.exception("closing a driver failed")
        del self._sessions[session]

    def _session(self, session: str) -> _Session:
        try:
            return self._sessions[session]
        except KeyError:
            raise ServiceError(
                f"unknown session {session!r}", hint="open one with create_session()", code="not_found"
            ) from None

    def _context(self, session: str, context: str) -> Context:
        state = self._session(session)
        try:
            return state.contexts[context]
        except KeyError:
            known = ", ".join(state.contexts) or "none"
            raise ServiceError(
                f"unknown context {context!r}", hint=f"known contexts: {known}", code="not_found"
            ) from None

    def _element(self, ctx: Context | ElementGroup, path: str) -> Any:
        try:
            return ctx[path]
        except (KeyError, GraphError):
            walk = getattr(ctx, "walk", None)
            known = (
                ", ".join(path for path, _ in walk()) if walk else ", ".join(ctx.id_of(e) for e in ctx.graph)
            )
            raise ServiceError(f"no element {path!r}", hint=f"elements: {known}", code="not_found") from None

    def _register(self, session: str, ctx: Context) -> ContextSummary:
        state = self._session(session)
        if len(state.contexts) >= self.config.max_contexts_per_session:
            raise ServiceError(
                f"quota of {self.config.max_contexts_per_session} contexts per session reached",
                hint="close contexts you no longer need with close_context()",
                code="quota",
            )
        identifier = uuid.uuid4().hex[:8]
        state.contexts[identifier] = ctx
        return self._summary(identifier, ctx)

    # --- Discovery -----------------------------------------------------------------------------------------

    def list_element_types(
        self, *, query: str = "", page: int = 1, page_size: int = 50
    ) -> Page[ElementTypeInfo]:
        """Element types, filtered by words of ``query``, a page at a time."""
        import lumos.processing
        import lumos.symbolic.source  # noqa: F401 - registers the symbolic source

        items = []
        words = query.lower().split()
        for type_id in sorted(registry):
            cls = registry.get(type_id)
            summary = (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else ""
            if words and not all(w in (type_id + " " + summary).lower() for w in words):
                continue
            ports = cls.ports()
            items.append(
                ElementTypeInfo(
                    type_id=type_id,
                    title=cls.__name__,
                    summary=summary,
                    inputs=[n for n, p in ports.items() if p.direction == "in"],
                    outputs=[n for n, p in ports.items() if p.direction == "out"],
                )
            )
        start = (max(page, 1) - 1) * page_size
        chunk = items[start : start + page_size]
        next_page = page + 1 if start + page_size < len(items) else None
        return Page[ElementTypeInfo](
            items=chunk, total=len(items), page=page, page_size=page_size, next_page=next_page
        )

    def _type(self, type_id: str) -> type[Element]:
        try:
            return registry.get(type_id)
        except UnknownTypeError:
            close = difflib.get_close_matches(type_id, list(registry), n=3, cutoff=0.4)
            hint = f"did you mean {', '.join(close)}?" if close else "list them with list_element_types()"
            raise ServiceError(f"unknown element type {type_id!r}", hint=hint, code="not_found") from None

    def get_element_schema(self, type_id: str) -> dict[str, Any]:
        """JSON Schema of an element type (parameters with units and bounds, ports)."""
        return self._type(type_id).schema()

    def search_docs(self, query: str, *, limit: int = 5) -> list[DocHit]:
        """Documentation pages matching ``query``."""
        return docs.search(query, limit=limit)

    # --- Contexts ------------------------------------------------------------------------------------------

    def create_context(self, session: str, *, name: str = "", mode: str = "simulation") -> ContextSummary:
        """Create an empty context."""
        try:
            ctx = Context(mode=mode, name=name or "context")  # type: ignore[arg-type]
        except ValueError as error:
            raise ServiceError(str(error), hint="mode is 'simulation' or 'operation'") from error
        return self._register(session, ctx)

    def list_contexts(self, session: str) -> list[ContextSummary]:
        """Contexts of the session."""
        state = self._session(session)
        return [self._summary(identifier, ctx) for identifier, ctx in state.contexts.items()]

    def close_context(self, session: str, context: str) -> None:
        """Forget a context."""
        self._context(session, context)
        del self._session(session).contexts[context]
        self._histories.pop(context, None)

    def presets_dir(self) -> Path:
        """Directory of the preset ``.lumos`` files."""
        if self.config.presets_dir:
            return Path(self.config.presets_dir)
        packaged = Path(lumos.__file__).resolve().parent / "presets"
        if packaged.is_dir():
            return packaged
        return Path(lumos.__file__).resolve().parents[2] / "examples" / "presets"

    def list_presets(self) -> list[PresetInfo]:
        """Preset contexts."""
        found = []
        for path in sorted(self.presets_dir().glob("*.lumos")):
            document = json.loads(path.read_text(encoding="utf-8"))
            meta = document.get("meta", {})
            types = list(dict.fromkeys(str(entry.get("type", "")) for entry in document.get("elements", [])))
            found.append(
                PresetInfo(
                    name=path.stem,
                    title=str(meta.get("name") or path.stem.replace("_", " ").capitalize()),
                    description=meta.get("description", ""),
                    outputs=meta.get("outputs", []),
                    elements=types,
                )
            )
        return found

    def load_preset(self, session: str, name: str) -> ContextSummary:
        """Load a preset into a new context."""
        path = self.presets_dir() / f"{name}.lumos"
        if not path.is_file():
            known = ", ".join(p.name for p in self.list_presets())
            raise ServiceError(f"unknown preset {name!r}", hint=f"presets: {known}", code="not_found")
        return self._register(session, lumos.io.load(path))

    def import_lumos(self, session: str, document: str | Mapping[str, Any]) -> ContextSummary:
        """Create a context from the text (or the mapping) of a ``.lumos`` document."""
        try:
            data = json.loads(document) if isinstance(document, str) else dict(document)
            ctx = from_document(data)
        except (ValueError, KeyError, TypeError) as error:
            raise ServiceError(
                f"invalid .lumos document: {error}", hint="export a valid one with export_lumos"
            ) from error
        if not isinstance(ctx, Context):
            raise ServiceError(
                "the document describes an element, not a context", hint="give a document of kind 'context'"
            )
        return self._register(session, ctx)

    def export_lumos(self, session: str, context: str) -> str:
        """Return the ``.lumos`` document of a context (JSON text)."""
        ctx = self._context(session, context)
        return json.dumps(lumos.io.to_document(ctx), indent=2, ensure_ascii=False)

    def export_python(self, session: str, context: str) -> str:
        """Return the equivalent Python script of a context."""
        return self._context(session, context).to_python()

    def describe_context(self, session: str, context: str) -> ContextSummary:
        """Elements, values, connections, mode and band of a context."""
        return self._summary(context, self._context(session, context))

    def _summary(self, identifier: str, ctx: Context) -> ContextSummary:
        elements = [self._element_summary(path, element) for path, element in ctx.walk()]
        connections = [
            ConnectionInfo(
                source=f"{ctx.id_of(c.source.owner)}.{c.source.name}",
                target=f"{ctx.id_of(c.target.owner)}.{c.target.name}",
            )
            for c in ctx._graph.connections
        ]
        band = None
        if ctx.observation.band is not None:
            band = [format_value(q) for q in ctx.observation.band]
        return ContextSummary(
            id=identifier,
            name=ctx.name or "",
            description=str(ctx.meta.get("description", "")),
            mode=ctx.mode,
            twin=ctx.twin,
            seed=ctx.seed,
            band=band,
            n_wavelengths=int(ctx.observation.n_wavelengths),
            elements=elements,
            connections=connections,
            feedbacks=[FeedbackInfo(source=f.source, target=f.target, mode=f.mode) for f in ctx.feedbacks],
            outputs=list(ctx.meta.get("outputs", [])),
            groups=self._groups(ctx),
            ui=json.loads(json.dumps(ctx.ui, default=str)),
            can_undo=bool(self._histories.get(identifier, _History()).undo),
            can_redo=bool(self._histories.get(identifier, _History()).redo),
        )

    def _groups(self, ctx: Context) -> dict[str, GroupInfo]:
        groups = {}
        for path, element in ctx.walk():
            if not isinstance(element, ElementGroup):
                continue
            document = entry_of(element)
            groups[path] = GroupInfo(
                connections=[
                    ConnectionInfo(source=c["from"], target=c["to"]) for c in document.get("connections", [])
                ],
                exposed=dict(document.get("exposed_ports", {})),
            )
        return groups

    def _element_summary(self, path: str, element: Any) -> ElementSummary:
        cls = type(element)
        groups: dict[str, dict[str, Any]] = {"param": {}, "setting": {}, "state": {}}
        for name, declared in cls.declared().items():
            value = getattr(element, name)
            unit = getattr(declared, "unit", None)
            if declared.kind == "setting" and not isinstance(
                value, u.Quantity | bool | int | float | str | None
            ):
                groups["setting"][name] = declared.encode(value)  # JSON form, accepted back by set_param
            else:
                groups[declared.kind][name] = format_value(value, unit)
        ports = element.instance_ports()
        if isinstance(element, ElementGroup):
            type_id = group_type_of(element) or GROUP_TYPE
        else:
            try:
                type_id = registry.type_id(cls)
            except UnknownTypeError:
                type_id = cls.__name__
        return ElementSummary(
            id=path,
            type_id=type_id,
            params=groups["param"],
            settings=groups["setting"],
            state=groups["state"],
            inputs=[n for n, p in ports.items() if p.direction == "in"],
            outputs=[n for n, p in ports.items() if p.direction == "out"],
            actions=sorted(actions_of(cls)),
            free=self._free_of(element),
        )

    @staticmethod
    def _free_of(element: Any) -> dict[str, dict[str, str] | None]:
        is_free = getattr(element, "is_free", None)
        if is_free is None or isinstance(element, ElementGroup):
            return {}
        found: dict[str, dict[str, str] | None] = {}
        for name, declared in type(element).declared().items():
            if isinstance(declared, Param) and is_free(name):
                found[name] = studies.prior_summary(element.prior(name), declared.unit)
        return found

    # --- History -------------------------------------------------------------------------------------------

    @contextlib.contextmanager
    def _editing(self, session: str, context: str) -> Iterator[Context]:
        """Edit a context; a successful outermost edit is recorded for undo."""
        ctx = self._context(session, context)
        history = self._histories.setdefault(context, _History())
        before = lumos.io.to_document(ctx) if history.depth == 0 else None
        history.depth += 1
        try:
            yield ctx
        finally:
            history.depth -= 1
        if before is not None:
            history.undo.append(before)
            del history.undo[:-HISTORY_DEPTH]
            history.redo.clear()

    def _rebuild(self, session: str, context: str, document: Mapping[str, Any]) -> ContextSummary:
        """Replace a context by the one ``document`` describes (same identifier)."""
        try:
            ctx = from_document(document)
        except (ValueError, KeyError, TypeError, GraphError, PortCompatibilityError) as error:
            raise ServiceError(
                f"the edit gives an invalid context: {error}", hint="check the elements involved"
            ) from error
        self._session(session).contexts[context] = ctx
        return self._summary(context, ctx)

    def _edit_document(
        self, session: str, context: str, change: Callable[[dict[str, Any]], dict[str, Any]]
    ) -> ContextSummary:
        with self._editing(session, context) as ctx:
            if ctx.drivers():
                raise ServiceError(
                    "this edit is not available while devices are attached",
                    hint="edit the context before associating the devices",
                )
            return self._rebuild(session, context, change(lumos.io.to_document(ctx)))

    def _step(self, session: str, context: str, *, forward: bool) -> ContextSummary:
        ctx = self._context(session, context)
        history = self._histories.setdefault(context, _History())
        source, target = (history.redo, history.undo) if forward else (history.undo, history.redo)
        if not source:
            word = "redo" if forward else "undo"
            raise ServiceError(
                f"nothing to {word}", hint="edit the context first" if not forward else "undo an edit first"
            )
        if ctx.drivers():
            raise ServiceError(
                "undo and redo are not available while devices are attached",
                hint="detach the devices, or undo before associating them",
            )
        target.append(lumos.io.to_document(ctx))
        return self._rebuild(session, context, source.pop())

    def undo(self, session: str, context: str) -> ContextSummary:
        """Undo the last edit of a context."""
        return self._step(session, context, forward=False)

    def redo(self, session: str, context: str) -> ContextSummary:
        """Redo the last undone edit of a context."""
        return self._step(session, context, forward=True)

    def update_ui(
        self, session: str, context: str, ui: Mapping[str, Any], *, record: bool = True
    ) -> ContextSummary:
        """Merge ``ui`` into the interface data of a context (positions of the blocks...).

        The library ignores these data; the ``.lumos`` file keeps them. With ``record=False`` the change is
        not an edit of its own for undo (e.g. the placement of a block just added).
        """
        ctx = self._context(session, context)
        with self._editing(session, context) if record else contextlib.nullcontext():
            merged = dict(ctx.ui)
            merged.update(json.loads(json.dumps(dict(ui))))
            ctx.ui = merged
        return self._summary(context, self._context(session, context))

    def set_context(
        self,
        session: str,
        context: str,
        *,
        name: str | None = None,
        description: str | None = None,
        seed: int | None = None,
    ) -> ContextSummary:
        """Rename a context, change its description or the seed of its random streams (REQ-CTX-6)."""
        if seed is not None and (isinstance(seed, bool) or int(seed) != seed or seed < 0):
            raise ServiceError(f"invalid seed {seed!r}", hint="the seed is a non-negative integer, e.g. 42")
        with self._editing(session, context) as ctx:
            if name is not None:
                ctx.name = name
            if description is not None:
                ctx.meta["description"] = description
            if seed is not None:
                ctx.seed = int(seed)
        return self._summary(context, self._context(session, context))

    def set_free(
        self,
        session: str,
        context: str,
        path: str,
        free: bool = True,
        prior: Mapping[str, Any] | None = None,
    ) -> ElementSummary:
        """Mark the parameter ``"element.field"`` as free with a prior, or fixed again (REQ-EL-3).

        ``prior`` is ``{"kind": "uniform" | "loguniform", "low": ..., "high": ...}`` or ``{"kind": "normal",
        "mean": ..., "std": ...}``, values with their units (``"1 mm"``). Monte-Carlo runs draw the free
        parameters from their priors.
        """
        ctx = self._context(session, context)
        owner, _, name = path.rpartition(".")
        element = self._element(ctx, owner)
        field = type(element).declared().get(name)
        if not isinstance(field, Param):
            physical = [n for n, f in type(element).declared().items() if isinstance(f, Param)]
            raise ServiceError(
                f"{owner} has no parameter {name!r}",
                hint=f"physical parameters: {', '.join(physical) or 'none'}",
            )
        parsed = studies.parse_prior(field, prior, name=path) if free and prior is not None else None
        with self._editing(session, context):
            element.set_free(name, bool(free), parsed)
        return self._element_summary(owner, element)

    # --- Editing -------------------------------------------------------------------------------------------

    def add_element(
        self,
        session: str,
        context: str,
        type_id: str,
        *,
        name: str | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> ElementSummary:
        """Add an element of ``type_id`` with the given parameter values (``"2 mW"``...)."""
        element = self._new_element(type_id, name, params)
        with self._editing(session, context) as ctx:
            try:
                ctx.add(element)
            except GraphError as error:
                raise ServiceError(str(error), hint=_hint_for(error)) from error
        return self._element_summary(ctx.id_of(element), element)

    def _new_element(self, type_id: str, name: str | None, params: Mapping[str, Any] | None) -> Element:
        cls = self._type(type_id)
        declared = cls.declared()
        values = {}
        for key, value in (params or {}).items():
            if key not in declared:
                raise ServiceError(
                    f"{type_id} has no parameter {key!r}", hint=f"parameters: {', '.join(declared)}"
                )
            values[key] = parse_value(declared[key], value, name=key)
        try:
            return cls(name=name, **values)
        except (TypeError, ValueError, u.UnitsError) as error:
            raise ServiceError(str(error), hint=_hint_for(error)) from error

    def create_group(
        self,
        session: str,
        context: str,
        name: str,
        *,
        elements: Iterable[Mapping[str, Any]] = (),
        connections: Iterable[tuple[str, str] | list[str]] = (),
        expose: Mapping[str, str] | None = None,
    ) -> ElementSummary:
        """Add a group of elements that behaves as one element (REQ-CTX-7).

        ``elements`` are ``{"type": type_id, "name": ..., "params": {...}}``, ``connections`` are pairs
        ``["child.port", "child.port"]`` inside the group and ``expose`` maps the names of the ports of the
        group to internal ports (``{"out": "laser.out"}``). Nothing is added if any part is invalid.
        """
        ctx = self._context(session, context)
        group = ElementGroup(name=name)
        for spec in elements:
            group.add(self._new_element(str(spec["type"]), spec.get("name"), spec.get("params")))
        try:
            for source, target in connections:
                group.connect(self._port(group, source), self._port(group, target))
            for port, internal in (expose or {}).items():
                group.expose(port, self._port(group, internal))
            with self._editing(session, context):
                ctx.add(group)
        except (PortCompatibilityError, GraphError, TypeError) as error:
            raise ServiceError(
                str(error), hint="connect and expose ports of the elements of the group ('child.port')"
            ) from error
        return self._element_summary(ctx.id_of(group), group)

    def remove_element(self, session: str, context: str, element: str) -> None:
        """Remove an element and its connections."""
        ctx = self._context(session, context)
        target = self._element(ctx, element)
        with self._editing(session, context):
            ctx.remove(target)

    def _port(self, ctx: Context | ElementGroup, reference: str) -> Any:
        owner, _, port = reference.rpartition(".")
        if not owner:
            raise ServiceError(
                f"{reference!r} is not a port", hint="write ports as 'element.port', e.g. 'laser.out'"
            )
        element = self._element(ctx, owner)
        try:
            return element.port(port)
        except KeyError:
            known = ", ".join(element.instance_ports())
            raise ServiceError(
                f"{owner} has no port {port!r}", hint=f"ports: {known}", code="not_found"
            ) from None

    def connect(self, session: str, context: str, source: str, target: str) -> ConnectionInfo:
        """Connect the output ``source`` to the input ``target`` (``"id.port"``)."""
        ctx = self._context(session, context)
        try:
            ports = self._port(ctx, source), self._port(ctx, target)
            with self._editing(session, context):
                ctx.connect(*ports)
        except (PortCompatibilityError, GraphError, TypeError) as error:
            raise ServiceError(str(error), hint=_hint_for(error)) from error
        return ConnectionInfo(source=source, target=target)

    def disconnect(self, session: str, context: str, source: str, target: str) -> None:
        """Remove a connection."""
        ctx = self._context(session, context)
        try:
            ports = self._port(ctx, source), self._port(ctx, target)
            with self._editing(session, context):
                ctx.disconnect(*ports)
        except (GraphError, KeyError) as error:
            raise ServiceError(str(error), hint="list the connections with describe_context") from error

    def set_param(self, session: str, context: str, path: str, value: Any) -> ElementSummary:
        """Set a parameter or setting ``"element.field"`` (quantities as ``"5 mm"``)."""
        ctx = self._context(session, context)
        owner, _, name = path.rpartition(".")
        element = self._element(ctx, owner)
        declared = type(element).declared()
        if name not in declared:
            raise ServiceError(f"{owner} has no field {name!r}", hint=f"fields: {', '.join(declared)}")
        parsed = parse_value(declared[name], value, name=path)
        try:
            with self._editing(session, context):
                setattr(element, name, parsed)
        except (TypeError, ValueError, u.UnitsError) as error:
            raise ServiceError(str(error), hint=_hint_for(error)) from error
        return self._element_summary(owner, element)

    def set_observation(
        self, session: str, context: str, *, minimum: Any, maximum: Any, n_wavelengths: int = 1
    ) -> ContextSummary:
        """Set the observation band and its number of wavelengths."""
        ctx = self._context(session, context)
        low = parse_quantity(minimum, u.m, name="minimum")
        high = parse_quantity(maximum, u.m, name="maximum")
        if not 0 < low < high or int(n_wavelengths) < 1:
            raise ServiceError(
                "invalid observation band", hint="give 0 < minimum < maximum and n_wavelengths >= 1"
            )
        try:
            with self._editing(session, context):
                ctx.observation.band = (low, high)
                ctx.observation.n_wavelengths = int(n_wavelengths)
        except ValueError as error:
            raise ServiceError(str(error), hint="give 0 < minimum < maximum") from error
        return self._summary(context, ctx)

    def add_feedback(
        self, session: str, context: str, source: str, target: str, mode: str = "set"
    ) -> ContextSummary:
        """Feed the output ``source`` (``"id.port"``) back to the field ``target`` of a loop (ADR-0020).

        ``mode="add"`` adds the value to the field at each step (an integrator), ``"set"`` replaces it. Run
        the loop with :meth:`run_loop`.
        """
        ctx = self._context(session, context)
        try:
            with self._editing(session, context):
                ctx.feedback(source, target, mode=mode)  # type: ignore[arg-type]
        except (KeyError, AttributeError, ValueError, GraphError) as error:
            message = str(error).strip("'\"")
            raise ServiceError(
                message,
                hint="source is an output 'element.port', target a field 'element.field' (describe_context), "
                "mode 'set' or 'add'",
            ) from error
        return self._summary(context, ctx)

    def remove_feedback(self, session: str, context: str, source: str, target: str) -> ContextSummary:
        """Remove the feedback from ``source`` to ``target``."""
        ctx = self._context(session, context)
        if not any(f.source == source and f.target == target for f in ctx.feedbacks):
            listed = ", ".join(f"{f.source} -> {f.target}" for f in ctx.feedbacks) or "none"
            raise ServiceError(f"no feedback {source} -> {target}", hint=f"feedbacks: {listed}")
        with self._editing(session, context):
            ctx.remove_feedback(source, target)
        return self._summary(context, ctx)

    def build_pipeline(
        self, session: str, context: str, source: str, steps: Iterable[Mapping[str, Any]]
    ) -> list[ElementSummary]:
        """Chain processing elements after the output ``source``.

        Each step is ``{"type": type_id, "name": ..., "params": {...}}``; the first input of each element is
        connected to the first output of the previous one.
        """
        import lumos.processing  # noqa: F401 - registers the processing elements

        added = []
        previous = source
        with self._editing(session, context):
            return self._chain(session, context, previous, steps, added)

    def _chain(
        self, session: str, context: str, previous: str, steps: Iterable[Mapping[str, Any]], added: list
    ) -> list[ElementSummary]:
        for step in steps:
            summary = self.add_element(
                session, context, str(step["type"]), name=step.get("name"), params=step.get("params")
            )
            if not summary.inputs or not summary.outputs:
                raise ServiceError(
                    f"{step['type']} is not a processing step", hint="use elements with ports in and out"
                )
            self.connect(session, context, previous, f"{summary.id}.{summary.inputs[0]}")
            previous = f"{summary.id}.{summary.outputs[0]}"
            added.append(summary)
        return added

    def rename_element(self, session: str, context: str, element: str, name: str) -> ContextSummary:
        """Rename an element (``"laser"`` or ``"group.child"``), keeping its connections and position."""
        return self._edit_document(session, context, lambda document: edits.rename(document, element, name))

    def group_elements(
        self, session: str, context: str, elements: Iterable[str], *, name: str
    ) -> ContextSummary:
        """Move elements of the context into a new group exposing their free ports (REQ-WEB-4)."""
        ctx = self._context(session, context)
        chosen = list(elements)
        ports = {}
        for identifier in chosen:
            if identifier in ctx.elements:
                found = ctx.elements[identifier].instance_ports()
                ports[identifier] = (
                    [n for n, p in found.items() if p.direction == "in"],
                    [n for n, p in found.items() if p.direction == "out"],
                )
        try:
            return self._edit_document(
                session, context, lambda document: edits.group(document, chosen, name, ports)
            )
        except ServiceError as error:
            if "cycle" not in error.message:
                raise
            raise ServiceError(
                "these elements cannot form a group: the light would leave the group and come back into it",
                hint="group elements that follow each other; an element between them must be grouped too",
            ) from error

    def ungroup(self, session: str, context: str, group: str) -> ContextSummary:
        """Dissolve a group: its elements join the context and keep their connections."""
        return self._edit_document(session, context, lambda document: edits.ungroup(document, group))

    def copy_elements(self, session: str, context: str, elements: Iterable[str]) -> dict[str, Any]:
        """Return a fragment holding ``elements``, the connections between them and their positions."""
        return edits.fragment_of(lumos.io.to_document(self._context(session, context)), list(elements))

    def paste_elements(self, session: str, context: str, fragment: Mapping[str, Any]) -> list[str]:
        """Add the elements of a fragment (from :meth:`copy_elements`) with free names; return the names."""
        pasted: list[str] = []

        def change(document: dict[str, Any]) -> dict[str, Any]:
            result, names = edits.paste(document, fragment)
            pasted.extend(names)
            return result

        self._edit_document(session, context, change)
        return pasted

    def export_element(self, session: str, context: str, element: str) -> str:
        """Return the ``.lumos`` document of an element or a group (a reusable block for lumos.io)."""
        target = self._element(self._context(session, context), element)
        return json.dumps(lumos.io.to_document(target), indent=2, ensure_ascii=False)

    def import_element(self, session: str, context: str, document: str | Mapping[str, Any]) -> ElementSummary:
        """Insert the element or group of a ``.lumos`` document (a block), renamed if its name is taken."""
        try:
            data = json.loads(document) if isinstance(document, str) else dict(document)
        except ValueError as error:
            raise ServiceError(
                f"invalid .lumos document: {error}", hint="give a block exported by LUMOS"
            ) from error
        kind = data.get("kind")
        if kind not in ("element", "group") or not isinstance(data.get(kind), Mapping):
            raise ServiceError(
                "the document is not a block",
                hint="give the .lumos file of an element or a group, not of a context",
            )
        entry = dict(data[kind])
        name = str(entry.get("name") or "")
        base = name if edits.NAME.match(name) else "block"
        entry["id"] = base
        names = self.paste_elements(session, context, {"elements": [entry], "connections": []})
        ctx = self._context(session, context)
        return self._element_summary(names[0], ctx[names[0]])

    # --- Verification --------------------------------------------------------------------------------------

    def validate_context(self, session: str, context: str) -> list[Issue]:
        """Problems that would make a run fail, with suggested fixes."""
        ctx = self._context(session, context)
        issues = []
        for path, element in ctx.walk():
            if element.container is not ctx:
                continue  # inside a group: checked by the group
            for name, port in element.instance_ports().items():
                if port.direction != "in" or port.multiple:
                    continue
                if not ctx._graph.incoming(element.port(name)):
                    issues.append(
                        Issue(
                            severity="error",
                            element=path,
                            message=f"input {path}.{name} is not connected",
                            hint=_hint_for(GraphError("not connected")),
                        )
                    )
            if getattr(type(element), "observation_dependent", False) and ctx.observation.band is None:
                issues.append(
                    Issue(
                        severity="error",
                        element=path,
                        message=f"{path} needs the observation band of the context",
                        hint=_hint_for(ValueError("band")),
                    )
                )
            driverless = actions_of(type(element)) and element.driver is None
            if driverless and ctx.mode_of(element) == "operation":
                issues.append(
                    Issue(
                        severity="warning",
                        element=path,
                        message=f"{path} has no driver in operation mode",
                        hint="associate devices (associate_devices) or force it to simulation",
                    )
                )
        return issues

    def explain_sampling(self, session: str, context: str, element: str) -> dict[str, str]:
        """How a propagation element will be computed: method (forced or chosen by the planner) and why."""
        from lumos.propagation.planner import plan_for

        ctx = self._context(session, context)
        target = self._element(ctx, element)
        if not hasattr(target, "distance"):
            raise ServiceError(f"{element} is not a propagation", hint="give a FreeSpace element")
        incoming = ctx._graph.incoming(target.port("in"))
        if not incoming:
            raise ServiceError(f"{element}.in is not connected", hint=_hint_for(GraphError("not connected")))
        source = incoming[0].source
        signal = self._compute(session, ctx, [source])[source]
        output_grid = target.output_grid or signal.grid
        plan = plan_for(signal, float(target.distance.to_value(u.m)), output_grid)
        forced = getattr(target, "method", "auto")
        method, reason = plan.method, plan.reason
        if forced != "auto":
            method = forced
            reason = f"forced by the element (method={forced!r}); the planner would choose {plan.method}"
        return {
            "method": method,
            "reason": reason,
            "planner_method": plan.method,
            "planner_reason": plan.reason,
            "cost": f"{plan.cost:.3g} operations (planner choice)",
            "input_grid": f"{signal.grid}",
            "output_grid": f"{plan.output_grid}",
        }

    def plan_propagation(self, session: str, context: str, element: str) -> dict[str, str]:
        """Alias of :meth:`explain_sampling`."""
        return self.explain_sampling(session, context, element)

    # --- Execution -----------------------------------------------------------------------------------------

    def run(
        self, session: str, context: str, outputs: Iterable[str], *, archive: bool | str = False
    ) -> JobInfo:
        """Start a run of ``outputs`` in the background; follow it with :meth:`get_job`."""
        names = list(outputs)

        def compute(ctx: Context, job: _Job, progress: Callable[[int, int, str], None]) -> None:
            job.result = ctx.run(
                names, archive=archive, archive_root=self.config.archive_root, progress=progress
            )
            if job.result.archive is not None:
                job.info.archive = str(job.result.archive.path)

        return self._start(session, context, "run", names, compute)

    def _start(
        self,
        session: str,
        context: str,
        kind: str,
        names: list[str],
        compute: Callable[[Context, _Job, Callable[[int, int, str], None]], None],
    ) -> JobInfo:
        """Start a job computing ``compute(ctx, job, progress)`` in the background, within the quotas."""
        state = self._session(session)
        ctx = self._context(session, context)
        running = [j for j in state.jobs.values() if j.info.status in ("queued", "running")]
        if len(running) >= self.config.max_jobs_per_session:
            raise ServiceError(
                f"quota of {self.config.max_jobs_per_session} runs at a time reached",
                hint="wait for the running jobs (wait_job) before starting another one",
                code="quota",
            )
        limit = self.config.max_concurrent_runs
        if limit is not None:
            busy = sum(
                job.info.status in ("queued", "running")
                for s in self._sessions.values()
                for job in s.jobs.values()
            )
            if busy >= limit:
                raise ServiceError(
                    f"the instance is busy: {busy} runs in progress (at most {limit})",
                    hint="try again when a run has ended",
                    code="quota",
                )
        self._check_quotas(ctx)
        job = _Job(
            JobInfo(id=uuid.uuid4().hex[:10], kind=kind, context=context, outputs=names, status="queued"),
            session,
        )
        state.jobs[job.info.id] = job

        def progress(done: int, total: int, element: str) -> None:
            if job.cancelled:
                raise _Cancelled("cancelled")
            job.info.progress = done / max(total, 1)
            job.info.message = f"computed {element} ({done}/{total})"
            if time.perf_counter() - job.started > self.config.max_run_seconds:
                raise TimeoutError(f"the run exceeded {self.config.max_run_seconds:g} s")

        origin = self._origin(state)

        def work() -> None:
            job.info.status = "running"
            job.started = time.perf_counter()
            try:
                with state.lock, ctx.as_origin(origin):
                    compute(ctx, job, progress)
                job.info.progress = 1.0
                job.info.status = "done"
            except _Cancelled:
                job.info.status = "cancelled"
                job.info.message = "cancelled by the user"
            except Exception as error:
                job.info.status = "failed"
                job.info.error = str(error)
                job.info.hint = _hint_for(error)
                logger.info("run %s failed: %s", job.info.id, error)
            finally:
                job.info.seconds = time.perf_counter() - job.started

        job.future = self._executor.submit(work)
        return job.info.model_copy()

    def run_study(
        self,
        session: str,
        context: str,
        *,
        kind: str,
        outputs: Iterable[str] | None = None,
        parameters: Mapping[str, Any] | None = None,
        mode: str = "grid",
        n: int | None = None,
        noise: str | None = None,
    ) -> JobInfo:
        """Start a study in the background: a sweep or a Monte-Carlo run (REQ-CTX-10).

        Follow the job with :meth:`get_job`, read its points with :meth:`get_study`.

        Parameters
        ----------
        kind
            ``"sweep"``: run the context for the values of ``parameters``; ``"sample"``: Monte-Carlo run of
            ``n`` draws of the free parameters from their priors (see :meth:`set_free`).
        outputs
            Ports to compute at each point (default: the outputs of the context).
        parameters
            For sweeps, the values of each path ``"element.field"``: a list such as ``["1 mm", "2 mm"]``, or
            a range such as ``{"start": "1 mm", "stop": "5 mm", "num": 5, "log": false}``.
        mode
            For sweeps, ``"grid"`` (every combination) or ``"zip"`` (the i-th values together).
        n
            For Monte-Carlo runs, the number of draws.
        noise
            ``"fixed"`` (the noise of the context at every point, default of sweeps) or ``"independent"`` (a
            new noise realization per point, default of Monte-Carlo runs).
        """
        ctx = self._context(session, context)
        if kind not in ("sweep", "sample"):
            raise ServiceError(
                f"unknown kind of study {kind!r}",
                hint="kind='sweep' (values of some parameters) or kind='sample' (Monte-Carlo run of n draws)",
            )
        names = list(outputs or self.default_outputs(session, context))
        if not names:
            raise ServiceError("no outputs to compute", hint="give outputs, e.g. outputs=['camera.out']")
        noise = noise or ("fixed" if kind == "sweep" else "independent")
        if noise not in ("fixed", "independent"):
            raise ServiceError(f"unknown noise {noise!r}", hint="noise='fixed' or 'independent'")
        values: dict[str, list[Any]] = {}
        if kind == "sweep":
            if not parameters:
                raise ServiceError(
                    "a sweep needs the values of at least one parameter",
                    hint="e.g. parameters={'lens.focal_length': {'start': '190 mm', 'stop': '210 mm', "
                    "'num': 5}}",
                )
            for path, spec in parameters.items():
                owner, _, name = path.rpartition(".")
                element = self._element(ctx, owner)
                field = type(element).declared().get(name)
                if field is None:
                    raise ServiceError(
                        f"{owner} has no field {name!r}",
                        hint=f"fields: {', '.join(type(element).declared())}",
                    )
                values[path] = studies.sweep_values(field, spec, name=path)
            count = studies.count_points([len(v) for v in values.values()], mode)
        else:
            if n is None or int(n) < 1:
                raise ServiceError("a Monte-Carlo run needs n >= 1 draws", hint="e.g. n=100")
            count = int(n)
            priors = ctx.free_parameters()
            missing = [path for path, prior in priors.items() if prior is None]
            if missing:
                raise ServiceError(
                    f"free parameters without a prior: {', '.join(missing)}",
                    hint="give them a prior with set_free_parameter (uniform, loguniform or normal)",
                )
            noisy = any(getattr(type(e), "stochastic", False) for _, e in ctx.walk())
            if not priors and (noise == "fixed" or not noisy):
                raise ServiceError(
                    "nothing varies between the draws",
                    hint="mark a parameter as free with a prior (set_free_parameter), "
                    "or add an element with noise",
                )
        if count > self.config.max_study_points:
            raise ServiceError(
                f"the study has {count} points",
                hint=f"this instance computes at most {self.config.max_study_points} points per study",
                code="quota",
            )
        seed = ctx.seed

        def compute(ctx: Context, job: _Job, progress: Callable[[int, int, str], None]) -> None:
            def point(done: int, total: int) -> None:
                progress(done, total, f"point {done}")  # cancellation and time limit, between points
                job.info.message = f"computed point {done}/{total}"

            if kind == "sweep":
                points = ctx.sweep(values, outputs=names, mode=mode, noise=noise, progress=point)  # type: ignore[arg-type]
            else:
                points = ctx.sample(count, outputs=names, noise=noise, progress=point)  # type: ignore[arg-type]
            job.study = StudyResult(
                job=job.info.id,
                context=context,
                kind=kind,
                mode=mode if kind == "sweep" else None,
                noise=noise,
                seed=seed,
                parameters=list(values) if kind == "sweep" else list(ctx.free_parameters()),
                outputs=names,
                points=[self._study_point(ctx, p, names) for p in points],
            )

        return self._start(session, context, kind, names, compute)

    def run_loop(
        self,
        session: str,
        context: str,
        *,
        steps: int,
        period: Any,
        outputs: Iterable[str] | None = None,
    ) -> JobInfo:
        """Run the context as a closed loop of ``steps`` iterations of ``period`` (ADR-0020).

        At each step the outputs are computed, then the feedbacks are applied and the clock advances. The
        fields driven by the feedbacks and the clock keep their final values; :meth:`undo` puts them back.
        Read the steps with :meth:`get_study` (parameter ``time``).
        """
        ctx = self._context(session, context)
        if not ctx.feedbacks:
            raise ServiceError(
                "the context has no feedback to loop on",
                hint="add one with add_feedback(source='ctrl.commands', target='dm.commands', mode='add')",
            )
        if int(steps) < 1:
            raise ServiceError(f"a loop needs at least one step, got {steps}", hint="e.g. steps=20")
        if int(steps) > self.config.max_study_points:
            raise ServiceError(
                f"the loop has {steps} steps",
                hint=f"this instance computes at most {self.config.max_study_points} steps per loop",
                code="quota",
            )
        seconds = float(parse_quantity(period, u.s, name="period").to(u.s).value)  # type: ignore[arg-type]
        if not seconds > 0:
            raise ServiceError(f"the period must be positive, got {period}", hint="e.g. period='2 ms'")
        names = list(outputs or self.default_outputs(session, context))
        if not names:
            raise ServiceError("no outputs to compute", hint="give outputs, e.g. outputs=['wfs.residual']")
        count = int(steps)
        seed = ctx.seed

        def compute(ctx: Context, job: _Job, progress: Callable[[int, int, str], None]) -> None:
            def step(done: int, total: int) -> None:
                progress(done, total, f"step {done}")
                job.info.message = f"computed step {done}/{total}"

            with self._editing(session, context):  # the loop changes the states and the clock: undoable
                records = ctx.loop(count, period=seconds * u.s, outputs=names, progress=step)
            job.study = StudyResult(
                job=job.info.id,
                context=context,
                kind="loop",
                noise="fixed",
                seed=seed,
                parameters=["time"],
                outputs=names,
                points=[
                    StudyPoint(
                        params={"time": str(format_value(r.time.to(u.s)))},
                        outputs={name: summarize(name, r.outputs[name]) for name in names},
                    )
                    for r in records
                ],
            )

        return self._start(session, context, "loop", names, compute)

    def _study_point(self, ctx: Context, point: Any, names: list[str]) -> StudyPoint:
        params = {}
        for path, value in point.params.items():
            element, field = ctx._parameter(path)
            unit = getattr(type(element).declared()[field], "unit", None)
            params[path] = str(format_value(value, unit))
        outputs = {name: summarize(name, point.outputs[name]) for name in names}
        return StudyPoint(params=params, realization=point.realization, outputs=outputs)

    def get_study(self, session: str, job: str) -> StudyResult:
        """Points of a finished study: values of the parameters, realization, summaries of the outputs."""
        record = self._session(session).jobs.get(job)
        if record is None:
            raise ServiceError(f"unknown job {job!r}", hint="start one with run_study()", code="not_found")
        if record.info.kind not in ("sweep", "sample", "loop"):
            raise ServiceError(
                f"job {job} is a {record.info.kind}, not a study", hint="read it with get_result()"
            )
        if record.info.status != "done" or record.study is None:
            raise ServiceError(
                f"job {job} is {record.info.status}", hint=record.info.hint or "wait for it with wait_job()"
            )
        return record.study

    def study_csv(self, session: str, job: str) -> str:
        """CSV table of a finished study (one row per point)."""
        return studies.study_table(self.get_study(session, job))

    def default_outputs(self, session: str, context: str) -> list[str]:
        """Return the outputs to run when none is named.

        Those listed by the context (``meta.outputs``), else the unconnected outputs of its elements.
        """
        ctx = self._context(session, context)
        listed = [str(name) for name in ctx.meta.get("outputs", [])]
        if listed:
            return listed
        used = {(id(c.source.owner), c.source.name) for c in ctx._graph.connections}
        free = []
        for element in ctx.graph:
            path = ctx.id_of(element)
            for name, port in element.instance_ports().items():
                if port.direction == "out" and (id(element), name) not in used:
                    free.append(f"{path}.{name}")
        return free

    def list_jobs(self, session: str) -> list[JobInfo]:
        """Return the runs of the session, oldest first."""
        return [job.info.model_copy() for job in self._session(session).jobs.values()]

    def _check_quotas(self, ctx: Context) -> None:
        """Refuse a computation whose grids or memory exceed the quotas of the configuration."""
        largest, where = 0, ""
        for path, element in ctx.walk():
            for name, side in _grid_sides(element):
                if side > largest:
                    largest, where = side, f"{path}.{name}"
                limit = self.config.max_grid_pixels
                if limit is not None and side > limit:
                    raise ServiceError(
                        f"{path}.{name} asks for a grid of {side} pixels per side",
                        hint=f"this instance computes grids of at most {limit} pixels per side",
                        code="quota",
                    )
        budget = self.config.max_memory_mb
        if budget is None or not largest:
            return
        n_wavelengths = max(int(ctx.observation.n_wavelengths), 1)
        needed = largest**2 * 16 * n_wavelengths * MEMORY_COPIES / 1e6
        if needed > budget:
            raise ServiceError(
                f"the computation needs about {needed:.0f} MB of memory ({where}: {largest}² pixels, "
                f"{n_wavelengths} wavelength(s)), more than the {budget:g} MB allowed",
                hint="use fewer pixels or fewer wavelengths",
                code="quota",
            )

    def _compute(self, session: str, ctx: Context, references: list[Any]) -> Any:
        """Compute ports synchronously within the quotas (grids, memory, time); errors become ServiceError."""
        state = self._session(session)
        self._check_quotas(ctx)
        started = time.perf_counter()

        def progress(done: int, total: int, element: str) -> None:
            if time.perf_counter() - started > self.config.max_run_seconds:
                raise TimeoutError(f"the computation exceeded {self.config.max_run_seconds:g} s")

        try:
            with state.lock, ctx.as_origin(self._origin(state)):
                return ctx.run(references, progress=progress)
        except ServiceError:
            raise
        except Exception as error:
            raise ServiceError(str(error), hint=_hint_for(error), code="failed") from error

    def cancel_job(self, session: str, job: str) -> JobInfo:
        """Ask a queued or running run to stop (at the end of the element being computed)."""
        record = self._session(session).jobs.get(job)
        if record is None:
            raise ServiceError(f"unknown job {job!r}", hint="start one with run()", code="not_found")
        if record.info.status not in ("queued", "running"):
            raise ServiceError(
                f"job {job} is already {record.info.status}", hint="only a run in progress can be cancelled"
            )
        record.cancelled = True
        if record.future is not None and record.future.cancel():
            record.info.status = "cancelled"
        return record.info.model_copy()

    def get_job(self, session: str, job: str) -> JobInfo:
        """Status, progress and outcome of a run."""
        state = self._session(session)
        try:
            return state.jobs[job].info.model_copy()
        except KeyError:
            raise ServiceError(
                f"unknown job {job!r}", hint="start one with run()", code="not_found"
            ) from None

    def wait_job(self, session: str, job: str, *, timeout: float | None = None) -> JobInfo:
        """Wait for the end of a run (or ``timeout`` seconds)."""
        state = self._session(session)
        record = state.jobs.get(job)
        if record is None:
            raise ServiceError(f"unknown job {job!r}", hint="start one with run()", code="not_found")
        if record.future is not None:
            with contextlib.suppress(concurrent.futures.TimeoutError):
                record.future.result(timeout=timeout)
        return record.info.model_copy()

    def get_result(self, session: str, job: str, port: str, *, figure: bool = False) -> PortSummary:
        """Summary of an output computed by a run."""
        record = self._session(session).jobs.get(job)
        if record is None:
            raise ServiceError(f"unknown job {job!r}", hint="start one with run()", code="not_found")
        if record.info.status != "done" or record.result is None:
            raise ServiceError(
                f"job {job} is {record.info.status}", hint=record.info.hint or "wait for it with wait_job()"
            )
        try:
            signal = record.result[port]
        except (KeyError, GraphError):
            raise ServiceError(
                f"the job did not compute {port!r}", hint=f"outputs: {', '.join(record.info.outputs)}"
            ) from None
        return summarize(port, signal, figure=figure)

    def inspect_port(self, session: str, context: str, port: str, *, figure: bool = True) -> PortSummary:
        """Compute (or reuse) the signal of an output port and summarise it, with a figure."""
        ctx = self._context(session, context)
        reference = self._port(ctx, port)
        signal = self._compute(session, ctx, [reference])[reference]
        return summarize(port, signal, figure=figure)

    def port_file(self, session: str, context: str, port: str, fmt: str) -> PortFile:
        """Compute the signal of ``port`` and return it as a file: ``png``, ``plt``, ``npz`` or ``json``."""
        self._check_format(fmt)
        ctx = self._context(session, context)
        reference = self._port(ctx, port)
        return self._file(port, self._compute(session, ctx, [reference])[reference], fmt)

    def result_file(self, session: str, job: str, port: str, fmt: str) -> PortFile:
        """Return an output computed by a run as a file (see :meth:`port_file`), without computing again."""
        self._check_format(fmt)
        record = self._session(session).jobs.get(job)
        if record is None:
            raise ServiceError(f"unknown job {job!r}", hint="start one with run()", code="not_found")
        if record.info.status != "done" or record.result is None:
            raise ServiceError(f"job {job} is {record.info.status}", hint="wait for the end of the run")
        try:
            signal = record.result[port]
        except (KeyError, GraphError):
            raise ServiceError(
                f"the job did not compute {port!r}", hint=f"outputs: {', '.join(record.info.outputs)}"
            ) from None
        return self._file(port, signal, fmt)

    @staticmethod
    def _check_format(fmt: str) -> None:
        if fmt not in FILE_FORMATS:
            raise ServiceError(f"unknown format {fmt!r}", hint=f"formats: {', '.join(FILE_FORMATS)}")

    def _file(self, port: str, signal: Any, fmt: str) -> PortFile:
        import io
        import tempfile

        import numpy as np

        from lumos.fields import DataProduct
        from lumos.results import save_data
        from lumos.service.summaries import figure_of

        name = f"{port}.{fmt}"
        if fmt in ("png", "plt"):
            figure = figure_of(signal, dpi=120)
            if figure is None:
                raise ServiceError(
                    f"{port} carries a {type(signal).__name__}, which has no figure", hint="download its data"
                )
            if fmt == "png":
                buffer = io.BytesIO()
                figure.savefig(buffer, format="png")
                return PortFile(filename=name, media_type="image/png", content=buffer.getvalue())
            from lumos._vendor import pltedit

            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / name
                pltedit.save(figure, path)
                return PortFile(
                    filename=name, media_type="application/octet-stream", content=path.read_bytes()
                )
        if fmt == "json":
            if not hasattr(signal, "to_json"):
                raise ServiceError(f"{port} has no JSON form", hint="download its data (npz)")
            document = {"signal": type(signal).__name__, "data": signal.to_json()}
            text = json.dumps(document, default=str)
            return PortFile(filename=name, media_type="application/json", content=text.encode())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / name
            if isinstance(signal, DataProduct):
                signal.to_npz(path)
            elif hasattr(signal, "intensity") and hasattr(signal, "grid"):
                from lumos.backend import to_numpy

                grid = signal.grid
                unit = signal.intensity_unit
                data: dict[str, Any] = {
                    "intensity": np.asarray(to_numpy(signal.intensity), dtype=float) * unit,
                    "x": grid.x,
                    "y": grid.y,
                }
                waves = getattr(signal, "wavelengths", None)
                if waves is None and hasattr(signal, "wavelength"):
                    waves = signal.wavelength
                if waves is not None:
                    data["wavelength"] = waves
                save_data(path, data)
            else:
                raise ServiceError(f"{port} has no data to download as npz", hint="download its JSON form")
            return PortFile(filename=name, media_type="application/octet-stream", content=path.read_bytes())

    def radial_profile(
        self,
        session: str,
        context: str,
        port: str,
        *,
        bin_width: Any = None,
        center: Any = None,
        normalize: str | None = None,
    ) -> dict[str, Any]:
        """Azimuthal mean of the intensity of a field at ``port``, in rings of width ``bin_width``.

        ``center`` is ``[x, y]`` (default: the origin of the grid); ``normalize="peak"`` divides by the
        maximum of the intensity, which gives contrasts. Radii are the middles of the rings, in SI
        units of the grid.
        """
        import numpy as np

        from lumos.backend import to_numpy
        from lumos.fields import IncoherentBundle, Wavecube, Wavefront
        from lumos.processing.profiles import radial_profile

        ctx = self._context(session, context)
        reference = self._port(ctx, port)
        signal = self._compute(session, ctx, [reference])[reference]
        if not isinstance(signal, Wavefront | Wavecube | IncoherentBundle):
            raise ServiceError(
                f"{port} carries a {type(signal).__name__}, not a field",
                hint="give the output of an optical element (a wavefront, a wavecube or a bundle)",
            )
        grid = signal.grid
        unit = u.rad if grid.plane == "angular" else u.m
        intensity = np.asarray(to_numpy(signal.intensity), dtype=float)
        intensity = intensity.reshape(-1, *intensity.shape[-2:]).sum(axis=0)
        width = None if bin_width is None else parse_quantity(bin_width, unit, name="bin_width")
        centre = None
        if center is not None:
            centre = tuple(parse_quantity(c, unit, name="center") for c in center)
        try:
            radii, profile = radial_profile(intensity, grid, center=centre, bin_width=width)  # type: ignore[arg-type]
        except ValueError as error:
            raise ServiceError(
                str(error), hint="give a positive bin_width in the unit of the grid"
            ) from error
        value_unit = signal.intensity_unit.to_string()
        if normalize == "peak":
            profile = profile / intensity.max()
            value_unit = "fraction of the peak"
        elif normalize is not None:
            raise ServiceError(f"unknown normalisation {normalize!r}", hint="use 'peak' or nothing")
        return {
            "port": port,
            "radii": [float(f"{r:.6g}") for r in np.atleast_1d(radii.to_value(unit))],
            "radius_unit": unit.to_string(),
            "values": [None if not np.isfinite(v) else float(f"{v:.6g}") for v in profile],
            "unit": value_unit,
            "bin_width": format_value(radii[0] * 2),
        }

    def journal(
        self, session: str, context: str, *, kind: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Last entries of the journal of a context (``kind``: ``"action"``, ``"log"``...)."""
        entries = self._context(session, context).journal.to_json()
        if kind is not None:
            entries = [e for e in entries if e.get("kind") == kind]
        return json.loads(json.dumps(entries[-limit:], default=str))

    def journal_since(self, session: str, context: str, start: int = 0) -> list[dict[str, Any]]:
        """Entries of the journal of a context from index ``start`` (to follow it as it grows)."""
        entries = list(self._context(session, context).journal)[start:]
        return json.loads(json.dumps([e.to_json() for e in entries], default=str))

    # --- Catalogues and processing -------------------------------------------------------------------------

    def resolve_object(self, name: str) -> dict[str, Any]:
        """Typed record of an astronomical object, with the provenance of every field."""
        from lumos.catalogs import resolve

        try:
            record = resolve(name)
        except Exception as error:
            raise ServiceError(
                f"cannot resolve {name!r}: {error}", hint="check the name, or the network access"
            ) from error
        fields: dict[str, Any] = {}
        for key, value in vars(record).items():
            if key.startswith("_"):
                continue
            fields[key] = format_value(getattr(value, "value", value))
        return {"type": type(record).__name__, "fields": fields}

    def evaluate_detection(
        self, session: str, context: str, port: str, *, peak: Any, resolution: Any, sigma: float = 5.0
    ) -> dict[str, Any]:
        """Contrast curve (detection limit versus separation) of the residual image of ``port``."""
        from lumos.processing.photometry import contrast_curve

        ctx = self._context(session, context)
        reference = self._port(ctx, port)
        image = self._compute(session, ctx, [reference])[reference]
        curve = contrast_curve(
            image,
            peak=parse_quantity(peak, name="peak"),
            resolution=parse_quantity(resolution, name="resolution"),
            sigma=sigma,
        )
        separations = curve.axes["separation"]
        return {
            "separation": [format_value(s) for s in separations],
            "contrast": [float(f"{c:.3g}") for c in curve.data.value],
            "sigma": sigma,
        }

    # --- Archives ------------------------------------------------------------------------------------------

    def archive_root(self) -> Path:
        """Directory of the archives of the runs."""
        from lumos.results import default_archive_root

        return Path(self.config.archive_root) if self.config.archive_root else default_archive_root()

    def list_archives(self, *, limit: int = 50) -> list[ArchiveInfo]:
        """Return the archives of the runs, most recent first."""
        import yaml

        root = self.archive_root()
        found = []
        for metadata in sorted(root.glob("*/*/metadata.yml"), reverse=True)[:limit]:
            data = yaml.safe_load(metadata.read_text(encoding="utf-8")) or {}
            found.append(
                ArchiveInfo(
                    path=str(metadata.parent),
                    name=metadata.parent.name,
                    created_at=str(data.get("created_at", "")),
                    outputs=list(data.get("outputs", [])),
                )
            )
        return found

    def _archive_path(self, path: str) -> Path:
        root = self.archive_root().resolve()
        target = Path(path).resolve()
        if root not in target.parents:
            raise ServiceError(
                "this path is outside the archives",
                hint="use a path given by list_archives",
                code="forbidden",
            )
        if not (target / "metadata.yml").is_file():
            raise ServiceError(
                f"no archive at {path}", hint="use a path given by list_archives", code="not_found"
            )
        return target

    def archive_zip(self, path: str) -> bytes:
        """Return an archive of the archive directory as the bytes of a ``.zip`` file."""
        import io
        import zipfile

        target = self._archive_path(path)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(target.rglob("*")):
                if file.is_file():
                    archive.write(file, Path(target.name) / file.relative_to(target))
        return buffer.getvalue()

    def read_archive_metadata(self, path: str) -> dict[str, Any]:
        """``metadata.yml`` of an archive (only inside the archive directory of the service)."""
        import yaml

        root = self.archive_root().resolve()
        target = Path(path).resolve()
        if root not in target.parents and target != root:
            raise ServiceError(
                "this path is outside the archives",
                hint="use a path given by list_archives",
                code="forbidden",
            )
        metadata = target / "metadata.yml"
        if not metadata.is_file():
            raise ServiceError(
                f"no archive at {path}", hint="use a path given by list_archives", code="not_found"
            )
        return json.loads(json.dumps(yaml.safe_load(metadata.read_text(encoding="utf-8")), default=str))

    # --- Operation -----------------------------------------------------------------------------------------

    def _require_operation(self) -> None:
        if not self.config.operation_enabled:
            raise ServiceError(
                "the operation of equipment is disabled",
                hint="enable it in the configuration of the server (e.g. 'lumos mcp --enable-operation')",
                code="forbidden",
            )

    def discover_devices(
        self, *, tcp: Iterable[str] = (), serial: bool | list[str] = False, visa: bool | list[str] = False
    ) -> list[DeviceModel]:
        """Devices answering on the given connections (operation must be enabled)."""
        from lumos.hardware.discovery import discover

        self._require_operation()
        self._devices = discover(tcp=tcp, serial=serial, visa=visa)
        return [DeviceModel(**vars(d)) for d in self._devices]

    def associate_devices(
        self, session: str, context: str, *, assignments: Mapping[str, str] | None = None
    ) -> dict[str, DeviceModel]:
        """Attach and connect the drivers of the discovered devices to the elements of a context."""
        from lumos.hardware.discovery import associate

        self._require_operation()
        ctx = self._context(session, context)
        try:
            found = associate(ctx, self._devices, assignments=assignments)
        except DriverError as error:
            raise ServiceError(str(error), hint="discover the devices first, or give assignments") from error
        return {path: DeviceModel(**vars(device)) for path, device in found.items()}

    def list_devices(self, session: str, context: str) -> list[dict[str, Any]]:
        """Drivers attached to the elements of a context, with their state."""
        ctx = self._context(session, context)
        return [{"element": path, **driver.status()} for path, driver in ctx.drivers().items()]

    def device_status(self, session: str, context: str, element: str) -> dict[str, Any]:
        """State of the driver of an element."""
        ctx = self._context(session, context)
        driver = self._element(ctx, element).driver
        if driver is None:
            raise ServiceError(f"{element} has no driver", hint="associate devices with associate_devices")
        return json.loads(json.dumps(driver.status(), default=str))

    def confirmation_needed(self, session: str, context: str, element: str) -> str | None:
        """Return the device that an action of ``element`` would drive, if it needs a human confirmation.

        ``None`` when the action only changes the simulation, or when the operation is disabled (the action is
        then refused by :meth:`execute_action`).
        """
        ctx = self._context(session, context)
        target = self._element(ctx, element)
        if not self.config.operation_enabled or ctx.mode_of(target) != "operation":
            return None
        driver = target.driver
        if driver is None:
            return f"of {element} (no driver attached)"
        status = driver.status()
        model, serial = status.get("model"), status.get("serial")
        return " ".join(str(v) for v in (model, serial) if v) or type(driver).__name__

    def execute_action(
        self,
        session: str,
        context: str,
        element: str,
        action: str,
        *,
        args: Iterable[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
        confirmed: bool = False,
    ) -> ActionResult:
        """Execute an action of an element; on equipment, only when enabled and confirmed by a human."""
        state = self._session(session)
        ctx = self._context(session, context)
        target = self._element(ctx, element)
        if action not in actions_of(type(target)):
            known = ", ".join(sorted(actions_of(type(target)))) or "none"
            raise ServiceError(
                f"{element} has no action {action!r}", hint=f"actions: {known}", code="not_found"
            )
        mode = ctx.mode_of(target)
        if mode == "operation":
            self._require_operation()
            if not confirmed:
                raise ServiceError(
                    f"{element}.{action} acts on real equipment and needs a human confirmation",
                    hint="ask the user to confirm, then call again with confirmed=true",
                    code="forbidden",
                )
            if target.driver is not None:
                target.driver.confirm = lambda command: True  # confirmed by the user for this action
        values = [parse_quantity(a) if isinstance(a, str | dict) and _has_unit(a) else a for a in args]
        named = {
            k: parse_quantity(v) if isinstance(v, str | dict) and _has_unit(v) else v
            for k, v in (kwargs or {}).items()
        }
        try:
            with state.lock, ctx.as_origin(self._origin(state)):
                value = getattr(target, action)(*values, **named)
        except (DriverError, ValueError, TypeError, u.UnitsError) as error:
            raise ServiceError(str(error), hint=_hint_for(error), code="failed") from error
        finally:
            if target.driver is not None:
                target.driver.confirm = None
        entry = ctx.journal.entries("action")[-1]
        return ActionResult(
            element=element,
            action=action,
            mode=mode,
            predicted_s=float(entry.predicted or 0.0),
            measured_s=None if entry.measured is None else float(entry.measured),
            value=None if value is None else str(format_value(value)),
        )

    def stop_all(self, session: str, context: str) -> None:
        """Emergency stop of every device of a context (always allowed)."""
        self._context(session, context).stop_all()

    def shutdown(self) -> None:
        """Close every session and stop the workers."""
        for session in list(self._sessions):
            self.close_session(session)
        self._executor.shutdown(wait=False, cancel_futures=True)


def _has_unit(value: Any) -> bool:
    if isinstance(value, dict):
        return "unit" in value
    try:
        quantity = u.Quantity(value)
    except (TypeError, ValueError):
        return False
    return quantity.unit != u.dimensionless_unscaled
