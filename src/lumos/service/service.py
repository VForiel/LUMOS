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
import difflib
import json
import logging
import threading
import time
import uuid
from collections.abc import Iterable, Mapping
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
from lumos.core.port import PortCompatibilityError
from lumos.core.registry import UnknownTypeError, registry
from lumos.io.lumos_format import from_document
from lumos.service import docs
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
    Issue,
    JobInfo,
    Page,
    PortSummary,
    PresetInfo,
    ServiceConfig,
    SessionInfo,
)
from lumos.service.summaries import summarize
from lumos.service.values import format_value, parse_quantity, parse_value

__all__ = ["Service"]

logger = logging.getLogger("lumos.service")


@dataclass
class _Job:
    info: JobInfo
    session: str
    result: Result | None = None
    future: concurrent.futures.Future[Any] | None = None
    started: float = 0.0


@dataclass
class _Session:
    info: SessionInfo
    contexts: dict[str, Context] = field(default_factory=dict)
    jobs: dict[str, _Job] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)


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

    # --- Sessions ------------------------------------------------------------------------------------------

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
            meta = json.loads(path.read_text(encoding="utf-8")).get("meta", {})
            found.append(
                PresetInfo(
                    name=path.stem, description=meta.get("description", ""), outputs=meta.get("outputs", [])
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
            band=band,
            n_wavelengths=int(ctx.observation.n_wavelengths),
            elements=elements,
            connections=connections,
            outputs=list(ctx.meta.get("outputs", [])),
        )

    def _element_summary(self, path: str, element: Any) -> ElementSummary:
        cls = type(element)
        groups: dict[str, dict[str, Any]] = {"param": {}, "setting": {}, "state": {}}
        for name, declared in cls.declared().items():
            value = getattr(element, name)
            unit = getattr(declared, "unit", None)
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
        )

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
        ctx = self._context(session, context)
        element = self._new_element(type_id, name, params)
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
            ctx.add(group)
        except (PortCompatibilityError, GraphError, TypeError) as error:
            raise ServiceError(
                str(error), hint="connect and expose ports of the elements of the group ('child.port')"
            ) from error
        return self._element_summary(ctx.id_of(group), group)

    def remove_element(self, session: str, context: str, element: str) -> None:
        """Remove an element and its connections."""
        ctx = self._context(session, context)
        ctx.remove(self._element(ctx, element))

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
            ctx.connect(self._port(ctx, source), self._port(ctx, target))
        except (PortCompatibilityError, GraphError, TypeError) as error:
            raise ServiceError(str(error), hint=_hint_for(error)) from error
        return ConnectionInfo(source=source, target=target)

    def disconnect(self, session: str, context: str, source: str, target: str) -> None:
        """Remove a connection."""
        ctx = self._context(session, context)
        try:
            ctx.disconnect(self._port(ctx, source), self._port(ctx, target))
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
        try:
            ctx.observation.band = (low, high)
            ctx.observation.n_wavelengths = int(n_wavelengths)
        except ValueError as error:
            raise ServiceError(str(error), hint="give 0 < minimum < maximum") from error
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
        """How a propagation element will be computed: method chosen by the planner and why (ADR-0005)."""
        from lumos.propagation.planner import plan_for

        ctx = self._context(session, context)
        target = self._element(ctx, element)
        if not hasattr(target, "distance"):
            raise ServiceError(f"{element} is not a propagation", hint="give a FreeSpace element")
        incoming = ctx._graph.incoming(target.port("in"))
        if not incoming:
            raise ServiceError(f"{element}.in is not connected", hint=_hint_for(GraphError("not connected")))
        source = incoming[0].source
        signal = ctx.run([source])[source]
        output_grid = target.output_grid or signal.grid
        plan = plan_for(signal, float(target.distance.to_value(u.m)), output_grid)
        return {
            "method": plan.method,
            "reason": plan.reason,
            "cost": f"{plan.cost:.3g} operations",
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
        state = self._session(session)
        ctx = self._context(session, context)
        running = [j for j in state.jobs.values() if j.info.status in ("queued", "running")]
        if len(running) >= self.config.max_jobs_per_session:
            raise ServiceError(
                f"quota of {self.config.max_jobs_per_session} runs at a time reached",
                hint="wait for the running jobs (wait_job) before starting another one",
                code="quota",
            )
        names = list(outputs)
        job = _Job(
            JobInfo(id=uuid.uuid4().hex[:10], context=context, outputs=names, status="queued"), session
        )
        state.jobs[job.info.id] = job

        def progress(done: int, total: int, element: str) -> None:
            job.info.progress = done / max(total, 1)
            job.info.message = f"computed {element} ({done}/{total})"
            if time.perf_counter() - job.started > self.config.max_run_seconds:
                raise TimeoutError(f"the run exceeded {self.config.max_run_seconds:g} s")

        def work() -> None:
            job.info.status = "running"
            job.started = time.perf_counter()
            try:
                with state.lock, ctx.as_origin(state.info.origin):
                    job.result = ctx.run(
                        names, archive=archive, archive_root=self.config.archive_root, progress=progress
                    )
                job.info.progress = 1.0
                job.info.status = "done"
                if job.result.archive is not None:
                    job.info.archive = str(job.result.archive.path)
            except Exception as error:
                job.info.status = "failed"
                job.info.error = str(error)
                job.info.hint = _hint_for(error)
                logger.info("run %s failed: %s", job.info.id, error)
            finally:
                job.info.seconds = time.perf_counter() - job.started

        job.future = self._executor.submit(work)
        return job.info.model_copy()

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
        state = self._session(session)
        ctx = self._context(session, context)
        reference = self._port(ctx, port)
        try:
            with state.lock, ctx.as_origin(state.info.origin):
                signal = ctx.run([reference])[reference]
        except Exception as error:
            raise ServiceError(str(error), hint=_hint_for(error), code="failed") from error
        return summarize(port, signal, figure=figure)

    def journal(
        self, session: str, context: str, *, kind: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Last entries of the journal of a context (``kind``: ``"action"``, ``"log"``...)."""
        entries = self._context(session, context).journal.to_json()
        if kind is not None:
            entries = [e for e in entries if e.get("kind") == kind]
        return json.loads(json.dumps(entries[-limit:], default=str))

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

        state = self._session(session)
        ctx = self._context(session, context)
        reference = self._port(ctx, port)
        with state.lock:
            image = ctx.run([reference])[reference]
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
            with state.lock, ctx.as_origin(state.info.origin):
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
