"""The MCP server of LUMOS: tools translating the Model Context Protocol into calls of the service layer.

Step P13.2 (REQ-MCP-1, REQ-MCP-2, ADR-0016). The server holds one session of :class:`~lumos.service.Service`
(origin ``mcp``) and a *current context*: the last context created, loaded or named by a tool, used when a
tool is called without ``context``. Every tool is a thin translation: the work, the checks and the safeguards
are in the service, so that the assistant, the web interface and Python give the same results.
"""

from __future__ import annotations

import base64
import functools
import inspect
import logging
import time
from collections.abc import Callable
from typing import Annotated, Any

import anyio
from mcp.server.mcpserver import Context as MCPContext
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, ImageContent, TextContent
from pydantic import BaseModel
from starlette.applications import Starlette

from lumos.mcp.operation import register_operation_tools
from lumos.mcp.prompts import register_prompts
from lumos.mcp.resources import register_resources
from lumos.service import Service, ServiceConfig, ServiceError
from lumos.service.models import (
    ArchiveInfo,
    ConnectionInfo,
    ContextSummary,
    DocHit,
    ElementSummary,
    ElementTypeInfo,
    Issue,
    JobInfo,
    Page,
    PortSummary,
    PresetInfo,
    StudyResult,
)
from lumos.service.service import acting_as
from lumos.service.summaries import job_text, study_text, summary_text

__all__ = ["INSTRUCTIONS", "LumosTools", "ViewLink", "create_http_app", "create_server"]

logger = logging.getLogger("lumos.mcp")


class ViewLink(BaseModel):
    """A link opening a context in the web interface, to give to the user (tool ``get_view_link``)."""

    url: str
    context: str
    message: str


#: Instructions given to the assistant when it connects.
INSTRUCTIONS = """\
LUMOS simulates light from a source or a scene to detected data through optical elements connected in a
graph (a "context"), and can drive real optical equipment.

How to work with it:
- Start from a preset (list_presets, load_preset) or an empty context (create_context). The last context
  created, loaded or named becomes the current one: the other tools use it when "context" is omitted.
- Find element types with list_element_types and their parameters, units and ports with get_element_schema.
- Write every physical quantity as text with its unit, e.g. "0.5 mm", "1.6 um", "2 mW", "10 mas"; a number
  alone is only accepted for dimensionless values.
- Ports are written "element.port" (e.g. "laser.out", "meter.in"); connect an output to an input.
- Check a context with validate_context before running it; run computes outputs in the background and
  reports progress; get_result and inspect_port summarise a computed signal with its unit.
- Errors give a hint saying how to fix the request: follow it.
"""


def _tool_error(error: BaseException) -> ToolError:
    if isinstance(error, ServiceError):
        text = error.message
        if error.hint:
            text += f"\nhint: {error.hint}"
        return ToolError(f"{text}\ncode: {error.code}")
    return ToolError(f"{type(error).__name__}: {error}\nhint: check the arguments of the tool")


def _guarded(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Turn the errors of ``fn`` into tool errors that the assistant reads (message, hint, code)."""
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def run_async(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except ToolError:
                raise
            except Exception as error:
                raise _tool_error(error) from error

        return run_async

    @functools.wraps(fn)
    def run(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as error:
            raise _tool_error(error) from error

    return run


def _as_mcp(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Journal what ``fn`` does with the origin ``mcp``, even in the session of a browser."""
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def run_async(*args: Any, **kwargs: Any) -> Any:
            with acting_as("mcp"):
                return await fn(*args, **kwargs)

        return run_async

    @functools.wraps(fn)
    def run(*args: Any, **kwargs: Any) -> Any:
        with acting_as("mcp"):
            return fn(*args, **kwargs)

    return run


def _job_result(job: JobInfo) -> CallToolResult:
    """Return a run as one line of text, with the :class:`JobInfo` as structured content."""
    return CallToolResult(
        content=[TextContent(type="text", text=job_text(job))], structured_content=job.model_dump(mode="json")
    )


def _study_result(study: StudyResult, max_rows: int) -> CallToolResult:
    """Return a study as a CSV table, with the :class:`StudyResult` as structured content."""
    structured = study.model_dump(
        mode="json", exclude={"points": {"__all__": {"outputs": {"__all__": {"figure_png"}}}}}
    )
    text = study_text(study, max_rows=max_rows)
    return CallToolResult(content=[TextContent(type="text", text=text)], structured_content=structured)


def _port_result(summary: PortSummary) -> CallToolResult:
    """Return a signal as a short text and a PNG figure, with its :class:`PortSummary` as structured data."""
    content: list[Any] = [TextContent(type="text", text=summary_text(summary))]
    if summary.figure_png is not None:
        data = base64.b64encode(summary.figure_png).decode("ascii")
        content.append(ImageContent(type="image", data=data, mime_type="image/png"))
    structured = summary.model_dump(mode="json", exclude={"figure_png"})
    return CallToolResult(content=content, structured_content=structured)


class LumosTools:
    """The state shared by the tools of one server: the service, its session and the current context.

    The current context is the *focus* of the session (:meth:`Service.focus`, P14.18): in the session of a
    browser it is the context open in the interface, and the interface follows the contexts the tools focus.

    Parameters
    ----------
    service
        Service layer executing the requests.
    """

    def __init__(self, service: Service, session: str | None = None) -> None:
        self.service = service
        self.session = session or service.create_session(origin="mcp").id

    @property
    def current(self) -> str | None:
        """The current context: the one in the foreground of the session."""
        return self.service.get_focus(self.session).context

    @current.setter
    def current(self, context: str | None) -> None:
        self.service.focus(self.session, context)

    def context(self, context: str | None) -> str:
        """Return ``context``, or the current context when it is ``None``; ``context`` becomes current."""
        if context is None:
            if self.current is None:
                raise ServiceError(
                    "no current context",
                    hint="create one with create_context or load one with load_preset",
                    code="not_found",
                )
            return self.current
        self.current = context  # checks that it exists
        return context

    def opened(self, summary: ContextSummary) -> ContextSummary:
        """Make a new context the current one."""
        self.current = summary.id
        return summary


def create_server(
    service: Service | None = None,
    *,
    config: ServiceConfig | None = None,
    name: str = "lumos",
    session: str | None = None,
    can_operate: bool = True,
    view_link: Callable[[str], str] | None = None,
) -> MCPServer:
    """Create the MCP server of LUMOS.

    Parameters
    ----------
    service
        Service layer to use (a new one with ``config`` by default).
    config
        Configuration of the new service (operation, quotas, directories).
    name
        Name of the server announced to the clients.
    session
        Existing session of ``service`` to work in (the web instance gives the session of the browser, so that
        the assistant sees the contexts of the interface); a new session of origin ``mcp`` by default.
    can_operate
        Whether the user may drive real equipment (operators); otherwise actions reach simulated elements.
    view_link
        Gives the URL opening a context of the session in the web interface (tool ``get_view_link``); ``None``
        for a server without web interface (stdio).

    Returns
    -------
    MCPServer
        Server to run over stdio (``server.run("stdio")``) or HTTP.
    """
    import lumos

    service = service or Service(config)
    state = LumosTools(service, session)
    server: MCPServer = MCPServer(name, title="LUMOS", instructions=INSTRUCTIONS, version=lumos.__version__)
    session = state.session

    def tool(fn: Callable[..., Any]) -> Callable[..., Any]:
        server.add_tool(_guarded(_as_mcp(fn)))
        return fn

    # --- Discovery -----------------------------------------------------------------------------------------

    @tool
    def list_element_types(query: str = "", page: int = 1, page_size: int = 50) -> Page[ElementTypeInfo]:
        """List the element types of LUMOS (sources, optics, detectors, processing...) with their ports.

        ``query`` keeps the types whose identifier or summary contains all its words; long lists come a page
        at a time (``next_page``).
        """
        return service.list_element_types(query=query, page=page, page_size=page_size)

    @tool
    def get_element_schema(type_id: str) -> dict[str, Any]:
        """JSON Schema of an element type: parameters with units (``x-lumos-unit``) and bounds, ports."""
        return service.get_element_schema(type_id)

    @tool
    def search_docs(query: str, limit: int = 5) -> list[DocHit]:
        """Search the documentation of LUMOS (guides, physics, element types); returns excerpts."""
        return service.search_docs(query, limit=limit)

    # --- Contexts ------------------------------------------------------------------------------------------

    @tool
    def create_context(name: str = "", mode: str = "simulation") -> ContextSummary:
        """Create an empty context (graph of elements), which becomes the current context.

        ``mode`` is ``"simulation"`` (default) or ``"operation"`` (drives real equipment, if enabled).
        """
        return state.opened(service.create_context(session, name=name, mode=mode))

    @tool
    def list_contexts() -> list[ContextSummary]:
        """List the contexts of this conversation."""
        return service.list_contexts(session)

    @tool
    def describe_context(context: str | None = None) -> ContextSummary:
        """Describe a context: elements with their values and units, connections, mode, observation band."""
        return service.describe_context(session, state.context(context))

    @tool
    def close_context(context: str | None = None) -> str:
        """Forget a context."""
        identifier = state.context(context)
        service.close_context(session, identifier)
        state.current = None
        return f"context {identifier} closed"

    @tool
    def get_view_link(context: str | None = None) -> ViewLink:
        """Give the link showing a context to the user in the web interface (current context by default).

        Share it with the user so that they see what you built, also when their page is closed or was never
        opened (e.g. when LUMOS is a connector of Claude.ai or ChatGPT). Anyone with the link opens this
        session: give it to the user only.
        """
        if view_link is None:
            raise ServiceError(
                "this MCP server has no web interface",
                hint="run `lumos web` and connect the assistant with its connector link (Assistant panel); "
                "or give the user the .lumos document of export_lumos to open in the interface",
                code="not_found",
            )
        identifier = state.context(context)
        url = view_link(identifier)
        return ViewLink(
            url=url, context=identifier, message=f"Open {url} to see the context {identifier} in LUMOS."
        )

    @tool
    def list_presets() -> list[PresetInfo]:
        """List the preset contexts (ready-made instruments and benches) with their main outputs."""
        return service.list_presets()

    @tool
    def load_preset(name: str) -> ContextSummary:
        """Load a preset into a new context, which becomes the current context."""
        return state.opened(service.load_preset(session, name))

    @tool
    def import_lumos(document: str) -> ContextSummary:
        """Create a context from the JSON text of a ``.lumos`` document; it becomes the current context."""
        return state.opened(service.import_lumos(session, document))

    @tool
    def export_lumos(context: str | None = None) -> str:
        """Return the ``.lumos`` document (JSON text) of a context, to save or share it."""
        return service.export_lumos(session, state.context(context))

    @tool
    def export_python(context: str | None = None) -> str:
        """Return a Python script building and running the same context with the LUMOS library."""
        return service.export_python(session, state.context(context))

    # --- Editing -------------------------------------------------------------------------------------------

    @tool
    def add_element(
        type_id: str,
        name: str | None = None,
        params: dict[str, Any] | None = None,
        context: str | None = None,
    ) -> ElementSummary:
        """Add an element of type ``type_id`` named ``name``.

        ``params`` gives parameter values with their units, e.g.
        ``{"power": "2 mW", "wavelength": "1.55 um"}``; the others keep their defaults
        (see get_element_schema).
        """
        return service.add_element(session, state.context(context), type_id, name=name, params=params)

    @tool
    def remove_element(element: str, context: str | None = None) -> str:
        """Remove an element and its connections."""
        service.remove_element(session, state.context(context), element)
        return f"{element} removed"

    @tool
    def connect(source: str, target: str, context: str | None = None) -> ConnectionInfo:
        """Connect the output ``source`` to the input ``target``, both written ``"element.port"``."""
        return service.connect(session, state.context(context), source, target)

    @tool
    def disconnect(source: str, target: str, context: str | None = None) -> str:
        """Remove the connection from ``source`` to ``target``."""
        service.disconnect(session, state.context(context), source, target)
        return f"{source} -> {target} disconnected"

    @tool
    def set_param(path: str, value: Any, context: str | None = None) -> ElementSummary:
        """Set a parameter ``"element.parameter"`` (``"group.child.parameter"`` in groups).

        Quantities are written with their unit, e.g. ``"0.5 mm"``; settings take plain values.
        """
        return service.set_param(session, state.context(context), path, value)

    @tool
    def set_observation(
        minimum: str, maximum: str, n_wavelengths: int = 1, context: str | None = None
    ) -> ContextSummary:
        """Set the observation band (e.g. ``"1 um"`` to ``"2 um"``) and its number of wavelengths."""
        return service.set_observation(
            session, state.context(context), minimum=minimum, maximum=maximum, n_wavelengths=n_wavelengths
        )

    @tool
    def create_group(
        name: str,
        elements: list[dict[str, Any]],
        expose: dict[str, str],
        connections: list[list[str]] | None = None,
        context: str | None = None,
    ) -> ElementSummary:
        """Add a group of elements behaving as one element (a photonic chip, an objective, a source...).

        ``elements``: ``[{"type": type_id, "name": ..., "params": {...}}]``; ``connections``: pairs
        ``["child.port", "child.port"]`` inside the group; ``expose``: ports of the group mapped to internal
        ports, e.g. ``{"in": "lens.in", "out": "camera.out"}``. Children are reached as ``"group.child"``.
        """
        return service.create_group(
            session,
            state.context(context),
            name,
            elements=elements,
            connections=[tuple(pair) for pair in connections or []],  # type: ignore[misc]
            expose=expose,
        )

    @tool
    def build_pipeline(
        source: str, steps: list[dict[str, Any]], context: str | None = None
    ) -> list[ElementSummary]:
        """Chain processing elements after the output ``source`` (e.g. ``"camera.out"``).

        Each step is ``{"type": type_id, "name": ..., "params": {...}}``; each one reads the previous output.
        """
        return service.build_pipeline(session, state.context(context), source, steps)

    # --- Verification --------------------------------------------------------------------------------------

    @tool
    def validate_context(context: str | None = None) -> list[Issue]:
        """List the problems that would make a run fail (unconnected inputs, missing band...) with fixes."""
        return service.validate_context(session, state.context(context))

    @tool
    def explain_sampling(element: str, context: str | None = None) -> dict[str, str]:
        """Explain how a propagation element is computed: method chosen, reason, cost, input and output grids.

        Use it to diagnose aliasing or under-sampling.
        """
        return service.explain_sampling(session, state.context(context), element)

    @tool
    def plan_propagation(element: str, context: str | None = None) -> dict[str, str]:
        """Propagation plan of an element (same as explain_sampling)."""
        return service.plan_propagation(session, state.context(context), element)

    # --- Execution -----------------------------------------------------------------------------------------

    @tool
    async def run(
        ctx: MCPContext,
        outputs: list[str] | None = None,
        archive: bool | str = False,
        wait_seconds: float = 300.0,
        context: str | None = None,
    ) -> Annotated[CallToolResult, JobInfo]:
        """Compute output ports (e.g. ``["camera.out"]``) and report the progress.

        Without ``outputs``, the main outputs of the context (those of its preset) are computed. ``archive``
        saves the run (``true`` or a name). Waits up to ``wait_seconds``; a longer run continues in the
        background: follow it with get_job. Read the outputs with get_result.
        """
        identifier = state.context(context)
        names = outputs or service.describe_context(session, identifier).outputs
        if not names:
            raise ServiceError(
                "no outputs to compute",
                hint="give outputs, e.g. outputs=['camera.out'] (see describe_context)",
            )
        job = service.run(session, identifier, names, archive=archive)
        return await follow(ctx, job, wait_seconds)

    async def follow(ctx: MCPContext, job: JobInfo, wait_seconds: float) -> CallToolResult:
        """Report the progress of a job for at most ``wait_seconds``, then return its state."""
        deadline = time.monotonic() + wait_seconds
        while job.status in ("queued", "running") and time.monotonic() < deadline:
            await ctx.report_progress(job.progress, 1.0, job.message or job.status)
            await anyio.sleep(0.05)
            job = service.get_job(session, job.id)
        if job.status == "done":
            await ctx.report_progress(1.0, 1.0, "done")
        return _job_result(job)

    # --- Studies -------------------------------------------------------------------------------------------

    @tool
    def set_seed(seed: int, context: str | None = None) -> ContextSummary:
        """Set the seed of every random process of a context (noise, turbulence, Monte-Carlo draws).

        Runs with the same seed give the same results; change it to draw other noise.
        """
        return service.set_context(session, state.context(context), seed=seed)

    @tool
    def set_free_parameter(
        path: str, free: bool = True, prior: dict[str, Any] | None = None, context: str | None = None
    ) -> ElementSummary:
        """Mark a parameter ``"element.field"`` as free with a prior, or fixed again (``free=false``).

        ``prior``: ``{"kind": "uniform" | "loguniform", "low": "190 mm", "high": "210 mm"}`` or
        ``{"kind": "normal", "mean": "200 mm", "std": "2 mm"}``. run_monte_carlo draws the free parameters.
        """
        return service.set_free(session, state.context(context), path, free, prior)

    @tool
    async def run_sweep(
        ctx: MCPContext,
        parameters: dict[str, Any],
        outputs: list[str] | None = None,
        mode: str = "grid",
        noise: str = "fixed",
        wait_seconds: float = 300.0,
        context: str | None = None,
    ) -> Annotated[CallToolResult, JobInfo]:
        """Run the context for several values of some parameters (a sweep) and report the progress.

        ``parameters`` maps paths to a list (``{"lens.focal_length": ["190 mm", "200 mm"]}``) or a range
        (``{"start": "190 mm", "stop": "210 mm", "num": 5, "log": false}``). ``mode``: ``grid`` (every
        combination) or ``zip`` (i-th values together). ``noise``: ``fixed`` (same noise at every point) or
        ``independent``. Read the points with get_study.
        """
        job = service.run_study(
            session,
            state.context(context),
            kind="sweep",
            parameters=parameters,
            outputs=outputs,
            mode=mode,
            noise=noise,
        )
        return await follow(ctx, job, wait_seconds)

    @tool
    async def run_monte_carlo(
        ctx: MCPContext,
        n: int,
        outputs: list[str] | None = None,
        noise: str = "independent",
        wait_seconds: float = 300.0,
        context: str | None = None,
    ) -> Annotated[CallToolResult, JobInfo]:
        """Monte-Carlo run: ``n`` draws of the free parameters from their priors (see set_free_parameter).

        With ``noise="independent"`` (default) each draw also has its own noise; without free parameters the
        run is repeated with independent noise. Read the points with get_study.
        """
        job = service.run_study(
            session, state.context(context), kind="sample", n=n, outputs=outputs, noise=noise
        )
        return await follow(ctx, job, wait_seconds)

    @tool
    def add_feedback(
        source: str, target: str, mode: str = "set", context: str | None = None
    ) -> ContextSummary:
        """Close a loop: feed the output ``source`` (``"ctrl.commands"``) back to the field ``target``.

        ``target`` is a field ``"element.field"`` (``"dm.commands"``); ``mode="add"`` adds the value at each
        step (an integrator), ``"set"`` replaces it. The graph stays acyclic: run the loop with run_loop.
        """
        return service.add_feedback(session, state.context(context), source, target, mode)

    @tool
    def remove_feedback(source: str, target: str, context: str | None = None) -> ContextSummary:
        """Remove the feedback from ``source`` to ``target``."""
        return service.remove_feedback(session, state.context(context), source, target)

    @tool
    async def run_loop(
        ctx: MCPContext,
        steps: int,
        period: str,
        outputs: list[str] | None = None,
        wait_seconds: float = 300.0,
        context: str | None = None,
    ) -> Annotated[CallToolResult, JobInfo]:
        """Run the context as a closed loop: ``steps`` iterations of ``period`` (e.g. ``"2 ms"``).

        Each step computes the outputs, applies the feedbacks and advances the clock; the fields driven by the
        feedbacks keep their final values. Read the steps (against time) with get_study.
        """
        job = service.run_loop(session, state.context(context), steps=steps, period=period, outputs=outputs)
        return await follow(ctx, job, wait_seconds)

    @tool
    def get_study(job: str, max_rows: int = 50) -> Annotated[CallToolResult, StudyResult]:
        """Points of a sweep or Monte-Carlo run: a CSV table (parameters, then outputs) and their data."""
        return _study_result(service.get_study(session, job), max_rows)

    @tool
    def get_job(job: str) -> Annotated[CallToolResult, JobInfo]:
        """Status, progress, duration and error (with a hint) of a run."""
        return _job_result(service.get_job(session, job))

    @tool
    def wait_job(job: str, timeout_seconds: float = 60.0) -> Annotated[CallToolResult, JobInfo]:
        """Wait for the end of a run (at most ``timeout_seconds``)."""
        return _job_result(service.wait_job(session, job, timeout=timeout_seconds))

    @tool
    def get_result(job: str, port: str, figure: bool = False) -> Annotated[CallToolResult, PortSummary]:
        """Summary of an output computed by a run: kind, unit, shape, statistics, values when few.

        ``figure=true`` adds a picture of the signal (intensity, image or curve).
        """
        return _port_result(service.get_result(session, job, port, figure=figure))

    @tool
    def inspect_port(
        port: str, figure: bool = True, context: str | None = None
    ) -> Annotated[CallToolResult, PortSummary]:
        """Compute (or reuse) the signal of an output port ``"element.port"`` and summarise it, with units.

        A picture of the signal is returned too, unless ``figure=false``.
        """
        return _port_result(service.inspect_port(session, state.context(context), port, figure=figure))

    @tool
    def radial_profile(
        port: str,
        bin_width: str | None = None,
        center: list[str] | None = None,
        normalize: str | None = None,
        context: str | None = None,
    ) -> dict[str, Any]:
        """Azimuthal mean of the intensity of a field at ``port``, ring by ring around ``center``.

        ``bin_width`` is the width of the rings in the unit of the grid (e.g. ``"20 um"`` in a focal plane,
        ``"10 mas"`` on the sky; default one pixel). ``normalize="peak"`` divides by the maximum intensity,
        which gives contrasts. Radii are the middles of the rings.
        """
        return service.radial_profile(
            session, state.context(context), port, bin_width=bin_width, center=center, normalize=normalize
        )

    @tool
    def get_journal(
        kind: str | None = None, limit: int = 50, context: str | None = None
    ) -> list[dict[str, Any]]:
        """Last entries of the journal of a context (``kind``: ``"action"``, ``"log"``...)."""
        return service.journal(session, state.context(context), kind=kind, limit=limit)

    # --- Catalogues and processing -------------------------------------------------------------------------

    @tool
    def resolve_object(name: str) -> dict[str, Any]:
        """Look up an astronomical object (star, planet...) in the catalogues; every field has its unit."""
        return service.resolve_object(name)

    @tool
    def evaluate_detection(
        port: str, peak: str, resolution: str, sigma: float = 5.0, context: str | None = None
    ) -> dict[str, Any]:
        """Contrast curve (detection limit versus separation) of the residual image at ``port``.

        ``peak`` is the peak of the unattenuated source (e.g. ``"4e6 adu"``) and ``resolution`` the resolution
        element in the units of the image grid (e.g. ``"20 um"`` or ``"40 mas"``).
        """
        return service.evaluate_detection(
            session, state.context(context), port, peak=peak, resolution=resolution, sigma=sigma
        )

    # --- Archives ------------------------------------------------------------------------------------------

    @tool
    def list_archives(limit: int = 50) -> list[ArchiveInfo]:
        """List the archives of runs, most recent first."""
        return service.list_archives(limit=limit)

    @tool
    def read_archive_metadata(path: str) -> dict[str, Any]:
        """Metadata of an archived run (parameters, versions, origin, outputs); path from list_archives."""
        return service.read_archive_metadata(path)

    register_operation_tools(server, state, tool, can_operate=can_operate)
    register_resources(server, state)
    register_prompts(server)
    server.lumos = state  # type: ignore[attr-defined]
    return server


def create_http_app(server: MCPServer, *, path: str = "/mcp", host: str = "127.0.0.1") -> Starlette:
    """Return the ASGI application serving ``server`` over streamable HTTP at ``path`` (step P13.6).

    The web backend mounts this application (P14), which then adds the authentication of its users; until
    then, serve it on the local machine only. ``host`` sets the protection against DNS rebinding: requests
    must name a local host when it is a loopback address.
    """
    return server.streamable_http_app(streamable_http_path=path, host=host)
