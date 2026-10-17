"""The context: container and coordinator of a LUMOS system (``REQ-CTX-*``).

A :class:`Context` holds the graph of elements and everything that is global to a simulation or an
experiment: the operating mode (simulation or operation) and the digital twin (REQ-EL-4, REQ-EL-7), the array
backend, the observation settings (spectral band, site, pointing; REQ-CTX-6) and the random seed from which
every element gets its own reproducible random stream.

::

    ctx = Context(seed=42)
    star = ctx.add(Star(name="vega"))
    star >> Telescope(name="tel") >> Camera(name="cam")   # connects unique ports, adds the elements
    image = ctx.run(["cam.out"])["cam.out"]
"""

from __future__ import annotations

import itertools
import logging
import math
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from astropy import units as u
from astropy.time import Time

from lumos.backend._core import known, use
from lumos.core.clock import Clock, duration_si
from lumos.core.driver import Command, DriverError
from lumos.core.graph import Graph, GraphError, Result
from lumos.core.journal import ORIGINS, Journal, JournalEntry, _Capture
from lumos.core.port import PortRef
from lumos.units import require_unit, to_si

__all__ = ["MODES", "Context", "Observation", "SweepPoint"]

logger = logging.getLogger("lumos.context")

Mode = Literal["simulation", "operation"]
MODES: tuple[str, ...] = ("simulation", "operation")


@dataclass(frozen=True)
class SweepPoint:
    """One point of a sweep: the field values and the computed outputs."""

    params: dict[str, Any]
    outputs: Result


class Observation:
    """Observation settings shared by the elements of a context (REQ-CTX-6).

    Attributes
    ----------
    band
        ``(min, max)`` wavelengths of the spectral band, or ``None``.
    n_wavelengths
        Number of spectral samples of the band.
    rule
        Quadrature rule of the band, ``"midpoint"`` or ``"gauss"`` (:func:`lumos.fields.spectral_samples`).
    extra
        Free-form JSON-compatible settings (site, platform, pointing...).
    """

    def __init__(self) -> None:
        self._listeners: list[Any] = []
        self._band: tuple[u.Quantity, u.Quantity] | None = None
        self.n_wavelengths = 1
        self.rule: Literal["midpoint", "gauss"] = "midpoint"
        self.extra: dict[str, Any] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute and notify the context (observation-dependent elements are recomputed)."""
        object.__setattr__(self, name, value)
        if not name.startswith("_"):
            self._changed()

    def _changed(self) -> None:
        for listener in getattr(self, "_listeners", []):
            listener()

    @property
    def band(self) -> tuple[u.Quantity, u.Quantity] | None:
        """Spectral band ``(min, max)``."""
        return self._band

    @band.setter
    def band(self, value: tuple[u.Quantity, u.Quantity] | None) -> None:
        if value is None:
            self._band = None
            self._changed()
            return
        low, high = value
        lo = float(
            to_si(
                require_unit(low, u.m, name="band minimum", equivalencies=u.spectral()),
                u.m,
                equivalencies=u.spectral(),
            )
        )
        hi = float(
            to_si(
                require_unit(high, u.m, name="band maximum", equivalencies=u.spectral()),
                u.m,
                equivalencies=u.spectral(),
            )
        )
        if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo < hi):
            raise ValueError("the band must satisfy 0 < min < max")
        self._band = (low, high)
        self._changed()

    def wavelengths(self) -> tuple[u.Quantity, np.ndarray]:
        """Return the wavelengths and normalised weights sampling the band."""
        from lumos.fields.spectral import spectral_samples

        if self._band is None:
            raise ValueError("no observation band is set (ctx.observation.band = (min, max))")
        low, high = self._band
        unit = low.unit
        lo, hi = (cast(u.Quantity, q.to(unit, equivalencies=u.spectral())) for q in (low, high))
        center = cast(u.Quantity, (lo + hi) / 2)
        width = cast(u.Quantity, hi - lo)
        return spectral_samples(center, width, int(self.n_wavelengths), rule=self.rule)


class Context:
    """Container and coordinator of elements.

    Parameters
    ----------
    mode
        ``"simulation"`` (default) or ``"operation"``.
    twin
        Whether the digital twin runs alongside the real equipment in operation mode (default ``True``).
    backend
        Array backend used by :meth:`run`: ``"auto"`` (default, the process default) or a backend name.
    seed
        Seed of the random streams; a random seed is drawn (and kept) if ``None``.
    start
        Absolute start date of the clock (``astropy.time.Time``, UTC); defaults to now.
    name
        Optional name of the context.
    """

    def __init__(
        self,
        *,
        mode: Mode = "simulation",
        twin: bool = True,
        backend: str = "auto",
        seed: int | None = None,
        start: Time | None = None,
        name: str | None = None,
    ) -> None:
        self._graph = Graph()
        self.clock = Clock(start)
        self._parallel: list[list[float]] = []
        self._overrides: dict[int, str] = {}
        self._twin_states: dict[int, dict[str, Any]] = {}
        self._links: list[tuple[str, str, Any, Any, Any]] = []
        self._recording: Any = None
        self._recording_start = 0.0
        self.journal = Journal()
        self._origins: list[str] = ["python"]
        self.mode = mode
        self.twin = twin
        self.backend = backend
        self._seed = int(np.random.SeedSequence().entropy) if seed is None else int(seed)  # type: ignore[arg-type]
        self.name = name
        self.meta: dict[str, Any] = {}
        self.ui: dict[str, Any] = {}
        self.observation = Observation()
        self.observation._listeners.append(self._observation_changed)

    # --- Settings -----------------------------------------------------------------------------

    @property
    def mode(self) -> Mode:
        """Operating mode: ``"simulation"`` or ``"operation"``."""
        return cast(Mode, self._mode)

    @mode.setter
    def mode(self, value: str) -> None:
        if value not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {value!r}")
        self._mode = cast(Mode, value)

    @property
    def twin(self) -> bool:
        """Whether the digital twin is enabled in operation mode."""
        return self._twin

    @twin.setter
    def twin(self, value: bool) -> None:
        self._twin = bool(value)

    @property
    def backend(self) -> str:
        """Backend used by :meth:`run` (``"auto"``: the process default)."""
        return self._backend

    @backend.setter
    def backend(self, value: str) -> None:
        if value != "auto" and value not in known():
            raise ValueError(f"unknown backend {value!r}; use 'auto' or one of {', '.join(known())}")
        self._backend = value

    # --- Elements -----------------------------------------------------------------------------

    @property
    def graph(self) -> Graph:
        """The graph of elements."""
        return self._graph

    def add(self, *elements: Any, id: str | None = None) -> Any:
        """Add elements and return them (the element itself when only one is given).

        ``id`` sets the identifier of a single element (by default its name, or its class name and a counter).
        """
        if id is not None and len(elements) != 1:
            raise ValueError("an explicit id can only be given for a single element")
        for element in elements:
            owner = getattr(element, "_lumos_container", None)
            if owner is not None and owner is not self:
                raise GraphError(f"{element!r} already belongs to another context")
            if owner is self:
                continue
            self._graph.add(element, id=id)
            element._lumos_container = self
        return elements[0] if len(elements) == 1 else elements

    def remove(self, element: Any) -> None:
        """Remove ``element`` and its connections."""
        self._graph.remove(element)
        element._lumos_container = None

    def connect(self, source: PortRef, target: PortRef) -> None:
        """Connect an output port to an input port (adding the target element if needed)."""
        if target.owner not in self._graph:
            self.add(target.owner)
        self._graph.connect(source, target)

    def id_of(self, element: Any) -> str:
        """Return the identifier of ``element`` in the context."""
        return self._graph.id_of(element)

    @property
    def elements(self) -> dict[str, Any]:
        """Elements by identifier."""
        return self._graph.elements

    def __getitem__(self, path: str) -> Any:
        """Return the element ``"id"`` or, inside groups, ``"group.child"``."""
        head, _, rest = path.partition(".")
        element = self._graph[head]
        return element[rest] if rest else element

    def port(self, path: str) -> PortRef:
        """Return the port ``"element.port"`` (``"group.child.port"`` inside groups)."""
        owner, _, name = path.rpartition(".")
        if not owner:
            raise KeyError(f"{path!r} is not of the form 'element.port'")
        return self[owner].port(name)

    def __contains__(self, item: object) -> bool:
        """Whether an identifier or an element is in the context."""
        return item in self._graph

    def __len__(self) -> int:
        """Return the number of elements."""
        return len(self._graph)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the elements."""
        return iter(self._graph)

    # --- Execution ----------------------------------------------------------------------------

    def run(
        self,
        outputs: Iterable[Any],
        *,
        archive: bool | str = False,
        archive_root: Any = None,
    ) -> Result:
        """Compute the requested outputs (ports, elements or ``"id.port"`` strings), see :meth:`Graph.run`.

        Records of the LUMOS loggers emitted during the run (numerical choices, physical warnings) are copied
        to the journal.

        Parameters
        ----------
        outputs
            Output ports, elements or ``"id.port"`` strings.
        archive
            ``True`` or a name: write an archive of the run (:func:`lumos.results.new_archive`) holding the
            context, the metadata, the journal and the computed outputs (in ``data/<id.port>``); it is
            returned as ``result.archive``.
        archive_root
            Directory of the archive (default: next to the executed script or notebook).
        """
        targets = [self.port(t) if isinstance(t, str) and "." in t else t for t in outputs]
        self.apply_links()
        root = logging.getLogger("lumos")
        handler = _Capture(self.journal, self._entry)
        previous_level = root.level
        root.addHandler(handler)
        if root.getEffectiveLevel() > logging.INFO:
            root.setLevel(logging.INFO)
        started = time.perf_counter()
        try:
            if self._backend == "auto":
                result = self._graph.run(targets)
            else:
                with use(self._backend):
                    result = self._graph.run(targets)
            if self._mode == "operation":
                self._measure(result)
        finally:
            root.removeHandler(handler)
            root.setLevel(previous_level)
        if archive:
            self._archive_run(result, archive, archive_root, time.perf_counter() - started)
        return result

    # --- Sequences -------------------------------------------------------------------------------

    @contextmanager
    def record(self) -> Iterator[Any]:
        """Record the actions of the block: ``with ctx.record() as sequence: ...`` (REQ-HW-7)."""
        from lumos.core.sequence import Sequence

        sequence = Sequence()
        previous, previous_start = self._recording, self._recording_start
        self._recording, self._recording_start = sequence, self.clock.elapsed_si
        try:
            yield sequence
        finally:
            self._recording, self._recording_start = previous, previous_start

    def replay(self, sequence: Any) -> list[Any]:
        """Execute the actions of a recorded sequence in the current mode; return their results."""
        results = []
        for step in sequence:
            command = step.command
            try:
                element = self[command.element]
            except (KeyError, GraphError) as error:
                raise KeyError(
                    f"the sequence needs the element {command.element!r}, absent from the context"
                ) from error
            results.append(getattr(element, command.action)(*command.args, **command.kwargs))
        return results

    # --- Digital twin ----------------------------------------------------------------------------

    def link(
        self,
        source: str,
        target: str,
        *,
        scale: Any = 1.0,
        offset: Any = None,
        function: Callable[[Any], Any] | None = None,
    ) -> None:
        """Make the field ``target`` follow the field ``source`` (paths ``"element.field"``).

        ``target = offset + scale * source``, or ``function(source)``: a stage carrying a lens drives a
        propagation distance, a phase controller drives phase shifters. Links are applied after every action
        and before every run, so that the digital twin follows the bench.
        """
        self._parameter(source)
        self._parameter(target)
        self._links.append((source, target, scale, offset, function))
        self.apply_links()

    def apply_links(self) -> None:
        """Copy the linked fields to their targets."""
        for source, target, scale, offset, function in self._links:
            owner, field = self._parameter(source)
            value = getattr(owner, field)
            value = function(value) if function is not None else value * scale
            if offset is not None and function is None:
                value = offset + value
            element, name = self._parameter(target)
            current = getattr(element, name)
            try:
                same = bool(np.all(current == value))
            except (TypeError, ValueError, u.UnitsError):
                same = False
            if not same:
                setattr(element, name, value)

    def _measure(self, result: Result) -> None:
        """Replace the outputs of measuring elements by their measurements (operation mode)."""
        for key in list(result):
            element = key.owner
            driver = getattr(element, "driver", None)
            if driver is None or not hasattr(driver, "acquire") or self.mode_of(element) != "operation":
                continue
            simulated = dict.__getitem__(result, key)
            if self._twin:
                self._feed(element, driver, simulated)
            measured = driver.acquire()
            result[key] = measured
            if self._twin:
                result.twin[key] = simulated
                residual = _residual(measured, simulated)
                if residual is not None:
                    result.residuals[key] = residual

    def _feed(self, element: Any, driver: Any, simulated: Any) -> None:
        """Send the prediction of the twin to an emulated measuring device."""
        hook = getattr(element, "twin_feed", None)
        if hook is None or not hasattr(driver, "feed"):
            return
        inputs = self._graph._inputs(element, {})
        data = hook(inputs, simulated)
        if data is None:
            return
        try:
            driver.feed(data)
        except DriverError as error:  # a physical device does not accept simulation commands
            logger.info("%s does not take the data of the twin: %s", self.path_of(element), error)

    def _archive_run(self, result: Result, archive: bool | str, root: Any, seconds: float) -> None:
        from lumos.results.archive import new_archive

        names = [result.name_of(key) for key in result]
        name = archive if isinstance(archive, str) else self.name
        run = new_archive(
            self, name, root=root, metadata={"outputs": names, "durations": {"run": max(seconds, 1e-9)}}
        )
        for key, name in zip(list(result), names, strict=True):
            try:
                run.save_data(name, dict.__getitem__(result, key))
            except TypeError as error:
                logger.warning("output %s is not saved in the archive: %s", name, error)
        result.archive = run

    # --- Journal ----------------------------------------------------------------------------------

    @property
    def origin(self) -> str:
        """Origin of the actions being executed: ``"python"``, ``"web"`` or ``"mcp"``."""
        return self._origins[-1]

    @contextmanager
    def as_origin(self, origin: str) -> Iterator[None]:
        """Attribute the actions of the block to ``origin`` in the journal."""
        if origin not in ORIGINS:
            raise ValueError(f"origin must be one of {ORIGINS}, got {origin!r}")
        self._origins.append(origin)
        try:
            yield
        finally:
            self._origins.pop()

    def _entry(self, kind: str) -> JournalEntry:
        return JournalEntry(
            kind=kind,
            elapsed=self.clock.elapsed_si,
            date=str(self.clock.time.isot),
            origin=self.origin,
        )

    def _log(self, entry: JournalEntry, message: str, *args: Any) -> None:
        self.journal.append(entry)
        logger.info(message, *args, extra={"lumos": entry.to_json()})

    # --- Time -------------------------------------------------------------------------------------

    @property
    def time(self) -> Time:
        """Current absolute date of the clock."""
        return self.clock.time

    @property
    def elapsed(self) -> u.Quantity:
        """Time elapsed since the start of the clock."""
        return self.clock.elapsed

    def wait(self, duration: u.Quantity) -> None:
        """Let ``duration`` pass (counts as one duration inside a :meth:`parallel` block)."""
        seconds = duration_si(duration)
        entry = self._entry("wait")
        entry.predicted = seconds
        self._log(entry, "wait %.6g s", seconds)
        self._advance(seconds)

    @contextmanager
    def parallel(self) -> Iterator[None]:
        """Run the actions of the block simultaneously: the clock advances by the longest duration."""
        self._parallel.append([])
        try:
            yield
        finally:
            durations = self._parallel.pop()
            self._advance(max(durations, default=0.0))

    def _advance(self, seconds: float) -> None:
        if self._parallel:
            self._parallel[-1].append(seconds)
            return
        if seconds == 0:
            return
        self.clock.advance(seconds)
        for element in self._walk():
            if type(element).time_dependent:
                element._on_field_changed("time")

    @property
    def seed(self) -> int:
        """Seed of the random streams; changing it recomputes the stochastic elements."""
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._seed = int(value)
        for element in self._walk():
            if getattr(type(element), "stochastic", False):
                element._on_field_changed("seed")

    def _observation_changed(self) -> None:
        for element in self._walk():
            if getattr(type(element), "observation_dependent", False):
                element._on_field_changed("observation")

    def walk(self) -> Iterator[tuple[str, Any]]:
        """Every element of the context with its path, including the children of groups."""
        for element in self._walk():
            yield self.path_of(element), element

    def _walk(self) -> Iterator[Any]:
        """Every element of the context, including the children of groups."""
        stack = list(self._graph)
        while stack:
            element = stack.pop()
            yield element
            inner = getattr(element, "graph", None)
            if isinstance(inner, Graph):
                stack.extend(inner)

    # --- Modes and actions ---------------------------------------------------------------------

    def simulate(self, element: Any, flag: bool = True) -> None:
        """Force ``element`` to simulation, even in operation mode (``flag=False`` removes the override)."""
        if flag:
            self._overrides[id(element)] = "simulation"
        else:
            self._overrides.pop(id(element), None)

    def mode_of(self, element: Any) -> Mode:
        """Effective mode of ``element``: the context mode unless the element is forced to simulation."""
        return cast(Mode, self._overrides.get(id(element), self._mode))

    def path_of(self, element: Any) -> str:
        """Path of ``element`` from the context: ``"id"`` or ``"group.child"``."""
        parts = []
        node = element
        while node is not None and node is not self:
            container = node.container
            if container is None:
                raise GraphError(f"{element!r} is not in this context")
            parts.append(container.id_of(node))
            node = container
        return ".".join(reversed(parts))

    def drivers(self) -> dict[str, Any]:
        """Drivers attached to the elements, by element path."""
        found = {}
        for element in self._walk():
            driver = getattr(element, "driver", None)
            if driver is not None:
                found[self.path_of(element)] = driver
        return found

    def stop_all(self) -> None:
        """Emergency stop: stop every driver, then bring the equipment to its safe state (REQ-HW-4)."""
        drivers = self.drivers()
        for driver in drivers.values():
            try:
                driver.stop()
            except Exception:  # keep stopping the others
                logger.exception("stop of %s failed", type(driver).__name__)
        for driver in drivers.values():
            try:
                driver.safe_state()
            except Exception:
                logger.exception("safe state of %s failed", type(driver).__name__)
        entry = self._entry("log")
        self._log(entry, "emergency stop of %d driver(s)", len(drivers))

    @contextmanager
    def operation_guard(self) -> Iterator[Context]:
        """Run a block of operations; on ``KeyboardInterrupt`` or any error, stop everything and secure it."""
        try:
            yield self
        except BaseException:
            self.stop_all()
            raise

    def twin_state(self, element: Any) -> dict[str, Any]:
        """State measured on the equipment at the last command, next to the twin's simulated state."""
        return dict(self._twin_states.get(id(element), {}))

    def _perform(self, element: Any, spec: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Execute an action of ``element`` (called by the :func:`~lumos.core.action.action` wrapper)."""
        predicted = spec.duration_si(element, args, kwargs)
        path = self.path_of(element)
        command = Command(path, spec.name, tuple(args), dict(kwargs))
        if self._recording is not None:
            self._recording.append((self.clock.elapsed_si - self._recording_start) * u.s, command)
        entry = self._entry("action")
        entry.element, entry.action, entry.command = path, spec.name, command
        entry.args = ", ".join([str(a) for a in args] + [f"{k}={v}" for k, v in kwargs.items()])
        entry.mode = self.mode_of(element)
        entry.predicted = predicted
        if entry.mode == "simulation":
            result = spec.method(element, *args, **kwargs)
            self.apply_links()
            self._log(entry, "%s.%s simulated (%.6g s)", path, spec.name, predicted)
            self._advance(predicted)
            return result
        driver = element.driver
        if driver is None:
            raise DriverError(f"{path} has no driver; attach one or use ctx.simulate({path!r})")
        if not driver.connected:
            raise DriverError(f"the driver of {path} is not connected")
        started = time.perf_counter()
        reply = driver.send(command)
        wall = time.perf_counter() - started
        measured = wall if reply.duration is None else duration_si(reply.duration, name="measured duration")
        if self._twin:
            result = spec.method(element, *args, **kwargs)
            self._twin_states[id(element)] = dict(reply.state)
            self.apply_links()
        else:
            result = reply.value
            for name, value in reply.state.items():
                setattr(element, name, value)
        entry.measured = measured
        entry.driver = type(driver).__name__
        self._log(
            entry,
            "%s.%s sent to %s (predicted %.6g s, measured %.6g s)",
            path,
            spec.name,
            entry.driver,
            predicted,
            measured,
        )
        self._advance(measured)
        return result

    # --- Random streams -----------------------------------------------------------------------

    def spawn_key(self, target: Any) -> tuple[int, ...]:
        """Spawn key of the random stream of an element (or identifier): the UTF-8 bytes of its path.

        The path is the identifier of a top-level element and ``"group.child"`` inside groups.
        """
        identifier = target if isinstance(target, str) else self.path_of(target)
        return tuple(identifier.encode("utf-8"))

    def rng(self, target: Any) -> np.random.Generator:
        """Return a new generator of the reproducible random stream of an element (or identifier).

        Streams derive from ``numpy.random.SeedSequence(seed, spawn_key=...)``: they are independent between
        elements and identical between runs with the same seed.
        """
        return np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=self.spawn_key(target)))

    # --- Sweeps and Monte-Carlo ----------------------------------------------------------------

    def _parameter(self, path: str) -> tuple[Any, str]:
        owner, _, field = path.rpartition(".")
        if not owner:
            raise KeyError(f"{path!r} is not of the form 'element.field'")
        element = self[owner]
        if field not in type(element).declared():
            raise AttributeError(f"{owner} has no field {field!r}")
        return element, field

    def sweep(
        self,
        parameters: Mapping[str, Sequence[Any]],
        *,
        outputs: Iterable[Any],
        mode: Literal["grid", "zip"] = "grid",
    ) -> list[SweepPoint]:
        """Run the context for several values of some fields and collect the outputs.

        Parameters
        ----------
        parameters
            Values of each field, by path ``"element.field"`` (``"group.child.field"`` inside groups).
        outputs
            Outputs to compute at each point (see :meth:`run`).
        mode
            ``"grid"``: every combination (Cartesian product, last parameter varying fastest); ``"zip"``: the
            i-th values of every parameter together.

        Returns
        -------
        list of SweepPoint
            One point per combination. The fields get their original values back afterwards.
        """
        if mode not in ("grid", "zip"):
            raise ValueError(f"mode must be 'grid' or 'zip', got {mode!r}")
        targets = [(path, *self._parameter(path)) for path in parameters]
        values = [list(parameters[path]) for path, _, _ in targets]
        if mode == "zip":
            lengths = {len(v) for v in values}
            if len(lengths) > 1:
                raise ValueError(
                    f"zip sweeps need sequences of the same length, got lengths {sorted(lengths)}"
                )
            combinations: Iterable[tuple[Any, ...]] = zip(*values, strict=True)
        else:
            combinations = itertools.product(*values)
        wanted = list(outputs)
        original = [getattr(element, field) for _, element, field in targets]
        points: list[SweepPoint] = []
        try:
            for combination in combinations:
                for (_, element, field), value in zip(targets, combination, strict=True):
                    setattr(element, field, value)
                result = self.run(wanted)
                params = {path: getattr(element, field) for path, element, field in targets}
                points.append(SweepPoint(params, result))
        finally:
            for (_, element, field), value in zip(targets, original, strict=True):
                setattr(element, field, value)
        return points

    def _ordered_elements(self) -> list[tuple[str, Any]]:
        """``(path, element)`` of every element, groups expanded depth-first in insertion order."""
        found: list[tuple[str, Any]] = []

        def visit(prefix: str, graph: Graph) -> None:
            for identifier, element in graph.elements.items():
                path = f"{prefix}{identifier}"
                found.append((path, element))
                inner = getattr(element, "graph", None)
                if isinstance(inner, Graph):
                    visit(f"{path}.", inner)

        visit("", self._graph)
        return found

    def free_parameters(self) -> dict[str, Any]:
        """Return the priors of the free parameters, by path ``"element.field"`` (``None`` without prior)."""
        found: dict[str, Any] = {}
        for path, element in self._ordered_elements():
            for name in type(element).declared():
                is_free = getattr(element, "is_free", None)
                try:
                    free = bool(is_free(name)) if is_free is not None else False
                except (TypeError, KeyError):
                    free = False
                if free:
                    found[f"{path}.{name}"] = element.prior(name)
        return found

    def sample(self, n: int, *, outputs: Iterable[Any]) -> list[SweepPoint]:
        """Monte-Carlo run: draw ``n`` values of every free parameter from its prior and run the context.

        Draws come from the reproducible stream ``ctx.rng("sample")``, parameter after parameter in the order
        of :meth:`free_parameters`.
        """
        priors = self.free_parameters()
        missing = [path for path, prior in priors.items() if prior is None]
        if missing:
            raise ValueError(f"free parameters without a prior cannot be sampled: {missing}")
        rng = self.rng("sample")
        draws = {path: prior.sample(rng, int(n)) for path, prior in priors.items()}
        return self.sweep(draws, outputs=outputs, mode="zip")

    def to_python(self) -> str:
        """Return a Python script rebuilding this context (see :func:`lumos.io.codegen.to_python`)."""
        from lumos.io.codegen import to_python

        return to_python(self)

    def __repr__(self) -> str:
        """Summarise the context."""
        name = f" {self.name!r}" if self.name else ""
        return f"<Context{name} mode={self._mode} elements={len(self)}>"


def _residual(measured: Any, simulated: Any) -> Any:
    """Measured minus simulated data, on the axes of the measurement (``None`` if not comparable)."""
    from lumos.fields.dataproduct import DataProduct

    if not (isinstance(measured, DataProduct) and isinstance(simulated, DataProduct)):
        return None
    if measured.data.size != simulated.data.size:
        return None
    try:
        difference = measured.data - simulated.data.to(measured.unit).reshape(measured.shape)
    except u.UnitsError:
        return None
    return DataProduct(cast(u.Quantity, difference), axes=measured.axes, meta={"residual": True})
