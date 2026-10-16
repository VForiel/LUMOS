"""The context: container and coordinator of a LUMOS system (REQ-CTX-*).

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

import logging
import math
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any, Literal, cast

import numpy as np
from astropy import units as u
from astropy.time import Time

from lumos.backend._core import known, use
from lumos.core.clock import Clock, duration_si
from lumos.core.driver import Command, DriverError
from lumos.core.graph import Graph, GraphError, Result
from lumos.core.port import PortRef
from lumos.units import require_unit, to_si

__all__ = ["MODES", "Context", "Observation"]

logger = logging.getLogger("lumos.context")

Mode = Literal["simulation", "operation"]
MODES: tuple[str, ...] = ("simulation", "operation")


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
        self._band: tuple[u.Quantity, u.Quantity] | None = None
        self.n_wavelengths = 1
        self.rule: Literal["midpoint", "gauss"] = "midpoint"
        self.extra: dict[str, Any] = {}

    @property
    def band(self) -> tuple[u.Quantity, u.Quantity] | None:
        """Spectral band ``(min, max)``."""
        return self._band

    @band.setter
    def band(self, value: tuple[u.Quantity, u.Quantity] | None) -> None:
        if value is None:
            self._band = None
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
        self.mode = mode
        self.twin = twin
        self.backend = backend
        self.seed = int(np.random.SeedSequence().entropy) if seed is None else int(seed)  # type: ignore[arg-type]
        self.name = name
        self.observation = Observation()

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

    def add(self, *elements: Any) -> Any:
        """Add elements and return them (the element itself when only one is given)."""
        for element in elements:
            owner = getattr(element, "_lumos_container", None)
            if owner is not None and owner is not self:
                raise GraphError(f"{element!r} already belongs to another context")
            if owner is self:
                continue
            self._graph.add(element)
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

    def run(self, outputs: Iterable[Any]) -> Result:
        """Compute the requested outputs (ports, elements or ``"id.port"`` strings), see :meth:`Graph.run`."""
        targets = [self.port(t) if isinstance(t, str) and "." in t else t for t in outputs]
        if self._backend == "auto":
            return self._graph.run(targets)
        with use(self._backend):
            return self._graph.run(targets)

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
        self._advance(duration_si(duration))

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

    def twin_state(self, element: Any) -> dict[str, Any]:
        """State measured on the equipment at the last command, next to the twin's simulated state."""
        return dict(self._twin_states.get(id(element), {}))

    def _perform(self, element: Any, spec: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Execute an action of ``element`` (called by the :func:`~lumos.core.action.action` wrapper)."""
        predicted = spec.duration_si(element, args, kwargs)
        path = self.path_of(element)
        if self.mode_of(element) == "simulation":
            result = spec.method(element, *args, **kwargs)
            logger.info("%s.%s simulated (%.6g s)", path, spec.name, predicted)
            self._advance(predicted)
            return result
        driver = element.driver
        if driver is None:
            raise DriverError(f"{path} has no driver; attach one or use ctx.simulate({path!r})")
        if not driver.connected:
            raise DriverError(f"the driver of {path} is not connected")
        command = Command(path, spec.name, tuple(args), dict(kwargs))
        started = time.perf_counter()
        reply = driver.send(command)
        wall = time.perf_counter() - started
        measured = wall if reply.duration is None else duration_si(reply.duration, name="measured duration")
        if self._twin:
            result = spec.method(element, *args, **kwargs)
            self._twin_states[id(element)] = dict(reply.state)
        else:
            result = reply.value
            for name, value in reply.state.items():
                setattr(element, name, value)
        logger.info(
            "%s.%s sent to %s (predicted %.6g s, measured %.6g s)",
            path,
            spec.name,
            type(driver).__name__,
            predicted,
            measured,
        )
        self._advance(measured)
        return result

    # --- Random streams -----------------------------------------------------------------------

    def spawn_key(self, target: Any) -> tuple[int, ...]:
        """Spawn key of the random stream of an element (or identifier): the UTF-8 bytes of its identifier."""
        identifier = target if isinstance(target, str) else self.id_of(target)
        return tuple(identifier.encode("utf-8"))

    def rng(self, target: Any) -> np.random.Generator:
        """Return a new generator of the reproducible random stream of an element (or identifier).

        Streams derive from ``numpy.random.SeedSequence(seed, spawn_key=...)``: they are independent between
        elements and identical between runs with the same seed.
        """
        return np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=self.spawn_key(target)))

    def __repr__(self) -> str:
        """Summarise the context."""
        name = f" {self.name!r}" if self.name else ""
        return f"<Context{name} mode={self._mode} elements={len(self)}>"
