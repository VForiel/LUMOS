"""Directed acyclic graph of elements connected port to port, with pull execution (REQ-CTX-1/2/4).

A :class:`Graph` holds elements under unique identifiers and the connections between their ports. A
connection goes from an output port to an input port whose signal type is the same or a base class
(REQ-SIG-11). The graph is acyclic: physical loops (adaptive optics, servo loops, iterative calibration) are
iterations in time driven by the context, not cycles of signals.

Execution is *pull*: :meth:`Graph.run` computes only the elements needed by the requested outputs, in
topological order. The outputs of every element are cached; changing a parameter or a state of an element,
or a connection, invalidates the cache of that element and of everything downstream.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from lumos.core.port import PortRef, check_compatible

__all__ = ["Connection", "CycleError", "Graph", "GraphError", "Result"]

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-]*$")


class GraphError(ValueError):
    """Raised when an operation would make the graph invalid."""


class CycleError(GraphError):
    """Raised when a connection would create a cycle."""


@dataclass(frozen=True)
class Connection:
    """A connection from an output port to an input port."""

    source: PortRef
    target: PortRef


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_\-]+", "_", text).strip("_")
    if not slug or not (slug[0].isalpha() or slug[0] == "_"):
        slug = f"e_{slug}"
    return slug


class Graph:
    """Elements and their connections."""

    def __init__(self) -> None:
        self._elements: dict[str, Any] = {}
        self._ids: dict[int, str] = {}
        self._connections: list[Connection] = []
        self._counters: dict[str, int] = {}
        self._cache: dict[int, Mapping[str, Any]] = {}

    # --- Elements ---------------------------------------------------------------------------

    def add(self, element: Any, *, id: str | None = None) -> str:
        """Add ``element`` and return its identifier.

        The identifier is ``id`` if given, otherwise the element name, otherwise the lower-case class name
        followed by a counter.
        """
        if self._ids.get(id_(element)) is not None:
            raise GraphError(f"{element!r} is already in the graph as {self._ids[id_(element)]!r}")
        if id is None:
            name = getattr(element, "name", None)
            if name:
                id = _slug(name)
            else:
                base = type(element).__name__.lower()
                count = self._counters.get(base, 0) + 1
                while f"{base}{count}" in self._elements:
                    count += 1
                self._counters[base] = count
                id = f"{base}{count}"
        if not _IDENTIFIER.match(id):
            raise GraphError(f"invalid identifier {id!r}: use letters, digits, '_' and '-'")
        if id in self._elements:
            raise GraphError(f"the identifier {id!r} is already used in the graph")
        self._elements[id] = element
        self._ids[id_(element)] = id
        if hasattr(element, "_add_listener"):
            element._add_listener(self._on_element_changed)
        return id

    def remove(self, element: Any) -> None:
        """Remove ``element`` and its connections."""
        key = self.id_of(element)
        self.invalidate(element)
        if hasattr(element, "_remove_listener"):
            element._remove_listener(self._on_element_changed)
        self._connections = [
            c for c in self._connections if c.source.owner is not element and c.target.owner is not element
        ]
        del self._elements[key]
        del self._ids[id_(element)]

    def id_of(self, element: Any) -> str:
        """Return the identifier of ``element`` in this graph."""
        try:
            return self._ids[id_(element)]
        except KeyError:
            raise GraphError(f"{element!r} is not in the graph") from None

    def __getitem__(self, key: str) -> Any:
        """Element with identifier ``key``."""
        try:
            return self._elements[key]
        except KeyError:
            raise KeyError(f"no element {key!r} in the graph; known: {', '.join(self._elements)}") from None

    def __contains__(self, item: object) -> bool:
        """Whether an identifier or an element is in the graph."""
        if isinstance(item, str):
            return item in self._elements
        return id_(item) in self._ids

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the elements in insertion order."""
        return iter(list(self._elements.values()))

    def __len__(self) -> int:
        """Return the number of elements."""
        return len(self._elements)

    @property
    def elements(self) -> dict[str, Any]:
        """Elements by identifier (a copy)."""
        return dict(self._elements)

    # --- Connections ------------------------------------------------------------------------

    @property
    def connections(self) -> list[Connection]:
        """Connections in creation order (a copy)."""
        return list(self._connections)

    def connect(self, source: PortRef, target: PortRef) -> Connection:
        """Connect the output ``source`` to the input ``target``.

        Raises
        ------
        lumos.core.port.PortCompatibilityError
            If the directions or signal types are incompatible.
        GraphError
            If an element is not in the graph or a single input is already connected.
        CycleError
            If the connection would create a cycle.
        """
        for ref in (source, target):
            if not isinstance(ref, PortRef):
                raise TypeError(f"connections are made between ports, got {ref!r}")
            if ref.owner not in self:
                raise GraphError(f"{ref.owner!r} is not in the graph; add it before connecting {ref!r}")
        check_compatible(source, target)
        if not target.port.multiple and self.incoming(target):
            raise GraphError(f"{target!r} is already connected and accepts a single connection")
        if target.owner is source.owner or self._reaches(target.owner, source.owner):
            raise CycleError(
                f"connecting {self.id_of(source.owner)}.{source.name} to "
                f"{self.id_of(target.owner)}.{target.name} would create a cycle"
            )
        connection = Connection(source, target)
        self._connections.append(connection)
        self.invalidate(target.owner)
        return connection

    def disconnect(self, source: PortRef, target: PortRef) -> None:
        """Remove the connection from ``source`` to ``target``."""
        for k, c in enumerate(self._connections):
            if c.source == source and c.target == target:
                del self._connections[k]
                self.invalidate(target.owner)
                return
        raise GraphError(f"{source!r} is not connected to {target!r}")

    def incoming(self, target: PortRef) -> list[Connection]:
        """Return the connections arriving at the input ``target``, in creation order."""
        return [c for c in self._connections if c.target == target]

    def upstream(self, element: Any) -> list[Any]:
        """Elements feeding ``element`` directly, in connection order, without duplicates."""
        return _unique(c.source.owner for c in self._connections if c.target.owner is element)

    def downstream(self, element: Any) -> list[Any]:
        """Elements fed directly by ``element``, in connection order, without duplicates."""
        return _unique(c.target.owner for c in self._connections if c.source.owner is element)

    def _reaches(self, start: Any, goal: Any) -> bool:
        stack, seen = [start], set()
        while stack:
            node = stack.pop()
            if node is goal:
                return True
            if id_(node) in seen:
                continue
            seen.add(id_(node))
            stack.extend(self.downstream(node))
        return False

    # --- Orders ---------------------------------------------------------------------------------

    def topological_order(self) -> list[Any]:
        """Elements ordered so that every connection goes forward (Kahn's algorithm, stable)."""
        indegree = {id_(e): 0 for e in self._elements.values()}
        for c in self._connections:
            indegree[id_(c.target.owner)] += 1
        ready = [e for e in self._elements.values() if indegree[id_(e)] == 0]
        order: list[Any] = []
        while ready:
            element = ready.pop(0)
            order.append(element)
            for c in self._connections:
                if c.source.owner is element:
                    key = id_(c.target.owner)
                    indegree[key] -= 1
                    if indegree[key] == 0:
                        ready.append(c.target.owner)
        return order

    def required(self, targets: Iterable[Any]) -> list[Any]:
        """Elements needed to compute ``targets`` (elements or output ports), in topological order."""
        wanted: set[int] = set()
        stack = [t.owner if isinstance(t, PortRef) else t for t in targets]
        for element in stack:
            if element not in self:
                raise GraphError(f"{element!r} is not in the graph")
        while stack:
            element = stack.pop()
            if id_(element) in wanted:
                continue
            wanted.add(id_(element))
            stack.extend(self.upstream(element))
        return [e for e in self.topological_order() if id_(e) in wanted]

    # --- Execution ------------------------------------------------------------------------------

    def _on_element_changed(self, element: Any, name: str) -> None:
        self.invalidate(element)

    def invalidate(self, element: Any = None) -> None:
        """Drop the cached outputs of ``element`` and everything downstream (everything if ``None``)."""
        if element is None:
            self._cache.clear()
            return
        stack = [element]
        seen: set[int] = set()
        while stack:
            node = stack.pop()
            if id_(node) in seen:
                continue
            seen.add(id_(node))
            self._cache.pop(id_(node), None)
            stack.extend(self.downstream(node))

    def is_cached(self, element: Any) -> bool:
        """Whether the outputs of ``element`` are cached."""
        return id_(element) in self._cache

    def resolve(self, target: Any) -> Any:
        """Turn ``"id"`` or ``"id.port"`` into an element or a port reference; other values pass through."""
        if not isinstance(target, str):
            return target
        key, _, port = target.rpartition(".") if "." in target else (target, "", "")
        if not port:
            return self[target]
        if key in self._elements:
            return self._elements[key].port(port)
        return self[target]

    def _inputs(self, element: Any, feeds: Mapping[PortRef, Any]) -> dict[str, Any]:
        inputs: dict[str, Any] = {}
        for name, port in element.instance_ports().items():
            if port.direction != "in":
                continue
            ref = element.port(name)
            connections = self.incoming(ref)
            if ref in feeds:
                if connections:
                    raise GraphError(f"input {self.id_of(element)}.{name} is both fed and connected")
                inputs[name] = feeds[ref]
                continue
            if not connections and not port.multiple:
                raise GraphError(f"input {self.id_of(element)}.{name} is not connected")
            signals = [self._cache[id_(c.source.owner)][c.source.name] for c in connections]
            inputs[name] = signals if port.multiple else signals[0]
        return inputs

    def _compute(self, element: Any, feeds: Mapping[PortRef, Any]) -> Mapping[str, Any]:
        return element.apply(**self._inputs(element, feeds))

    def run(
        self,
        targets: Iterable[Any],
        *,
        feeds: Mapping[PortRef, Any] | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> Result:
        """Compute the requested outputs.

        Parameters
        ----------
        targets
            Output ports, elements (all their outputs) or identifiers ``"id"`` / ``"id.port"``.
        feeds
            Signals given directly to unconnected input ports (used by groups for their exposed inputs).
            The caller invalidates the elements whose feeds changed.
        progress
            Called as ``progress(done, total, element_id)`` after each element (for long runs).

        Returns
        -------
        Result
            The signals by output port, also reachable by ``"id.port"``.
        """
        resolved = [self.resolve(t) for t in targets]
        feeds = {} if feeds is None else feeds
        required = self.required(resolved)
        for index, element in enumerate(required, start=1):
            if id_(element) not in self._cache:
                self._cache[id_(element)] = dict(self._compute(element, feeds))
            if progress is not None:
                progress(index, len(required), self.id_of(element))
        result = Result(self)
        for target in resolved:
            if isinstance(target, PortRef):
                result[target] = self._cache[id_(target.owner)][target.name]
            else:
                for name, signal in self._cache[id_(target)].items():
                    result[target.port(name)] = signal
        return result


class Result(dict[PortRef, Any]):
    """Signals computed by :meth:`Graph.run`, by port reference or ``"id.port"`` string.

    Attributes
    ----------
    archive
        The :class:`~lumos.results.RunArchive` written by ``ctx.run(..., archive=...)``, or ``None``.
    twin
        In operation with the digital twin, the simulated outputs of the measuring elements (the result
        itself holds the measurements).
    residuals
        Measured minus simulated outputs, where both are comparable data products.
    """

    def __init__(self, graph: Graph) -> None:
        super().__init__()
        self._graph = graph
        self.archive: Any = None
        self.twin = PortMap(graph)
        self.residuals = PortMap(graph)

    def name_of(self, key: PortRef) -> str:
        """Return ``"id.port"`` for an output port of the graph."""
        return f"{self._graph.id_of(key.owner)}.{key.name}"

    def __getitem__(self, key: Any) -> Any:
        """Return the signal of an output port given as a reference or as ``"id.port"``."""
        if isinstance(key, str):
            key = self._graph.resolve(key)
        return super().__getitem__(key)


class PortMap(dict[PortRef, Any]):
    """Signals by output port, also reachable by ``"id.port"``."""

    def __init__(self, graph: Graph) -> None:
        super().__init__()
        self._graph = graph

    def __getitem__(self, key: Any) -> Any:
        """Return the signal of an output port given as a reference or as ``"id.port"``."""
        if isinstance(key, str):
            key = self._graph.resolve(key)
        return super().__getitem__(key)


def id_(obj: Any) -> int:
    """Identity key of an object (elements may define ``__eq__``)."""
    return id(obj)


def _unique(items: Iterable[Any]) -> list[Any]:
    seen: set[int] = set()
    result = []
    for item in items:
        if id_(item) not in seen:
            seen.add(id_(item))
            result.append(item)
    return result
