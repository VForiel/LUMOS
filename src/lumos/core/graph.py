"""Directed acyclic graph of elements connected port to port (REQ-CTX-1, REQ-CTX-2).

A :class:`Graph` holds elements under unique identifiers and the connections between their ports. A
connection goes from an output port to an input port whose signal type is the same or a base class
(REQ-SIG-11). The graph is acyclic: physical loops (adaptive optics, servo loops, iterative calibration) are
iterations in time driven by the context, not cycles of signals.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from lumos.core.port import PortRef, check_compatible

__all__ = ["Connection", "CycleError", "Graph", "GraphError"]

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
        return id

    def remove(self, element: Any) -> None:
        """Remove ``element`` and its connections."""
        key = self.id_of(element)
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
        return connection

    def disconnect(self, source: PortRef, target: PortRef) -> None:
        """Remove the connection from ``source`` to ``target``."""
        for k, c in enumerate(self._connections):
            if c.source == source and c.target == target:
                del self._connections[k]
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
