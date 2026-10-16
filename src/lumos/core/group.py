"""Element groups: composite elements exposing some of their internal ports (REQ-CTX-7).

An :class:`ElementGroup` holds a sub-graph of elements (a planetary system, a photonic chip and its
components, a multi-lens objective...) and **exposes** chosen internal ports as its own ports. It is an
:class:`~lumos.core.element.Element`: it can be added to a context or to another group, connected with ``>>``
and saved to ``.lumos`` (``kind: "group"``).

::

    chip = ElementGroup(name="chip")
    splitter, shifter = chip.add(YSplitter(name="y"), PhaseShifter(name="phi"))
    splitter.a >> shifter
    chip.expose("in", splitter.in_)
    chip.expose("out", shifter.out)

When the group runs, the signals arriving on its exposed inputs are fed to the internal ports and the
internal graph computes the exposed outputs with its own cache. A change of a parameter of any child
invalidates the results of the group, and therefore everything downstream of it, in the enclosing graph.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from lumos.core.element import Element
from lumos.core.graph import Graph, GraphError
from lumos.core.port import Input, Output, Port, PortRef

__all__ = ["GROUP_TYPE", "ElementGroup", "group_class", "group_type_of", "register_group"]

#: Type identifier of groups in ``.lumos`` documents.
GROUP_TYPE = "lumos.group"
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Specialised group classes by identifier (saved as ``"group_type"`` in ``.lumos`` group entries).
_GROUP_CLASSES: dict[str, type[ElementGroup]] = {}


def register_group(type_id: str) -> Any:
    """Return a class decorator registering a specialised group class (such as a scene) under ``type_id``.

    A registered subclass is saved as an ordinary group entry with ``"group_type": type_id`` and recreated
    with its class when loaded. Its children, connections and exposed ports are saved like those of any
    group; the subclass must keep its configuration in its children.
    """

    def decorator(cls: type[ElementGroup]) -> type[ElementGroup]:
        if not (isinstance(cls, type) and issubclass(cls, ElementGroup)):
            raise TypeError("only ElementGroup subclasses can be registered as group types")
        existing = _GROUP_CLASSES.get(type_id)
        if existing is not None and existing is not cls:
            raise ValueError(f"group type {type_id!r} is already registered for {existing.__qualname__}")
        _GROUP_CLASSES[type_id] = cls
        setattr(cls, "_lumos_group_type", type_id)  # noqa: B010 - class attribute added at registration
        return cls

    return decorator


def group_class(type_id: str) -> type[ElementGroup]:
    """Return the group class registered under ``type_id``."""
    try:
        return _GROUP_CLASSES[type_id]
    except KeyError:
        raise LookupError(f"unknown group type {type_id!r}; known: {', '.join(_GROUP_CLASSES)}") from None


def group_type_of(group: ElementGroup) -> str | None:
    """Return the registered type of ``group``'s class, or ``None`` for a plain group."""
    type_id = type(group).__dict__.get("_lumos_group_type")
    return type_id if isinstance(type_id, str) else None


class ElementGroup(Element):
    """Composite element made of a sub-graph of elements.

    Parameters
    ----------
    name
        Name of the group (also its default identifier in a context).
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name=name)
        self._graph = Graph()
        self._exposed: dict[str, tuple[Port, PortRef]] = {}

    @classmethod
    def _empty(cls, name: str | None = None) -> ElementGroup:
        """Create an instance without children, to be filled from a ``.lumos`` entry."""
        group = cls.__new__(cls)
        ElementGroup.__init__(group, name=name)
        return group

    # --- Children -----------------------------------------------------------------------------

    @property
    def graph(self) -> Graph:
        """Internal graph of the group."""
        return self._graph

    def add(self, *elements: Any, id: str | None = None) -> Any:
        """Add children and return them (the child itself when only one is given)."""
        if id is not None and len(elements) != 1:
            raise ValueError("an explicit id can only be given for a single element")
        for element in elements:
            owner = getattr(element, "_lumos_container", None)
            if owner is self:
                continue
            if owner is not None:
                raise GraphError(f"{element!r} already belongs to another context or group")
            self._graph.add(element, id=id)
            element._lumos_container = self
            element._add_listener(self._child_changed)
        return elements[0] if len(elements) == 1 else elements

    def remove(self, element: Any) -> None:
        """Remove a child, its connections and the ports it exposes."""
        self._graph.remove(element)
        element._remove_listener(self._child_changed)
        element._lumos_container = None
        self._exposed = {name: e for name, e in self._exposed.items() if e[1].owner is not element}
        self._on_field_changed("children")

    def connect(self, source: PortRef, target: PortRef) -> None:
        """Connect two internal ports (adding the target child if needed)."""
        if target.owner not in self._graph:
            self.add(target.owner)
        self._graph.connect(source, target)
        self._on_field_changed("connections")

    def __contains__(self, item: object) -> bool:
        """Whether an identifier or an element is a direct child."""
        return item in self._graph

    def __getitem__(self, path: str) -> Any:
        """Return the child ``"id"`` or a deeper descendant ``"id.child"``."""
        head, _, rest = path.partition(".")
        child = self._graph[head]
        return child[rest] if rest else child

    def id_of(self, element: Any) -> str:
        """Return the identifier of a child."""
        return self._graph.id_of(element)

    def _child_changed(self, element: Any, name: str) -> None:
        self._on_field_changed(f"{self._graph.id_of(element)}.{name}")

    # --- Exposed ports --------------------------------------------------------------------------

    def expose(self, name: str, internal: PortRef) -> PortRef:
        """Expose the internal port ``internal`` as the group port ``name`` and return the group port."""
        if not _NAME.match(name):
            raise GraphError(f"invalid port name {name!r}")
        if name in self._exposed:
            raise GraphError(f"the group already exposes a port named {name!r}")
        if not isinstance(internal, PortRef) or internal.owner not in self._graph:
            raise GraphError(f"{internal!r} is not in the group; add its element first")
        source = internal.port
        port: Port
        if source.direction == "in":
            port = Input(source.signal_type, name=name, multiple=source.multiple, doc=source.doc)
        else:
            port = Output(source.signal_type, name=name, doc=source.doc)
        port.attribute = name
        self._exposed[name] = (port, internal)
        self._on_field_changed("ports")
        return PortRef(self, port)

    def exposed(self) -> dict[str, PortRef]:
        """Return the internal ports behind the exposed ports, by exposed name."""
        return {name: internal for name, (_, internal) in self._exposed.items()}

    def instance_ports(self) -> Mapping[str, Port]:
        """Return the exposed ports of this group."""
        return {name: port for name, (port, _) in self._exposed.items()}

    def __getattr__(self, name: str) -> Any:
        """Reach exposed ports as attributes (``group.out``) when the name is not taken."""
        exposed = self.__dict__.get("_exposed", {})
        if name in exposed:
            return PortRef(self, exposed[name][0])
        raise AttributeError(f"{type(self).__name__} has no attribute or exposed port {name!r}")

    # --- Execution ------------------------------------------------------------------------------

    def apply(self, **inputs: Any) -> Mapping[str, Any]:
        """Run the internal graph on the signals arriving at the exposed inputs."""
        feeds: dict[PortRef, Any] = {}
        for name, (port, internal) in self._exposed.items():
            if port.direction != "in":
                continue
            if name not in inputs:
                raise TypeError(f"{self!r}: missing input {name!r}")
            feeds[internal] = inputs[name]
            self._graph.invalidate(internal.owner)
        outputs = {
            name: internal for name, (port, internal) in self._exposed.items() if port.direction == "out"
        }
        result = self._graph.run(list(outputs.values()), feeds=feeds)
        return {name: result[internal] for name, internal in outputs.items()}

    def __repr__(self) -> str:
        """Represent the group by its name and number of children."""
        name = f" {self.name!r}" if self.name else ""
        return f"<ElementGroup{name} children={len(self._graph)}>"
