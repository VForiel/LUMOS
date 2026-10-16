"""Typed input and output ports of elements (REQ-SIG-11).

Ports are declared as class attributes::

    class BeamSplitter(Element):
        in_ = Input(OpticalSignal)          # port name "in" (trailing underscore stripped)
        transmitted = Output(OpticalSignal)
        reflected = Output(OpticalSignal)

Reading a port on an instance returns a :class:`PortRef`, which identifies "this port of this object" and
is what connections are made of (``ctx.connect(source.out, splitter.in_)``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Literal

from lumos.core.signal import Signal

__all__ = [
    "Input",
    "Output",
    "Port",
    "PortCompatibilityError",
    "PortRef",
    "check_compatible",
    "compatible",
    "ports_of",
]


class PortCompatibilityError(TypeError):
    """Raised when two ports cannot be connected."""


class Port:
    """Declaration of a port: its name, direction and signal type."""

    direction: ClassVar[Literal["in", "out"]]
    multiple: bool = False  # whether several connections may be attached to this port

    def __init__(self, signal_type: type[Signal] = Signal, *, name: str | None = None, doc: str = "") -> None:
        if not (isinstance(signal_type, type) and issubclass(signal_type, Signal)):
            raise TypeError(f"signal_type must be a subclass of Signal, got {signal_type!r}")
        self.signal_type = signal_type
        self.name = name or "<unnamed>"
        self._explicit_name = name is not None
        self.doc = doc

    def __set_name__(self, owner: type, attribute: str) -> None:
        """Derive the port name from the attribute name unless one was given."""
        self.attribute = attribute
        if not self._explicit_name:
            self.name = attribute.rstrip("_") or attribute

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        """Return the declaration on the class, a bound :class:`PortRef` on an instance."""
        if instance is None:
            return self
        return PortRef(instance, self)

    def __set__(self, instance: Any, value: Any) -> None:
        """Refuse assignment: ports are part of the class definition."""
        raise AttributeError(f"port {self.name!r} cannot be reassigned")

    def __repr__(self) -> str:
        """Represent the port declaration."""
        return f"{type(self).__name__}({self.signal_type.__name__}, name={self.name!r})"


class Input(Port):
    """An input port.

    Parameters
    ----------
    signal_type
        Accepted signal type (subtypes are accepted too).
    name
        Port name; defaults to the attribute name without its trailing underscore.
    multiple
        Whether several connections may feed this input (fan-in). By default an input takes one connection.
    doc
        Description of the port.
    """

    direction: ClassVar[Literal["in", "out"]] = "in"

    def __init__(
        self,
        signal_type: type[Signal] = Signal,
        *,
        name: str | None = None,
        multiple: bool = False,
        doc: str = "",
    ) -> None:
        super().__init__(signal_type, name=name, doc=doc)
        self.multiple = multiple


class Output(Port):
    """An output port. An output can feed any number of inputs (fan-out)."""

    direction: ClassVar[Literal["in", "out"]] = "out"
    multiple = True


class PortRef:
    """A port bound to the object owning it."""

    __slots__ = ("owner", "port")

    def __init__(self, owner: Any, port: Port) -> None:
        self.owner = owner
        self.port = port

    @property
    def name(self) -> str:
        """Name of the port."""
        return self.port.name

    @property
    def direction(self) -> Literal["in", "out"]:
        """``"in"`` or ``"out"``."""
        return self.port.direction

    @property
    def signal_type(self) -> type[Signal]:
        """Signal type of the port."""
        return self.port.signal_type

    def __eq__(self, other: object) -> bool:
        """Two references are equal when they designate the same port of the same object."""
        return isinstance(other, PortRef) and other.owner is self.owner and other.port is self.port

    def __hash__(self) -> int:
        """Hash consistent with equality."""
        return hash((id(self.owner), id(self.port)))

    def __rshift__(self, other: Any) -> Any:
        """Connect this output port to ``other`` (an element with a unique input, or an input port)."""
        from lumos.core.element import _connect_shorthand

        return _connect_shorthand(self, other)

    def __repr__(self) -> str:
        """Represent the reference as ``Owner.port``."""
        owner_name = getattr(self.owner, "name", None) or type(self.owner).__name__
        return f"<{owner_name}.{self.name}>"


def ports_of(cls: type) -> Mapping[str, Port]:
    """Return the ports declared by ``cls`` and its bases, by port name, base classes first."""
    ports: dict[str, Port] = {}
    for klass in reversed(cls.__mro__):
        for attribute in vars(klass).values():
            if isinstance(attribute, Port):
                ports[attribute.name] = attribute
    return ports


def _as_port(port: Port | PortRef) -> Port:
    return port.port if isinstance(port, PortRef) else port


def _incompatibility(source: Port | PortRef, target: Port | PortRef) -> str | None:
    src, dst = _as_port(source), _as_port(target)
    if src.direction != "out":
        return f"{source!r} is an input; connections start from an output"
    if dst.direction != "in":
        return f"{target!r} is an output; connections end on an input"
    if not issubclass(src.signal_type, dst.signal_type):
        produced, expected = src.signal_type.__name__, dst.signal_type.__name__
        return f"{source!r} produces {produced} but {target!r} expects {expected}"
    return None


def compatible(source: Port | PortRef, target: Port | PortRef) -> bool:
    """Whether ``source`` (an output) can feed ``target`` (an input)."""
    return _incompatibility(source, target) is None


def check_compatible(source: Port | PortRef, target: Port | PortRef) -> None:
    """Raise :class:`PortCompatibilityError` if ``source`` cannot feed ``target``."""
    problem = _incompatibility(source, target)
    if problem is not None:
        raise PortCompatibilityError(problem)
