"""Base class of every LUMOS element: equipment, optical component, detector or processing block.

An element declares its fields (:class:`~lumos.core.param.Param`, :class:`~lumos.core.param.State`,
:class:`~lumos.core.param.Setting`) and its ports (:class:`~lumos.core.port.Input`,
:class:`~lumos.core.port.Output`) as class attributes, and implements its behaviour as a **pure** transfer
function::

    class Attenuator(Element):
        \"\"\"Attenuate a field.\"\"\"

        transmission = Param(default=0.5, bounds=(0, 1))
        in_ = Input(Wavefront)
        out = Output(Wavefront)

        @staticmethod
        def transfer(values, inputs):
            return {"out": inputs["in"] * values.transmission**0.5}

``transfer`` must be a ``staticmethod``: it cannot read or modify the element itself. It receives an
immutable snapshot of the SI values of all fields (:class:`Values`) and the input signals by port name, and
returns the output signals by port name. This separation between the element's state and its computation is
what allows JIT compilation, vectorisation and automatic differentiation (ADR-0002).
"""

from __future__ import annotations

import difflib
import html
import inspect
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, ClassVar

from lumos.core.param import MISSING, Field, Param, ParamHost, Setting
from lumos.core.port import Port, PortRef
from lumos.units import quantity_to_json

__all__ = ["Element", "Values"]

_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
_NUMBER_SCHEMA: dict[str, Any] = {"oneOf": [{"type": "number"}, {"enum": ["nan", "inf", "-inf"]}]}
_COMPLEX_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"re": _NUMBER_SCHEMA, "im": _NUMBER_SCHEMA},
    "required": ["re", "im"],
    "additionalProperties": False,
}
_ARRAY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "dtype": {"type": "string"},
        "shape": {"type": "array", "items": {"type": "integer", "minimum": 0}},
        "data": {"type": "string"},
    },
    "required": ["dtype", "shape", "data"],
    "additionalProperties": False,
}
_SETTING_TYPES = {bool: "boolean", int: "integer", float: "number", str: "string"}


class Values(Mapping[str, Any]):
    """Immutable snapshot of the SI values of an element's fields.

    Values are reachable by key (``values["focal_length"]``) or attribute (``values.focal_length``).
    Physical fields are floats, complex numbers or read-only arrays in unscaled SI units; settings are
    returned as they are.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_data", dict(data))

    def __getattr__(self, name: str) -> Any:
        """Return the value of field ``name``."""
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse any modification."""
        raise AttributeError("Values snapshots are immutable")

    def __getitem__(self, name: str) -> Any:
        """Return the value of field ``name``."""
        return self._data[name]

    def __iter__(self) -> Iterator[str]:
        """Iterate over field names."""
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of fields."""
        return len(self._data)

    def __repr__(self) -> str:
        """Represent the snapshot."""
        return f"Values({self._data!r})"


def _own_docstring(cls: type) -> str:
    doc = cls.__dict__.get("__doc__")
    return inspect.cleandoc(doc) if doc else ""


class Element(ParamHost):
    """Base class of all LUMOS elements.

    Parameters
    ----------
    name
        Optional human-readable name of the element.
    **values
        Values of the declared fields; fields without a default value are required.
    """

    _lumos_ports: ClassVar[dict[str, Port]] = {}
    #: Whether the transfer function reads the elapsed time ``values.time`` (seconds since the start of the
    #: context). Time-dependent elements are recomputed whenever the clock advances.
    time_dependent: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Collect fields and ports and check the class definition."""
        super().__init_subclass__(**kwargs)

        transfer = cls.__dict__.get("transfer")
        if transfer is not None and not isinstance(transfer, staticmethod):
            raise TypeError(
                f"{cls.__name__}.transfer must be a staticmethod (transfer functions are pure, ADR-0002)"
            )

        ports: dict[str, Port] = {}
        for klass in reversed(cls.__mro__):
            for attribute, port in vars(klass).items():
                if not isinstance(port, Port):
                    continue
                existing = ports.get(port.name)
                if existing is not None and existing.attribute != attribute:
                    raise TypeError(f"{cls.__name__}: two ports are named {port.name!r}")
                ports[port.name] = port
        cls._lumos_ports = ports

        reserved = _RESERVED_NAMES & set(cls._lumos_fields)
        if reserved:
            raise TypeError(f"{cls.__name__}: field name(s) reserved by the Element API: {sorted(reserved)}")
        clashes = set(cls._lumos_fields) & set(ports)
        if clashes:
            raise TypeError(f"{cls.__name__}: field(s) and port(s) share the name(s) {sorted(clashes)}")
        if cls.time_dependent and "time" in cls._lumos_fields:
            raise TypeError(f"{cls.__name__}: time-dependent elements cannot declare a field named 'time'")

    def __init__(self, name: str | None = None, **values: Any) -> None:
        self._lumos_listeners: list[Callable[[Element, str], None]] = []
        self._lumos_container: Any = None
        self._lumos_driver: Any = None
        self.name = name
        super().__init__(**values)

    @property
    def context(self) -> Any:
        """The :class:`~lumos.Context` holding this element (through its groups), or ``None``."""
        container = self._lumos_container
        while container is not None and not hasattr(container, "mode"):
            container = container._lumos_container  # an ElementGroup: go up
        return container

    @property
    def driver(self) -> Any:
        """Driver of the equipment represented by this element (used in operation mode), or ``None``."""
        return self._lumos_driver

    @driver.setter
    def driver(self, value: Any) -> None:
        from lumos.core.driver import Driver

        if value is not None and not isinstance(value, Driver):
            raise TypeError(f"driver must be a lumos.core.driver.Driver or None, got {type(value).__name__}")
        self._lumos_driver = value

    @property
    def container(self) -> Any:
        """The context or group directly holding this element, or ``None``."""
        return self._lumos_container

    def __rshift__(self, other: Any) -> Any:
        """Connect the unique output of this element to the unique input of ``other`` (an element or port).

        Both must end up in the same context: ``other`` is added to the context of this element if needed.
        Returns ``other`` (its owner for a port) so that chains read ``a >> b >> c``.
        """
        return _connect_shorthand(self, other)

    # --- Change notification ----------------------------------------------------------------

    def _on_field_changed(self, name: str) -> None:
        """Notify the listeners (graphs, contexts) that field ``name`` changed."""
        for listener in list(self._lumos_listeners):
            listener(self, name)

    def _add_listener(self, listener: Callable[[Element, str], None]) -> None:
        self._lumos_listeners.append(listener)

    def _remove_listener(self, listener: Callable[[Element, str], None]) -> None:
        if listener in self._lumos_listeners:
            self._lumos_listeners.remove(listener)

    # --- Attributes -----------------------------------------------------------------------

    @property
    def name(self) -> str | None:
        """Human-readable name of the element (or ``None``)."""
        return self._name

    @name.setter
    def name(self, value: str | None) -> None:
        if value is not None and not isinstance(value, str):
            raise TypeError(f"name must be a string or None, got {type(value).__name__}")
        self._name = value

    def __setattr__(self, key: str, value: Any) -> None:
        """Only allow declared fields, the name and private attributes, to catch typos early."""
        cls = type(self)
        if (
            key.startswith("_")
            or key in ("name", "driver")
            or key in cls._lumos_fields
            or isinstance(getattr(cls, key, None), Port)
        ):
            object.__setattr__(self, key, value)
            return
        candidates = [*cls._lumos_fields, "name"]
        suggestion = difflib.get_close_matches(key, candidates, n=1)
        hint = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
        raise AttributeError(f"{cls.__name__} has no field {key!r}{hint}")

    @classmethod
    def ports(cls) -> Mapping[str, Port]:
        """Return the declared ports by name, inputs and outputs in definition order."""
        return dict(cls._lumos_ports)

    def instance_ports(self) -> Mapping[str, Port]:
        """Return the ports of this instance (the class ports; groups add the ports they expose)."""
        return type(self).ports()

    def port(self, name: str) -> PortRef:
        """Return the port ``name`` of this instance, bound to it."""
        ports = self.instance_ports()
        if name not in ports:
            raise KeyError(f"{self!r} has no port {name!r}; ports: {', '.join(ports)}")
        return PortRef(self, ports[name])

    def values(self) -> Values:
        """Return an immutable snapshot of the SI values of all fields.

        Time-dependent elements also get ``time``, the elapsed time of their context in seconds (0 outside a
        context).
        """
        data = {name: self.si(name) for name in type(self)._lumos_fields}
        if type(self).time_dependent:
            context = self.context
            data["time"] = 0.0 if context is None else context.clock.elapsed_si
        return Values(data)

    @classmethod
    def actions(cls) -> Mapping[str, Any]:
        """Return the timed actions declared by the class, by name (see :func:`lumos.core.action.action`)."""
        from lumos.core.action import actions_of

        return actions_of(cls)

    # --- Behaviour --------------------------------------------------------------------------

    @staticmethod
    def transfer(values: Values, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        """Compute the output signals from the field values and the input signals.

        Parameters
        ----------
        values
            SI values of the element's fields.
        inputs
            Input signals by port name (a sequence of signals for ports accepting several connections).

        Returns
        -------
        Mapping
            Output signals by port name.
        """
        raise NotImplementedError("this element does not define a transfer function")

    def apply(self, **inputs: Any) -> Mapping[str, Any]:
        """Run the transfer function on ``inputs``, checking port names and signal types."""
        ports = type(self)._lumos_ports
        input_ports = {name: port for name, port in ports.items() if port.direction == "in"}
        output_ports = {name: port for name, port in ports.items() if port.direction == "out"}

        unexpected = sorted(set(inputs) - set(input_ports))
        if unexpected:
            raise TypeError(f"{self!r}: unexpected input(s) {unexpected}")
        missing = sorted(set(input_ports) - set(inputs))
        if missing:
            raise TypeError(f"{self!r}: missing input(s) {missing}")
        for name, port in input_ports.items():
            signals = inputs[name] if port.multiple else [inputs[name]]
            if port.multiple and not isinstance(signals, Sequence):
                raise TypeError(f"{self!r}: input {name!r} accepts several signals and expects a sequence")
            for signal in signals:
                if not isinstance(signal, port.signal_type):
                    raise TypeError(
                        f"{self!r}: input {name!r} expects {port.signal_type.__name__}, "
                        f"got {type(signal).__name__}"
                    )

        outputs = type(self).transfer(self.values(), dict(inputs))
        if not isinstance(outputs, Mapping) or set(outputs) != set(output_ports):
            produced = sorted(outputs) if isinstance(outputs, Mapping) else type(outputs).__name__
            raise TypeError(
                f"{self!r}: transfer returned {produced}, expected outputs {sorted(output_ports)}"
            )
        for name, port in output_ports.items():
            if not isinstance(outputs[name], port.signal_type):
                raise TypeError(
                    f"{self!r}: output {name!r} must be {port.signal_type.__name__}, "
                    f"got {type(outputs[name]).__name__}"
                )
        return outputs

    # --- Serialisation -----------------------------------------------------------------------------

    def to_lumos(self, *, registry: Any = None) -> dict[str, Any]:
        """Return the ``.lumos`` document describing this element (see :mod:`lumos.io`)."""
        from lumos.io import to_document  # deferred: lumos.io depends on this module

        return to_document(self, registry=registry)

    @classmethod
    def from_lumos(cls, document: Mapping[str, Any], *, registry: Any = None) -> Element:
        """Create an element from a ``.lumos`` document, checking that it is an instance of ``cls``."""
        from lumos.io import from_document  # deferred: lumos.io depends on this module

        element = from_document(document, registry=registry)
        if not isinstance(element, cls):
            raise TypeError(f"the document describes a {type(element).__name__}, not a {cls.__name__}")
        return element

    # --- Introspection ----------------------------------------------------------------------------

    def __repr__(self) -> str:
        """Represent the element by its class and name."""
        name = f" {self.name!r}" if self.name else ""
        return f"<{type(self).__name__}{name}>"

    def _fields_by_kind(self) -> dict[str, list[tuple[str, Field]]]:
        groups: dict[str, list[tuple[str, Field]]] = {"param": [], "setting": [], "state": []}
        for name, field in type(self)._lumos_fields.items():
            groups[field.kind].append((name, field))
        return groups

    def _display_value(self, name: str, field: Field) -> str:
        value = getattr(self, name)
        text = repr(value) if isinstance(field, Setting) else str(value)
        if isinstance(field, Param) and self.is_free(name):
            text += "  [free]"
        return text

    def describe(self) -> str:
        """Return a readable description of the element: fields, values and ports."""
        lines = [f"{type(self).__name__}" + (f" {self.name!r}" if self.name else "")]
        titles = {"param": "params", "setting": "settings", "state": "state"}
        for kind, fields in self._fields_by_kind().items():
            if not fields:
                continue
            lines.append(f"  {titles[kind]}:")
            for name, field in fields:
                doc = f"  # {field.doc}" if field.doc else ""
                lines.append(f"    {name} = {self._display_value(name, field)}{doc}")
        for direction, title in (("in", "inputs"), ("out", "outputs")):
            ports = [p for p in type(self)._lumos_ports.values() if p.direction == direction]
            if ports:
                listed = ", ".join(f"{p.name} ({p.signal_type.__name__})" for p in ports)
                lines.append(f"  {title}: {listed}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Return an HTML table for Jupyter."""
        title = html.escape(type(self).__name__ + (f" '{self.name}'" if self.name else ""))
        rows = [
            "<tr>"
            f"<td><code>{html.escape(name)}</code></td>"
            f"<td>{html.escape(field.kind)}</td>"
            f"<td>{html.escape(self._display_value(name, field))}</td>"
            f"<td>{html.escape(field.doc)}</td>"
            "</tr>"
            for name, field in type(self)._lumos_fields.items()
        ]
        rows += [
            "<tr>"
            f"<td><code>{html.escape(port.name)}</code></td>"
            f"<td>{'input' if port.direction == 'in' else 'output'}</td>"
            f"<td>{html.escape(port.signal_type.__name__)}</td>"
            f"<td>{html.escape(port.doc)}</td>"
            "</tr>"
            for port in type(self)._lumos_ports.values()
        ]
        return (
            f"<div><strong>{title}</strong><table>"
            "<thead><tr><th>Name</th><th>Kind</th><th>Value</th><th>Description</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
        )

    @classmethod
    def schema(cls) -> dict[str, Any]:
        """Return the JSON Schema (draft 2020-12) of the serialised form of this element.

        The serialised body is ``{"name": ..., "params": {...}, "settings": {...}, "state": {...}}``, where
        physical values are encoded with :func:`lumos.units.quantity_to_json`. Information that JSON Schema
        cannot express (units, bounds, dimension...) is given in ``x-lumos-*`` keywords; unit compatibility
        and bounds are checked when the element is loaded.
        """
        groups: dict[str, dict[str, Any]] = {kind: {} for kind in ("param", "setting", "state")}
        required: dict[str, list[str]] = {kind: [] for kind in groups}
        for name, field in cls._lumos_fields.items():
            groups[field.kind][name] = _field_schema(field)
            if field.default is MISSING:
                required[field.kind].append(name)

        def group(kind: str) -> dict[str, Any]:
            return {
                "type": "object",
                "properties": groups[kind],
                "required": required[kind],
                "additionalProperties": False,
            }

        ports = cls._lumos_ports.values()
        return {
            "$schema": _SCHEMA_DIALECT,
            "title": cls.__name__,
            "description": _own_docstring(cls),
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "params": group("param"),
                "settings": group("setting"),
                "state": group("state"),
            },
            "required": ["params", "settings", "state"],
            "additionalProperties": False,
            "x-lumos-ports": {
                direction: [
                    {"name": p.name, "signal": p.signal_type.__name__, "multiple": p.multiple, "doc": p.doc}
                    for p in ports
                    if p.direction == code
                ]
                for direction, code in (("inputs", "in"), ("outputs", "out"))
            },
        }


def _field_schema(field: Field) -> dict[str, Any]:
    if isinstance(field, Setting):
        schema: dict[str, Any] = {"description": field.doc, "x-lumos-kind": field.kind}
        if field.type in _SETTING_TYPES:
            schema["type"] = _SETTING_TYPES[field.type]
        if field.choices is not None:
            schema["enum"] = list(field.choices)
        if field.default is not MISSING:
            schema["default"] = field.default
        return schema

    assert isinstance(field, Param)
    if field.shape == ():
        value_schema = _COMPLEX_SCHEMA if field.dtype == "complex" else _NUMBER_SCHEMA
    elif field.shape is None:
        value_schema = {"anyOf": [_NUMBER_SCHEMA, _COMPLEX_SCHEMA, _ARRAY_SCHEMA]}
    else:
        value_schema = _ARRAY_SCHEMA
    schema = {
        "type": "object",
        "properties": {"value": value_schema, "unit": {"type": "string"}},
        "required": ["value", "unit"],
        "additionalProperties": False,
        "description": field.doc,
        "x-lumos-kind": field.kind,
        "x-lumos-unit": field.unit.to_string(),
        "x-lumos-dimension": str(field.unit.physical_type),
        "x-lumos-bounds": [None if b is None else quantity_to_json(b) for b in field.bounds],
        "x-lumos-inclusive": list(field.inclusive),
        "x-lumos-shape": None if field.shape is None else list(field.shape),
        "x-lumos-dtype": field.dtype,
        "x-lumos-free": field.free,
        "x-lumos-differentiable": field.differentiable,
    }
    if field.default is not MISSING:
        schema["default"] = quantity_to_json(field.default)
    return schema


def _unique_port(element: Any, direction: str) -> PortRef:
    from lumos.core.graph import GraphError

    ports = [name for name, port in element.instance_ports().items() if port.direction == direction]
    if len(ports) != 1:
        kind = "output" if direction == "out" else "input"
        raise GraphError(
            f"'>>' is ambiguous: {element!r} has {len(ports)} {kind} ports {ports}; "
            "use ctx.connect() or name a port"
        )
    return element.port(ports[0])


def _connect_shorthand(left: Any, right: Any) -> Any:
    """Implement ``left >> right`` for elements and ports."""
    from lumos.core.graph import GraphError

    source = left if isinstance(left, PortRef) else _unique_port(left, "out")
    context = source.owner.container
    if context is None:
        raise GraphError(
            f"{source.owner!r} is not in a context or group; add it with ctx.add() before using '>>'"
        )
    target_owner = right.owner if isinstance(right, PortRef) else right
    if target_owner not in context:
        context.add(target_owner)
    target = right if isinstance(right, PortRef) else _unique_port(right, "in")
    context.connect(source, target)
    return target_owner


_RESERVED_NAMES = {name for name in dir(Element) if not name.startswith("_")} | {"name"}
