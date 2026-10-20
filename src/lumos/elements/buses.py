r"""Splitting and gathering the lanes of a bus (step P15.7, ADR-0023).

A :class:`~lumos.fields.Bus` carries N lanes on one connection. Two pure elements change how they are wired,
without acting on the light:

- :class:`Demux` splits a bus of ``n_lanes`` lanes into the ports ``out0``... ``out{N-1}``, each holding one
  lane (a one-lane bus, so the lane keeps its name and the position of its telescope);
- :class:`Mux` puts the lanes received on ``in0``... ``in{N-1}`` side by side in one bus, in the order of its
  ports. It never sums nor interferes them: only a physical component (coupler, MMI, beam splitter,
  recombiner) mixes lanes. An input receiving a bus contributes all its lanes; an unconnected input is an
  empty lane (zero amplitude, the same kind of signal as the others), which the validation reports.

On packed buses both are slices and concatenations of one array. Use them to put different components on the
arms of an interferometer (``TelescopeArray → Demux → optics of each arm → Mux → chip``) or to send part of
the lanes of a chip to another one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from lumos.core.element import Element, Values
from lumos.core.param import Setting
from lumos.core.port import Input, Output, Port
from lumos.core.registry import register
from lumos.core.signal import OpticalSignal
from lumos.fields.bus import Bus
from lumos.fields.modal import ModalField

__all__ = ["Demux", "Mux"]

#: Most lanes of a Demux or a Mux (the number of their ports).
MAX_LANES = 64


def _lane_count(value: Any) -> int:
    n = int(value)
    if not 1 <= n <= MAX_LANES:
        raise ValueError(f"n_lanes must be between 1 and {MAX_LANES}, got {n}")
    return n


class _Repeated(Element):
    """An element whose ports ``<prefix>0``... ``<prefix>{N-1}`` follow its setting ``n_lanes``."""

    n_lanes = Setting(int, default=2, doc="number of lanes (and of repeated ports)")

    #: Direction and prefix of the repeated ports.
    repeated: ClassVar[tuple[str, str]] = ("out", "out")
    _cache: ClassVar[dict[int, Port]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Give every subclass its own cache of repeated ports (one port object per index)."""
        super().__init_subclass__(**kwargs)
        cls._cache = {}

    @classmethod
    def _make_port(cls, index: int) -> Port:
        raise NotImplementedError

    @classmethod
    def ports_for(cls, n_lanes: int) -> dict[str, Port]:
        """Return the ports for ``n_lanes`` lanes (repeated port ``k`` is the same object for any count)."""
        n = _lane_count(n_lanes)
        direction, prefix = cls.repeated
        repeated = {}
        for index in range(n):
            if index not in cls._cache:
                port = cls._make_port(index)
                port.name = port.attribute = f"{prefix}{index}"
                cls._cache[index] = port
            repeated[f"{prefix}{index}"] = cls._cache[index]
        fixed = dict(cls._lumos_ports)
        return {**repeated, **fixed} if direction == "in" else {**fixed, **repeated}

    def instance_ports(self) -> Mapping[str, Port]:
        """Return the fixed port and the ``n_lanes`` repeated ports of this element."""
        return type(self).ports_for(self.n_lanes)

    def port_lanes(self, name: str) -> int | None:
        """Return ``n_lanes`` for the bus side (``in`` of a Demux, ``out`` of a Mux)."""
        return int(self.n_lanes) if name == ("in" if self.repeated[0] == "out" else "out") else None

    @classmethod
    def schema(cls) -> dict[str, Any]:
        """Return the schema of the element, the repeated ports given by a pattern and a count."""
        schema = super().schema()
        direction, prefix = cls.repeated
        template = cls._make_port(0)
        entry = {
            "name": f"{prefix}{{i}}",
            "count": "n_lanes",
            "signal": template.signal_type.__name__,
            "multiple": template.multiple,
            "bus": template.bus,
            "optional": getattr(template, "optional", False),
            "doc": template.doc,
        }
        key = "inputs" if direction == "in" else "outputs"
        schema["x-lumos-ports"][key].insert(
            0 if direction == "in" else len(schema["x-lumos-ports"][key]), entry
        )
        return schema


@register("lumos.buses.Demux")
class Demux(_Repeated):
    """Split a bus of N lanes into N ports of one lane each (lane names and positions kept)."""

    repeated = ("out", "out")
    in_ = Input(OpticalSignal, bus=True, doc="bus of n_lanes lanes")

    @classmethod
    def _make_port(cls, index: int) -> Port:
        return Output(OpticalSignal, doc=f"lane {index} of the bus")

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Give each lane of the bus its own port."""
        signal = inputs["in"]
        bus = signal if isinstance(signal, Bus) else Bus([signal])
        n = int(values.n_lanes)
        if len(bus) != n:
            raise ValueError(f"this Demux splits {n} lanes, but it receives a bus of {len(bus)} lanes")
        return {f"out{k}": bus.select([k]) for k in range(n)}


@register("lumos.buses.Mux")
class Mux(_Repeated):
    """Put N lanes side by side in one bus, in the order of the ports, without mixing them."""

    repeated = ("in", "in")
    out = Output(OpticalSignal, bus=True, doc="bus of the lanes, in the order of the inputs")

    @classmethod
    def _make_port(cls, index: int) -> Port:
        return Input(
            OpticalSignal,
            bus=True,
            optional=True,
            doc=f"lane {index} (a lane or a bus; empty if unconnected)",
        )

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Concatenate the lanes received, with empty lanes for the unconnected inputs."""
        n = int(values.n_lanes)
        received = [inputs.get(f"in{k}") for k in range(n)]
        connected = [signal for signal in received if signal is not None]
        if not connected:
            raise ValueError("the Mux has no input: connect at least one lane")
        template = connected[0][0] if isinstance(connected[0], Bus) else connected[0]
        if isinstance(template, ModalField):  # an unconnected guide: one empty mode
            template = template._with(template.amplitudes[..., :1], None)
        parts = []
        for k, signal in enumerate(received):
            if signal is None:
                parts.append(Bus([template * 0], names=[f"in{k}"]))
            elif isinstance(signal, Bus):
                parts.append(signal)
            else:
                parts.append(Bus([signal], names=[f"in{k}"]))
        bus = Bus.concatenate(parts)
        names = list(bus.names)
        # lanes received alone are named after their place in the bus, like the lanes of a new bus
        offset = 0
        for signal, part in zip(received, parts, strict=True):
            if not isinstance(signal, Bus):
                names[offset] = f"T{offset + 1}"
            offset += len(part)
        return {"out": Bus(bus.lanes, names=names, positions=bus.positions)}
