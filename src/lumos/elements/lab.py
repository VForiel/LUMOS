r"""Laboratory equipment: stages, piezo, shutter, filter wheel, laser source, phase controller (step P12.5).

These elements describe actuators of an optical bench. Their actions are timed (the clock advances by the
predicted duration in simulation) and are sent to a driver in operation mode (the LumosLabs drivers of
:mod:`lumos.hardware.drivers.lumoslabs`, or a vendor driver). ``lumoslabs_model`` names the LumosLabs device
that each element stands for, used by the discovery to associate devices with elements.

- :class:`LinearStage` and :class:`RotationStage`: trapezoidal moves, duration :math:`d/v + v/a` (or
  :math:`2\sqrt{d/a}` for short moves), homing in :math:`|x|/v + 0.1` s;
- :class:`Piezo`: three axes, moves taken as immediate;
- :class:`Shutter` (8 ms to open, 10 ms to close) and :class:`FilterWheel` (0.4 s per slot, shortest way):
  they transmit or attenuate the light of their input;
- :class:`LaserSource`: a Gaussian beam emitted only when ``emitting`` (emission is a dangerous action that
  drivers confirm);
- :class:`PhaseController`: heater voltages of a photonic chip; :meth:`PhaseController.phase` gives the phase
  :math:`\kappa (V_i^2 + c\,(V_{i-1}^2 + V_{i+1}^2))` of each channel.

``ctx.link`` lets these states drive parameters of the optical model (a stage carrying a lens, a phase
controller setting phase shifters), so that the digital twin follows the bench.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from astropy import units as u

from lumos.core.action import action
from lumos.core.element import Element, Values
from lumos.core.param import Param, Setting, State
from lumos.core.port import Input, Output
from lumos.core.registry import register
from lumos.core.signal import OpticalSignal
from lumos.fields.grid import Grid
from lumos.fields.radiance import RadianceField
from lumos.fields.wavefront import Wavefront

__all__ = [
    "FilterWheel",
    "LaserSource",
    "LinearStage",
    "PhaseController",
    "Piezo",
    "RotationStage",
    "Shutter",
    "move_duration",
]


def move_duration(distance: u.Quantity, velocity: u.Quantity, acceleration: u.Quantity) -> u.Quantity:
    """Duration of a trapezoidal move (triangular when the distance is shorter than v²/a)."""
    d = abs(float(np.asarray(distance.si.value)))
    v = float(np.asarray(velocity.si.value))
    a = float(np.asarray(acceleration.si.value))
    if d == 0:
        return 0 * u.s
    if d >= v * v / a:
        return (d / v + v / a) * u.s
    return 2 * math.sqrt(d / a) * u.s


def _attenuate(signal: Any, power: float) -> Any:
    """Return ``signal`` with its power multiplied by ``power``."""
    if isinstance(signal, RadianceField):
        return signal * power
    return signal * math.sqrt(power)


class _Stage(Element):
    homed = State(default=0.0, doc="1 once homed")

    @action(duration=lambda self: abs(self.position) / self.velocity + 0.1 * u.s)
    def home(self) -> None:
        """Find the reference switch: the position becomes 0."""
        self.position = cast(Any, self.position) * 0
        self.homed = 1.0

    @action(
        duration=lambda self, target: move_duration(target - self.position, self.velocity, self.acceleration)
    )
    def move(self, target: u.Quantity) -> None:
        """Move to ``target``."""
        self.position = target


@register("lumos.lab.LinearStage")
class LinearStage(_Stage):
    """Motorised linear stage (LumosLabs ``LL-LS100``: 100 mm)."""

    lumoslabs_model = "LL-LS100"
    position = State(
        unit=u.mm, default=0 * u.mm, bounds=(0 * u.mm, 100 * u.mm), doc="position of the carriage"
    )
    velocity = Param(
        unit=u.mm / u.s,
        default=10 * u.mm / u.s,
        bounds=(0 * u.mm / u.s, 50 * u.mm / u.s),
        doc="cruise velocity of the moves",
    )
    acceleration = Param(
        unit=u.mm / u.s**2,
        default=50 * u.mm / u.s**2,
        bounds=(0 * u.mm / u.s**2, None),
        doc="acceleration of the moves",
    )


@register("lumos.lab.RotationStage")
class RotationStage(_Stage):
    """Motorised rotation stage (LumosLabs ``LL-RS360``: ±360°)."""

    lumoslabs_model = "LL-RS360"
    position = State(
        unit=u.deg, default=0 * u.deg, bounds=(-360 * u.deg, 360 * u.deg), doc="angle of the stage"
    )
    velocity = Param(
        unit=u.deg / u.s,
        default=20 * u.deg / u.s,
        bounds=(0 * u.deg / u.s, 90 * u.deg / u.s),
        doc="angular cruise velocity of the moves",
    )
    acceleration = Param(
        unit=u.deg / u.s**2,
        default=100 * u.deg / u.s**2,
        bounds=(0 * u.deg / u.s**2, None),
        doc="angular acceleration of the moves",
    )


@register("lumos.lab.Piezo")
class Piezo(Element):
    """Three-axis piezo controller (LumosLabs ``LL-PZ3``: 0–30 µm per axis)."""

    lumoslabs_model = "LL-PZ3"
    x = State(unit=u.um, default=0 * u.um, bounds=(0 * u.um, 30 * u.um), doc="position along x")
    y = State(unit=u.um, default=0 * u.um, bounds=(0 * u.um, 30 * u.um), doc="position along y")
    z = State(unit=u.um, default=0 * u.um, bounds=(0 * u.um, 30 * u.um), doc="position along z")

    @action()
    def move(self, axis: str, position: u.Quantity) -> None:
        """Move ``axis`` (``"X"``, ``"Y"`` or ``"Z"``) to ``position``."""
        name = str(axis).lower()
        if name not in ("x", "y", "z"):
            raise ValueError(f"unknown axis {axis!r}: X, Y or Z")
        setattr(self, name, position)


@register("lumos.lab.Shutter")
class Shutter(Element):
    """Shutter (LumosLabs ``LL-SH1``): transmits its input when open, blocks it when closed."""

    lumoslabs_model = "LL-SH1"
    is_open = State(default=0.0, doc="1 when open")
    in_ = Input(OpticalSignal)
    out = Output(OpticalSignal)

    @action(duration=8 * u.ms)
    def open(self) -> None:
        """Open the shutter."""
        self.is_open = 1.0

    @action(duration=10 * u.ms)
    def close(self) -> None:
        """Close the shutter."""
        self.is_open = 0.0

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Transmit the light when open."""
        return {"out": _attenuate(inputs["in"], 1.0 if values.is_open else 0.0)}


def _wheel_duration(self: Any, slot: int) -> u.Quantity:
    distance = abs(int(slot) - int(self.slot))
    return 0.4 * min(distance, 6 - distance) * u.s


@register("lumos.lab.FilterWheel")
class FilterWheel(Element):
    """Filter wheel of six slots (LumosLabs ``LL-FW6``) holding neutral filters of ``transmissions``."""

    lumoslabs_model = "LL-FW6"
    slot = State(default=1.0, bounds=(1, 6), doc="selected slot, 1 to 6")
    transmissions = Param(
        shape=(6,),
        default=[1.0, 0.5, 0.1, 0.01, 1e-3, 1e-4],
        bounds=(0, 1),
        doc="power transmission of each position of the wheel",
    )
    in_ = Input(OpticalSignal)
    out = Output(OpticalSignal)

    @action(duration=_wheel_duration)
    def select(self, slot: int) -> None:
        """Turn to ``slot`` (1 to 6) by the shortest way."""
        if not 1 <= int(slot) <= 6:
            raise ValueError(f"slot {slot} outside [1, 6]")
        self.slot = float(int(slot))

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Attenuate the light by the filter of the selected slot."""
        power = float(np.asarray(values.transmissions)[int(values.slot) - 1])
        return {"out": _attenuate(inputs["in"], power)}


@register("lumos.lab.LaserSource")
class LaserSource(Element):
    """Laser source (LumosLabs ``LL-LD``): a Gaussian beam of ``power`` emitted when ``emitting``."""

    lumoslabs_model = "LL-LD"
    wavelength = Param(
        unit=u.nm,
        default=635 * u.nm,
        bounds=(0 * u.m, None),
        inclusive=(False, True),
        doc="vacuum wavelength",
    )
    power = Param(unit=u.mW, default=1 * u.mW, bounds=(0 * u.mW, 10 * u.mW), doc="set point")
    waist = Param(
        unit=u.mm,
        default=1 * u.mm,
        bounds=(0 * u.m, None),
        inclusive=(False, True),
        doc="1/e² radius of the beam at its waist",
    )
    pixel_scale = Param(
        unit=u.um,
        default=20 * u.um,
        bounds=(0 * u.m, None),
        inclusive=(False, True),
        doc="pixel size of the grid of the beam",
    )
    n_pixels = Setting(int, default=256, doc="pixels per side of the grid of the beam")
    emitting = State(default=0.0, doc="1 while emitting")
    out = Output(Wavefront)

    @action()
    def emit(self, on: bool) -> None:
        """Switch the emission on or off."""
        self.emitting = 1.0 if on else 0.0

    @action()
    def set_power(self, power: u.Quantity) -> None:
        """Change the power set point."""
        self.power = power

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the beam at its waist (zero when not emitting)."""
        grid = Grid(int(values.n_pixels), dx=values.pixel_scale * u.m)
        power = values.power if values.emitting else 0.0
        beam = Wavefront.gaussian(grid, values.wavelength * u.m, waist=values.waist * u.m, power=power * u.W)
        return {"out": beam}


@register("lumos.lab.PhaseController")
class PhaseController(Element):
    """Heater voltages of a photonic chip (LumosLabs ``LL-PCU``, 8 channels of 0–10 V)."""

    lumoslabs_model = "LL-PCU"
    voltages = State(
        unit=u.V,
        shape=(8,),
        default=[0.0] * 8 * u.V,
        bounds=(0 * u.V, 10 * u.V),
        doc="voltage of each channel",
    )
    kappa = Param(unit=u.rad / u.V**2, default=math.pi / 25 * u.rad / u.V**2, doc="phase per V²")
    crosstalk = Param(default=0.05, bounds=(0, 1), doc="fraction of the heating of the neighbours")

    @action(duration=5 * u.ms)
    def set_voltage(self, channel: int, voltage: u.Quantity) -> None:
        """Set the voltage of ``channel`` (1 to 8)."""
        if not 1 <= int(channel) <= 8:
            raise ValueError(f"channel {channel} outside [1, 8]")
        voltages = self.voltages.copy()
        voltages[int(channel) - 1] = voltage
        self.voltages = voltages

    def phase(self, channel: int) -> u.Quantity:
        """Phase produced on ``channel``, including the thermal crosstalk of its neighbours."""
        squares = self.voltages.to_value(u.V) ** 2
        i = int(channel) - 1
        neighbours = sum(squares[j] for j in (i - 1, i + 1) if 0 <= j < squares.size)
        return (self.kappa * (squares[i] + float(self.crosstalk) * neighbours) * u.V**2).to(u.rad)
