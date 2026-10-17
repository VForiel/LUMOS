r"""Emulated LumosLabs devices (step P12.4, §5.8.1 of the specifications).

Each class documents its behavioural model; the commands are those of ``docs/hardware/lumoslabs_protocol.md``.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np
from astropy import units as u

from lumos.hardware import protocol
from lumos.hardware.emulators.base import Emulator, EmulatorError

__all__ = [
    "Camera",
    "FilterWheel",
    "LaserSource",
    "LinearStage",
    "PhaseController",
    "Piezo",
    "PowerMeter",
    "RotationStage",
    "Shutter",
]


def trapezoid(distance: float, velocity: float, acceleration: float) -> tuple[float, Any]:
    """Duration of a move and the travelled distance ``s(t)`` of a trapezoidal (or triangular) profile."""
    d, v, a = abs(distance), velocity, acceleration
    if d == 0:
        return 0.0, lambda t: 0.0
    if d >= v * v / a:
        t1 = v / a
        total = d / v + v / a

        def travelled(t: float) -> float:
            if t <= t1:
                return 0.5 * a * t * t
            if t <= total - t1:
                return 0.5 * a * t1 * t1 + v * (t - t1)
            return d - 0.5 * a * (total - t) ** 2

        return total, travelled
    half = math.sqrt(d / a)

    def triangular(t: float) -> float:
        if t <= half:
            return 0.5 * a * t * t
        return d - 0.5 * a * (2 * half - t) ** 2

    return 2 * half, triangular


class _Stage(Emulator):
    """Motorised stage: homing, trapezoidal moves, encoder noise of ``encoder_noise``."""

    unit: ClassVar[str] = "mm"
    default_velocity: ClassVar[float] = 10.0
    default_acceleration: ClassVar[float] = 50.0
    max_velocity: ClassVar[float] = 50.0
    encoder_noise: ClassVar[float] = 1e-5

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self.position = 0.0
        self.velocity = self.default_velocity
        self.acceleration = self.default_acceleration
        self.homed = False

    def _move(self, target: float, after: Any = None) -> None:
        start = self.position
        duration, travelled = trapezoid(target - start, self.velocity, self.acceleration)
        sign = 1.0 if target >= start else -1.0

        def progress(fraction: float) -> None:
            self.position = start + sign * travelled(fraction * duration)

        def done() -> None:
            self.position = target
            if after is not None:
                after()

        self.start(duration, progress, done)

    def cmd_home(self) -> None:
        """``HOME``: move to 0 and find the reference switch (0.1 s)."""
        self.require_ready()
        start = self.position
        duration = abs(start) / self.velocity + 0.1

        def progress(fraction: float) -> None:
            self.position = start * (1 - min(fraction * duration / max(duration - 0.1, 1e-12), 1.0))

        def done() -> None:
            self.position = 0.0
            self.homed = True

        self.start(duration, progress, done)

    def cmd_move(self, target: Any) -> None:
        """``MOVE <position>``."""
        value = self.quantity(target, self.unit, "position")
        self.require_ready()
        if not self.homed:
            raise EmulatorError(301, "not homed: send HOME first")
        self.check_range("POS", value, self.unit)
        self._move(value)

    def cmd_pos_q(self) -> u.Quantity:
        """``POS?``: encoder reading."""
        reading = self.position + self.offset() + self.encoder_noise * self.rng.standard_normal()
        return reading * u.Unit(self.unit)

    def cmd_vel(self, velocity: Any) -> None:
        """``VEL <velocity>``."""
        value = self.quantity(velocity, f"{self.unit}/s", "velocity")
        if not 0 < value <= self.max_velocity:
            raise EmulatorError(200, f"velocity {value:g} outside ]0, {self.max_velocity:g}] {self.unit}/s")
        self.velocity = value

    def cmd_vel_q(self) -> u.Quantity:
        """``VEL?``."""
        return self.velocity * u.Unit(f"{self.unit}/s")

    def cmd_acc(self, acceleration: Any) -> None:
        """``ACC <acceleration>``."""
        value = self.quantity(acceleration, f"{self.unit}/s2", "acceleration")
        if value <= 0:
            raise EmulatorError(200, "the acceleration must be positive")
        self.acceleration = value

    def cmd_acc_q(self) -> u.Quantity:
        """``ACC?``."""
        return self.acceleration * u.Unit(f"{self.unit}/s2")


class LinearStage(_Stage):
    """``LL-LS100``: linear stage of 100 mm, 10 mm/s and 50 mm/s² by default, encoder noise 10 nm."""

    model = "LL-LS100"
    description = "linear stage, 100 mm"
    limits: ClassVar = {"POS": (0 * u.mm, 100 * u.mm)}


class RotationStage(_Stage):
    """``LL-RS360``: rotation stage, ±360°, 20 °/s and 100 °/s² by default."""

    model = "LL-RS360"
    description = "rotation stage, ±360 deg"
    unit = "deg"
    default_velocity = 20.0
    default_acceleration = 100.0
    max_velocity = 90.0
    limits: ClassVar = {"POS": (-360 * u.deg, 360 * u.deg)}


class Piezo(Emulator):
    """``LL-PZ3``: three-axis piezo controller of 0–30 µm.

    The position lags by 1 % of the range behind the direction of motion (hysteresis), with a noise of 5 nm.
    """

    model = "LL-PZ3"
    description = "three-axis piezo controller, 30 µm"
    limits: ClassVar = {"POS": (0 * u.um, 30 * u.um)}
    hysteresis = 0.3  # µm
    noise = 0.005  # µm

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self.targets = {"X": 0.0, "Y": 0.0, "Z": 0.0}
        self.actual = dict(self.targets)

    def _axis(self, axis: Any) -> str:
        name = str(axis).upper()
        if name not in self.targets:
            raise EmulatorError(101, f"unknown axis {axis}: X, Y or Z")
        return name

    def cmd_move(self, axis: Any, position: Any) -> None:
        """``MOVE <axis> <position>``."""
        name = self._axis(axis)
        value = self.quantity(position, "um", "position")
        self.require_ready()
        self.check_range("POS", value, "um")
        step = value - self.targets[name]
        lag = 0.0 if step == 0 else -math.copysign(self.hysteresis, step)
        self.targets[name] = value
        self.actual[name] = value + lag

    def cmd_pos_q(self, axis: Any) -> u.Quantity:
        """``POS? <axis>``."""
        name = self._axis(axis)
        return (self.actual[name] + self.offset() + self.noise * self.rng.standard_normal()) * u.um


class Shutter(Emulator):
    """``LL-SH1``: opens in 8 ms, closes in 10 ms."""

    model = "LL-SH1"
    description = "shutter"
    opening, closing = 0.008, 0.010

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self.state = "CLOSED"

    def _go(self, state: str, duration: float) -> None:
        self.require_ready()

        def progress(fraction: float) -> None:
            self.state = "MOVING" if fraction < 1 else state

        def done() -> None:
            self.state = state

        self.start(duration, progress, done)

    def cmd_open(self) -> None:
        """``OPEN``."""
        self._go("OPEN", self.opening)

    def cmd_close(self) -> None:
        """``CLOSE``."""
        self._go("CLOSED", self.closing)

    def cmd_state_q(self) -> str:
        """``STATE?``."""
        return self.state

    def halt(self) -> None:
        """Close at once on an emergency stop (safe state)."""
        self.state = "CLOSED"


class FilterWheel(Emulator):
    """``LL-FW6``: six slots, 0.4 s per step, shortest direction."""

    model = "LL-FW6"
    description = "filter wheel, 6 positions"
    step_time = 0.4
    limits: ClassVar = {"SLOT": (1 * u.one, 6 * u.one)}

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self.slot = 1

    def cmd_slot(self, slot: Any) -> None:
        """``SLOT <n>``."""
        if not isinstance(slot, int):
            raise EmulatorError(101, "SLOT needs an integer from 1 to 6")
        if not 1 <= slot <= 6:
            raise EmulatorError(200, f"slot {slot} outside [1, 6]")
        self.require_ready()
        distance = abs(slot - self.slot)
        steps = min(distance, 6 - distance)

        def done() -> None:
            self.slot = slot

        self.start(steps * self.step_time, lambda fraction: None, done)

    def cmd_slot_q(self) -> int:
        """``SLOT?``."""
        return self.slot


class LaserSource(Emulator):
    """``LL-LD``: laser source of 635 nm, up to 10 mW.

    Warm-up P(t) = P_set (1 - exp(-t/τ)) with τ = 2 s; fluctuations of 0.2 %.
    """

    model = "LL-LD"
    description = "laser source, 635 nm, 10 mW"
    limits: ClassVar = {"POWER": (0 * u.mW, 10 * u.mW)}
    warm_up = 2.0
    stability = 0.002
    wavelength = 635.0

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self.setpoint = 0.0
        self.emitting = False
        self.switched_on = 0.0

    def cmd_emit(self, state: Any) -> None:
        """``EMIT ON|OFF``."""
        value = str(state).upper()
        if value not in ("ON", "OFF"):
            raise EmulatorError(101, "EMIT ON or EMIT OFF")
        if value == "ON":
            self.require_ready()
            if not self.emitting:
                self.switched_on = self.now
            self.emitting = True
        else:
            self.emitting = False

    def cmd_emit_q(self) -> str:
        """``EMIT?``."""
        return "ON" if self.emitting else "OFF"

    def cmd_power(self, power: Any) -> None:
        """``POWER <set point>``."""
        value = self.quantity(power, "mW", "power")
        self.check_range("POWER", value, "mW")
        self.setpoint = value

    def cmd_power_q(self) -> u.Quantity:
        """``POWER?``: emitted power."""
        if not self.emitting:
            return 0.0 * u.mW
        elapsed = self.now - self.switched_on
        power = self.setpoint * (1 - math.exp(-elapsed / self.warm_up))
        return (power * (1 + self.stability * self.rng.standard_normal()) + self.offset()) * u.mW

    def cmd_wl_q(self) -> u.Quantity:
        """``WL?``."""
        return self.wavelength * u.nm

    def halt(self) -> None:
        """Stop emitting on an emergency stop (safe state)."""
        self.emitting = False


class PowerMeter(Emulator):
    """``LL-PM``: power meter.

    Reads (1 + 2 %) P + 1 nW + a noise of 0.2 nW for 1 s of integration (∝ 1/√t), clipped at the range; the
    incident power P is sent by the twin (``SIM:POWER``).
    """

    model = "LL-PM"
    description = "power meter"
    gain_error = 0.02
    zero_offset = 1e-9
    noise = 2e-10

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self.incident = 0.0
        self.integration = 0.1
        self.range = 1e-2
        self.wavelength = 635.0

    def cmd_sim_power(self, power: Any) -> None:
        """``SIM:POWER <power>``."""
        self.incident = self.quantity(power, "W", "power")

    def cmd_int(self, time: Any) -> None:
        """``INT <integration time>``."""
        value = self.quantity(time, "s", "integration time")
        if not 1e-4 <= value <= 10:
            raise EmulatorError(200, "integration time outside [0.1 ms, 10 s]")
        self.integration = value

    def cmd_int_q(self) -> u.Quantity:
        """``INT?``."""
        return self.integration * u.s

    def cmd_range(self, power: Any) -> None:
        """``RANGE <full scale>``."""
        value = self.quantity(power, "W", "range")
        if value <= 0:
            raise EmulatorError(200, "the range must be positive")
        self.range = value

    def cmd_range_q(self) -> u.Quantity:
        """``RANGE?``."""
        return self.range * u.W

    def cmd_wl(self, wavelength: Any) -> None:
        """``WL <correction wavelength>``."""
        self.wavelength = self.quantity(wavelength, "nm", "wavelength")

    def cmd_read_q(self) -> u.Quantity:
        """``READ?``."""
        sigma = self.noise / math.sqrt(self.integration)
        reading = (1 + self.gain_error) * self.incident + self.zero_offset + self.offset()
        reading += sigma * self.rng.standard_normal()
        return min(reading, self.range) * u.W


class Camera(Emulator):
    """``LL-CAM``: scientific camera.

    Counts bias + [Poisson((S + D) t) + read noise] / gain with a bias of 100 adu, a dark current D of 5 e⁻/s
    (500 e⁻/s more on 0.5 % hot pixels), a read noise of 3 e⁻ and 2 e⁻/adu; the image S (e⁻/s) is sent by the
    twin (``SIM:IMAGE``).
    """

    model = "LL-CAM"
    description = "scientific camera"
    bias = 100.0
    dark = 5.0
    hot = 500.0
    read_noise = 3.0
    limits: ClassVar = {"EXP": (1e-6 * u.s, 100 * u.s), "GAIN": (0.1 * u.one, 100 * u.one)}

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self.exposure = 0.01
        self.gain = 2.0
        self._set_image(np.zeros((32, 32)))
        self.frame = np.zeros((32, 32), dtype=np.uint16)

    def _set_image(self, image: np.ndarray) -> None:
        self.image = np.asarray(image, dtype=float)
        if getattr(self, "hot_pixels", np.zeros(0)).shape != self.image.shape:
            hot = np.random.default_rng(len(self.serial) + self.image.size).random(self.image.shape) < 0.005
            self.hot_pixels = np.where(hot, self.hot, 0.0)
        self.roi = (0, 0, self.image.shape[1], self.image.shape[0])

    def cmd_sim_image(self, dtype: Any, shape: Any, data: Any) -> None:
        """``SIM:IMAGE <dtype> <h>x<w> b64:...`` (e⁻/s)."""
        self._set_image(protocol.decode_array(str(dtype), str(shape), str(data)))

    def cmd_exp(self, time: Any) -> None:
        """``EXP <exposure>``."""
        value = self.quantity(time, "s", "exposure")
        self.check_range("EXP", value, "s")
        self.exposure = value

    def cmd_exp_q(self) -> u.Quantity:
        """``EXP?``."""
        return self.exposure * u.s

    def cmd_gain(self, gain: Any) -> None:
        """``GAIN <e-/adu>``."""
        value = float(gain.value) if isinstance(gain, u.Quantity) else float(gain)
        self.check_range("GAIN", value, "")
        self.gain = value

    def cmd_gain_q(self) -> float:
        """``GAIN?``."""
        return self.gain

    def cmd_roi(self, x: Any, y: Any, width: Any, height: Any) -> None:
        """``ROI <x> <y> <width> <height>``."""
        values = [int(v) for v in (x, y, width, height)]
        ny, nx = self.image.shape
        if values[0] < 0 or values[1] < 0 or values[2] < 1 or values[3] < 1:
            raise EmulatorError(200, "invalid region of interest")
        if values[0] + values[2] > nx or values[1] + values[3] > ny:
            raise EmulatorError(200, f"the region leaves the sensor of {nx} x {ny} pixels")
        self.roi = (values[0], values[1], values[2], values[3])

    def cmd_roi_q(self) -> tuple[int, int, int, int]:
        """``ROI?``."""
        return self.roi

    def cmd_acq(self) -> None:
        """``ACQ``: expose for ``EXP``."""
        self.require_ready()
        exposure = self.exposure

        def done() -> None:
            electrons = self.rng.poisson((self.image + self.dark + self.hot_pixels) * exposure).astype(float)
            electrons += self.read_noise * self.rng.standard_normal(electrons.shape)
            counts = np.clip(np.round(self.bias + self.offset() + electrons / self.gain), 0, 65535)
            x, y, w, h = self.roi
            self.frame = counts.astype(np.uint16)[y : y + h, x : x + w]

        self.start(exposure, lambda fraction: None, done)

    def cmd_frame_q(self) -> tuple[str, str, str]:
        """``FRAME?``: the last image in adu."""
        return protocol.encode_array(self.frame)


class PhaseController(Emulator):
    """``LL-PCU``: phase controller with 8 heaters of 0–10 V.

    φ_i = κ (V_i² + c (V_{i-1}² + V_{i+1}²)) with κ = π/25 rad/V² and c = 5 %; settling time 5 ms.
    """

    model = "LL-PCU"
    description = "phase controller of photonic chips, 8 channels"
    limits: ClassVar = {"VOLT": (0 * u.V, 10 * u.V)}
    channels = 8
    kappa = math.pi / 25
    crosstalk = 0.05
    settling = 0.005

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self.voltages = np.zeros(self.channels)

    def _channel(self, channel: Any) -> int:
        if not isinstance(channel, int) or not 1 <= channel <= self.channels:
            raise EmulatorError(101, f"channel must be an integer from 1 to {self.channels}")
        return channel - 1

    def cmd_chan_q(self) -> int:
        """``CHAN?``."""
        return self.channels

    def cmd_volt(self, channel: Any, voltage: Any) -> None:
        """``VOLT <channel> <voltage>``."""
        index = self._channel(channel)
        value = self.quantity(voltage, "V", "voltage")
        self.check_range("VOLT", value, "V")
        self.require_ready()
        self.voltages[index] = value
        self.start(self.settling, lambda fraction: None, lambda: None)

    def cmd_volt_q(self, channel: Any) -> u.Quantity:
        """``VOLT? <channel>``."""
        return float(self.voltages[self._channel(channel)]) * u.V

    def cmd_phase_q(self, channel: Any) -> u.Quantity:
        """``PHASE? <channel>``."""
        i = self._channel(channel)
        squares = self.voltages**2
        neighbours = sum(squares[j] for j in (i - 1, i + 1) if 0 <= j < self.channels)
        return (self.kappa * (squares[i] + self.crosstalk * neighbours) + self.offset()) * u.rad

    def halt(self) -> None:
        """Switch the heaters off on an emergency stop."""
        self.voltages[:] = 0.0
