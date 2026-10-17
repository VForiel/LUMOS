"""Drivers of the LumosLabs devices (step P12.5, REQ-HW-3, ADR-0014).

Every driver speaks the LumosLabs protocol over a :class:`~lumos.hardware.transport.Transport` (a URL such as
``tcp://127.0.0.1:5100`` or ``serial://COM3``, or a transport object): exactly the code path of a physical
device. On connection it checks the model (``*IDN?``) and reads the limits of the device, which then apply
to every command (:meth:`lumos.core.driver.Driver.check`). Long operations (moves, exposures) are followed by
``BUSY?`` polling every ``poll`` seconds; an emergency stop from another thread interrupts the polling and
sends ``STOP``; :meth:`LumosLabsDriver.reset` sends ``*RST``.

The actions of the lab elements (:mod:`lumos.elements.lab`) and of the detectors map to ``do_<action>``
methods; the measuring devices also offer :meth:`PowerMeterDriver.acquire` and :meth:`CameraDriver.acquire`
(``DataProduct`` measurements) and ``feed`` (data of the digital twin, for the emulators).
"""

from __future__ import annotations

import logging
import time
from typing import Any, ClassVar

import numpy as np
from astropy import units as u

from lumos.core.driver import Command, Driver, DriverError, DriverStopped, DriverTimeout, Reply
from lumos.fields.dataproduct import DataProduct
from lumos.hardware import protocol
from lumos.hardware.transport import Transport, open_transport

__all__ = [
    "CameraDriver",
    "FilterWheelDriver",
    "LaserDriver",
    "LinearStageDriver",
    "LumosLabsDriver",
    "PhaseControllerDriver",
    "PiezoDriver",
    "PowerMeterDriver",
    "RotationStageDriver",
    "ShutterDriver",
]

logger = logging.getLogger("lumos.hardware")


class LumosLabsDriver(Driver):
    """Base of the LumosLabs drivers.

    Parameters
    ----------
    transport
        Transport, or its URL (``tcp://host:port``, ``serial://port``, ``visa://resource``).
    poll
        Period of the ``BUSY?`` polling during long operations, in seconds.
    **options
        Options of :class:`~lumos.core.driver.Driver` (timeout, retries, dry_run, confirm).
    """

    model: ClassVar[str] = ""
    #: Query commands by name, e.g. ``{"position": "POS?"}``.
    queries: ClassVar[dict[str, str]] = {}

    def __init__(self, transport: str | Transport, *, poll: float = 0.02, **options: Any) -> None:
        super().__init__(**options)
        self.transport = (
            open_transport(transport, timeout=self.timeout_s) if isinstance(transport, str) else transport
        )
        self.poll = float(poll)
        self.identity: protocol.Identity | None = None

    # --- Connection ----------------------------------------------------------------------------------------

    def _connect(self) -> None:
        self.transport.open()
        identity = protocol.parse_identity(str(self.ask("*IDN?")[0]))
        if identity.model != self.model:
            self.transport.close()
            raise DriverError(
                f"{type(self).__name__} expected a {self.model}, found {identity.model} at {self.url}"
            )
        self.identity = identity
        self.read_limits()

    def _disconnect(self) -> None:
        self.transport.close()

    @property
    def url(self) -> str:
        """URL of the transport."""
        return self.transport.url

    def read_limits(self) -> None:
        """Read the limits of the device into :attr:`limits` (nothing by default)."""

    # --- Protocol ------------------------------------------------------------------------------------------

    def ask(self, verb: str, *arguments: Any) -> list[Any]:
        """Send a command and return the values of the reply (``ERR`` replies raise)."""
        line = protocol.format_command(verb, *arguments)
        return protocol.parse_reply(self.transport.query(line, self.timeout_s))

    def run(self, verb: str, *arguments: Any, expected: float = 0.0) -> None:
        """Send a long operation and poll ``BUSY?`` until it ends (interrupted by an emergency stop)."""
        self.ask(verb, *arguments)
        deadline = time.monotonic() + 2 * expected + self.timeout_s
        while True:
            if not self.wait(self.poll):
                raise DriverStopped(f"{self.model} {verb} interrupted by an emergency stop")
            if self.ask("BUSY?") == [0]:
                return
            if time.monotonic() > deadline:
                raise DriverTimeout(f"{self.model} {verb} did not finish in time")

    # --- Driver interface ----------------------------------------------------------------------------------

    def _send(self, command: Command) -> Reply:
        handler = getattr(self, "do_" + command.action, None)
        if handler is None:
            raise DriverError(f"{type(self).__name__} cannot execute {command.action!r}")
        reply = handler(*command.args, **command.kwargs)
        return reply if isinstance(reply, Reply) else Reply(value=reply)

    def _query(self, name: str, *arguments: Any) -> Any:
        if name not in self.queries:
            raise DriverError(f"{type(self).__name__} cannot read {name!r}; known: {', '.join(self.queries)}")
        values = self.ask(self.queries[name], *arguments)
        return values[0] if len(values) == 1 else values

    def _stop(self) -> None:
        if not self.transport.is_open:
            return
        try:
            self.ask("STOP")
        except Exception:  # the device may be unreachable: the stop must not raise
            logger.exception("%s: STOP failed", self.model)

    def reset(self) -> None:
        """Acknowledge an emergency stop: also clears it on the device (``*RST``)."""
        super().reset()
        if self.connected:
            self.ask("*RST")

    def _status(self) -> dict[str, Any]:
        serial = None if self.identity is None else self.identity.serial
        return {"model": self.model, "serial": serial, "url": self.url}


def _limits(driver: LumosLabsDriver, name: str) -> tuple[Any, Any]:
    low, high = driver.ask("LIMITS?", name)
    return low, high


# --- Motion -----------------------------------------------------------------------------------------------


class _StageDriver(LumosLabsDriver):
    unit: ClassVar[u.UnitBase] = u.mm
    signatures: ClassVar = {"move": ("target",)}
    queries: ClassVar = {"position": "POS?", "velocity": "VEL?", "acceleration": "ACC?"}

    def read_limits(self) -> None:
        """Range of the position."""
        self.limits["move.target"] = _limits(self, "POS")

    def _profile(self) -> None:
        element = self.element
        if element is not None and hasattr(element, "velocity"):
            self.ask("VEL", element.velocity.to(self.unit / u.s))
            self.ask("ACC", element.acceleration.to(self.unit / u.s**2))

    def do_home(self) -> Reply:
        """Find the reference."""
        self.run("HOME")
        return Reply(state={"position": self._query("position"), "homed": 1.0})

    def do_move(self, target: u.Quantity) -> Reply:
        """Move to ``target`` with the velocity and acceleration of the element."""
        self._profile()
        self.run("MOVE", target.to(self.unit))
        return Reply(state={"position": self._query("position")})


class LinearStageDriver(_StageDriver):
    """Driver of the ``LL-LS100`` linear stage."""

    model = "LL-LS100"


class RotationStageDriver(_StageDriver):
    """Driver of the ``LL-RS360`` rotation stage."""

    model = "LL-RS360"
    unit = u.deg


class PiezoDriver(LumosLabsDriver):
    """Driver of the ``LL-PZ3`` piezo controller."""

    model = "LL-PZ3"
    signatures: ClassVar = {"move": ("axis", "position")}
    queries: ClassVar = {"position": "POS?"}

    def read_limits(self) -> None:
        """Range of the axes."""
        self.limits["move.position"] = _limits(self, "POS")

    def do_move(self, axis: str, position: u.Quantity) -> Reply:
        """Move ``axis`` to ``position``."""
        name = str(axis).upper()
        self.ask("MOVE", name, position.to(u.um))
        return Reply(state={name.lower(): self._query("position", name)})


# --- Shutter, filter wheel, laser -------------------------------------------------------------------------


class ShutterDriver(LumosLabsDriver):
    """Driver of the ``LL-SH1`` shutter (safe state: closed)."""

    model = "LL-SH1"
    queries: ClassVar = {"state": "STATE?"}

    def do_open(self) -> Reply:
        """Open."""
        self.run("OPEN", expected=0.008)
        return Reply(state={"is_open": 1.0})

    def do_close(self) -> Reply:
        """Close."""
        self.run("CLOSE", expected=0.010)
        return Reply(state={"is_open": 0.0})

    def _safe_state(self) -> None:
        self.ask("*RST")
        self.ask("CLOSE")


class FilterWheelDriver(LumosLabsDriver):
    """Driver of the ``LL-FW6`` filter wheel."""

    model = "LL-FW6"
    signatures: ClassVar = {"select": ("slot",)}
    queries: ClassVar = {"slot": "SLOT?"}

    def read_limits(self) -> None:
        """Slots 1 to 6."""
        self.limits["select.slot"] = (1, 6)

    def do_select(self, slot: int) -> Reply:
        """Turn to ``slot``."""
        self.run("SLOT", int(slot), expected=1.2)
        return Reply(state={"slot": float(self._query("slot"))})


class LaserDriver(LumosLabsDriver):
    """Driver of the ``LL-LD`` laser: switching the emission on needs a confirmation; safe state: off."""

    model = "LL-LD"
    signatures: ClassVar = {"set_power": ("power",), "emit": ("on",)}
    queries: ClassVar = {"power": "POWER?", "wavelength": "WL?", "emission": "EMIT?"}

    def read_limits(self) -> None:
        """Range of the power set point."""
        self.limits["set_power.power"] = _limits(self, "POWER")

    def is_dangerous(self, command: Command) -> bool:
        """Only switching the emission on is dangerous."""
        if command.action != "emit":
            return False
        on = command.args[0] if command.args else command.kwargs.get("on", False)
        return bool(on)

    def do_emit(self, on: bool) -> Reply:
        """Switch the emission on or off."""
        self.ask("EMIT", "ON" if on else "OFF")
        return Reply(state={"emitting": 1.0 if on else 0.0})

    def do_set_power(self, power: u.Quantity) -> Reply:
        """Change the set point."""
        self.ask("POWER", power.to(u.mW))
        return Reply(state={"power": power})

    def _safe_state(self) -> None:
        self.ask("EMIT", "OFF")


# --- Measurements -----------------------------------------------------------------------------------------


class PowerMeterDriver(LumosLabsDriver):
    """Driver of the ``LL-PM`` power meter."""

    model = "LL-PM"
    queries: ClassVar = {"power": "READ?", "integration": "INT?", "range": "RANGE?"}

    def configure(
        self,
        *,
        integration: u.Quantity | None = None,
        range: u.Quantity | None = None,
        wavelength: u.Quantity | None = None,
    ) -> None:
        """Set the integration time, the full scale and the correction wavelength."""
        if integration is not None:
            self.ask("INT", integration.to(u.s))
        if range is not None:
            self.ask("RANGE", range.to(u.W))
        if wavelength is not None:
            self.ask("WL", wavelength.to(u.nm))

    def feed(self, power: u.Quantity) -> None:
        """Give the emulator the incident power computed by the twin (``SIM:POWER``)."""
        self.ask("SIM:POWER", power.to(u.W))

    def acquire(self) -> DataProduct:
        """Read the power (a one-channel product in W, like :class:`~lumos.elements.detectors.PowerMeter`)."""
        self._require_connection()
        value = self.ask("READ?")[0]
        return DataProduct(np.atleast_1d(value.to(u.W)), meta={"device": self.model, "url": self.url})


class CameraDriver(LumosLabsDriver):
    """Driver of the ``LL-CAM`` camera."""

    model = "LL-CAM"
    queries: ClassVar = {"exposure": "EXP?", "gain": "GAIN?", "roi": "ROI?"}

    def read_limits(self) -> None:
        """Range of the exposure time."""
        self.limits["configure.exposure"] = _limits(self, "EXP")

    def configure(
        self,
        *,
        exposure: u.Quantity | None = None,
        gain: float | None = None,
        roi: tuple[int, ...] | None = None,
    ) -> None:
        """Set the exposure time, the gain (e⁻/adu) and the region of interest ``(x, y, width, height)``."""
        if exposure is not None:
            self.check(Command("camera", "configure", (), {"exposure": exposure}))
            self.ask("EXP", exposure.to(u.s))
        if gain is not None:
            self.ask("GAIN", float(gain))
        if roi is not None:
            self.ask("ROI", *(int(v) for v in roi))

    def feed(self, image: u.Quantity) -> None:
        """Give the emulator the image of the twin, in e⁻/s per pixel (``SIM:IMAGE``)."""
        rates = np.asarray(image.to_value(u.electron / u.s), dtype=np.float32)
        self.ask("SIM:IMAGE", *protocol.encode_array(rates))

    def do_expose(self) -> Reply:
        """Take an exposure (``ACQ``) and wait for its end."""
        exposure = float(np.asarray(self.ask("EXP?")[0].to_value(u.s)))
        self.run("ACQ", expected=exposure)
        return Reply()

    def frame(self) -> DataProduct:
        """Return the last image (adu), with pixel axes."""
        image = protocol.decode_array(*(str(v) for v in self.ask("FRAME?")))
        ny, nx = image.shape
        axes = {"y": np.arange(ny) * u.pix, "x": np.arange(nx) * u.pix}
        return DataProduct(
            image.astype(float) * u.adu, axes=axes, meta={"device": self.model, "url": self.url}
        )

    def acquire(self) -> DataProduct:
        """Take an exposure with the settings of the element (exposure, gain) and return the image."""
        self._require_connection()
        element = self.element
        if element is not None and hasattr(element, "exposure"):
            self.configure(exposure=element.exposure, gain=float(element.gain.to_value(u.electron / u.adu)))
        self.do_expose()
        return self.frame()


class PhaseControllerDriver(LumosLabsDriver):
    """Driver of the ``LL-PCU`` phase controller (safe state: heaters off)."""

    model = "LL-PCU"
    signatures: ClassVar = {"set_voltage": ("channel", "voltage")}
    queries: ClassVar = {"phase": "PHASE?", "voltage": "VOLT?", "channels": "CHAN?"}

    def read_limits(self) -> None:
        """Range of the voltages."""
        self.limits["set_voltage.voltage"] = _limits(self, "VOLT")

    def do_set_voltage(self, channel: int, voltage: u.Quantity) -> Reply:
        """Set the voltage of ``channel``."""
        self.run("VOLT", int(channel), voltage.to(u.V), expected=0.005)
        return Reply()

    def _safe_state(self) -> None:
        self.ask("STOP")  # the controller switches its heaters off on a stop
        self.ask("*RST")
