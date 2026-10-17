"""Base of the LumosLabs emulators: protocol handling, timed operations, emergency stop and faults."""

from __future__ import annotations

import threading
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from astropy import units as u

from lumos.core.driver import ConnectionLost
from lumos.hardware import protocol

__all__ = ["Emulator", "EmulatorError", "Operation"]


class EmulatorError(Exception):
    """Error answered as ``ERR <code> <message>``."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = int(code)
        self.message = message


@dataclass
class Operation:
    """A long operation: started at ``start`` for ``duration`` seconds.

    ``progress(fraction)`` updates the state of the device while it runs (``fraction`` in [0, 1]);
    ``done()`` is called once at the end (not after an emergency stop).
    """

    start: float
    duration: float
    progress: Callable[[float], None]
    done: Callable[[], None]


class Emulator:
    """Emulated device answering the LumosLabs protocol.

    Parameters
    ----------
    serial
        Serial number.
    clock
        Function returning the time in seconds (``time.monotonic``; a fake clock in tests).
    latency
        Real time waited before each reply, in seconds.
    seed
        Seed of the noise (default: derived from the serial number).
    """

    model: ClassVar[str] = ""
    description: ClassVar[str] = ""
    firmware: ClassVar[str] = "1.0"
    limits: ClassVar[dict[str, tuple[u.Quantity, u.Quantity]]] = {}

    def __init__(
        self,
        *,
        serial: str = "SN0001",
        clock: Callable[[], float] = time.monotonic,
        latency: float = 0.001,
        seed: int | None = None,
    ) -> None:
        self.serial = serial
        self.clock = clock
        self.latency = float(latency)
        self.rng = np.random.default_rng(zlib.crc32(serial.encode()) if seed is None else seed)
        self.stopped = False
        self.faults: dict[str, float] = {}
        self._operation: Operation | None = None
        self._lock = threading.RLock()

    # --- Protocol ------------------------------------------------------------------------------------------

    def handle(self, line: str) -> str:
        """Answer one command line (raises :class:`ConnectionLost` for an injected disconnection)."""
        with self._lock:
            if self.faults.pop("DISCONNECT", None) is not None:
                raise ConnectionLost(f"{self.model} {self.serial}: injected disconnection")
            delay = self.latency + self.faults.get("DELAY", 0.0)
            if delay > 0:
                time.sleep(delay)
            try:
                verb, arguments = protocol.parse_command(line)
                method = getattr(self, "cmd_" + _method_name(verb), None)
                if method is None:
                    raise EmulatorError(100, f"unknown command {verb}")
                self._update()
                try:
                    result = method(*arguments)
                except TypeError as error:
                    raise EmulatorError(101, f"bad arguments for {verb}: {error}") from error
            except EmulatorError as error:
                return protocol.format_error(error.code, error.message)
            except protocol.ProtocolError as error:
                return protocol.format_error(101, str(error))
            if result is None:
                return protocol.format_reply()
            return protocol.format_reply(*(result if isinstance(result, tuple) else (result,)))

    def inject(self, kind: str, value: float = 0.0) -> None:
        """Inject a fault: ``DISCONNECT``, ``DELAY`` (s), ``OFFSET``, ``STUCK`` or ``CLEAR``."""
        kind = kind.upper()
        if kind == "CLEAR":
            self.faults.clear()
        elif kind in ("DISCONNECT", "DELAY", "OFFSET", "STUCK"):
            self.faults[kind] = float(value)
        else:
            raise EmulatorError(101, f"unknown fault {kind}")

    # --- Helpers -------------------------------------------------------------------------------------------

    @property
    def now(self) -> float:
        """Current time of the device clock."""
        return float(self.clock())

    def quantity(self, value: Any, unit: u.UnitBase | str, name: str) -> float:
        """Return ``value`` in ``unit`` (a float), or answer ``ERR 101``/``ERR 102``."""
        if not isinstance(value, u.Quantity):
            raise EmulatorError(101, f"{name} needs a quantity in {unit}, e.g. 1[{unit}]")
        try:
            return float(np.asarray(value.to_value(unit)))
        except u.UnitConversionError as error:
            raise EmulatorError(102, f"{name} in {value.unit} cannot be converted to {unit}") from error

    def check_range(self, name: str, value: float, unit: str) -> None:
        """Answer ``ERR 200`` if ``value`` (in ``unit``) is outside ``limits[name]``."""
        low, high = (float(np.asarray(q.to_value(unit))) for q in self.limits[name])
        if not low <= value <= high:
            raise EmulatorError(200, f"{name} {value:g} {unit} outside [{low:g}, {high:g}] {unit}")

    def offset(self) -> float:
        """Offset injected in the measurements."""
        return self.faults.get("OFFSET", 0.0)

    def start(self, duration: float, progress: Callable[[float], None], done: Callable[[], None]) -> None:
        """Start a long operation (``ERR 400`` after a stop, ``ERR 300`` if another one runs)."""
        if self.stopped:
            raise EmulatorError(400, "emergency stop latched: send *RST")
        if self.busy:
            raise EmulatorError(300, "busy")
        self._operation = Operation(self.now, max(float(duration), 0.0), progress, done)
        self._update()

    def require_ready(self) -> None:
        """Answer ``ERR 400`` after an emergency stop and ``ERR 300`` during an operation."""
        if self.stopped:
            raise EmulatorError(400, "emergency stop latched: send *RST")
        if self.busy:
            raise EmulatorError(300, "busy")

    @property
    def busy(self) -> bool:
        """Whether a long operation is running."""
        self._update()
        return self._operation is not None

    def _update(self) -> None:
        operation = self._operation
        if operation is None:
            return
        elapsed = self.now - operation.start
        fraction = 1.0 if operation.duration == 0 else min(max(elapsed / operation.duration, 0.0), 1.0)
        operation.progress(fraction)
        if fraction >= 1.0 and "STUCK" not in self.faults:
            self._operation = None
            operation.done()

    # --- Common commands -----------------------------------------------------------------------------------

    def cmd_star_idn_q(self) -> str:
        """``*IDN?``."""
        return protocol.format_identity(
            protocol.Identity(protocol.VENDOR, self.model, self.serial, self.firmware)
        )

    def cmd_star_rst(self) -> None:
        """``*RST``: stop the operation and clear the emergency stop."""
        self._operation = None
        self.stopped = False
        self.reset()

    def cmd_status_q(self) -> str:
        """``STATUS?``."""
        if self.stopped:
            return "STOPPED"
        return "BUSY" if self.busy else "READY"

    def cmd_busy_q(self) -> int:
        """``BUSY?``."""
        return int(self.busy)

    def cmd_stop(self) -> None:
        """``STOP``: freeze the operation in progress and latch the emergency stop."""
        self._update()
        self._operation = None
        self.stopped = True
        self.halt()

    def cmd_limits_q(self, name: str) -> tuple[u.Quantity, u.Quantity]:
        """``LIMITS? <name>``."""
        key = str(name).upper()
        if key not in self.limits:
            raise EmulatorError(101, f"no limits named {name}; known: {', '.join(self.limits)}")
        return self.limits[key]

    def cmd_sim_fault(self, kind: str, value: Any = 0.0) -> None:
        """``SIM:FAULT <kind> [value]``."""
        amount = float(np.asarray(value.value)) if isinstance(value, u.Quantity) else float(value)
        self.inject(str(kind), amount)

    # --- Hooks of the devices ------------------------------------------------------------------------------

    def reset(self) -> None:
        """Device-specific part of ``*RST``."""

    def halt(self) -> None:
        """Device-specific part of ``STOP`` (laser off...)."""


def _method_name(verb: str) -> str:
    return verb.lower().replace("*", "star_").replace("?", "_q").replace(":", "_")
