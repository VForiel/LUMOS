"""Drivers of real or emulated equipment (REQ-HW-1).

A :class:`Driver` connects an element to the equipment it represents. In operation mode, the actions of the
element become :class:`Command` objects sent to the driver, which answers with a :class:`Reply`: the measured
duration, the state read back from the equipment and an optional value. Concrete drivers implement the
protected methods ``_connect``, ``_disconnect``, ``_send``, ``_query`` and ``_stop`` on top of a transport
(serial, TCP, VISA, vendor SDK; phase 12).

:class:`SimulatedDriver` sends nothing: it records the commands and answers with the virtual state of its
element, which makes it possible to test a sequence of commands without any equipment.

Safety (REQ-HW-4, ADR-0014)
---------------------------
Every driver inherits these checks, applied in :meth:`Driver.send` to every command, whether it comes from an
element action or is sent directly:

- **limits**: ``limits`` maps ``"action.argument"`` to ``(low, high)`` quantities; ``signatures`` gives the
  names of the positional arguments of each action. A value outside its range raises :class:`LimitError`;
- **dry run**: with ``dry_run=True`` the command is checked and logged (as a warning), never sent;
- **confirmations**: the actions listed in ``dangerous`` are sent only if the ``confirm`` callback accepts
  the command (:class:`ActionRefused` otherwise, also when no callback is set);
- **recovery**: a :class:`ConnectionLost` from the equipment triggers a reconnection and a new attempt, at
  most ``retries`` times; :class:`DriverTimeout` reports an equipment that does not answer;
- **emergency stop**: :meth:`Driver.stop` may be called from another thread; it interrupts the waits of a
  command in progress (:meth:`Driver.wait`), which then raises :class:`DriverStopped`. The driver refuses new
  commands until :meth:`Driver.reset`;
- **safe state**: :meth:`Driver.safe_state` brings the equipment to a safe state (laser off, shutter
  closed...).

States: ``"disconnected"``, ``"ready"``, ``"busy"`` (a command in progress), ``"stopped"`` (after an emergency
stop) and ``"error"`` (after a failure).
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from astropy import units as u

from lumos.units import quantity_from_json, quantity_to_json

__all__ = [
    "ActionRefused",
    "Command",
    "ConnectionLost",
    "Driver",
    "DriverError",
    "DriverStopped",
    "DriverTimeout",
    "LimitError",
    "Reply",
    "SimulatedDriver",
]

logger = logging.getLogger("lumos.hardware")

_QUANTITY = "__quantity__"


class DriverError(RuntimeError):
    """Raised when a driver cannot execute a command (missing, disconnected, failing equipment)."""


class LimitError(DriverError):
    """A command argument is outside the software limits of the driver."""


class ActionRefused(DriverError):
    """A dangerous action was not confirmed."""


class ConnectionLost(DriverError):
    """The connection to the equipment was lost (the driver reconnects and retries)."""


class DriverTimeout(DriverError):
    """The equipment did not answer in time."""


class DriverStopped(DriverError):
    """A command was interrupted by an emergency stop."""


def _encode(value: Any) -> Any:
    if isinstance(value, u.Quantity):
        return {_QUANTITY: quantity_to_json(value)}
    if isinstance(value, list | tuple):
        return [_encode(item) for item in value]
    if isinstance(value, Mapping):
        return {str(k): _encode(v) for k, v in value.items()}
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {_QUANTITY}:
            return quantity_from_json(value[_QUANTITY])
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


@dataclass(frozen=True, eq=False)
class Command:
    """A command for the equipment of an element.

    Attributes
    ----------
    element
        Path of the element in the context (``"id"`` or ``"group.child"``).
    action
        Name of the action.
    args, kwargs
        Arguments of the action (quantities keep their units).
    """

    element: str
    action: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        """Compare element, action and arguments (quantities by value and unit)."""
        if not isinstance(other, Command):
            return NotImplemented
        return (
            self.element == other.element
            and self.action == other.action
            and _same(self.args, other.args)
            and set(self.kwargs) == set(other.kwargs)
            and all(_same(self.kwargs[k], other.kwargs[k]) for k in self.kwargs)
        )

    __hash__ = None  # type: ignore[assignment]

    def to_json(self) -> dict[str, Any]:
        """Encode the command as a JSON-compatible dictionary."""
        return {
            "element": self.element,
            "action": self.action,
            "args": _encode(list(self.args)),
            "kwargs": _encode(self.kwargs),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Command:
        """Decode a dictionary produced by :meth:`to_json`."""
        return cls(
            data["element"], data["action"], tuple(_decode(data["args"])), dict(_decode(data["kwargs"]))
        )

    def __str__(self) -> str:
        """Readable form ``element.action(args)``."""
        arguments = [repr(a) if not isinstance(a, u.Quantity) else str(a) for a in self.args]
        arguments += [f"{k}={v}" for k, v in self.kwargs.items()]
        return f"{self.element}.{self.action}({', '.join(arguments)})"


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, tuple | list) and isinstance(b, tuple | list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, u.Quantity) or isinstance(b, u.Quantity):
        try:
            qa, qb = u.Quantity(a), u.Quantity(b)
            return bool(qa.unit == qb.unit and np.all(np.asarray(qa.value) == np.asarray(qb.value)))
        except (TypeError, u.UnitsError):
            return False
    return bool(a == b)


@dataclass
class Reply:
    """Answer of a driver to a command.

    Attributes
    ----------
    duration
        Measured duration of the command (``None``: the wall-clock time of the call is used).
    state
        State read back from the equipment, by field name.
    value
        Value returned by the action, if any.
    """

    duration: u.Quantity | None = None
    state: dict[str, Any] = field(default_factory=dict)
    value: Any = None


class Driver(ABC):
    """Base class of drivers (see the module documentation for the safety mechanisms).

    Parameters
    ----------
    timeout
        Time allowed to the equipment to answer (used by the transports).
    retries
        Reconnections attempted after a lost connection, for one command.
    dry_run
        Check and log the commands without sending them.
    confirm
        Callback ``confirm(command) -> bool`` asked before the dangerous actions.
    """

    #: Positional argument names of each action, used to check the limits.
    signatures: ClassVar[Mapping[str, tuple[str, ...]]] = {}
    #: Software limits: ``"action.argument"`` -> ``(low, high)``.
    limits: ClassVar[Mapping[str, tuple[Any, Any]]] = {}
    #: Actions that need a confirmation.
    dangerous: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        *,
        timeout: u.Quantity = 5 * u.s,
        retries: int = 2,
        dry_run: bool = False,
        confirm: Callable[[Command], bool] | None = None,
    ) -> None:
        self._connected = False
        self._state = "disconnected"
        self._stop_event = threading.Event()
        self.timeout = timeout
        self.retries = int(retries)
        self.dry_run = bool(dry_run)
        self.confirm = confirm

    @property
    def connected(self) -> bool:
        """Whether the driver is connected to its equipment."""
        return self._connected

    @property
    def state(self) -> str:
        """``"disconnected"``, ``"ready"``, ``"busy"``, ``"stopped"`` or ``"error"``."""
        return self._state

    @property
    def timeout_s(self) -> float:
        """Timeout in seconds."""
        return float(np.asarray(u.Quantity(self.timeout).to_value(u.s)))

    def connect(self) -> None:
        """Open the connection to the equipment."""
        self._connect()
        self._connected = True
        self._state = "ready"
        self._stop_event.clear()
        logger.info("%s connected", type(self).__name__)

    def disconnect(self) -> None:
        """Close the connection."""
        if self._connected:
            self._disconnect()
        self._connected = False
        self._state = "disconnected"

    def _require_connection(self) -> None:
        if not self._connected:
            raise DriverError(f"{type(self).__name__} is not connected; call connect() first")

    # --- Checks --------------------------------------------------------------------------------------------

    def check(self, command: Command) -> None:
        """Check ``command`` against the limits and the confirmation rules (raises if refused)."""
        names = self.signatures.get(command.action, ())
        arguments = dict(zip(names, command.args, strict=False))
        arguments.update(command.kwargs)
        for key, (low, high) in self.limits.items():
            action_name, _, argument = key.partition(".")
            if action_name != command.action or argument not in arguments:
                continue
            value = arguments[argument]
            try:
                inside = bool(low <= value <= high)
            except (TypeError, u.UnitsError) as error:
                raise LimitError(
                    f"{command.element}.{key} = {value!r} cannot be compared with its limits"
                ) from error
            if not inside:
                raise LimitError(f"{command.element}.{key} = {value} is outside its limits [{low}, {high}]")
        if command.action in self.dangerous:
            if self.confirm is None:
                raise ActionRefused(f"{command} is dangerous and needs a confirmation (set driver.confirm)")
            if not self.confirm(command):
                raise ActionRefused(f"{command} was not confirmed")

    # --- Commands ------------------------------------------------------------------------------------------

    def send(self, command: Command) -> Reply:
        """Check ``command``, execute it on the equipment and return its reply."""
        if self._state == "stopped":
            raise DriverError(f"{type(self).__name__} is in emergency stop: call reset() before new commands")
        self._require_connection()
        self.check(command)
        if self.dry_run:
            logger.warning("%s dry-run: %s (not sent)", type(self).__name__, command)
            return Reply(value="dry-run")
        attempts = 0
        while True:
            self._state = "busy"
            try:
                reply = self._send(command)
            except ConnectionLost as error:
                if attempts >= self.retries:
                    self._state = "error"
                    raise DriverError(f"{command} failed after {self.retries} retries: {error}") from error
                attempts += 1
                logger.warning(
                    "%s: %s; reconnect and retry (%d/%d)", type(self).__name__, error, attempts, self.retries
                )
                self._reconnect()
                continue
            except DriverStopped:
                self._state = "stopped"
                raise
            except Exception:
                self._state = "stopped" if self._stop_event.is_set() else "error"
                raise
            self._state = "stopped" if self._stop_event.is_set() else "ready"
            return reply if isinstance(reply, Reply) else Reply(value=reply)

    def _reconnect(self) -> None:
        try:
            self._disconnect()
        except Exception:  # the connection is already broken
            logger.debug("%s: closing the broken connection failed", type(self).__name__, exc_info=True)
        self._connect()
        logger.info("%s reconnected", type(self).__name__)

    def query(self, name: str) -> Any:
        """Read the quantity ``name`` from the equipment."""
        self._require_connection()
        return self._query(name)

    def status(self) -> dict[str, Any]:
        """Return a summary of the driver and equipment state."""
        summary = {"driver": type(self).__name__, "connected": self._connected, "state": self._state}
        return {**summary, **self._status()}

    # --- Emergency stop and safe state ---------------------------------------------------------------------

    def wait(self, seconds: float) -> bool:
        """Wait ``seconds``; return ``False`` at once if an emergency stop occurs (for implementations)."""
        return not self._stop_event.wait(max(float(seconds), 0.0))

    @property
    def stop_requested(self) -> bool:
        """Whether an emergency stop is in progress."""
        return self._stop_event.is_set()

    def stop(self) -> None:
        """Stop any motion or emission immediately (emergency stop, callable from another thread)."""
        logger.warning("%s: stop requested", type(self).__name__)
        self._stop_event.set()
        if self._connected:
            self._state = "stopped"
        self._stop()

    def reset(self) -> None:
        """Acknowledge an emergency stop: commands are accepted again."""
        self._stop_event.clear()
        self._state = "ready" if self._connected else "disconnected"
        logger.info("%s reset after a stop", type(self).__name__)

    def safe_state(self) -> None:
        """Bring the equipment to its safe state (laser off, shutter closed, motion stopped)."""
        logger.warning("%s: safe state", type(self).__name__)
        self._safe_state()

    # --- Implementation ------------------------------------------------------------------------------------

    @abstractmethod
    def _connect(self) -> None: ...

    @abstractmethod
    def _disconnect(self) -> None: ...

    @abstractmethod
    def _send(self, command: Command) -> Reply: ...

    @abstractmethod
    def _query(self, name: str) -> Any: ...

    @abstractmethod
    def _stop(self) -> None: ...

    def _safe_state(self) -> None:  # noqa: B027 - optional hook, nothing to do by default
        """Bring the equipment to its safe state (nothing by default)."""

    def _status(self) -> dict[str, Any]:
        return {}


class SimulatedDriver(Driver):
    """Driver that records commands and answers with the virtual state of ``element``."""

    def __init__(self, element: Any = None) -> None:
        super().__init__()
        self.element = element
        self.commands: list[Command] = []

    def _connect(self) -> None:
        pass

    def _disconnect(self) -> None:
        pass

    def _send(self, command: Command) -> Reply:
        self.commands.append(command)
        logger.info("simulated command %s", command)
        return Reply()

    def _query(self, name: str) -> Any:
        if self.element is None:
            raise DriverError("this simulated driver is not attached to an element")
        return getattr(self.element, name)

    def _stop(self) -> None:
        pass

    def _status(self) -> dict[str, Any]:
        return {"commands": len(self.commands)}
