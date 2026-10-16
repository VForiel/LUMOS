"""Drivers of real or emulated equipment (REQ-HW-1).

A :class:`Driver` connects an element to the equipment it represents. In operation mode, the actions of the
element become :class:`Command` objects sent to the driver, which answers with a :class:`Reply`: the measured
duration, the state read back from the equipment and an optional value. Concrete drivers implement the
protected methods ``_connect``, ``_disconnect``, ``_send``, ``_query`` and ``_stop`` on top of a transport
(serial, TCP, VISA, vendor SDK; phase 12).

:class:`SimulatedDriver` sends nothing: it records the commands and answers with the virtual state of its
element, which makes it possible to test a sequence of commands without any equipment.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from astropy import units as u

from lumos.units import quantity_from_json, quantity_to_json

__all__ = ["Command", "Driver", "DriverError", "Reply", "SimulatedDriver"]

logger = logging.getLogger("lumos.hardware")

_QUANTITY = "__quantity__"


class DriverError(RuntimeError):
    """Raised when a driver cannot execute a command (missing, disconnected, failing equipment)."""


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
    """Base class of drivers."""

    def __init__(self) -> None:
        self._connected = False

    @property
    def connected(self) -> bool:
        """Whether the driver is connected to its equipment."""
        return self._connected

    def connect(self) -> None:
        """Open the connection to the equipment."""
        self._connect()
        self._connected = True
        logger.info("%s connected", type(self).__name__)

    def disconnect(self) -> None:
        """Close the connection."""
        if self._connected:
            self._disconnect()
        self._connected = False

    def _require_connection(self) -> None:
        if not self._connected:
            raise DriverError(f"{type(self).__name__} is not connected; call connect() first")

    def send(self, command: Command) -> Reply:
        """Execute ``command`` on the equipment and return its reply."""
        self._require_connection()
        reply = self._send(command)
        return reply if isinstance(reply, Reply) else Reply(value=reply)

    def query(self, name: str) -> Any:
        """Read the quantity ``name`` from the equipment."""
        self._require_connection()
        return self._query(name)

    def status(self) -> dict[str, Any]:
        """Return a summary of the driver and equipment state."""
        return {"driver": type(self).__name__, "connected": self._connected, **self._status()}

    def stop(self) -> None:
        """Stop any motion or emission immediately (emergency stop)."""
        logger.warning("%s: stop requested", type(self).__name__)
        self._stop()

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
