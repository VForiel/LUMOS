r"""The LumosLabs ASCII protocol (step P12.3, REQ-HW-3, ADR-0014).

Specification: ``docs/hardware/lumoslabs_protocol.md``. In short:

- one command per line, ``VERB [arg ...]``, ``\n``-terminated; verbs are case-insensitive, queries end
  with ``?``;
- arguments and values are tokens separated by spaces: integers (``3``), reals (``0.25``, ``1e-12``), words
  (``ON``) and **quantities** ``value[unit]`` (``12.5[mm]``, ``4[mm/s]``), units in astropy notation;
- replies are ``OK [value ...]`` or ``ERR <code> <message>``; the codes are listed in :data:`ERRORS`;
- arrays (images) travel as three tokens: dtype, shape (``3x4``) and ``b64:`` followed by the base64 of the
  little-endian bytes (:func:`encode_array`).
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from astropy import units as u

from lumos.core.driver import DriverError, DriverStopped, LimitError

__all__ = [
    "ERRORS",
    "DeviceBusy",
    "DeviceError",
    "DeviceLimitError",
    "DeviceStopped",
    "Identity",
    "ProtocolError",
    "decode_array",
    "encode_array",
    "format_command",
    "format_error",
    "format_identity",
    "format_reply",
    "format_token",
    "parse_command",
    "parse_identity",
    "parse_reply",
    "parse_token",
]

VENDOR = "LUMOSLABS"


class ProtocolError(DriverError):
    """A line does not follow the protocol."""


class DeviceError(DriverError):
    """An ``ERR`` reply of a device, with its numeric ``code``."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"ERR {code}: {message}")
        self.code = int(code)
        self.detail = message


class DeviceLimitError(DeviceError, LimitError):
    """``ERR 200``: a value outside the range of the device."""


class DeviceStopped(DeviceError, DriverStopped):
    """``ERR 400``: the emergency stop of the device is latched."""


class DeviceBusy(DeviceError):
    """``ERR 300``: the device is executing another command."""


#: Error codes: name and exception raised by :func:`parse_reply`.
ERRORS: dict[int, tuple[str, type[DeviceError]]] = {
    100: ("unknown command", DeviceError),
    101: ("bad argument", DeviceError),
    102: ("bad unit", DeviceError),
    200: ("out of range", DeviceLimitError),
    300: ("busy", DeviceBusy),
    301: ("not homed", DeviceError),
    400: ("emergency stop latched", DeviceStopped),
    500: ("hardware fault", DeviceError),
    501: ("motion stalled", DeviceError),
}

_QUANTITY = re.compile(r"^([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\[([^\]\s]+)\]$")
_INTEGER = re.compile(r"^[-+]?\d+$")
_REAL = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$|^[-+]?(?:inf|nan)$", re.IGNORECASE)


def _number(value: float) -> str:
    return f"{float(value):.15g}"


def format_token(value: Any) -> str:
    """Return the token of a value: quantity ``12.5[mm]``, number, boolean (``1``/``0``) or word."""
    if isinstance(value, u.Quantity):
        if not value.isscalar:
            raise ProtocolError("only scalar quantities are tokens; send arrays with encode_array")
        unit = cast(u.UnitBase, value.unit).to_string().replace(" ", "")
        return f"{_number(float(np.asarray(value.value)))}[{unit}]"
    if isinstance(value, bool | np.bool_):
        return "1" if value else "0"
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        return _number(float(value))
    text = str(value)
    if not text or any(c.isspace() for c in text):
        raise ProtocolError(f"a word token cannot be empty nor contain spaces: {text!r}")
    return text


def parse_token(token: str) -> Any:
    """Return the value of a token (quantity, int, float or word)."""
    match = _QUANTITY.match(token)
    if match:
        try:
            unit = u.Unit(match.group(2))
        except ValueError as error:
            raise ProtocolError(f"unknown unit in {token!r}") from error
        return float(match.group(1)) * unit
    if _INTEGER.match(token):
        return int(token)
    if _REAL.match(token):
        return float(token)
    return token


def format_command(verb: str, *arguments: Any) -> str:
    """Return the line of a command."""
    return " ".join([verb.upper(), *(format_token(a) for a in arguments)])


def parse_command(line: str) -> tuple[str, list[Any]]:
    """Return the verb (upper case) and the arguments of a command line."""
    tokens = line.strip().split()
    if not tokens:
        raise ProtocolError("empty command")
    return tokens[0].upper(), [parse_token(t) for t in tokens[1:]]


def format_reply(*values: Any) -> str:
    """Return an ``OK`` reply with its values."""
    return " ".join(["OK", *(format_token(v) for v in values)])


def format_error(code: int, message: str) -> str:
    """Return an ``ERR`` reply."""
    return f"ERR {int(code)} {message}"


def parse_reply(line: str) -> list[Any]:
    """Return the values of an ``OK`` reply; raise the exception of an ``ERR`` reply."""
    tokens = line.strip().split()
    if not tokens:
        raise ProtocolError("empty reply")
    if tokens[0] == "OK":
        return [parse_token(t) for t in tokens[1:]]
    if tokens[0] == "ERR" and len(tokens) >= 2 and _INTEGER.match(tokens[1]):
        code = int(tokens[1])
        message = " ".join(tokens[2:]) or ERRORS.get(code, ("error",))[0]
        exception = ERRORS.get(code, ("error", DeviceError))[1]
        raise exception(code, message)
    raise ProtocolError(f"invalid reply {line!r}")


@dataclass(frozen=True)
class Identity:
    """Answer to ``*IDN?``: vendor, model, serial number, firmware version."""

    vendor: str
    model: str
    serial: str
    firmware: str


def parse_identity(text: str) -> Identity:
    """Parse ``VENDOR,MODEL,SERIAL,FIRMWARE``."""
    parts = text.strip().split(",")
    if len(parts) != 4:
        raise ProtocolError(f"invalid identity {text!r}: expected VENDOR,MODEL,SERIAL,FIRMWARE")
    return Identity(*(p.strip() for p in parts))


def format_identity(identity: Identity) -> str:
    """Return ``VENDOR,MODEL,SERIAL,FIRMWARE``."""
    return ",".join((identity.vendor, identity.model, identity.serial, identity.firmware))


def encode_array(array: Any) -> tuple[str, str, str]:
    """Return the three tokens of an array: dtype, shape (``3x4``) and ``b64:`` + base64 bytes."""
    data = np.ascontiguousarray(array)
    little = data.astype(data.dtype.newbyteorder("<"), copy=False)
    shape = "x".join(str(n) for n in data.shape) or "1"
    return little.dtype.name, shape, "b64:" + base64.b64encode(little.tobytes()).decode("ascii")


def decode_array(dtype: str, shape: str, data: str) -> np.ndarray:
    """Return the array of the tokens written by :func:`encode_array`."""
    if not str(data).startswith("b64:"):
        raise ProtocolError("array data must start with 'b64:'")
    kind = np.dtype(str(dtype)).newbyteorder("<")
    if kind.kind not in "biuf":
        raise ProtocolError(f"unsupported array dtype {dtype!r}")
    dimensions = tuple(int(n) for n in str(shape).split("x"))
    raw = base64.b64decode(str(data)[4:], validate=True)
    array = np.frombuffer(raw, dtype=kind)
    if array.size != int(np.prod(dimensions)):
        raise ProtocolError(f"array of {array.size} values does not match the shape {shape}")
    return array.reshape(dimensions).astype(kind.newbyteorder("="))
