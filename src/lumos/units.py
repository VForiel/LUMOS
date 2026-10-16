"""Physical units at the boundary of LUMOS.

Users always give and receive :class:`astropy.units.Quantity` objects. Inside LUMOS, performance-critical
code works on plain floats or arrays expressed in **unscaled SI units** (metres, seconds, radians...). This
module provides the conversions between both worlds and the JSON encoding of quantities used by the
``.lumos`` file format.

JSON encoding
-------------
A quantity is encoded as ``{"value": <value>, "unit": "<astropy unit string>"}``, keeping the user's unit.
``<value>`` is:

- a JSON number for a finite real scalar, or one of the strings ``"nan"``, ``"inf"``, ``"-inf"``
  (strict JSON has no non-finite numbers);
- ``{"re": <number|string>, "im": <number|string>}`` for a complex scalar;
- ``{"dtype": "<f8", "shape": [n, ...], "data": "<base64>"}`` for an array: little-endian, C-order raw
  bytes encoded in base64, which preserves the exact values and the dtype.
"""

from __future__ import annotations

import base64
import binascii
import math
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from astropy import units as u
from astropy.units.function.core import FunctionUnitBase

__all__ = ["quantity_from_json", "quantity_to_json", "require_unit", "si_unit", "to_si"]

# Array dtypes accepted when decoding: booleans, integers, floats and complex numbers only. Object or
# structured dtypes are refused, so a file can never make NumPy build arbitrary Python objects.
_ALLOWED_DTYPE_KINDS = frozenset("biufc")
_NON_FINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


def _as_unit(unit: Any) -> u.UnitBase:
    result = u.Unit(unit)
    if isinstance(result, FunctionUnitBase):
        raise TypeError(f"function units such as '{result}' are not supported here")
    return result  # type: ignore[return-value]


def require_unit(
    value: Any,
    unit: Any,
    *,
    name: str = "value",
    equivalencies: Any = None,
) -> u.Quantity:
    """Check that ``value`` is a quantity convertible to ``unit`` and return it.

    The quantity is returned unchanged (same object, same unit), so the user's unit is preserved.
    Plain numbers are only accepted when ``unit`` is dimensionless; they are then wrapped in a
    dimensionless quantity.

    Parameters
    ----------
    value
        The value to check.
    unit
        Expected unit, or any unit with the same physical dimension.
    name
        Name of the checked value, used in error messages.
    equivalencies
        Optional astropy equivalencies, e.g. ``u.spectral()`` to accept a frequency for a wavelength.

    Returns
    -------
    astropy.units.Quantity
        The validated quantity.

    Raises
    ------
    TypeError
        If ``value`` is not a quantity (or not numeric) while a dimensional unit is expected.
    astropy.units.UnitsError
        If the unit of ``value`` cannot be converted to ``unit``.
    """
    expected = _as_unit(unit)
    equivalencies = equivalencies or []
    if isinstance(value, u.Quantity):
        value_unit = cast(u.UnitBase, value.unit)
        if not value_unit.is_equivalent(expected, equivalencies=equivalencies):
            raise u.UnitConversionError(
                f"{name}: expected a quantity convertible to '{expected}', got '{value_unit}'"
            )
        return value

    kind = np.asarray(value).dtype.kind
    if kind not in "iufc":
        raise TypeError(f"{name}: expected a numeric quantity in '{expected}', got {type(value).__name__}")
    if expected.physical_type != "dimensionless":
        raise TypeError(
            f"{name}: expected an astropy Quantity convertible to '{expected}' "
            f"(e.g. {value!r} * u.{expected.to_string()}), got a plain {type(value).__name__}"
        )
    return u.Quantity(value, u.dimensionless_unscaled)


def si_unit(unit: Any) -> u.UnitBase:
    """Return the unscaled SI unit with the same dimension as ``unit``.

    For example ``nm`` gives ``m``, ``deg`` gives ``rad``, ``Jy`` gives ``kg / s2`` and ``%`` gives the
    dimensionless unit. Irreducible non-SI units (such as ``ph``) are kept.
    """
    decomposed = _as_unit(unit).decompose()
    return u.CompositeUnit(1.0, decomposed.bases, decomposed.powers)


def to_si(quantity: u.Quantity, unit: Any = None, *, equivalencies: Any = None) -> Any:
    """Return the numerical value of ``quantity`` in unscaled SI units.

    Parameters
    ----------
    quantity
        The quantity to convert.
    unit
        Optional target dimension; defaults to the dimension of ``quantity``. Use it with
        ``equivalencies`` to convert across dimensions (e.g. frequency to wavelength).
    equivalencies
        Optional astropy equivalencies.

    Returns
    -------
    float, complex or numpy.ndarray
        A Python scalar for scalar quantities, an array otherwise.
    """
    target = si_unit(quantity.unit if unit is None else unit)
    value: Any = quantity.to_value(target, equivalencies=equivalencies or [])
    if np.ndim(value) == 0:
        return complex(value) if np.iscomplexobj(value) else float(value)
    return value


# --- JSON encoding ---------------------------------------------------------------


def _encode_float(value: float) -> float | str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def _decode_float(value: Any) -> float:
    if isinstance(value, str):
        try:
            return _NON_FINITE[value]
        except KeyError:
            raise ValueError(f"invalid number {value!r}") from None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"invalid number {value!r}")
    return float(value)


def quantity_to_json(quantity: u.Quantity) -> dict[str, Any]:
    """Encode a quantity as a strict-JSON-compatible dictionary (see the module docstring)."""
    quantity = u.Quantity(quantity, copy=False)
    unit = quantity.unit.to_string()  # type: ignore[union-attr]
    raw = quantity.value
    if np.ndim(raw) == 0:
        if np.iscomplexobj(raw):
            c = complex(raw)
            value: Any = {"re": _encode_float(c.real), "im": _encode_float(c.imag)}
        else:
            value = _encode_float(float(raw))
    else:
        array = np.asarray(raw)
        array = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
        value = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
        }
    return {"value": value, "unit": unit}


def _decode_array(value: Mapping[str, Any]) -> np.ndarray:
    try:
        dtype = np.dtype(value["dtype"])
        shape = tuple(int(n) for n in value["shape"])
        raw = base64.b64decode(value["data"], validate=True)
    except (KeyError, TypeError, binascii.Error) as error:
        raise ValueError(f"invalid array encoding: {error}") from error
    if dtype.kind not in _ALLOWED_DTYPE_KINDS or dtype.fields is not None:
        raise ValueError(f"unsupported array dtype {dtype.str!r}")
    if any(n < 0 for n in shape) or len(raw) != dtype.itemsize * math.prod(shape):
        raise ValueError(f"array data size does not match dtype {dtype.str!r} and shape {list(shape)}")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def quantity_from_json(data: Mapping[str, Any]) -> u.Quantity:
    """Decode a dictionary produced by :func:`quantity_to_json`.

    Raises
    ------
    ValueError
        If the dictionary is malformed or the unit string cannot be parsed.
    """
    if not isinstance(data, Mapping) or "value" not in data or "unit" not in data:
        raise ValueError("a quantity must be a mapping with 'value' and 'unit' keys")
    try:
        unit = _as_unit(data["unit"])
    except (ValueError, TypeError) as error:
        raise ValueError(f"invalid unit {data['unit']!r}: {error}") from error

    value = data["value"]
    if isinstance(value, Mapping) and "dtype" in value:
        array = _decode_array(value)
        # Pass the dtype explicitly: astropy would otherwise turn integer arrays into floats.
        return u.Quantity(array, unit, dtype=cast(Any, array.dtype), copy=False)
    if isinstance(value, Mapping) and set(value) == {"re", "im"}:
        return u.Quantity(complex(_decode_float(value["re"]), _decode_float(value["im"])), unit)
    return u.Quantity(_decode_float(value), unit)
