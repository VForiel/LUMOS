"""Values given to and returned by the service: quantities written as text, parsed and checked."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from astropy import units as u

from lumos.core.param import Param, Setting
from lumos.service.errors import ServiceError

__all__ = ["format_value", "parse_quantity", "parse_value"]


def parse_quantity(value: Any, unit: Any = None, *, name: str = "value") -> u.Quantity:
    """Parse ``"5 mm"``, ``{"value": 5, "unit": "mm"}``, a quantity, or a number in ``unit``."""
    try:
        if isinstance(value, u.Quantity):
            quantity = value
        elif isinstance(value, dict):
            quantity = u.Quantity(value["value"], value.get("unit", ""))
        elif isinstance(value, str):
            quantity = u.Quantity(value)
        elif isinstance(value, list | tuple | int | float | np.ndarray):
            quantity = u.Quantity(value, unit if unit is not None else "")
        else:
            raise TypeError(f"unsupported value {value!r}")
    except (TypeError, ValueError, KeyError) as error:
        expected = f"e.g. '1 {unit}'" if unit is not None else "e.g. '5 mm'"
        raise ServiceError(
            f"{name}: cannot read {value!r} as a quantity", hint=f"write it as text, {expected}"
        ) from error
    if unit is not None and not cast(u.UnitBase, quantity.unit).is_equivalent(
        unit, equivalencies=u.spectral()
    ):
        dimension = u.Unit(unit).physical_type
        raise ServiceError(
            f"{name}: {quantity} is not a {dimension}",
            hint=f"give a {dimension} such as '1 {u.Unit(unit).to_string()}'",
        )
    return quantity


def parse_value(field: Any, value: Any, *, name: str) -> Any:
    """Return the Python value of ``value`` for the field ``field`` of an element."""
    if isinstance(field, Param):
        quantity = parse_quantity(value, field.unit, name=name)
        return quantity.to(field.unit, equivalencies=u.spectral())
    if isinstance(field, Setting):
        if isinstance(value, dict) and not ({"value", "unit"} >= set(value)):
            return field.decode(value)  # structured settings (grids, symbolic fields...)
        if field.type is object:
            if isinstance(value, str | dict | u.Quantity):
                return (
                    parse_quantity(value, name=name)
                    if not isinstance(value, dict) or "unit" in value
                    else value
                )
            return value
        try:
            if field.type is bool and isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return field.type(value)
        except (TypeError, ValueError) as error:
            raise ServiceError(f"{name}: expected a {field.type.__name__}, got {value!r}", hint="") from error
    return value


def _number(value: float) -> str:
    return f"{value:.6g}"


def format_value(value: Any, unit: Any = None) -> Any:
    """Return a JSON-friendly form: ``"0.5 mm"`` for scalar quantities, lists for arrays, plain values."""
    if isinstance(value, u.Quantity):
        if unit is not None:
            value = value.to(unit, equivalencies=u.spectral())
        unit_text = cast(u.UnitBase, value.unit).to_string()
        if value.isscalar:
            return f"{_number(float(np.asarray(value.value)))} {unit_text}".strip()
        flat = np.asarray(value.value, dtype=float).ravel()
        shown = [float(f"{v:.6g}") for v in flat[:16]]
        suffix = "..." if flat.size > 16 else ""
        return f"{shown}{suffix} {unit_text}".strip()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    if isinstance(value, list | tuple):
        return [format_value(v) for v in value]
    return repr(value)
