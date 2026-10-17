"""Settings of processing elements."""

from __future__ import annotations

from typing import Any

from astropy import units as u

from lumos.core.param import Setting
from lumos.units import quantity_from_json, quantity_to_json

__all__ = ["QuantitySetting"]


class QuantitySetting(Setting):
    """A scalar quantity whose unit depends on the data it applies to (pixels, angles, lengths...)."""

    def __init__(self, *, default: u.Quantity, doc: str = "") -> None:
        super().__init__(object, default=default, doc=doc)

    def validate(self, value: Any) -> tuple[Any, Any]:
        """Accept a scalar quantity."""
        if not isinstance(value, u.Quantity) or not value.isscalar:
            raise TypeError(f"{self.name}: expected a scalar Quantity, got {value!r}")
        return value, value

    def encode(self, value: Any) -> Any:
        """Return the JSON form of the quantity."""
        return quantity_to_json(value)

    def decode(self, data: Any) -> Any:
        """Return the quantity of a JSON form."""
        return quantity_from_json(data)
