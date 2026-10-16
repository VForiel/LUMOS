"""Prior distributions attached to free parameters (REQ-EL-3).

Priors describe what is known about a free parameter before calibration or inference. They carry physical
units, so they can be checked against the parameter they are attached to. Plain numbers are accepted for
dimensionless parameters. Sampling and log-densities are added with the inference blocks (phase 17).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from astropy import units as u

from lumos.units import require_unit

__all__ = ["LogUniform", "Normal", "Prior", "Uniform"]


def _as_quantity(value: Any, name: str) -> u.Quantity:
    if isinstance(value, u.Quantity):
        return value
    return require_unit(value, u.dimensionless_unscaled, name=name)


class Prior:
    """Base class of prior distributions.

    Subclasses are frozen dataclasses whose first field fixes the unit of the distribution.
    """

    def _normalise(self, first: str, second: str) -> tuple[u.Quantity, u.Quantity]:
        a = _as_quantity(getattr(self, first), first)
        b = require_unit(_as_quantity(getattr(self, second), second), a.unit, name=second).to(a.unit)
        object.__setattr__(self, first, a)
        object.__setattr__(self, second, b)
        return a, b

    @property
    def unit(self) -> u.UnitBase:
        """Unit of the values described by the prior."""
        first = next(iter(self.__dataclass_fields__))  # type: ignore[attr-defined]
        return getattr(self, first).unit


@dataclass(frozen=True)
class Uniform(Prior):
    """Uniform distribution between ``low`` and ``high`` (``low < high``)."""

    low: u.Quantity
    high: u.Quantity

    def __post_init__(self) -> None:
        """Check the bounds and express both in the unit of ``low``."""
        low, high = self._normalise("low", "high")
        if not low.value < high.value:
            raise ValueError(f"Uniform prior needs low < high, got low={low} and high={high}")


@dataclass(frozen=True)
class LogUniform(Prior):
    """Log-uniform distribution between ``low`` and ``high`` (``0 < low < high``).

    Suited to scale parameters known only to an order of magnitude (fluxes, contrasts...).
    """

    low: u.Quantity
    high: u.Quantity

    def __post_init__(self) -> None:
        """Check the bounds and express both in the unit of ``low``."""
        low, high = self._normalise("low", "high")
        if not 0 < low.value < high.value:
            raise ValueError(f"LogUniform prior needs 0 < low < high, got low={low} and high={high}")


@dataclass(frozen=True)
class Normal(Prior):
    """Normal (Gaussian) distribution of mean ``mean`` and standard deviation ``std > 0``."""

    mean: u.Quantity
    std: u.Quantity

    def __post_init__(self) -> None:
        """Check the standard deviation and express it in the unit of ``mean``."""
        _, std = self._normalise("mean", "std")
        if not std.value > 0:
            raise ValueError(f"Normal prior needs std > 0, got {std}")
