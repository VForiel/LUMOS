"""Prior distributions attached to free parameters (REQ-EL-3).

Priors describe what is known about a free parameter before calibration or inference. They carry physical
units, so they can be checked against the parameter they are attached to. Plain numbers are accepted for
dimensionless parameters. Sampling and log-densities are added with the inference blocks (phase 17).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from astropy import units as u

from lumos.units import quantity_from_json, quantity_to_json, require_unit

__all__ = ["LogUniform", "Normal", "Prior", "Uniform", "prior_from_json", "prior_to_json"]


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


_PRIOR_KINDS: dict[str, type[Prior]] = {"uniform": Uniform, "loguniform": LogUniform, "normal": Normal}


def prior_to_json(prior: Prior) -> dict[str, Any]:
    """Encode a prior as ``{"kind": ..., <field>: <quantity JSON>, ...}``."""
    kind = next(name for name, cls in _PRIOR_KINDS.items() if type(prior) is cls)
    fields = prior.__dataclass_fields__  # type: ignore[attr-defined]
    return {"kind": kind, **{name: quantity_to_json(getattr(prior, name)) for name in fields}}


def prior_from_json(data: Mapping[str, Any]) -> Prior:
    """Decode a prior encoded by :func:`prior_to_json`.

    Raises
    ------
    ValueError
        If the encoding is malformed or the prior is invalid.
    """
    if not isinstance(data, Mapping) or data.get("kind") not in _PRIOR_KINDS:
        raise ValueError(f"invalid prior {data!r}: 'kind' must be one of {sorted(_PRIOR_KINDS)}")
    cls = _PRIOR_KINDS[data["kind"]]
    names = list(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    if set(data) != {"kind", *names}:
        raise ValueError(f"invalid {data['kind']} prior: expected the fields {names}")
    return cls(*(quantity_from_json(data[name]) for name in names))
