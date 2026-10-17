"""Source of a symbolic field in a context (step P10.6, REQ-SYM-1, REQ-SYM-4).

:class:`SymbolicSource` emits a :class:`~lumos.symbolic.SymbolicField`. The elements downstream keep it
symbolic while they have closed forms, and the field is converted to numbers where needed
(:mod:`lumos.symbolic.integration`). The field is saved in ``.lumos`` files as a tree of allowed operations
(:mod:`lumos.symbolic.serialize`), never as code.
"""

from __future__ import annotations

from typing import Any

from astropy import units as u

from lumos.core.element import Element, Values
from lumos.core.param import Setting
from lumos.core.port import Output
from lumos.core.registry import register
from lumos.symbolic.field import SymbolicField

__all__ = ["SymbolicFieldSetting", "SymbolicSource"]


class SymbolicFieldSetting(Setting):
    """A setting holding a :class:`~lumos.symbolic.SymbolicField` (saved as a JSON expression tree)."""

    def __init__(self, *, doc: str = "") -> None:
        super().__init__(object, doc=doc)

    def validate(self, value: Any) -> tuple[Any, Any]:
        """Accept a symbolic field."""
        if not isinstance(value, SymbolicField):
            raise TypeError(
                f"{self.name}: expected a lumos.symbolic.SymbolicField, got {type(value).__name__}"
            )
        return value, value

    def encode(self, value: Any) -> Any:
        """Return the JSON form of the field."""
        from lumos.symbolic.serialize import field_to_json

        return field_to_json(value)

    def decode(self, data: Any) -> Any:
        """Return the field of a JSON form (only allowed operations are rebuilt)."""
        from lumos.symbolic.serialize import field_from_json

        return field_from_json(data)


@register("lumos.symbolic.SymbolicSource")
class SymbolicSource(Element):
    """Source emitting a symbolic field.

    Parameters
    ----------
    field
        The :class:`~lumos.symbolic.SymbolicField` emitted. Give it a grid hint (``grid=``) if an element
        downstream needs numbers.
    """

    field = SymbolicFieldSetting(doc="symbolic field emitted")
    out = Output(SymbolicField)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the field."""
        return {"out": values.field}

    @staticmethod
    def transfer_symbolic(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the field (a source has no input)."""
        return {"out": values.field}

    @classmethod
    def gaussian_beam(
        cls,
        *,
        waist: u.Quantity,
        wavelength: u.Quantity,
        power: u.Quantity,
        distance_from_waist: u.Quantity = 0 * u.m,
        family: str = "hg",
        indices: tuple[int, int] = (0, 0),
        grid: Any = None,
        name: str | None = None,
    ) -> SymbolicSource:
        """Return a source of an HG or LG beam (:func:`lumos.symbolic.gaussian_beam`)."""
        from lumos.symbolic.closed_forms import gaussian_beam

        field = gaussian_beam(
            waist=waist,
            wavelength=wavelength,
            power=power,
            distance_from_waist=distance_from_waist,
            family=family,
            indices=indices,
            grid=grid,
        )
        return cls(name=name, field=field)

    @classmethod
    def plane_wave(
        cls, *, amplitude: u.Quantity, wavelength: u.Quantity, grid: Any = None, name: str | None = None
    ) -> SymbolicSource:
        """Return a source of a uniform plane wave (:func:`lumos.symbolic.plane_wave`)."""
        from lumos.symbolic.closed_forms import plane_wave

        return cls(name=name, field=plane_wave(amplitude=amplitude, wavelength=wavelength, grid=grid))
