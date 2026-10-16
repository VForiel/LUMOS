r"""Coherent monochromatic fields sampled on a grid (REQ-SIG-2).

A :class:`Wavefront` is the complex amplitude :math:`U(x, y)` of a monochromatic field at one vacuum
wavelength :math:`\lambda`, sampled on a :class:`~lumos.fields.Grid`. With the convention of §9 of the
specifications, the physical field is :math:`E(\mathbf r, t) = \mathrm{Re}\{U(\mathbf r)\,e^{-i\omega t}\}`
and a wave travelling towards :math:`+z` varies as :math:`e^{+ikz}` with :math:`k = 2\pi/\lambda`.

Normalisation
-------------
The field is stored without units in an array of the current backend. Its normalisation is a metadata,
:attr:`Wavefront.power_unit` (``W`` or ``photon / s``), chosen such that

.. math:: P = \sum_{i,j} |U_{ij}|^2\,\Delta x\,\Delta y

is the power carried by the field. :math:`|U|^2` is then an irradiance (W m⁻²) on a spatial grid, or a
radiant intensity (W sr⁻¹) on an angular grid.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from astropy import units as u

from lumos.backend._core import backend_of, convert, current, get
from lumos.backend.memory import dtypes
from lumos.core.signal import OpticalSignal
from lumos.fields.grid import Grid
from lumos.units import quantity_from_json, quantity_to_json, require_unit, to_si

__all__ = ["Wavefront"]

_PHOTON_RATE = u.photon / u.s


def _power_unit(unit: Any) -> u.UnitBase:
    unit = cast(u.UnitBase, u.Unit(unit))
    if not (unit.is_equivalent(u.W) or unit.is_equivalent(_PHOTON_RATE)):
        raise u.UnitsError(f"power_unit must be a power (W) or a photon rate (photon / s), got '{unit}'")
    return unit


def _wavelength(value: Any) -> u.Quantity:
    quantity = require_unit(value, u.m, name="wavelength", equivalencies=u.spectral())
    if not quantity.unit.is_equivalent(u.m):  # type: ignore[union-attr]
        quantity = quantity.to(u.m, equivalencies=u.spectral())
    si = float(to_si(quantity, u.m))
    if not (math.isfinite(si) and si > 0):
        raise ValueError(f"wavelength must be positive and finite, got {quantity}")
    return quantity


# Exact values of the 2019 SI.
_PLANCK = 6.62607015e-34  # J s
_SPEED_OF_LIGHT = 299_792_458.0  # m / s


def _photon_energy(wavelength_si: float) -> float:
    """Energy ``hc/λ`` of one photon, in joules."""
    return _PLANCK * _SPEED_OF_LIGHT / wavelength_si


def _array_backend(field: Any) -> str:
    """Keep arrays of a non-NumPy backend where they are; put user data in the current backend."""
    if isinstance(field, np.ndarray):
        return current().name
    try:
        return backend_of(field)
    except TypeError:
        return current().name


class Wavefront(OpticalSignal):
    """Complex scalar field at one wavelength, sampled on a grid.

    Parameters
    ----------
    field
        Complex amplitude, shape ``grid.shape``: any array-like. A NumPy array or a list is copied into the
        current backend, an array of another backend stays on it; the dtype follows
        ``lumos.config.precision``.
    grid
        Sampling of the field.
    wavelength
        Vacuum wavelength (a frequency or a photon energy is converted with ``u.spectral()``).
    power_unit
        Normalisation of the field: ``u.W`` (default) or ``u.photon / u.s``.

    Notes
    -----
    Wavefronts are immutable values: operations return new wavefronts. They are JAX pytrees (the field is the
    only leaf), so pure functions of wavefronts can be compiled with :func:`lumos.jit` and differentiated.
    """

    __slots__ = ("_field", "_grid", "_power_unit", "_wavelength")
    __array_ufunc__ = None  # let ``ndarray * wavefront`` call Wavefront.__rmul__

    _field: Any
    _grid: Grid
    _wavelength: u.Quantity
    _power_unit: u.UnitBase

    def __init__(
        self,
        field: Any,
        grid: Grid,
        wavelength: u.Quantity,
        *,
        power_unit: Any = u.W,
    ) -> None:
        if not isinstance(grid, Grid):
            raise TypeError(f"grid must be a lumos.fields.Grid, got {type(grid).__name__}")
        target = get(_array_backend(field))
        _, complex_dtype = dtypes(backend=target.name)
        array = target.asarray(field if target.is_array(field) else np.asarray(field), dtype=complex_dtype)
        self._init(array, grid, _wavelength(wavelength), _power_unit(power_unit))

    def _init(self, field: Any, grid: Grid, wavelength: u.Quantity, power_unit: u.UnitBase) -> None:
        if tuple(field.shape) != grid.shape:
            raise ValueError(
                f"the field shape {tuple(field.shape)} does not match the grid shape {grid.shape}"
            )
        self._field = field
        self._grid = grid
        self._wavelength = wavelength
        self._power_unit = power_unit

    def _with_field(self, field: Any, power_unit: u.UnitBase | None = None) -> Wavefront:
        new = object.__new__(type(self))
        new._init(field, self._grid, self._wavelength, self._power_unit if power_unit is None else power_unit)
        return new

    # --- Factories --------------------------------------------------------------------------------

    @classmethod
    def plane(cls, grid: Grid, wavelength: u.Quantity, *, power: u.Quantity) -> Wavefront:
        """Uniform, flat-phase field carrying ``power`` (in W or photon / s) over the grid."""
        power_unit, value = cls._power_value(power)
        amplitude = math.sqrt(value / (grid.size * grid.dx_si * grid.dy_si))
        return cls(np.full(grid.shape, amplitude), grid, wavelength, power_unit=power_unit)

    @classmethod
    def gaussian(
        cls,
        grid: Grid,
        wavelength: u.Quantity,
        *,
        waist: u.Quantity,
        power: u.Quantity,
    ) -> Wavefront:
        r"""Gaussian beam at its waist, :math:`U = \sqrt{2P/(\pi w^2)}\,\exp(-r^2/w^2)`, centred on the grid.

        The intensity integrates to ``power`` over the infinite plane; the sampled power differs from it only
        when the beam is truncated by the grid or undersampled.
        """
        if grid.plane != "spatial":
            raise ValueError("a Gaussian beam waist is defined on a spatial grid")
        power_unit, value = cls._power_value(power)
        w = float(to_si(require_unit(waist, u.m, name="waist")))
        if not (math.isfinite(w) and w > 0):
            raise ValueError("waist must be positive and finite")
        x, y = grid.axes(backend="numpy")
        r2 = (x[None, :] - grid.x0_si) ** 2 + (y[:, None] - grid.y0_si) ** 2
        field = math.sqrt(2 * value / (math.pi * w**2)) * np.exp(-r2 / w**2)
        return cls(field, grid, wavelength, power_unit=power_unit)

    @staticmethod
    def _power_value(power: u.Quantity) -> tuple[u.UnitBase, float]:
        if not isinstance(power, u.Quantity):
            raise TypeError(f"power must be a Quantity in W or photon / s, got {power!r}")
        unit = _power_unit(u.W if power.unit.is_equivalent(u.W) else _PHOTON_RATE)  # type: ignore[union-attr]
        value = float(cast(Any, power.to_value(unit)))
        if not (math.isfinite(value) and value >= 0):
            raise ValueError(f"power must be non-negative and finite, got {power}")
        return unit, value

    # --- Attributes -------------------------------------------------------------------------------

    @property
    def field(self) -> Any:
        """Complex amplitude, a backend array of shape ``grid.shape`` (in sqrt(power_unit) / grid.unit)."""
        return self._field

    @property
    def grid(self) -> Grid:
        """Sampling grid."""
        return self._grid

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of :attr:`field`."""
        return tuple(self._field.shape)

    @property
    def backend(self) -> str:
        """Name of the backend holding the field."""
        return backend_of(self._field)

    @property
    def wavelength(self) -> u.Quantity:
        """Vacuum wavelength."""
        return self._wavelength

    @property
    def wavelength_si(self) -> float:
        """Vacuum wavelength in metres."""
        return float(to_si(self._wavelength))

    @property
    def wavenumber_si(self) -> float:
        """Vacuum wavenumber ``k = 2π/λ`` in rad / m."""
        return 2 * math.pi / self.wavelength_si

    @property
    def power_unit(self) -> u.UnitBase:
        """Normalisation of the field: ``W`` or ``photon / s``."""
        return self._power_unit

    @property
    def intensity_unit(self) -> u.UnitBase:
        """Unit of :attr:`intensity`: power per m² (spatial grid) or per sr (angular grid)."""
        per = u.m**2 if self._grid.plane == "spatial" else u.sr
        return cast(u.UnitBase, self._power_unit / per)

    # --- Derived quantities -----------------------------------------------------------------------

    @property
    def _xp(self) -> Any:
        return get(self.backend).xp

    @property
    def intensity(self) -> Any:
        """:math:`|U|^2`, a real backend array in :attr:`intensity_unit`."""
        xp = self._xp
        return xp.real(self._field * xp.conj(self._field))

    @property
    def amplitude(self) -> Any:
        """:math:`|U|`, a real backend array."""
        return self._xp.abs(self._field)

    @property
    def phase(self) -> Any:
        r""":math:`\arg U` in radians, in ``(-π, π]``."""
        xp = self._xp
        return xp.atan2(xp.imag(self._field), xp.real(self._field))

    @property
    def power(self) -> u.Quantity:
        r"""Power :math:`\sum |U|^2 \Delta x \Delta y` carried by the field, in :attr:`power_unit`."""
        total = float(self._xp.sum(self.intensity)) * self._grid.dx_si * self._grid.dy_si
        return u.Quantity(total, self._power_unit)

    # --- Operations -------------------------------------------------------------------------------

    def normalized(self, power: u.Quantity) -> Wavefront:
        """Return the field scaled to carry ``power`` (its unit becomes the power unit)."""
        unit, value = self._power_value(power)
        scaled = self.with_power_unit(unit)
        current_power = float(cast(Any, scaled.power.value))
        if current_power == 0:
            raise ValueError("cannot normalise a field that is identically zero")
        return scaled._with_field(scaled._field * math.sqrt(value / current_power))

    def with_power_unit(self, unit: Any) -> Wavefront:
        """Return the same light expressed in another power unit, using the photon energy ``hc/λ``."""
        unit = _power_unit(unit)
        if unit == self._power_unit:
            return self
        # factor such that P[new] = factor × P[old]
        old = self._power_unit
        energy = _photon_energy(self.wavelength_si)
        factor: Any
        if old.is_equivalent(unit):
            factor = old.to(unit)
        elif old.is_equivalent(u.W):  # W -> photon / s
            factor = cast(Any, old.to(u.W)) / energy * cast(Any, _PHOTON_RATE.to(unit))
        else:  # photon / s -> W
            factor = cast(Any, old.to(_PHOTON_RATE)) * energy * cast(Any, u.W.to(unit))
        return self._with_field(self._field * math.sqrt(float(factor)), unit)

    def tilt(self, theta_x: u.Quantity, theta_y: u.Quantity) -> Wavefront:
        r"""Multiply by the phase of a plane wave travelling at angles :math:`(\theta_x, \theta_y)` to z.

        The factor is :math:`\exp[ik(x\sin\theta_x + y\sin\theta_y)]`, null at the coordinate origin; the
        angles are measured in the x-z and y-z planes.
        """
        if self._grid.plane != "spatial":
            raise ValueError("tilt() needs a field on a spatial grid")
        tx = float(to_si(require_unit(theta_x, u.rad, name="theta_x")))
        ty = float(to_si(require_unit(theta_y, u.rad, name="theta_y")))
        xx, yy = self._grid.coords(backend=self.backend)
        k = self.wavenumber_si
        ramp = self._xp.exp(1j * k * (xx * math.sin(tx) + yy * math.sin(ty)))
        return self._with_field(self._field * ramp)

    def to_backend(self, name: str) -> Wavefront:
        """Return the same field held by backend ``name``."""
        return self._with_field(convert(self._field, name))

    def __mul__(self, other: Any) -> Wavefront:
        """Multiply by a scalar or a (complex) transmission array of shape ``grid.shape``."""
        if isinstance(other, Wavefront):
            return NotImplemented
        return self._with_field(self._field * other)

    __rmul__ = __mul__

    def __add__(self, other: Any) -> Wavefront:
        """Coherent superposition of two fields on the same grid, wavelength and power unit."""
        if not isinstance(other, Wavefront):
            return NotImplemented
        if other._grid != self._grid:
            raise ValueError("cannot add wavefronts sampled on different grids")
        if other.wavelength_si != self.wavelength_si:
            raise ValueError("cannot coherently add wavefronts of different wavelengths")
        if other._power_unit != self._power_unit:
            raise ValueError("cannot add wavefronts with different power units; use with_power_unit()")
        return self._with_field(self._field + other._field)

    def __neg__(self) -> Wavefront:
        """Field with opposite sign (a π phase shift)."""
        return self._with_field(-self._field)

    def __repr__(self) -> str:
        """Summarise the sampling, wavelength and backend."""
        return (
            f"Wavefront(shape={self.shape}, wavelength={self._wavelength}, power_unit={self._power_unit}, "
            f"backend={self.backend!r}, grid={self._grid!r})"
        )

    # --- JAX pytree -------------------------------------------------------------------------------

    def _tree_flatten(self) -> tuple[tuple[Any], tuple[Any, ...]]:
        aux = (self._grid, float(self._wavelength.value), self._wavelength.unit, self._power_unit)
        return (self._field,), aux

    @classmethod
    def _tree_unflatten(cls, aux: tuple[Any, ...], children: Any) -> Wavefront:
        grid, value, unit, power_unit = aux
        new = object.__new__(cls)
        new._field = children[0]  # may be a JAX tracer or a sentinel: skip validation
        new._grid = grid
        new._wavelength = u.Quantity(value, unit)
        new._power_unit = power_unit
        return new

    # --- Serialisation ----------------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        """Encode the wavefront as a strict-JSON-compatible dictionary (the field inline, in base64)."""
        field = get(self.backend).to_numpy(self._field)
        return {
            "grid": self._grid.to_json(),
            "wavelength": quantity_to_json(self._wavelength),
            "power_unit": self._power_unit.to_string(),
            "field": quantity_to_json(u.Quantity(field, u.dimensionless_unscaled)),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Wavefront:
        """Decode a dictionary produced by :meth:`to_json` into the current backend."""
        field = quantity_from_json(data["field"]).value
        return cls(
            field,
            Grid.from_json(data["grid"]),
            quantity_from_json(data["wavelength"]),
            power_unit=u.Unit(data["power_unit"]),
        )
