r"""Spectral radiance maps of incoherent scenes (REQ-SIG-6).

A :class:`RadianceField` describes light that has no phase: the spectral radiance
:math:`L(x, y, \lambda)` of a scene, optionally with the Stokes parameters :math:`(I, Q, U, V)`, sampled at
wavelengths :math:`\lambda_k` on a :class:`~lumos.fields.Grid`. On an angular grid (the sky) it is the
radiance seen from the observer; on a spatial grid (a laboratory source) it is the radiance leaving each
point of the source plane. Its data has shape ``(nλ, ny, nx)`` or ``(nλ, 4, ny, nx)``.

Units
-----
The radiance may be given per wavelength or per frequency, in energy or photon units, e.g.
``W m⁻² sr⁻¹ m⁻¹``, ``photon s⁻¹ m⁻² sr⁻¹ m⁻¹``, ``erg s⁻¹ cm⁻² sr⁻¹ Å⁻¹`` or ``Jy sr⁻¹``. Conversions use
the wavelength of each channel (Rybicki & Lightman, *Radiative Processes in Astrophysics*, §1.3):

.. math:: L_\lambda = L_\nu \frac{c}{\lambda^2}, \qquad L^{\mathrm{ph}} = L \frac{\lambda}{h c} .

Pixels are piecewise-constant: :meth:`RadianceField.resample` computes each new pixel as the area-weighted
mean of the old pixels it overlaps, which conserves the flux :math:`\sum L\,\Delta A` (§9 of the
specifications).
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from astropy import units as u

from lumos.backend._core import backend_of, convert, current, get
from lumos.backend.memory import dtypes
from lumos.core.signal import OpticalSignal
from lumos.fields.grid import Grid
from lumos.fields.wavecube import _wavelengths
from lumos.fields.wavefront import _PLANCK, _SPEED_OF_LIGHT
from lumos.units import quantity_from_json, quantity_to_json, to_si

__all__ = ["CANONICAL_RADIANCE", "RadianceField"]

#: Canonical SI spectral radiance, per wavelength.
CANONICAL_RADIANCE = cast(u.UnitBase, u.W / u.m**2 / u.sr / u.m)

_Kind = Literal["energy_wavelength", "energy_frequency", "photon_wavelength", "photon_frequency"]
_KINDS: dict[_Kind, u.UnitBase] = {
    "energy_wavelength": CANONICAL_RADIANCE,
    "energy_frequency": cast(u.UnitBase, u.W / u.m**2 / u.sr / u.Hz),
    "photon_wavelength": cast(u.UnitBase, u.photon / u.s / u.m**2 / u.sr / u.m),
    "photon_frequency": cast(u.UnitBase, u.photon / u.s / u.m**2 / u.sr / u.Hz),
}


def _kind(unit: Any) -> tuple[u.UnitBase, _Kind]:
    unit = cast(u.UnitBase, u.Unit(unit))
    for kind, reference in _KINDS.items():
        if unit.is_equivalent(reference):
            return unit, kind
    raise u.UnitsError(
        f"'{unit}' is not a spectral radiance (e.g. W m-2 sr-1 m-1, photon s-1 m-2 sr-1 m-1 or Jy sr-1)"
    )


def _to_canonical_factors(unit: u.UnitBase, wavelengths_si: np.ndarray) -> np.ndarray:
    """Factors converting radiance values in ``unit`` to W m⁻² sr⁻¹ m⁻¹, one per wavelength."""
    unit, kind = _kind(unit)
    scale = float(cast(Any, unit.to(_KINDS[kind])))
    lam = wavelengths_si
    photon_energy = _PLANCK * _SPEED_OF_LIGHT / lam
    per_frequency = _SPEED_OF_LIGHT / lam**2  # d nu / d lambda
    factors = {
        "energy_wavelength": np.ones_like(lam),
        "energy_frequency": per_frequency,
        "photon_wavelength": photon_energy,
        "photon_frequency": photon_energy * per_frequency,
    }[kind]
    return scale * factors


def _edges(n: int, step: float, center: float) -> np.ndarray:
    return (np.arange(n + 1) - n // 2 - 0.5) * step + center


def _overlap(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Overlap lengths between new (rows) and old (columns) 1D pixels given by their edges."""
    low = np.maximum(new[:-1, None], old[None, :-1])
    high = np.minimum(new[1:, None], old[None, 1:])
    return np.clip(high - low, 0.0, None)


class RadianceField(OpticalSignal):
    """Spectral radiance of an incoherent scene sampled on a grid.

    Parameters
    ----------
    radiance
        Spectral radiance, a quantity of shape ``(nλ, ny, nx)`` or ``(nλ, 4, ny, nx)`` (Stokes I, Q, U, V).
    grid
        Angular (sky) or spatial (source plane) sampling.
    wavelengths
        ``nλ`` vacuum wavelengths of the channels.

    Attributes
    ----------
    data
        Values of the radiance in :attr:`unit`, as a real backend array.
    """

    __slots__ = ("_data", "_grid", "_unit", "_wavelengths")
    __array_ufunc__ = None

    _data: Any
    _grid: Grid
    _unit: u.UnitBase
    _wavelengths: u.Quantity

    def __init__(self, radiance: u.Quantity, grid: Grid, wavelengths: u.Quantity) -> None:
        if not isinstance(radiance, u.Quantity):
            raise TypeError("radiance must be an astropy Quantity in spectral radiance units")
        if not isinstance(grid, Grid):
            raise TypeError(f"grid must be a lumos.fields.Grid, got {type(grid).__name__}")
        unit, _ = _kind(radiance.unit)
        lam = _wavelengths(wavelengths)
        target = current()
        real, _ = dtypes(backend=target.name)
        self._init(target.asarray(np.asarray(radiance.value), dtype=real), grid, lam, unit)

    def _init(self, data: Any, grid: Grid, wavelengths: u.Quantity, unit: u.UnitBase) -> None:
        n = len(wavelengths)
        allowed = ((n, *grid.shape), (n, 4, *grid.shape))
        if tuple(data.shape) not in allowed:
            raise ValueError(f"the radiance shape {tuple(data.shape)} must be one of {allowed}")
        self._data = data
        self._grid = grid
        self._wavelengths = wavelengths
        self._unit = unit

    def _with(self, data: Any, grid: Grid | None = None, unit: u.UnitBase | None = None) -> RadianceField:
        new = object.__new__(type(self))
        new._init(
            data,
            self._grid if grid is None else grid,
            self._wavelengths,
            self._unit if unit is None else unit,
        )
        return new

    # --- Attributes -------------------------------------------------------------------------------

    @property
    def data(self) -> Any:
        """Radiance values in :attr:`unit` (real backend array)."""
        return self._data

    @property
    def radiance(self) -> u.Quantity:
        """Radiance as a NumPy quantity in :attr:`unit`."""
        return u.Quantity(get(self.backend).to_numpy(self._data), self._unit)

    @property
    def unit(self) -> u.UnitBase:
        """Unit of :attr:`data`."""
        return self._unit

    @property
    def grid(self) -> Grid:
        """Sampling grid."""
        return self._grid

    @property
    def wavelengths(self) -> u.Quantity:
        """Wavelengths of the channels."""
        return self._wavelengths

    @property
    def wavelengths_si(self) -> np.ndarray:
        """Wavelengths in metres."""
        return np.asarray(to_si(self._wavelengths), dtype=float)

    @property
    def n_wavelengths(self) -> int:
        """Number of spectral channels."""
        return len(self._wavelengths)

    @property
    def has_stokes(self) -> bool:
        """Whether the data holds the four Stokes parameters along axis 1."""
        return len(self._data.shape) == 4

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of :attr:`data`."""
        return tuple(self._data.shape)

    @property
    def backend(self) -> str:
        """Name of the backend holding the data."""
        return backend_of(self._data)

    # --- Conversions and integrals ---------------------------------------------------------------

    def to(self, unit: Any) -> RadianceField:
        """Return the same radiance expressed in ``unit`` (per wavelength or frequency, energy or photons)."""
        new_unit, _ = _kind(unit)
        lam = self.wavelengths_si
        factors = _to_canonical_factors(self._unit, lam) / _to_canonical_factors(new_unit, lam)
        shape = (-1,) + (1,) * (len(self.shape) - 1)
        real, _ = dtypes(backend=self.backend)
        return self._with(
            self._data * get(self.backend).asarray(factors.reshape(shape), dtype=real), unit=new_unit
        )

    def integrated(self) -> u.Quantity:
        r"""Return :math:`\sum L\,\Delta A` per wavelength (Stokes I only), shape ``(nλ,)``.

        On an angular grid it is the spectral irradiance at the observer (e.g. W m⁻² m⁻¹); on a spatial grid
        it is the spectral radiant intensity of the source (e.g. W sr⁻¹ m⁻¹).
        """
        xp = get(self.backend).xp
        intensity = self._data[:, 0] if self.has_stokes else self._data
        sums = np.asarray(get(self.backend).to_numpy(xp.sum(intensity, axis=(1, 2))), dtype=float)
        area_unit = u.sr if self._grid.plane == "angular" else u.m**2
        return u.Quantity(sums * self._grid.dx_si * self._grid.dy_si, self._unit * area_unit)

    def resample(self, grid: Grid) -> RadianceField:
        """Resample onto ``grid`` by area-weighted averaging, conserving the flux inside the overlap."""
        if grid.plane != self._grid.plane:
            raise ValueError(f"cannot resample a {self._grid.plane} plane onto a {grid.plane} plane")
        old, new = self._grid, grid
        wx = _overlap(_edges(old.nx, old.dx_si, old.x0_si), _edges(new.nx, new.dx_si, new.x0_si))
        wy = _overlap(_edges(old.ny, old.dy_si, old.y0_si), _edges(new.ny, new.dy_si, new.y0_si))
        wx /= new.dx_si
        wy /= new.dy_si
        selected = get(self.backend)
        real, _ = dtypes(backend=selected.name)
        xp = selected.xp
        wx_b = selected.asarray(wx.T, dtype=real)  # (old nx, new nx)
        wy_b = selected.asarray(wy, dtype=real)  # (new ny, old ny)
        out = xp.matmul(xp.matmul(wy_b, self._data), wx_b)
        return self._with(out, grid=grid)

    def to_backend(self, name: str) -> RadianceField:
        """Return the same radiance held by backend ``name``."""
        return self._with(convert(self._data, name))

    def __mul__(self, other: Any) -> RadianceField:
        """Multiply by a dimensionless scalar or transmission map (broadcast on the data)."""
        if isinstance(other, OpticalSignal):
            return NotImplemented
        return self._with(self._data * other)

    __rmul__ = __mul__

    def __repr__(self) -> str:
        """Summarise the radiance field."""
        return (
            f"RadianceField(shape={self.shape}, unit={self._unit}, wavelengths={self._wavelengths}, "
            f"grid={self._grid!r})"
        )

    # --- JAX pytree -------------------------------------------------------------------------------

    def _tree_flatten(self) -> tuple[tuple[Any], tuple[Any, ...]]:
        values = tuple(float(v) for v in np.asarray(self._wavelengths.value))
        return (self._data,), (self._grid, values, self._wavelengths.unit, self._unit)

    @classmethod
    def _tree_unflatten(cls, aux: tuple[Any, ...], children: Any) -> RadianceField:
        grid, values, lam_unit, unit = aux
        new = object.__new__(cls)
        new._data = children[0]
        new._grid = grid
        new._wavelengths = u.Quantity(np.asarray(values), lam_unit)
        new._unit = unit
        return new

    # --- Serialisation ----------------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        """Encode the radiance field as a strict-JSON-compatible dictionary."""
        return {
            "grid": self._grid.to_json(),
            "wavelengths": quantity_to_json(self._wavelengths),
            "radiance": quantity_to_json(self.radiance),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> RadianceField:
        """Decode a dictionary produced by :meth:`to_json` into the current backend."""
        return cls(
            quantity_from_json(data["radiance"]),
            Grid.from_json(data["grid"]),
            quantity_from_json(data["wavelengths"]),
        )
