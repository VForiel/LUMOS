r"""Polychromatic fields: stacks of wavefronts on a spectral axis (REQ-SIG-4).

A :class:`Wavecube` holds the complex amplitudes :math:`U_k` of a field at the vacuum wavelengths
:math:`\lambda_k`, all sampled on the **same** :class:`~lumos.fields.Grid`, with spectral weights
:math:`w_k`. Its field has shape ``(nλ, ny, nx)`` (scalar) or ``(nλ, 2, ny, nx)`` (polarised).

Each channel is normalised like a :class:`~lumos.fields.Wavefront`:
:math:`P_k = \sum |U_k|^2 \Delta x \Delta y` is the power carried by channel :math:`k`. Fields at different
wavelengths do not interfere in a time-averaged intensity (Born & Wolf, *Principles of Optics*, §10.3), so
the detected quantities are weighted **incoherent** sums over the channels:

.. math:: I = \sum_k w_k |U_k|^2, \qquad P = \sum_k w_k P_k .

The weights hold quadrature weights, a spectral transmission or a detector response (default: 1); see
:func:`~lumos.fields.spectral_samples` to sample a band.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any, cast

import numpy as np
from astropy import units as u

from lumos.backend._core import backend_of, convert, get
from lumos.backend.memory import dtypes
from lumos.core.signal import OpticalSignal
from lumos.fields.grid import Grid
from lumos.fields.wavefront import (
    _PHOTON_RATE,
    Wavefront,
    _array_backend,
    _photon_energy,
    _power_unit,
)
from lumos.units import quantity_from_json, quantity_to_json, require_unit, to_si

__all__ = ["Wavecube"]


def _wavelengths(value: Any) -> u.Quantity:
    quantity = require_unit(value, u.m, name="wavelengths", equivalencies=u.spectral())
    if not cast(u.UnitBase, quantity.unit).is_equivalent(u.m):
        quantity = quantity.to(u.m, equivalencies=u.spectral())
    quantity = cast(u.Quantity, np.atleast_1d(quantity))
    if quantity.ndim != 1:
        raise ValueError("wavelengths must be a 1D quantity")
    si = np.asarray(to_si(quantity), dtype=float)
    if not np.all(np.isfinite(si) & (si > 0)):
        raise ValueError("wavelengths must be positive and finite")
    return quantity


def _item(quantity: u.Quantity, k: int) -> u.Quantity:
    return cast(u.Quantity, quantity[k])


class Wavecube(OpticalSignal):
    """Complex field at several wavelengths, sampled on a common grid.

    Parameters
    ----------
    field
        Complex amplitudes, shape ``(nλ, ny, nx)`` or ``(nλ, 2, ny, nx)`` (Jones components).
    grid
        Common sampling of every channel.
    wavelengths
        ``nλ`` vacuum wavelengths (or frequencies, converted with ``u.spectral()``).
    weights
        ``nλ`` dimensionless spectral weights used by the integrated quantities (default: ones).
    power_unit
        Normalisation of every channel: ``u.W`` (default) or ``u.photon / u.s``.

    Notes
    -----
    Wavecubes are immutable JAX pytrees whose only leaf is the field.
    """

    __slots__ = ("_field", "_grid", "_power_unit", "_wavelengths", "_weights")
    __array_ufunc__ = None

    _field: Any
    _grid: Grid
    _wavelengths: u.Quantity
    _weights: np.ndarray
    _power_unit: u.UnitBase

    def __init__(
        self,
        field: Any,
        grid: Grid,
        wavelengths: u.Quantity,
        *,
        weights: Any = None,
        power_unit: Any = u.W,
    ) -> None:
        if not isinstance(grid, Grid):
            raise TypeError(f"grid must be a lumos.fields.Grid, got {type(grid).__name__}")
        lam = _wavelengths(wavelengths)
        w = np.ones(len(lam)) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != (len(lam),):
            raise ValueError(f"weights must have one value per wavelength ({len(lam)}), got shape {w.shape}")
        if not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite")
        target = get(_array_backend(field))
        _, complex_dtype = dtypes(backend=target.name)
        array = target.asarray(field if target.is_array(field) else np.asarray(field), dtype=complex_dtype)
        self._init(array, grid, lam, w, _power_unit(power_unit))

    def _init(
        self,
        field: Any,
        grid: Grid,
        wavelengths: u.Quantity,
        weights: np.ndarray,
        power_unit: u.UnitBase,
    ) -> None:
        n = len(wavelengths)
        allowed = ((n, *grid.shape), (n, 2, *grid.shape))
        if tuple(field.shape) not in allowed:
            raise ValueError(f"the field shape {tuple(field.shape)} must be one of {allowed}")
        self._field = field
        self._grid = grid
        self._wavelengths = wavelengths
        self._weights = weights
        self._power_unit = power_unit

    def _with_field(self, field: Any, power_unit: u.UnitBase | None = None) -> Wavecube:
        new = object.__new__(type(self))
        unit = self._power_unit if power_unit is None else power_unit
        new._init(field, self._grid, self._wavelengths, self._weights, unit)
        return new

    # --- Factories --------------------------------------------------------------------------------

    @classmethod
    def from_wavefronts(cls, wavefronts: Sequence[Wavefront], *, weights: Any = None) -> Wavecube:
        """Stack wavefronts sharing the same grid, power unit and polarisation."""
        if not wavefronts:
            raise ValueError("at least one wavefront is needed")
        first = wavefronts[0]
        for wf in wavefronts[1:]:
            if wf.grid != first.grid:
                raise ValueError("every wavefront of a Wavecube must share the same grid")
            if wf.power_unit != first.power_unit:
                raise ValueError("every wavefront of a Wavecube must have the same power unit")
            if wf.polarized != first.polarized:
                raise ValueError("cannot mix polarised and scalar wavefronts")
        xp = get(first.backend).xp
        field = xp.stack([convert(wf.field, first.backend) for wf in wavefronts])
        wavelengths = u.Quantity([wf.wavelength.to(u.m) for wf in wavefronts]).to(first.wavelength.unit)
        return cls(field, first.grid, wavelengths, weights=weights, power_unit=first.power_unit)

    @classmethod
    def plane(
        cls,
        grid: Grid,
        wavelengths: u.Quantity,
        *,
        power: u.Quantity,
        weights: Any = None,
        polarization: Any = None,
    ) -> Wavecube:
        """Uniform flat-phase field; ``power`` is the power of each channel (a scalar or one per channel)."""
        lam = _wavelengths(wavelengths)
        if not isinstance(power, u.Quantity):
            raise TypeError(f"power must be a Quantity in W or photon / s, got {power!r}")
        powers = cast(u.Quantity, power * np.ones(len(lam)))
        wfs = [
            Wavefront.plane(grid, _item(lam, k), power=_item(powers, k), polarization=polarization)
            for k in range(len(lam))
        ]
        return cls.from_wavefronts(wfs, weights=weights)

    # --- Attributes -------------------------------------------------------------------------------

    @property
    def field(self) -> Any:
        """Complex amplitudes, a backend array of shape ``(nλ, [2,] ny, nx)``."""
        return self._field

    @property
    def grid(self) -> Grid:
        """Common sampling grid."""
        return self._grid

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of :attr:`field`."""
        return tuple(self._field.shape)

    @property
    def polarized(self) -> bool:
        """Whether each channel holds the Jones components ``(Ex, Ey)``."""
        return len(self._field.shape) == 4

    @property
    def backend(self) -> str:
        """Name of the backend holding the field."""
        return backend_of(self._field)

    @property
    def n_wavelengths(self) -> int:
        """Number of spectral channels."""
        return len(self._wavelengths)

    @property
    def wavelengths(self) -> u.Quantity:
        """Vacuum wavelengths of the channels."""
        return self._wavelengths

    @property
    def wavelengths_si(self) -> np.ndarray:
        """Vacuum wavelengths in metres."""
        return np.asarray(to_si(self._wavelengths), dtype=float)

    @property
    def weights(self) -> np.ndarray:
        """Spectral weights of the channels (a copy)."""
        return self._weights.copy()

    @property
    def power_unit(self) -> u.UnitBase:
        """Normalisation of every channel: ``W`` or ``photon / s``."""
        return self._power_unit

    @property
    def intensity_unit(self) -> u.UnitBase:
        """Unit of the intensities: power per m² (spatial grid) or per sr (angular grid)."""
        per = u.m**2 if self._grid.plane == "spatial" else u.sr
        return cast(u.UnitBase, self._power_unit / per)

    def __len__(self) -> int:
        """Return the number of spectral channels."""
        return self.n_wavelengths

    def __getitem__(self, k: int) -> Wavefront:
        """Channel ``k`` as a :class:`~lumos.fields.Wavefront`."""
        wf = object.__new__(Wavefront)
        wf._init(self._field[k], self._grid, _item(self._wavelengths, k), self._power_unit)
        return wf

    def wavefronts(self) -> Iterator[Wavefront]:
        """Iterate over the channels as wavefronts."""
        for k in range(self.n_wavelengths):
            yield self[k]

    # --- Derived quantities -----------------------------------------------------------------------

    @property
    def _xp(self) -> Any:
        return get(self.backend).xp

    @property
    def intensity(self) -> Any:
        """Intensity of each channel, shape ``(nλ, ny, nx)`` (polarisations summed)."""
        xp = self._xp
        density = xp.real(self._field * xp.conj(self._field))
        return xp.sum(density, axis=1) if self.polarized else density

    @property
    def integrated_intensity(self) -> Any:
        r"""Weighted incoherent sum :math:`\sum_k w_k |U_k|^2`, shape ``grid.shape``."""
        real, _ = dtypes(backend=self.backend)
        weights = get(self.backend).asarray(self._weights, dtype=real)
        return self._xp.sum(weights[:, None, None] * self.intensity, axis=0)

    @property
    def power(self) -> u.Quantity:
        """Power of each channel, shape ``(nλ,)``, in :attr:`power_unit`."""
        xp = self._xp
        per_channel = np.asarray(get(self.backend).to_numpy(xp.sum(self.intensity, axis=(1, 2))), dtype=float)
        return u.Quantity(per_channel * self._grid.dx_si * self._grid.dy_si, self._power_unit)

    @property
    def total_power(self) -> u.Quantity:
        r"""Weighted total power :math:`\sum_k w_k P_k`."""
        return u.Quantity(float(np.sum(self._weights * self.power.value)), self._power_unit)

    # --- Operations -------------------------------------------------------------------------------

    def _channel_shape(self, values: np.ndarray) -> np.ndarray:
        return values.reshape((-1,) + (1,) * (len(self.shape) - 1))

    def with_power_unit(self, unit: Any) -> Wavecube:
        """Return the same light in another power unit, using the photon energy of each channel."""
        unit = _power_unit(unit)
        if unit == self._power_unit:
            return self
        old = self._power_unit
        energies = np.array([_photon_energy(lam) for lam in self.wavelengths_si])
        if old.is_equivalent(unit):
            factors = np.full(self.n_wavelengths, float(cast(Any, old.to(unit))))
        elif old.is_equivalent(u.W):
            factors = cast(Any, old.to(u.W)) / energies * cast(Any, _PHOTON_RATE.to(unit))
        else:
            factors = cast(Any, old.to(_PHOTON_RATE)) * energies * cast(Any, u.W.to(unit))
        scale = get(self.backend).asarray(self._channel_shape(np.sqrt(np.asarray(factors, dtype=float))))
        return self._with_field(self._field * scale, unit)

    def tilt(self, theta_x: u.Quantity, theta_y: u.Quantity) -> Wavecube:
        r"""Multiply each channel by :math:`\exp[ik_k(x\sin\theta_x + y\sin\theta_y)]` (chromatic tilt)."""
        if self._grid.plane != "spatial":
            raise ValueError("tilt() needs a field on a spatial grid")
        tx = float(to_si(require_unit(theta_x, u.rad, name="theta_x")))
        ty = float(to_si(require_unit(theta_y, u.rad, name="theta_y")))
        xx, yy = self._grid.coords(backend=self.backend)
        k = get(self.backend).asarray(2 * math.pi / self.wavelengths_si)
        xp = self._xp
        ramp = xp.exp(1j * k[:, None, None] * (xx * math.sin(tx) + yy * math.sin(ty))[None])
        if self.polarized:
            ramp = ramp[:, None]
        return self._with_field(self._field * ramp)

    def to_backend(self, name: str) -> Wavecube:
        """Return the same cube held by backend ``name``."""
        return self._with_field(convert(self._field, name))

    def __mul__(self, other: Any) -> Wavecube:
        """Multiply by a scalar, an achromatic mask ``(ny, nx)`` or a chromatic mask ``(nλ, ny, nx)``."""
        if isinstance(other, Wavecube | Wavefront):
            return NotImplemented
        shape = tuple(getattr(other, "shape", ()))
        if self.polarized and len(shape) == 3:
            other = other[:, None]  # chromatic mask applied to both components
        return self._with_field(self._field * other)

    __rmul__ = __mul__

    def __repr__(self) -> str:
        """Summarise the cube."""
        return (
            f"Wavecube(shape={self.shape}, wavelengths={self._wavelengths}, power_unit={self._power_unit}, "
            f"backend={self.backend!r}, grid={self._grid!r})"
        )

    # --- JAX pytree -------------------------------------------------------------------------------

    def _tree_flatten(self) -> tuple[tuple[Any], tuple[Any, ...]]:
        aux = (
            self._grid,
            tuple(float(v) for v in np.asarray(self._wavelengths.value)),
            self._wavelengths.unit,
            tuple(float(v) for v in self._weights),
            self._power_unit,
        )
        return (self._field,), aux

    @classmethod
    def _tree_unflatten(cls, aux: tuple[Any, ...], children: Any) -> Wavecube:
        grid, values, unit, weights, power_unit = aux
        new = object.__new__(cls)
        new._field = children[0]
        new._grid = grid
        new._wavelengths = u.Quantity(np.asarray(values), unit)
        new._weights = np.asarray(weights)
        new._power_unit = power_unit
        return new

    # --- Serialisation ----------------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        """Encode the cube as a strict-JSON-compatible dictionary (the field inline, in base64)."""
        field = get(self.backend).to_numpy(self._field)
        return {
            "grid": self._grid.to_json(),
            "wavelengths": quantity_to_json(self._wavelengths),
            "weights": quantity_to_json(u.Quantity(self._weights)),
            "power_unit": self._power_unit.to_string(),
            "field": quantity_to_json(u.Quantity(field, u.dimensionless_unscaled)),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Wavecube:
        """Decode a dictionary produced by :meth:`to_json` into the current backend."""
        return cls(
            quantity_from_json(data["field"]).value,
            Grid.from_json(data["grid"]),
            quantity_from_json(data["wavelengths"]),
            weights=quantity_from_json(data["weights"]).value,
            power_unit=u.Unit(data["power_unit"]),
        )
