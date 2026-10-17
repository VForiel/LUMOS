r"""Point sources at infinity (REQ-SCN-2).

A :class:`SourceList` describes :math:`M` unresolved sources at infinity by their directions
:math:`(\theta_x, \theta_y)` (direction cosines, as in far-field grids) and their spectral **photon**
flux densities :math:`F_{mk}` (photon s⁻¹ m⁻² m⁻¹) at the wavelengths :math:`\lambda_k` of the channels.
Each channel covers the bandwidth :math:`b_k`, so a collecting area :math:`A` receives
:math:`A \sum_k F_{mk} b_k` photons per second from source :math:`m`.

A telescope turns a source list into the plane waves sampled on its pupil grid (an
:class:`~lumos.fields.IncoherentBundle`).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from astropy import units as u

from lumos.core.signal import OpticalSignal
from lumos.units import quantity_from_json, quantity_to_json, require_unit, to_si

__all__ = ["PHOTON_FLUX_DENSITY", "SourceList"]

#: Unit of the spectral photon flux densities.
PHOTON_FLUX_DENSITY = cast(u.UnitBase, u.photon / u.s / u.m**2 / u.m)


def _vector(value: Any, unit: Any, name: str) -> np.ndarray:
    return np.atleast_1d(np.asarray(to_si(require_unit(value, unit, name=name)), dtype=float))


class SourceList(OpticalSignal):
    """Unresolved sources at infinity with their spectra.

    Parameters
    ----------
    theta_x, theta_y
        Directions of the ``M`` sources (angles, small-angle direction cosines).
    wavelengths
        ``nλ`` vacuum wavelengths of the channels.
    bandwidths
        ``nλ`` widths of the channels.
    flux
        Spectral flux densities, shape ``(M, nλ)``: photon s⁻¹ m⁻² m⁻¹, or any spectral flux density
        (per wavelength or frequency, energy or photons), converted at each wavelength.
    names
        Optional names of the sources.
    """

    def __init__(
        self,
        theta_x: u.Quantity,
        theta_y: u.Quantity,
        wavelengths: u.Quantity,
        bandwidths: u.Quantity,
        flux: u.Quantity,
        names: list[str] | None = None,
    ) -> None:
        self._tx = _vector(theta_x, u.rad, "theta_x")
        self._ty = _vector(theta_y, u.rad, "theta_y")
        self._lam = _vector(wavelengths, u.m, "wavelengths")
        self._bw = _vector(bandwidths, u.m, "bandwidths")
        if self._tx.shape != self._ty.shape or self._lam.shape != self._bw.shape:
            raise ValueError("theta_x/theta_y and wavelengths/bandwidths must have matching shapes")
        if not isinstance(flux, u.Quantity):
            raise TypeError("flux must be a spectral flux density quantity")
        values = np.asarray(flux.value, dtype=float)
        if values.ndim == 1 and len(self._tx) == 1:
            values = values[None, :]
        if values.shape != (len(self._tx), len(self._lam)):
            raise ValueError(
                f"flux must have shape (M, nλ) = {(len(self._tx), len(self._lam))}, got {values.shape}"
            )
        spectral = u.spectral_density(self._lam * u.m)
        converted = u.Quantity(values, flux.unit).to(PHOTON_FLUX_DENSITY, equivalencies=spectral)
        self._flux = np.asarray(converted.value, dtype=float)
        self._names = list(names) if names is not None else [f"source{m}" for m in range(len(self._tx))]
        if len(self._names) != len(self._tx):
            raise ValueError("one name per source is needed")

    @property
    def n_sources(self) -> int:
        """Number of sources ``M``."""
        return len(self._tx)

    @property
    def names(self) -> list[str]:
        """Names of the sources."""
        return list(self._names)

    @property
    def theta_x(self) -> u.Quantity:
        """Directions along x."""
        return self._tx * u.rad

    @property
    def theta_y(self) -> u.Quantity:
        """Directions along y."""
        return self._ty * u.rad

    @property
    def wavelengths(self) -> u.Quantity:
        """Wavelengths of the channels."""
        return (self._lam * u.m).to(u.um)

    @property
    def bandwidths(self) -> u.Quantity:
        """Widths of the channels."""
        return (self._bw * u.m).to(u.um)

    @property
    def flux(self) -> u.Quantity:
        """Spectral photon flux densities, shape ``(M, nλ)``."""
        return self._flux * PHOTON_FLUX_DENSITY

    def photon_rate(self, area: u.Quantity) -> u.Quantity:
        """Photons per second collected by ``area`` from each source, shape ``(M,)``."""
        a = float(to_si(require_unit(area, u.m**2, name="area")))
        return (self._flux @ self._bw) * a * u.photon / u.s

    def __add__(self, other: Any) -> SourceList:
        """Concatenate two source lists sampled at the same wavelengths."""
        if not isinstance(other, SourceList):
            return NotImplemented
        if not (np.array_equal(other._lam, self._lam) and np.array_equal(other._bw, self._bw)):
            raise ValueError("cannot combine source lists with different spectral channels")
        return SourceList(
            np.concatenate([self._tx, other._tx]) * u.rad,
            np.concatenate([self._ty, other._ty]) * u.rad,
            self._lam * u.m,
            self._bw * u.m,
            np.concatenate([self._flux, other._flux]) * PHOTON_FLUX_DENSITY,
            names=self._names + other._names,
        )

    def __repr__(self) -> str:
        """Summarise the list."""
        return f"SourceList(n_sources={self.n_sources}, n_wavelengths={len(self._lam)}, names={self._names})"

    def to_json(self) -> dict[str, Any]:
        """Encode the list as a strict-JSON-compatible dictionary."""
        return {
            "theta_x": quantity_to_json(self._tx * u.rad),
            "theta_y": quantity_to_json(self._ty * u.rad),
            "wavelengths": quantity_to_json(self._lam * u.m),
            "bandwidths": quantity_to_json(self._bw * u.m),
            "flux": quantity_to_json(self._flux * PHOTON_FLUX_DENSITY),
            "names": list(self._names),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SourceList:
        """Decode a dictionary produced by :meth:`to_json`."""
        return cls(
            quantity_from_json(data["theta_x"]),
            quantity_from_json(data["theta_y"]),
            quantity_from_json(data["wavelengths"]),
            quantity_from_json(data["bandwidths"]),
            quantity_from_json(data["flux"]),
            names=list(data["names"]),
        )
