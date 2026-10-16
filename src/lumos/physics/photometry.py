r"""Photometry: magnitude systems, photometric bands and phase functions (REQ-SCN-5).

Magnitudes
----------
- **AB** magnitudes are defined by :math:`m_\mathrm{AB} = -2.5\log_{10}(F_\nu / 3631\ \mathrm{Jy})`
  (Oke & Gunn 1983, ApJ 266, 713).
- **Vega** magnitudes of the Johnson-Cousins-Glass system use the zero points of Bessell, Castelli & Plez
  (1998, A&A 333, 231, table A2): :math:`F_\nu = F_{\nu,0} 10^{-0.4 m}` at the effective wavelength of the
  band.

Bands are described by their effective wavelength and an approximate full width; filter transmission curves
come with the catalogue services (phase 9).

Phase function
--------------
A Lambertian sphere seen at phase angle :math:`\alpha` reflects the fraction
:math:`\Phi(\alpha) = [\sin\alpha + (\pi - \alpha)\cos\alpha] / \pi` of its full-phase flux (Sobolev 1975).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy import units as u

from lumos.units import require_unit, to_si

__all__ = [
    "AB_ZERO_POINT",
    "BANDS",
    "Band",
    "ab_magnitude",
    "flux_from_magnitude",
    "lambert_phase",
    "magnitude_from_flux",
    "photon_flux_from_fnu_si",
]

H = 6.62607015e-34
C = 299_792_458.0

#: Zero point of AB magnitudes.
AB_ZERO_POINT = 3631 * u.Jy


@dataclass(frozen=True)
class Band:
    """A photometric band: effective wavelength, approximate width and Vega zero point (``F_ν`` at m = 0)."""

    name: str
    effective_wavelength: u.Quantity
    width: u.Quantity
    zero_point: u.Quantity


def _band(name: str, lam_um: float, width_um: float, zero_jy: float) -> Band:
    return Band(name, lam_um * u.um, width_um * u.um, zero_jy * u.Jy)


#: Johnson-Cousins-Glass bands: effective wavelengths and zero points of Bessell, Castelli & Plez (1998,
#: table A2); widths are approximate full widths of the filters.
BANDS: dict[str, Band] = {
    band.name: band
    for band in (
        _band("U", 0.366, 0.066, 1790),
        _band("B", 0.438, 0.094, 4063),
        _band("V", 0.545, 0.088, 3636),
        _band("R", 0.641, 0.138, 3064),
        _band("I", 0.798, 0.149, 2416),
        _band("J", 1.22, 0.213, 1589),
        _band("H", 1.63, 0.307, 1021),
        _band("K", 2.19, 0.390, 640),
    )
}


def _band_of(band: str | Band) -> Band:
    if isinstance(band, Band):
        return band
    if band not in BANDS:
        raise ValueError(f"unknown band {band!r}; known bands: {', '.join(BANDS)}")
    return BANDS[band]


def ab_magnitude(flux_density: u.Quantity) -> float:
    """AB magnitude of a spectral flux density per unit frequency."""
    fnu = float(to_si(require_unit(flux_density, u.Jy, name="flux_density")))
    return -2.5 * math.log10(fnu / float(to_si(AB_ZERO_POINT)))


def flux_from_magnitude(magnitude: float, band: str | Band) -> u.Quantity:
    """Spectral flux density ``F_ν`` of a Vega magnitude in ``band`` (at its effective wavelength)."""
    return _band_of(band).zero_point * 10 ** (-0.4 * float(magnitude))


def magnitude_from_flux(flux_density: u.Quantity, band: str | Band) -> float:
    """Vega magnitude of a spectral flux density ``F_ν`` in ``band``."""
    zero = _band_of(band).zero_point
    ratio = float((require_unit(flux_density, u.Jy, name="flux_density") / zero).decompose())
    return -2.5 * math.log10(ratio)


def photon_flux_from_fnu_si(fnu: Any, wavelength: Any) -> Any:
    r"""Photon flux density per unit wavelength (photon s⁻¹ m⁻² m⁻¹) from :math:`F_\nu` (W m⁻² Hz⁻¹).

    :math:`F_\lambda = F_\nu c/\lambda^2` and one photon carries :math:`hc/\lambda`, so the photon flux is
    :math:`F_\nu / (h \lambda)`.
    """
    return np.asarray(fnu) / (H * np.asarray(wavelength))


def lambert_phase(phase_angle: u.Quantity) -> Any:
    """Phase function of a Lambertian sphere at ``phase_angle`` (1 at full phase)."""
    alpha = np.asarray(to_si(require_unit(phase_angle, u.rad, name="phase_angle")))
    return (np.sin(alpha) + (math.pi - alpha) * np.cos(alpha)) / math.pi
