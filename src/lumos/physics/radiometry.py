r"""Radiometry: radiance, irradiance, étendue and photon counting (step P8.2, REQ-SCN-5).

Definitions (McCluney, *Introduction to Radiometry and Photometry*, 2nd ed., ch. 2):

- the **radiance** :math:`L` is the power per unit projected area and per unit solid angle
  (W m⁻² sr⁻¹; *spectral* radiance :math:`L_\lambda` per unit wavelength, W m⁻² sr⁻¹ m⁻¹). It is
  invariant along a ray in a lossless medium;
- a pupil of area :math:`A` receiving the radiance :math:`L` from a solid angle :math:`\Omega` small enough
  for the obliquity to be negligible collects the power :math:`P = L\,A\,\Omega`; the product
  :math:`G = A\,\Omega` is the **étendue**, which for a cone of half-angle :math:`\alpha` filled by a
  Lambertian source is :math:`G = \pi A \sin^2\alpha` (Welford & Winston, *High Collection Nonimaging
  Optics*, §2.6);
- a Lambertian surface of radiance :math:`L` has the exitance :math:`M = \pi L`; a uniform disk of
  radiance :math:`L` subtending the angular radius :math:`\theta` produces the irradiance
  :math:`E = \pi L \sin^2\theta` on a surface facing it;
- the solid angle of a cone of half-angle :math:`\alpha` is :math:`2\pi(1 - \cos\alpha)`;
- a photon of wavelength :math:`\lambda` carries the energy :math:`hc/\lambda`: spectral quantities in watts
  become photon rates by multiplying by :math:`\lambda / hc` at each wavelength.

The blackbody spectral radiance is Planck's law (:mod:`lumos.physics.blackbody`); magnitudes and bands are in
:mod:`lumos.physics.photometry`.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from astropy import units as u

from lumos.physics.blackbody import planck_lambda
from lumos.physics.photometry import AB_ZERO_POINT, ab_magnitude, flux_from_magnitude, magnitude_from_flux
from lumos.units import require_unit, to_si

__all__ = [
    "ab_magnitude",
    "blackbody_radiance",
    "cone_solid_angle",
    "disk_irradiance",
    "etendue",
    "flux_from_ab_magnitude",
    "flux_from_magnitude",
    "integrate_spectrum",
    "lambertian_exitance",
    "magnitude_from_flux",
    "photon_energy",
    "pupil_power",
    "to_energy",
    "to_photons",
]

_H = 6.62607015e-34  # J s, exact (SI 2019)
_C = 299_792_458.0  # m / s, exact
_PHOTON = u.photon


def blackbody_radiance(temperature: u.Quantity, wavelength: u.Quantity) -> u.Quantity:
    r"""Return the spectral radiance :math:`B_\lambda(T)` of a blackbody (W m⁻² sr⁻¹ m⁻¹), Planck's law."""
    return planck_lambda(wavelength, temperature)


def photon_energy(wavelength: u.Quantity) -> u.Quantity:
    r"""Energy :math:`hc/\lambda` of a photon of vacuum wavelength ``wavelength``."""
    lam = np.asarray(to_si(require_unit(wavelength, u.m, name="wavelength", equivalencies=u.spectral())))
    return u.Quantity(_H * _C / lam, u.J)


def _photon_factor(wavelength: u.Quantity) -> np.ndarray:
    """Photons per joule at each wavelength."""
    lam = np.asarray(to_si(require_unit(wavelength, u.m, name="wavelength", equivalencies=u.spectral())))
    return lam / (_H * _C)


def to_photons(quantity: u.Quantity, wavelength: u.Quantity) -> u.Quantity:
    """Convert an energy-based quantity (W, W m⁻² sr⁻¹ m⁻¹...) to the photon-based one at ``wavelength``.

    The power unit W is replaced by photon s⁻¹; any other factor of the unit is kept.
    """
    if not isinstance(quantity, u.Quantity):
        raise TypeError("quantity must be an astropy Quantity")
    rest = cast(u.UnitBase, (quantity.unit / u.W).decompose())
    si = np.asarray(cast(Any, quantity).to_value(u.W * rest))
    return u.Quantity(si * _photon_factor(wavelength), _PHOTON / u.s * rest)


def to_energy(quantity: u.Quantity, wavelength: u.Quantity) -> u.Quantity:
    """Convert a photon-based quantity (photon s⁻¹, photon s⁻¹ m⁻²...) to the energy-based one (W...)."""
    if not isinstance(quantity, u.Quantity):
        raise TypeError("quantity must be an astropy Quantity")
    rest = cast(u.UnitBase, (quantity.unit / (_PHOTON / u.s)).decompose())
    values = np.asarray(cast(Any, quantity).to_value(_PHOTON / u.s * rest))
    return u.Quantity(values / _photon_factor(wavelength), u.W * rest)


def cone_solid_angle(half_angle: u.Quantity) -> u.Quantity:
    r"""Solid angle :math:`2\pi(1 - \cos\alpha)` of a cone of half-angle :math:`\alpha`."""
    alpha = np.asarray(to_si(require_unit(half_angle, u.rad, name="half_angle")))
    return u.Quantity(2 * math.pi * (1 - np.cos(alpha)), u.sr)


def etendue(area: u.Quantity, half_angle: u.Quantity) -> u.Quantity:
    r"""Étendue :math:`\pi A \sin^2\alpha` of an area receiving a Lambertian cone of half-angle α."""
    a = np.asarray(to_si(require_unit(area, u.m**2, name="area")))
    alpha = np.asarray(to_si(require_unit(half_angle, u.rad, name="half_angle")))
    return u.Quantity(math.pi * a * np.sin(alpha) ** 2, u.m**2 * u.sr)


def lambertian_exitance(radiance: u.Quantity) -> u.Quantity:
    r"""Return the exitance :math:`M = \pi L` of a Lambertian surface (any radiance unit, spectral or not)."""
    return cast(u.Quantity, radiance * (math.pi * u.sr))


def disk_irradiance(radiance: u.Quantity, angular_radius: u.Quantity) -> u.Quantity:
    r"""Irradiance :math:`\pi L \sin^2\theta` of a uniform disk of angular radius θ, facing the surface."""
    theta = np.asarray(to_si(require_unit(angular_radius, u.rad, name="angular_radius")))
    return cast(u.Quantity, radiance * (math.pi * np.sin(theta) ** 2 * u.sr))


def pupil_power(radiance: u.Quantity, *, area: u.Quantity, solid_angle: u.Quantity) -> u.Quantity:
    r"""Power :math:`L\,A\,\Omega` collected by a pupil of area ``A`` from a small solid angle Ω."""
    a = require_unit(area, u.m**2, name="area")
    omega = require_unit(solid_angle, u.sr, name="solid_angle")
    product: Any = radiance * a * omega
    return cast(u.Quantity, product.to(cast(Any, radiance.unit) * u.m**2 * u.sr))


def integrate_spectrum(spectrum: u.Quantity, wavelengths: u.Quantity) -> u.Quantity:
    """Integrate a spectral quantity over wavelength with the trapezoidal rule (last axis)."""
    if np.shape(spectrum)[-1:] != np.shape(wavelengths):
        raise ValueError(
            f"spectrum shape {np.shape(spectrum)} does not match the shape of the wavelengths "
            f"{np.shape(wavelengths)}"
        )
    lam = require_unit(wavelengths, u.m, name="wavelengths")
    values = cast(u.Quantity, spectrum).value
    integral = np.trapezoid(values, cast(Any, lam).value, axis=-1)
    return cast(u.Quantity, u.Quantity(integral, cast(Any, spectrum.unit) * cast(Any, lam).unit).decompose())


def flux_from_ab_magnitude(magnitude: float) -> u.Quantity:
    r"""Spectral flux density :math:`F_\nu = 3631\ \mathrm{Jy} \times 10^{-0.4 m}` of an AB magnitude."""
    return cast(u.Quantity, AB_ZERO_POINT * 10 ** (-0.4 * float(magnitude)))
