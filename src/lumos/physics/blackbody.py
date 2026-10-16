r"""Thermal emission (Planck's law).

The spectral radiance of a blackbody per unit wavelength is

.. math:: B_\lambda(T) = \frac{2 h c^2}{\lambda^5} \frac{1}{e^{hc/\lambda k T} - 1}

in W m⁻² sr⁻¹ m⁻¹, and its photon radiance is :math:`B_\lambda \lambda / hc` (Rybicki & Lightman, *Radiative
Processes in Astrophysics*, §1.5). Constants are the exact 2019 SI values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from astropy import units as u

from lumos.units import require_unit, to_si

__all__ = ["planck_lambda", "planck_lambda_si", "planck_photon_lambda"]

H = 6.62607015e-34  # J s
C = 299_792_458.0  # m / s
K_B = 1.380649e-23  # J / K


def planck_lambda_si(wavelength: Any, temperature: Any) -> Any:
    """Planck spectral radiance in W m⁻² sr⁻¹ m⁻¹ for SI wavelengths (m) and temperatures (K)."""
    lam = np.asarray(wavelength, dtype=float)
    t = np.asarray(temperature, dtype=float)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        x = H * C / (lam * K_B * t)
        radiance = 2 * H * C**2 / lam**5 / np.expm1(x)
    return np.where(t > 0, radiance, 0.0)


def planck_lambda(wavelength: u.Quantity, temperature: u.Quantity) -> u.Quantity:
    """Spectral radiance of a blackbody per unit wavelength."""
    lam = to_si(
        require_unit(wavelength, u.m, name="wavelength", equivalencies=u.spectral()),
        u.m,
        equivalencies=u.spectral(),
    )
    t = to_si(
        require_unit(temperature, u.K, name="temperature", equivalencies=u.temperature()),
        u.K,
        equivalencies=u.temperature(),
    )
    return u.Quantity(planck_lambda_si(lam, t), u.W / u.m**2 / u.sr / u.m)


def planck_photon_lambda(wavelength: u.Quantity, temperature: u.Quantity) -> u.Quantity:
    """Photon spectral radiance of a blackbody per unit wavelength (photon s⁻¹ m⁻² sr⁻¹ m⁻¹)."""
    lam = to_si(
        require_unit(wavelength, u.m, name="wavelength", equivalencies=u.spectral()),
        u.m,
        equivalencies=u.spectral(),
    )
    energy = planck_lambda(wavelength, temperature).to_value(u.W / u.m**2 / u.sr / u.m)
    return u.Quantity(energy * np.asarray(lam) / (H * C), u.photon / u.s / u.m**2 / u.sr / u.m)
