r"""Sampling of a spectral band for polychromatic computations (REQ-SIG-4).

A polychromatic quantity is integrated over a band with a quadrature rule: wavelengths :math:`\lambda_k` and
weights :math:`w_k` such that

.. math:: \frac{1}{\Delta\lambda}\int_{\lambda_0 - \Delta\lambda/2}^{\lambda_0 + \Delta\lambda/2}
          f(\lambda)\,d\lambda \approx \sum_k w_k f(\lambda_k), \qquad \sum_k w_k = 1 .

- ``"midpoint"``: ``n`` equal panels sampled at their centres, equal weights ``1/n``; exact for linear
  spectra, robust for any spectrum.
- ``"gauss"``: Gauss-Legendre nodes and weights; exact for polynomials of degree ``2n - 1``, the most
  accurate choice for smooth spectra.
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
from astropy import units as u

from lumos.units import require_unit, to_si

__all__ = ["spectral_samples"]


def spectral_samples(
    center: u.Quantity,
    bandwidth: u.Quantity,
    n: int,
    *,
    rule: Literal["midpoint", "gauss"] = "midpoint",
) -> tuple[u.Quantity, np.ndarray]:
    """Return wavelengths and normalised weights sampling a band.

    Parameters
    ----------
    center
        Central wavelength of the band.
    bandwidth
        Full width of the band; the band must stay at positive wavelengths.
    n
        Number of samples.
    rule
        ``"midpoint"`` or ``"gauss"`` (Gauss-Legendre).

    Returns
    -------
    wavelengths : astropy.units.Quantity
        ``n`` wavelengths in the unit of ``center``.
    weights : numpy.ndarray
        ``n`` weights summing to 1 (band average).
    """
    unit = require_unit(center, u.m, name="center").unit
    lam0 = float(to_si(center))
    width = float(to_si(require_unit(bandwidth, u.m, name="bandwidth")))
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if not (width >= 0 and lam0 - width / 2 > 0):
        raise ValueError("the band must have a non-negative width and stay at positive wavelengths")
    if rule == "midpoint":
        nodes = (np.arange(n) + 0.5) / n - 0.5
        weights = np.full(n, 1.0 / n)
    elif rule == "gauss":
        x, w = np.polynomial.legendre.leggauss(int(n))
        nodes, weights = x / 2, w / 2
    else:
        raise ValueError(f"unknown quadrature rule {rule!r}; use 'midpoint' or 'gauss'")
    wavelengths = (lam0 + width * nodes) * u.m
    return cast(u.Quantity, wavelengths.to(unit)), weights
