r"""Zernike polynomials in Noll's ordering and normalisation (Noll 1976, JOSA 66, 207).

Mode :math:`j \ge 1` has the radial order :math:`n` and the azimuthal frequency :math:`m` given by
:func:`noll_indices` (:math:`m > 0` for cosines, :math:`m < 0` for sines, even :math:`j` for cosines):

.. math::
    Z_j = \sqrt{n + 1}\, R_n^0(\rho) \quad (m = 0), \qquad
    Z_j = \sqrt{2(n + 1)}\, R_n^{|m|}(\rho) \times \begin{cases} \cos m\theta \\ \sin |m|\theta \end{cases},

    R_n^{m}(\rho) = \sum_{s=0}^{(n-m)/2}
    \frac{(-1)^s (n - s)!}{s!\,(\frac{n+m}{2} - s)!\,(\frac{n-m}{2} - s)!}\, \rho^{n - 2s},

so that :math:`\frac{1}{\pi}\int_{\rho \le 1} Z_i Z_j = \delta_{ij}`: a coefficient is the RMS of its mode
over the pupil.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from astropy import units as u

from lumos.fields.grid import Grid
from lumos.units import require_unit, to_si

__all__ = ["noll_indices", "zernike", "zernike_basis"]


def noll_indices(j: int) -> tuple[int, int]:
    """Return the radial order ``n`` and the signed azimuthal frequency ``m`` of Noll mode ``j``."""
    if int(j) != j or j < 1:
        raise ValueError(f"Noll indices start at 1, got {j!r}")
    j = int(j)
    n = int((math.isqrt(8 * (j - 1) + 1) - 1) // 2)
    p = j - n * (n + 1) // 2  # rank in the row, from 1
    parity = n % 2
    m = 2 * ((p + parity) // 2) - parity
    if m == 0:
        return n, 0
    return n, m if j % 2 == 0 else -m


def _radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    result = np.zeros_like(rho, dtype=float)
    for s in range((n - m) // 2 + 1):
        coefficient = (-1) ** s * math.factorial(n - s)
        coefficient /= math.factorial(s) * math.factorial((n + m) // 2 - s) * math.factorial((n - m) // 2 - s)
        result = result + coefficient * rho ** (n - 2 * s)
    return result


def zernike(j: int, rho: Any, theta: Any) -> np.ndarray:
    """Evaluate Noll mode ``j`` at polar coordinates ``rho`` (pupil radius 1) and ``theta`` (radians)."""
    n, m = noll_indices(j)
    rho = np.asarray(rho, dtype=float)
    theta = np.asarray(theta, dtype=float)
    radial = _radial(n, abs(m), rho)
    if m == 0:
        return math.sqrt(n + 1) * radial
    angular = np.cos(m * theta) if m > 0 else np.sin(-m * theta)
    return math.sqrt(2 * (n + 1)) * radial * angular


def zernike_basis(grid: Grid, diameter: u.Quantity, n_modes: int) -> np.ndarray:
    """Sample modes 1 … ``n_modes`` on ``grid`` over a pupil of ``diameter`` centred on the origin.

    Returns a NumPy array of shape ``(n_modes, ny, nx)``, zero outside the pupil (``r > diameter / 2``).
    """
    radius = float(to_si(require_unit(diameter, u.m, name="diameter"))) / 2
    if not radius > 0:
        raise ValueError("the pupil diameter must be positive")
    xx, yy = (np.asarray(c, dtype=float) for c in grid.coords(backend="numpy"))
    rho = np.hypot(xx, yy) / radius
    theta = np.arctan2(yy, xx)
    inside = rho <= 1 + 1e-12
    basis = np.zeros((int(n_modes), *grid.shape))
    for k in range(int(n_modes)):
        basis[k][inside] = zernike(k + 1, rho[inside], theta[inside])
    return basis
