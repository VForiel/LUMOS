r"""Linearly polarised (LP) modes of weakly guiding step-index fibres (Gloge 1971, Appl. Opt. 10, 2252).

A fibre of core radius :math:`a`, numerical aperture :math:`\mathrm{NA} = \sqrt{n_\mathrm{co}^2 -
n_\mathrm{cl}^2}` has the normalised frequency :math:`V = 2\pi a\,\mathrm{NA}/\lambda`. Mode
:math:`\mathrm{LP}_{\ell m}` has the transverse field (with :math:`\rho = r/a`)

.. math::
    \psi(\rho, \phi) = \begin{cases} J_\ell(U\rho)/J_\ell(U) & \rho \le 1 \\
    K_\ell(W\rho)/K_\ell(W) & \rho > 1 \end{cases} \times \begin{cases} \cos \ell\phi \\ \sin \ell\phi
    \end{cases},

where :math:`U` is the :math:`m`-th root of the characteristic equation

.. math:: U \frac{J_{\ell-1}(U)}{J_\ell(U)} = -W \frac{K_{\ell-1}(W)}{K_\ell(W)}, \qquad U^2 + W^2 = V^2,

solved here in the pole-free form :math:`U J_{\ell-1}(U) K_\ell(W) + W K_{\ell-1}(W) J_\ell(U) = 0`. The
normalised propagation constant :math:`b = W^2/V^2` gives the effective index
:math:`n_\mathrm{eff}^2 = n_\mathrm{cl}^2 + b\,\mathrm{NA}^2`. A mode is guided above its cutoff, a zero of
:math:`J_{\ell-1}` (:math:`J_1` for :math:`\ell = 0`).

The Gaussian approximation of LP01 has the waist :math:`w/a = 0.65 + 1.619 V^{-3/2} + 2.879 V^{-6}`
(Marcuse 1977, Bell Syst. Tech. J. 56, 703).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.special import jv, kv

__all__ = ["LPMode", "lp_modes", "lp_radial", "marcuse_waist"]


@dataclass(frozen=True)
class LPMode:
    """Guided mode LP(ell, m) at the normalised frequency ``v``, with the roots ``u`` and ``w``."""

    ell: int
    m: int
    u: float
    w: float
    v: float

    @property
    def b(self) -> float:
        """Normalised propagation constant ``W² / V²``."""
        return self.w**2 / self.v**2

    @property
    def label(self) -> str:
        """Name of the mode, e.g. ``"LP11"``."""
        return f"LP{self.ell}{self.m}"


def _characteristic(ell: int, v: float, x: float) -> float:
    w = math.sqrt(max(v * v - x * x, 0.0))
    return float(x * jv(ell - 1, x) * kv(ell, w) + w * kv(ell - 1, w) * jv(ell, x))


def _root(function: Callable[[float], float], low: float, high: float) -> float:
    root: Any = brentq(function, low, high, xtol=1e-15, rtol=np.float64(1e-15))
    return float(root)


def _roots(ell: int, v: float) -> list[float]:
    samples = max(400, int(200 * v))
    xs = np.linspace(1e-9 * v, v * (1 - 1e-12), samples)
    values = [_characteristic(ell, v, float(x)) for x in xs]
    return [
        _root(lambda x: _characteristic(ell, v, x), float(xs[i]), float(xs[i + 1]))
        for i in range(samples - 1)
        if np.sign(values[i]) != np.sign(values[i + 1]) and values[i] != 0
    ]


def lp_modes(v: float) -> list[LPMode]:
    """Return the guided LP modes at the normalised frequency ``v``, by decreasing propagation constant."""
    if not v > 0:
        raise ValueError(f"the normalised frequency must be positive, got {v}")
    modes: list[LPMode] = []
    ell = 0
    while True:
        roots = _roots(ell, v)
        if not roots:
            break
        modes += [LPMode(ell, m, x, math.sqrt(v * v - x * x), v) for m, x in enumerate(roots, start=1)]
        ell += 1
    return sorted(modes, key=lambda mode: -mode.b)


def lp_radial(mode: LPMode, rho: np.ndarray) -> np.ndarray:
    """Radial profile of ``mode`` at ``rho = r / a`` (1 at the core-cladding interface)."""
    rho = np.asarray(rho, dtype=float)
    core = jv(mode.ell, mode.u * rho) / jv(mode.ell, mode.u)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        cladding = kv(mode.ell, mode.w * np.maximum(rho, 1.0)) / kv(mode.ell, mode.w)
    return np.where(rho <= 1, core, cladding)


def marcuse_waist(v: float) -> float:
    """Waist of the Gaussian approximation of LP01 in units of the core radius (Marcuse 1977)."""
    return 0.65 + 1.619 * v**-1.5 + 2.879 * v**-6
