r"""Scattering matrices of guided-wave components (REQ-SIG-7).

A component with :math:`n_\mathrm{in}` input and :math:`n_\mathrm{out}` output channels maps the modal
amplitudes :math:`a` to :math:`S a`, with :math:`S` of shape ``(n_out, n_in)`` or, for a wavelength-dependent
component, ``(nλ, n_out, n_in)``. Lossless reciprocal components have unitary matrices,
:math:`S^\dagger S = 1` (Pozar, *Microwave Engineering*, §4.3), and conserve the power.

Phases follow the :math:`e^{-i\omega t}` convention of §9 of the specifications: :math:`e^{+i\phi}` is a
delay.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from astropy import units as u

from lumos.units import require_unit, to_si

__all__ = ["coupler", "is_unitary", "phase_shifter"]


def coupler(ratio: float) -> np.ndarray:
    r"""Symmetric lossless directional coupler with power coupling ratio ``ratio`` (``κ²``).

    .. math:: S = \begin{pmatrix} t & i\kappa \\ i\kappa & t \end{pmatrix}, \qquad t = \sqrt{1 - \kappa^2}

    (Okamoto, *Fundamentals of Optical Waveguides*, §4.4). ``ratio = 0.5`` is the 50/50 coupler.
    """
    ratio = float(ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"the coupling ratio must be in [0, 1], got {ratio}")
    t, k = math.sqrt(1.0 - ratio), math.sqrt(ratio)
    return np.array([[t, 1j * k], [1j * k, t]], dtype=complex)


def phase_shifter(phases: Sequence[u.Quantity] | u.Quantity) -> np.ndarray:
    """Diagonal matrix applying the phase ``phases[j]`` to channel ``j``."""
    values = np.atleast_1d(np.asarray([float(to_si(require_unit(p, u.rad, name="phase"))) for p in phases]))
    return np.diag(np.exp(1j * values))


def is_unitary(matrix: Any, *, atol: float = 1e-12) -> bool:
    """Whether a square matrix (or every matrix of a stack) satisfies ``S^† S = 1`` within ``atol``."""
    s = np.asarray(matrix)
    if s.ndim < 2 or s.shape[-1] != s.shape[-2]:
        return False
    product = np.conj(np.swapaxes(s, -1, -2)) @ s
    return bool(np.allclose(product, np.eye(s.shape[-1]), atol=atol, rtol=0.0))
