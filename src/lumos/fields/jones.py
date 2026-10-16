r"""Polarisation: Jones vectors, Jones matrices and Stokes parameters (REQ-SIG-3).

A polarised :class:`~lumos.fields.Wavefront` holds the Jones components :math:`(E_x, E_y)` in the laboratory
basis along the first axis of its field, of shape ``(2, ny, nx)``. Polarising elements multiply it by a
:math:`2 \times 2` Jones matrix, uniform (shape ``(2, 2)``) or varying across the grid (shape
``(2, 2, ny, nx)``), with :func:`apply`.

Conventions
-----------
- Angles are measured from +x towards +y.
- With the time dependence :math:`e^{-i\omega t}` of §9 of the specifications, a phase factor
  :math:`e^{+i\phi}` is a delay. A retarder of retardance :math:`\Gamma` with its fast axis along x is
  :math:`\mathrm{diag}(1, e^{i\Gamma})`.
- Stokes parameters:

  .. math:: S_0 = |E_x|^2 + |E_y|^2,\quad S_1 = |E_x|^2 - |E_y|^2,\quad
            S_2 = 2\,\mathrm{Re}(E_x^* E_y),\quad S_3 = 2\,\mathrm{Im}(E_x^* E_y).

  The right circular state :math:`R = (1, i)/\sqrt 2` turns clockwise for an observer facing the source and
  has :math:`S_3 = +S_0` (Born & Wolf, *Principles of Optics*, §1.4.2).
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from astropy import units as u

from lumos.backend._core import get
from lumos.backend.memory import dtypes
from lumos.fields.wavecube import Wavecube
from lumos.fields.wavefront import Wavefront
from lumos.units import require_unit, to_si

__all__ = [
    "apply",
    "degree_of_polarization",
    "half_wave_plate",
    "linear",
    "polarizer",
    "quarter_wave_plate",
    "retarder",
    "rotator",
    "stokes",
    "vector",
]

_SQRT_HALF = 1 / math.sqrt(2)
_STATES = {
    "H": np.array([1.0, 0.0], dtype=complex),
    "V": np.array([0.0, 1.0], dtype=complex),
    "D": np.array([_SQRT_HALF, _SQRT_HALF], dtype=complex),
    "A": np.array([_SQRT_HALF, -_SQRT_HALF], dtype=complex),
    "R": np.array([_SQRT_HALF, 1j * _SQRT_HALF], dtype=complex),
    "L": np.array([_SQRT_HALF, -1j * _SQRT_HALF], dtype=complex),
}


def _angle(value: u.Quantity, name: str) -> float:
    return float(to_si(require_unit(value, u.rad, name=name)))


# --- Jones vectors -------------------------------------------------------------------------------------


def vector(name: str) -> np.ndarray:
    """Return the normalised Jones vector of a basis state.

    Parameters
    ----------
    name
        ``"H"`` (horizontal, x), ``"V"`` (vertical, y), ``"D"`` (diagonal, +45°), ``"A"`` (anti-diagonal,
        -45°), ``"R"`` (right circular) or ``"L"`` (left circular).
    """
    try:
        return _STATES[name].copy()
    except KeyError:
        raise ValueError(f"unknown polarisation state {name!r}; use one of {', '.join(_STATES)}") from None


def linear(angle: u.Quantity) -> np.ndarray:
    """Return the Jones vector of a linear polarisation at ``angle`` from x."""
    theta = _angle(angle, "angle")
    return np.array([math.cos(theta), math.sin(theta)], dtype=complex)


# --- Jones matrices --------------------------------------------------------------------------------------


def _rotation(theta: float) -> np.ndarray:
    """Passive rotation into a frame turned by ``theta``."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, s], [-s, c]], dtype=complex)


def _oriented(diagonal: tuple[complex, complex], theta: float) -> np.ndarray:
    return _rotation(-theta) @ np.diag(np.asarray(diagonal, dtype=complex)) @ _rotation(theta)


def polarizer(angle: u.Quantity) -> np.ndarray:
    """Ideal linear polariser transmitting the polarisation at ``angle`` (a projector)."""
    return _oriented((1.0, 0.0), _angle(angle, "angle"))


def retarder(retardance: u.Quantity, angle: u.Quantity) -> np.ndarray:
    r"""Linear retarder of retardance :math:`\Gamma` with its fast axis at ``angle``.

    The slow axis is delayed by :math:`e^{i\Gamma}`.
    """
    gamma = _angle(retardance, "retardance")
    return _oriented((1.0, complex(math.cos(gamma), math.sin(gamma))), _angle(angle, "angle"))


def half_wave_plate(angle: u.Quantity) -> np.ndarray:
    """Half-wave plate (retardance π) with its fast axis at ``angle``."""
    return retarder(math.pi * u.rad, angle)


def quarter_wave_plate(angle: u.Quantity) -> np.ndarray:
    """Quarter-wave plate (retardance π/2) with its fast axis at ``angle``."""
    return retarder(math.pi / 2 * u.rad, angle)


def rotator(angle: u.Quantity) -> np.ndarray:
    """Polarisation rotator turning every linear polarisation by ``angle`` (e.g. a Faraday rotator)."""
    c, s = math.cos(_angle(angle, "angle")), math.sin(_angle(angle, "angle"))
    return np.array([[c, -s], [s, c]], dtype=complex)


def apply[F: Wavefront | Wavecube](matrix: Any, wavefront: F) -> F:
    """Apply a Jones matrix to a polarised wavefront or wavecube (every channel).

    Parameters
    ----------
    matrix
        Shape ``(2, 2)`` (uniform) or ``(2, 2, ny, nx)`` (varying across the grid), any backend.
    wavefront
        Polarised :class:`~lumos.fields.Wavefront` (field ``(2, ny, nx)``) or
        :class:`~lumos.fields.Wavecube` (field ``(nλ, 2, ny, nx)``).
    """
    if not wavefront.polarized:
        raise ValueError("Jones matrices act on polarised fields; use with_polarization() first")
    shape = tuple(np.shape(matrix)) if not hasattr(matrix, "shape") else tuple(matrix.shape)
    if shape not in ((2, 2), (2, 2, *wavefront.grid.shape)):
        raise ValueError(f"a Jones matrix has shape (2, 2) or (2, 2, ny, nx), got {shape}")
    target = get(wavefront.backend)
    _, complex_dtype = dtypes(backend=target.name)
    m = target.asarray(matrix if target.is_array(matrix) else np.asarray(matrix), dtype=complex_dtype)
    ex, ey = wavefront.field[..., 0, :, :], wavefront.field[..., 1, :, :]
    out = target.xp.stack([m[0, 0] * ex + m[0, 1] * ey, m[1, 0] * ex + m[1, 1] * ey], axis=-3)
    return cast(F, wavefront._with_field(out))


# --- Stokes parameters ----------------------------------------------------------------------------------


def stokes(wavefront: Wavefront | Wavecube) -> Any:
    """Return the Stokes parameters ``(S0, S1, S2, S3)``, shape ``(4, ny, nx)`` or ``(nλ, 4, ny, nx)``.

    They are in the intensity unit of the wavefront (:attr:`Wavefront.intensity_unit`).
    """
    if not wavefront.polarized:
        raise ValueError("Stokes parameters need a polarised wavefront")
    xp = get(wavefront.backend).xp
    ex, ey = wavefront.field[..., 0, :, :], wavefront.field[..., 1, :, :]
    ix = xp.real(ex * xp.conj(ex))
    iy = xp.real(ey * xp.conj(ey))
    cross = xp.conj(ex) * ey
    return xp.stack([ix + iy, ix - iy, 2 * xp.real(cross), 2 * xp.imag(cross)], axis=-3)


def degree_of_polarization(stokes_vector: Any) -> Any:
    r"""Return :math:`\sqrt{S_1^2 + S_2^2 + S_3^2} / S_0` for Stokes parameters along the first axis."""
    s = np.asarray(stokes_vector) if isinstance(stokes_vector, list | tuple) else stokes_vector
    return (s[1] ** 2 + s[2] ** 2 + s[3] ** 2) ** 0.5 / s[0]
