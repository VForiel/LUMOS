r"""Band-limited angular spectrum propagation (REQ-PROP-2, step P4.5).

The angular spectrum method is exact for free-space propagation of the sampled field (no paraxial
approximation; Goodman, *Introduction to Fourier Optics*, §3.10): the spectrum
:math:`A(f_x, f_y) = \iint U e^{-2\pi i (f_x x + f_y y)} dx\, dy` is multiplied by the transfer function

.. math:: H(f_x, f_y) = \exp\left(2\pi i z \sqrt{\lambda^{-2} - f_x^2 - f_y^2}\right)

(evanescent components decay for :math:`z > 0` and are removed when back-propagating), and the field is
recovered by the inverse transform.

Band limit
----------
Sampling :math:`H` on the frequency grid of a window of width :math:`2S` (the field zero-padded by 2) aliases
its chirp at long distances. Following Matsushima & Shimobaba (2009, Opt. Express 17, 19662), frequencies are
kept only up to

.. math:: |f_x| \le \frac{1}{\lambda \sqrt{(2 \Delta f_x z)^2 + 1}}, \qquad \Delta f_x = \frac{1}{2 S_x},

and likewise along y. The method is accurate when the spectrum of the field lies inside this band, which
the planner checks.

Output grids
------------
On the input grid the inverse transform is an FFT. On any other grid (scaled, shifted or rectangular output)
the inverse transform is evaluated as a matrix Fourier transform at the output coordinates, the
"scaled/shifted" angular spectrum: exact for the band-limited field, valid as long as the output plane stays
inside the padded window.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from lumos.backend import fft
from lumos.fields.grid import Grid
from lumos.propagation._common import Method, axis_coordinates, crop, frequencies, mft_axes, pad

__all__ = ["METHOD", "angular_spectrum", "band_limit"]

_PAD = 2


def band_limit(n: int, step: float, wavelength: float, z: float) -> float:
    """Highest frequency kept along one axis for a window of ``n`` samples of ``step`` (Matsushima 2009)."""
    delta_f = 1.0 / (_PAD * n * step)
    return 1.0 / (wavelength * math.sqrt((2 * delta_f * z) ** 2 + 1))


def transfer_function(
    fx: np.ndarray, fy: np.ndarray, wavelength: float, z: float, nx: int, ny: int, dx: float, dy: float
) -> np.ndarray:
    """Band-limited transfer function on the centred frequency grid ``(fy, fx)``."""
    f2 = fx[None, :] ** 2 + fy[:, None] ** 2
    arg = 1.0 / wavelength**2 - f2
    propagating = arg >= 0
    kz = 2 * math.pi * np.sqrt(np.abs(arg))
    h = np.where(propagating, np.exp(1j * kz * z), np.exp(-kz * abs(z)) if z > 0 else 0.0)
    limit_x = band_limit(nx, dx, wavelength, z)
    limit_y = band_limit(ny, dy, wavelength, z)
    inside = (np.abs(fx)[None, :] <= limit_x) & (np.abs(fy)[:, None] <= limit_y)
    return np.where(inside, h, 0.0)


def angular_spectrum(
    field: Any, grid_in: Grid, grid_out: Grid, wavelength: float, z: float | None, xp: Any
) -> Any:
    """Propagate ``field`` by ``z`` with the band-limited angular spectrum (any output grid)."""
    if z is None:
        raise ValueError("the angular spectrum method needs a finite distance")
    padded, offsets = pad(field, _PAD, xp)
    ny, nx = padded.shape[-2:]
    fx = frequencies(nx, grid_in.dx_si)
    fy = frequencies(ny, grid_in.dy_si)
    h = transfer_function(fx, fy, wavelength, z, grid_in.nx, grid_in.ny, grid_in.dx_si, grid_in.dy_si)
    spectrum = fft.fft2c(padded) * xp.asarray(h, dtype=padded.dtype)
    if grid_out == grid_in:
        return crop(fft.ifft2c(spectrum), offsets, grid_in.shape)
    # Phase of the input origin: the centred FFT measures coordinates from x0, the MFT from 0.
    xo = axis_coordinates(grid_out.nx, grid_out.dx_si, grid_out.x0_si - grid_in.x0_si)
    yo = axis_coordinates(grid_out.ny, grid_out.dy_si, grid_out.y0_si - grid_in.y0_si)
    df = 1.0 / (nx * grid_in.dx_si * ny * grid_in.dy_si)
    # A(f) in continuous units is fft2c(U)·dx·dy; the inverse integral is Σ A H e^{+2πi f x} df_x df_y.
    return mft_axes(spectrum, xo, yo, fx, fy, +1.0, xp) * (grid_in.dx_si * grid_in.dy_si * df)


METHOD = Method(
    name="asm",
    kernel=angular_spectrum,
    any_output_grid=True,
    far_field=False,
    finite_distance=True,
    description="band-limited angular spectrum (exact, scaled/shifted output by matrix Fourier transform)",
)
