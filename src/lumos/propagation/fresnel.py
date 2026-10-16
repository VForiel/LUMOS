r"""Fresnel (paraxial) propagation (REQ-PROP-2, step P4.4).

In the Fresnel approximation the field at distance :math:`z` is the convolution of the input field with the
paraxial impulse response (Goodman, *Introduction to Fourier Optics*, §4.2):

.. math:: U(x, y, z) = \iint U(x', y', 0)\, h(x - x', y - y')\, dx'\, dy', \qquad
          h(x, y) = \frac{e^{ikz}}{i\lambda z} e^{ik(x^2 + y^2)/2z},

whose transfer function is :math:`H(f_x, f_y) = e^{ikz} e^{-i\pi\lambda z (f_x^2 + f_y^2)}`. The
approximation holds when the neglected term of the phase is small, :math:`k \rho^4 / 8 z^3 \ll 1` for lateral
distances :math:`\rho`.

Three discretisations are provided (Voelz, *Computational Fourier Optics*, 2011, §5):

- ``"fresnel_tf"`` multiplies the spectrum by :math:`H` (FFT, output on the input grid). It is accurate when
  the chirp of :math:`H` is sampled over the spectrum of the field, typically for short distances,
  :math:`\Delta x \gtrsim \lambda z / L`;
- ``"fresnel_ir"`` convolves with the sampled :math:`h` (FFT, output on the input grid), accurate for longer
  distances, :math:`\Delta x \lesssim \lambda z / L`;
- ``"fresnel_mft"`` expands the quadratic phase and evaluates one Fourier integral with a matrix Fourier
  transform, which gives the field on **any** output grid:

  .. math:: U(x, y, z) = \frac{e^{ikz}}{i\lambda z} e^{ik(x^2 + y^2)/2z} \iint U(x', y')\,
            e^{ik(x'^2 + y'^2)/2z} e^{-2\pi i (x x' + y y')/\lambda z} dx'\, dy',

  accurate when the input chirp is sampled over the support of the field, :math:`k \rho\, \Delta x / z < \pi`.

The FFT methods zero-pad the field by a factor 2 to avoid circular wrap-around. Negative distances
back-propagate.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from lumos.backend import fft
from lumos.backend._core import backend_of
from lumos.fields.grid import Grid
from lumos.propagation._common import KERNEL_CACHE, Method, axis_coordinates, crop, frequencies, pad
from lumos.propagation.fraunhofer import fraunhofer

__all__ = ["FRESNEL_IR", "FRESNEL_MFT", "FRESNEL_TF", "fresnel_ir", "fresnel_mft", "fresnel_tf"]

_PAD = 2


def _finite(z: float | None) -> float:
    if z is None or z == 0:
        raise ValueError("Fresnel propagation needs a finite non-zero distance")
    return z


def fresnel_tf(field: Any, grid_in: Grid, grid_out: Grid, wavelength: float, z: float | None, xp: Any) -> Any:
    """Fresnel propagation with the transfer function (output on the input grid)."""
    z = _finite(z)
    padded, offsets = pad(field, _PAD, xp)
    ny, nx = padded.shape[-2:]
    fx = frequencies(nx, grid_in.dx_si)
    fy = frequencies(ny, grid_in.dy_si)
    k = 2 * math.pi / wavelength

    def build() -> np.ndarray:
        f2 = fx[None, :] ** 2 + fy[:, None] ** 2
        return np.exp(1j * (k * z - math.pi * wavelength * z * f2))

    key = ("fresnel_tf", nx, ny, grid_in.dx_si, grid_in.dy_si, wavelength, z)
    transfer = KERNEL_CACHE.get(key, build, xp, padded.dtype)
    out = fft.ifft2c(fft.fft2c(padded) * transfer)
    return crop(out, offsets, grid_in.shape)


def fresnel_ir(field: Any, grid_in: Grid, grid_out: Grid, wavelength: float, z: float | None, xp: Any) -> Any:
    """Fresnel propagation by convolution with the sampled impulse response (output on the input grid)."""
    z = _finite(z)
    padded, offsets = pad(field, _PAD, xp)
    ny, nx = padded.shape[-2:]
    k = 2 * math.pi / wavelength

    def build() -> np.ndarray:
        x = axis_coordinates(nx, grid_in.dx_si, 0.0)
        y = axis_coordinates(ny, grid_in.dy_si, 0.0)
        r2 = x[None, :] ** 2 + y[:, None] ** 2
        h = np.exp(1j * (k * z + k * r2 / (2 * z))) / (1j * wavelength * z) * (grid_in.dx_si * grid_in.dy_si)
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(h)))

    key = ("fresnel_ir", nx, ny, grid_in.dx_si, grid_in.dy_si, wavelength, z)
    response = KERNEL_CACHE.get(key, build, xp, padded.dtype)
    out = fft.ifft2c(fft.fft2c(padded) * response)
    return crop(out, offsets, grid_in.shape)


def fresnel_mft(
    field: Any, grid_in: Grid, grid_out: Grid, wavelength: float, z: float | None, xp: Any
) -> Any:
    """One-step Fresnel propagation with a matrix Fourier transform (any output grid)."""
    z = _finite(z)
    k = 2 * math.pi / wavelength
    xx, yy = grid_in.coords(backend=backend_of(field))
    chirp = xp.exp(1j * k * (xx**2 + yy**2) / (2 * z))
    return fraunhofer(field * chirp, grid_in, grid_out, wavelength, z, xp)


FRESNEL_TF = Method(
    name="fresnel_tf",
    kernel=fresnel_tf,
    any_output_grid=False,
    far_field=False,
    finite_distance=True,
    description="Fresnel transfer function (FFT, short distances, output on the input grid)",
)
FRESNEL_IR = Method(
    name="fresnel_ir",
    kernel=fresnel_ir,
    any_output_grid=False,
    far_field=False,
    finite_distance=True,
    description="Fresnel impulse response (FFT, long distances, output on the input grid)",
)
FRESNEL_MFT = Method(
    name="fresnel_mft",
    kernel=fresnel_mft,
    any_output_grid=True,
    far_field=False,
    finite_distance=True,
    description="one-step Fresnel with a matrix Fourier transform (any output grid)",
)
