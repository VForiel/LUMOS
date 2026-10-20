r"""Focal-plane masks of coronagraphs: pupil → focal plane → mask → Lyot plane.

With pupil coordinates :math:`\tilde{\mathbf{x}} = \mathbf{x}/D` and focal coordinates :math:`\mathbf{u}` in
units of :math:`\lambda/D`, the field after a focal-plane mask :math:`M(\mathbf{u})` is, back in the pupil
(Lyot) plane,

.. math::
    P'(\tilde{\mathbf{x}}) = \int M(\mathbf{u})\, E(\mathbf{u})\,
    e^{2i\pi\,\tilde{\mathbf{x}}\cdot\mathbf{u}}\,d^2u, \qquad
    E(\mathbf{u}) = \int P(\tilde{\mathbf{x}})\, e^{-2i\pi\,\tilde{\mathbf{x}}\cdot\mathbf{u}}\,d^2\tilde{x}.

Masks such as vortices and quadrant phase masks cover the whole focal plane and are singular at its centre,
so the integral is split with smooth radial windows :math:`W_k` (1 inside :math:`a_k`, 0 outside
:math:`b_k`, raised cosine between): a zero-padded FFT computes the part :math:`M (1 - W_0)` of the whole
plane, and matrix Fourier transforms (Soummer et al. 2007, Opt. Express 15, 15935) of finer and finer
sampling compute the parts :math:`M (W_{k-1} - W_k)` and :math:`M W_{K}` near the axis. This multi-scale
scheme follows the one used for vortex coronagraphs by Krist et al. (2012, Proc. SPIE 8151). The residual of
ideal masks is then set by the sampling of the pupil (it decreases as :math:`1/N^2` for :math:`N` samples
across the pupil), not by the focal-plane computation.

A batch of fields (the sources of an extended scene) is transformed a few fields at a time, which bounds the
memory of the zero-padded planes. The coarse FFT skips the transforms of the rows and columns that only hold
padding (the forward transform along :math:`x` needs the rows of the pupil only, the inverse one along
:math:`x` the rows kept in the pupil only), and the matrix Fourier transforms of the batch are computed as a
few large matrix products.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import scipy.fft

from lumos._config import config

__all__ = ["DEFAULT_LEVELS", "lyot_plane", "window"]

#: (inner radius, outer radius, sampling) of the windows of the fine levels, in λ/D.
DEFAULT_LEVELS: tuple[tuple[float, float, float], ...] = (
    (8.0, 12.0, 1 / 16),
    (0.8, 1.2, 1 / 160),
    (0.08, 0.12, 1 / 1600),
)

Mask = Callable[[np.ndarray, np.ndarray], np.ndarray]

#: Fields transformed together: bounds the memory of the zero-padded planes (16 fields of 512² are 64 MB).
CHUNK = 16


def window(r: np.ndarray, inner: float, outer: float) -> np.ndarray:
    """Radial window: 1 for ``r <= inner``, 0 for ``r >= outer``, a raised cosine between."""
    t = np.clip((r - inner) / (outer - inner), 0.0, 1.0)
    return 0.5 * (1 + np.cos(math.pi * t))


def _centered_axis(n: int, step: float) -> np.ndarray:
    return (np.arange(n) - n // 2) * step


def lyot_plane(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mask: Mask,
    *,
    pad: int = 4,
    levels: tuple[tuple[float, float, float], ...] = DEFAULT_LEVELS,
) -> np.ndarray:
    """Return the field in the Lyot plane, on the same samples as ``field``.

    Parameters
    ----------
    field
        Complex pupil field, shape ``(..., ny, nx)``, on the uniform pupil coordinates ``x`` and ``y``, in
        units of the pupil diameter (``x / D``).
    mask
        Focal-plane transmission ``mask(ux, uy)`` for coordinates in λ/D (arrays of the same shape).
    pad
        Zero-padding factor of the coarse FFT (its sampling is ``1 / (pad n dx)`` λ/D).
    levels
        ``(inner, outer, sampling)`` of the windows and of the finer levels, from the outside in.
    """
    field = np.asarray(field, dtype=complex)
    ny, nx = field.shape[-2:]
    dx = float(x[1] - x[0]) if nx > 1 else 1.0
    dy = float(y[1] - y[0]) if ny > 1 else dx
    if not math.isclose(dx, dy, rel_tol=1e-9):
        raise ValueError("coronagraph masks need square pupil pixels")
    batch = field.reshape(-1, ny, nx)

    # coarse level: whole focal plane, zero-padded FFT, mask × (1 - W0)
    n_pad = int(pad) * max(nx, ny)
    oy, ox = n_pad // 2 - ny // 2, n_pad // 2 - nx // 2
    du = 1.0 / (n_pad * dx)
    u = _centered_axis(n_pad, du)
    ux, uy = np.meshgrid(u, u)
    weight = mask(ux, uy)
    if levels:
        weight = weight * (1 - window(np.hypot(ux, uy), levels[0][0], levels[0][1]))
    # the centred transform (origin of x at x[nx // 2], u = 0 at index n_pad // 2) in FFT order: the weight
    # moves with the spectrum, and the pupil sits at the rows and columns where the centred grid puts them
    weight = np.fft.ifftshift(weight)
    where = np.empty(n_pad, dtype=int)
    where[np.fft.ifftshift(np.arange(n_pad))] = np.arange(n_pad)
    rows, cols = where[oy : oy + ny], where[ox : ox + nx]

    # fine levels: matrix Fourier transforms over the windowed central regions
    x0, y0 = float(x[nx // 2]), float(y[ny // 2])
    fine = []
    for index, (_inner, outer, sampling) in enumerate(levels):
        n_u = 2 * math.ceil(outer / sampling) + 1
        u = _centered_axis(n_u, sampling)
        ux, uy = np.meshgrid(u, u)
        r = np.hypot(ux, uy)
        level_weight = window(r, levels[index][0], levels[index][1])
        if index + 1 < len(levels):
            level_weight = level_weight - window(r, levels[index + 1][0], levels[index + 1][1])
        level_weight = level_weight * mask(ux, uy) * (dx * dy * sampling**2)
        ax = np.exp(-2j * math.pi * np.outer(u, x - x0))  # (n_u, nx)
        ay = np.exp(-2j * math.pi * np.outer(u, y - y0))  # (n_u, ny)
        fine.append((ax, ay, level_weight))

    result = np.empty_like(batch)
    workers = config.fft_workers
    for first in range(0, len(batch), CHUNK):
        part = batch[first : first + CHUNK]
        k = len(part)
        rows_x = np.zeros((k, ny, n_pad), dtype=complex)
        rows_x[:, :, cols] = part
        plane = np.zeros((k, n_pad, n_pad), dtype=complex)
        plane[:, rows, :] = scipy.fft.fft(rows_x, axis=-1, workers=workers, overwrite_x=True)
        plane = scipy.fft.fft(plane, axis=-2, workers=workers, overwrite_x=True)
        plane *= weight
        kept = np.asarray(scipy.fft.ifft(plane, axis=-2, workers=workers, overwrite_x=True))[:, rows, :]
        out = np.asarray(scipy.fft.ifft(kept, axis=-1, workers=workers, overwrite_x=True))[:, :, cols]
        for ax, ay, level_weight in fine:
            out += _mft_pair(part, ax, ay, level_weight)
        result[first : first + k] = out
    return result.reshape(field.shape)


def _mft_pair(part: np.ndarray, ax: np.ndarray, ay: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Return ``Ayᴴ ((Ay F Axᵀ) ∘ weight) Ax*`` for every field ``F`` of ``part`` (large matrix products)."""
    k, ny, nx = part.shape
    n_u = ax.shape[0]
    along_x = (part.reshape(k * ny, nx) @ ax.T).reshape(k, ny, n_u)  # F Axᵀ
    spectrum = ay @ along_x.transpose(1, 0, 2).reshape(ny, k * n_u)  # Ay F Axᵀ, columns by field
    spectrum = spectrum.reshape(n_u, k, n_u).transpose(1, 0, 2) * weight  # (k, n_u, n_u)
    back_x = (spectrum.reshape(k * n_u, n_u) @ np.conj(ax)).reshape(k, n_u, nx)  # (…) Ax*
    back = np.conj(ay).T @ back_x.transpose(1, 0, 2).reshape(n_u, k * nx)  # Ayᴴ (…)
    return back.reshape(ny, k, nx).transpose(1, 0, 2)
