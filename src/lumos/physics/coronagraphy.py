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
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

__all__ = ["DEFAULT_LEVELS", "lyot_plane", "window"]

#: (inner radius, outer radius, sampling) of the windows of the fine levels, in λ/D.
DEFAULT_LEVELS: tuple[tuple[float, float, float], ...] = (
    (8.0, 12.0, 1 / 16),
    (0.8, 1.2, 1 / 160),
    (0.08, 0.12, 1 / 1600),
)

Mask = Callable[[np.ndarray, np.ndarray], np.ndarray]


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

    # coarse level: whole focal plane, zero-padded FFT, mask × (1 - W0)
    n_pad = int(pad) * max(nx, ny)
    padded = np.zeros((*field.shape[:-2], n_pad, n_pad), dtype=complex)
    oy, ox = n_pad // 2 - ny // 2, n_pad // 2 - nx // 2
    padded[..., oy : oy + ny, ox : ox + nx] = field
    du = 1.0 / (n_pad * dx)
    u = _centered_axis(n_pad, du)
    ux, uy = np.meshgrid(u, u)
    weight = mask(ux, uy)
    if levels:
        weight = weight * (1 - window(np.hypot(ux, uy), levels[0][0], levels[0][1]))
    # phase ramps place the origin of x at the pupil coordinates x[nx // 2] and u = 0 at index n_pad // 2
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded, axes=(-2, -1))), axes=(-2, -1))
    back = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum * weight, axes=(-2, -1))), axes=(-2, -1))
    result = back[..., oy : oy + ny, ox : ox + nx]
    # the centred FFT puts the origin of the transform at pixel n // 2: the fine levels do the same
    x0, y0 = float(x[nx // 2]), float(y[ny // 2])

    # fine levels: matrix Fourier transforms over the windowed central regions
    for index, (_inner, outer, sampling) in enumerate(levels):
        n_u = 2 * math.ceil(outer / sampling) + 1
        u = _centered_axis(n_u, sampling)
        ux, uy = np.meshgrid(u, u)
        r = np.hypot(ux, uy)
        weight = window(r, levels[index][0], levels[index][1])
        if index + 1 < len(levels):
            weight = weight - window(r, levels[index + 1][0], levels[index + 1][1])
        weight = weight * mask(ux, uy)
        ax = np.exp(-2j * math.pi * np.outer(u, x - x0))  # (n_u, nx)
        ay = np.exp(-2j * math.pi * np.outer(u, y - y0))  # (n_u, ny)
        spectrum = ay @ field @ ax.T * dx * dy
        result = result + np.conj(ay).T @ (spectrum * weight) @ np.conj(ax) * sampling**2
    return result
