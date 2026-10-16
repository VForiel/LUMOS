"""Azimuthal averages of maps sampled on a :class:`~lumos.fields.Grid`."""

from __future__ import annotations

from typing import Any

import numpy as np
from astropy import units as u

from lumos.backend import to_numpy
from lumos.fields.grid import Grid
from lumos.units import require_unit, to_si

__all__ = ["radial_profile"]


def radial_profile(
    image: Any,
    grid: Grid,
    *,
    center: tuple[u.Quantity, u.Quantity] | None = None,
    bin_width: u.Quantity | None = None,
) -> tuple[u.Quantity, np.ndarray]:
    r"""Average ``image`` over rings centred on ``center``.

    Pixel :math:`(i, j)` belongs to ring :math:`k = \lfloor r_{ij} / w \rfloor`, where :math:`r_{ij}` is
    the distance from its centre to ``center`` and :math:`w` the ring width; the profile is the mean of the
    pixels of each ring. Rings beyond the farthest pixel are not returned; rings holding no pixel (possible
    near the centre for widths smaller than a pixel) are ``NaN``.

    Parameters
    ----------
    image
        Map of shape ``grid.shape`` (``(ny, nx)``), an array of any backend.
    grid
        Sampling of the map.
    center
        ``(x, y)`` centre of the rings, in the plane of the grid; defaults to the origin ``(0, 0)``.
    bin_width
        Width of the rings; defaults to the smaller pixel step.

    Returns
    -------
    radii : Quantity
        Radii of the middle of the rings, ``(k + 1/2) w``, in the unit of the grid.
    profile : numpy.ndarray
        Mean value of ``image`` in each ring.
    """
    values = np.asarray(to_numpy(image))
    if values.shape != grid.shape:
        raise ValueError(f"image shape {values.shape} does not match the grid shape {grid.shape}")
    unit = grid.unit
    cx, cy = (0.0, 0.0)
    if center is not None:
        cx, cy = (float(to_si(require_unit(c, unit, name="center"))) for c in center)
    width = min(grid.dx_si, grid.dy_si)
    if bin_width is not None:
        width = float(to_si(require_unit(bin_width, unit, name="bin_width")))
        if not width > 0:
            raise ValueError("bin_width must be positive")
    x, y = (np.asarray(c) for c in grid.coords(backend="numpy"))
    r = np.hypot(x - cx, y - cy)
    rings = np.floor(r / width).astype(int).ravel()
    n = int(rings.max()) + 1
    counts = np.bincount(rings, minlength=n)
    sums = np.bincount(rings, weights=values.ravel().astype(float), minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    scale = float(to_si(1.0 * unit))
    radii = ((np.arange(n) + 0.5) * width / scale) * unit
    return radii, profile
