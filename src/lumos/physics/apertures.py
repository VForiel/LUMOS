r"""Aperture masks on sampling grids (§5.9, step P6.2).

Masks are amplitude transmissions between 0 and 1 sampled on a :class:`~lumos.fields.Grid` (in its SI unit:
metres for pupils). Edges are **anti-aliased**: each pixel is split into ``supersample × supersample``
sub-pixels and its value is the fraction of sub-pixels inside the shape, which makes the sampled area
converge to the exact area and avoids the aliasing of binary edges in the far field.

Shapes: disks, annuli, regular polygons, radial spider vanes and primaries made of hexagonal segments on a
hexagonal lattice (ring by ring around the centre).
"""

from __future__ import annotations

import math

import numpy as np

from lumos.fields.grid import Grid

__all__ = ["annulus", "disk", "hex_centers", "hex_segments", "regular_polygon", "spiders"]


def _subpixels(grid: Grid, supersample: int) -> tuple[np.ndarray, np.ndarray]:
    """SI coordinates of the sub-pixel centres, shape ``(ny·s, nx·s)``."""
    s = int(supersample)
    if s < 1:
        raise ValueError("supersample must be a positive integer")
    offsets = (np.arange(s) + 0.5) / s - 0.5
    x, y = (np.asarray(a) for a in grid.axes(backend="numpy"))
    xs = (x[:, None] + offsets[None, :] * grid.dx_si).ravel()
    ys = (y[:, None] + offsets[None, :] * grid.dy_si).ravel()
    return np.meshgrid(xs, ys, indexing="xy")


def _reduce(inside: np.ndarray, grid: Grid, supersample: int) -> np.ndarray:
    s = int(supersample)
    return inside.reshape(grid.ny, s, grid.nx, s).mean(axis=(1, 3))


def disk(
    grid: Grid, diameter: float, *, center: tuple[float, float] = (0.0, 0.0), supersample: int = 4
) -> np.ndarray:
    """Disk of ``diameter`` (SI) centred on ``center``."""
    xx, yy = _subpixels(grid, supersample)
    inside = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= (diameter / 2) ** 2
    return _reduce(inside, grid, supersample)


def annulus(grid: Grid, outer: float, inner: float, *, supersample: int = 4) -> np.ndarray:
    """Annulus between the diameters ``inner`` and ``outer`` (SI), centred on the origin."""
    xx, yy = _subpixels(grid, supersample)
    r2 = xx**2 + yy**2
    inside = (r2 <= (outer / 2) ** 2) & (r2 >= (inner / 2) ** 2) if inner > 0 else r2 <= (outer / 2) ** 2
    return _reduce(inside, grid, supersample)


def _polygon_inside(
    xx: np.ndarray, yy: np.ndarray, n: int, circumradius: float, rotation: float, center: tuple[float, float]
) -> np.ndarray:
    apothem = circumradius * math.cos(math.pi / n)
    inside = np.ones(xx.shape, dtype=bool)
    for k in range(n):
        normal = rotation + math.pi / n + 2 * math.pi * k / n
        inside &= (xx - center[0]) * math.cos(normal) + (yy - center[1]) * math.sin(normal) <= apothem
    return inside


def regular_polygon(
    grid: Grid,
    n: int,
    circumradius: float,
    *,
    rotation: float = 0.0,
    center: tuple[float, float] = (0.0, 0.0),
    supersample: int = 4,
) -> np.ndarray:
    """Regular ``n``-gon with a vertex at angle ``rotation`` from +x, inscribed in ``circumradius`` (SI)."""
    if n < 3:
        raise ValueError("a polygon has at least 3 sides")
    xx, yy = _subpixels(grid, supersample)
    return _reduce(_polygon_inside(xx, yy, n, circumradius, rotation, center), grid, supersample)


def spiders(grid: Grid, n: int, width: float, *, rotation: float = 0.0, supersample: int = 4) -> np.ndarray:
    """Coverage (1 = blocked) of ``n`` radial vanes of ``width`` (SI) starting at the centre.

    The first vane points at angle ``rotation`` from +x.
    """
    xx, yy = _subpixels(grid, supersample)
    blocked = np.zeros(xx.shape, dtype=bool)
    for k in range(int(n)):
        angle = rotation + 2 * math.pi * k / n
        along = xx * math.cos(angle) + yy * math.sin(angle)
        across = -xx * math.sin(angle) + yy * math.cos(angle)
        blocked |= (along >= 0) & (np.abs(across) <= width / 2)
    return _reduce(blocked, grid, supersample)


def hex_centers(
    pitch: float,
    rings: int,
    *,
    missing_center: bool = False,
    inner_radius: float = 0.0,
    outer_radius: float = 0.0,
    segment_circumradius: float = 0.0,
) -> list[tuple[float, float]]:
    """Centres of hexagonal segments ring by ring on a lattice of ``pitch`` (SI).

    Segments have two vertices along x; neighbours are at angles 30° + k·60°. With ``outer_radius`` (or
    ``inner_radius``), only segments lying entirely inside (outside) the circle are kept.
    """
    a1 = (pitch * math.cos(math.pi / 6), pitch * math.sin(math.pi / 6))
    a2 = (0.0, pitch)
    centers = []
    for i in range(-rings, rings + 1):
        for j in range(-rings, rings + 1):
            distance = (abs(i) + abs(j) + abs(i + j)) // 2
            if distance > rings or (missing_center and distance == 0):
                continue
            x, y = i * a1[0] + j * a2[0], i * a1[1] + j * a2[1]
            r = math.hypot(x, y)
            if outer_radius > 0 and r + segment_circumradius > outer_radius:
                continue
            if inner_radius > 0 and r - segment_circumradius < inner_radius:
                continue
            centers.append((x, y))
    return centers


def hex_segments(
    grid: Grid,
    *,
    flat_to_flat: float,
    gap: float,
    rings: int,
    missing_center: bool = False,
    inner_radius: float = 0.0,
    outer_radius: float = 0.0,
    supersample: int = 4,
) -> tuple[np.ndarray, int]:
    """Primary made of hexagonal segments of ``flat_to_flat`` width separated by ``gap`` (SI).

    Returns the mask and the number of segments.
    """
    circumradius = flat_to_flat / math.sqrt(3)
    centers = hex_centers(
        flat_to_flat + gap,
        rings,
        missing_center=missing_center,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        segment_circumradius=circumradius,
    )
    xx, yy = _subpixels(grid, supersample)
    xs, ys = xx[0], yy[:, 0]  # sorted sub-pixel axes
    inside = np.zeros(xx.shape, dtype=bool)
    reach = circumradius * 1.01
    for cx, cy in centers:
        # only the sub-pixels of the bounding box of the segment are tested
        i0, i1 = np.searchsorted(xs, [cx - reach, cx + reach])
        j0, j1 = np.searchsorted(ys, [cy - reach, cy + reach])
        if i0 >= i1 or j0 >= j1:
            continue
        box = (slice(j0, j1), slice(i0, i1))
        inside[box] |= _polygon_inside(xx[box], yy[box], 6, circumradius, 0.0, (cx, cy))
    return _reduce(inside, grid, supersample), len(centers)
