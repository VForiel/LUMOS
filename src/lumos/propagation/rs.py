r"""Direct Rayleigh-Sommerfeld propagation, the reference method (REQ-PROP-1, step P4.2).

The first Rayleigh-Sommerfeld solution gives the field at :math:`(x, y, z)` from the field at :math:`z = 0`
(Goodman, *Introduction to Fourier Optics*, 3rd ed., eq. 3-42):

.. math:: U(x, y, z) = \iint U(x', y', 0)\, h(r)\, dx'\, dy', \qquad
          h(r) = \frac{z}{2\pi r^2}\left(\frac{1}{r} - ik\right) e^{ikr},

with :math:`r = \sqrt{(x - x')^2 + (y - y')^2 + z^2}`. The integral is evaluated as a direct sum over the
input pixels for every output point, which is exact for the sampled field (no paraxial or far-field
approximation) and works between any grids. Its cost is :math:`N_\mathrm{in} N_\mathrm{out}`, so it is used
for validation and for small grids; the output points are processed in chunks to bound the memory.

The sum is accurate when the phase of :math:`h` changes by much less than :math:`2\pi` across one input pixel,
i.e. :math:`k\,\rho\,\Delta x / r \ll 2\pi` where :math:`\rho` is the lateral distance.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from lumos.fields.grid import Grid
from lumos.propagation._common import Method, axis_coordinates, complex_dtype

__all__ = ["METHOD", "rayleigh_sommerfeld"]

_CHUNK_BYTES = 64 * 2**20


def rayleigh_sommerfeld(
    field: Any, grid_in: Grid, grid_out: Grid, wavelength: float, z: float | None, xp: Any
) -> Any:
    """Propagate ``field`` by ``z > 0`` with the direct Rayleigh-Sommerfeld sum (see the module docstring)."""
    if z is None or not z > 0:
        raise ValueError("the Rayleigh-Sommerfeld method needs a positive finite distance")
    k = 2 * math.pi / wavelength
    dtype = complex_dtype(field)
    xi = axis_coordinates(grid_in.nx, grid_in.dx_si, grid_in.x0_si)
    yi = axis_coordinates(grid_in.ny, grid_in.dy_si, grid_in.y0_si)
    xo = axis_coordinates(grid_out.nx, grid_out.dx_si, grid_out.x0_si)
    yo = axis_coordinates(grid_out.ny, grid_out.dy_si, grid_out.y0_si)
    xin = xp.asarray(np.broadcast_to(xi[None, :], grid_in.shape).ravel())
    yin = xp.asarray(np.broadcast_to(yi[:, None], grid_in.shape).ravel())
    xout = np.broadcast_to(xo[None, :], grid_out.shape).ravel()
    yout = np.broadcast_to(yo[:, None], grid_out.shape).ravel()
    leading = field.shape[:-2]
    flat = xp.reshape(field, (-1, grid_in.size))  # (batch, n_in)
    area = grid_in.dx_si * grid_in.dy_si
    chunk = max(1, _CHUNK_BYTES // (16 * grid_in.size))
    pieces = []
    for start in range(0, grid_out.size, chunk):
        sx = xp.asarray(xout[start : start + chunk])[:, None]
        sy = xp.asarray(yout[start : start + chunk])[:, None]
        r = xp.sqrt((sx - xin[None, :]) ** 2 + (sy - yin[None, :]) ** 2 + z**2)
        kernel = (z / (2 * math.pi) / r**2) * (1 / r - 1j * k) * xp.exp(1j * k * r) * area
        pieces.append(xp.matmul(flat, xp.matrix_transpose(xp.astype(kernel, dtype))))
    out = xp.concat(pieces, axis=-1)
    return xp.reshape(out, (*leading, *grid_out.shape))


METHOD = Method(
    name="rs",
    kernel=rayleigh_sommerfeld,
    any_output_grid=True,
    far_field=False,
    finite_distance=True,
    description="direct Rayleigh-Sommerfeld sum (exact, cost N_in × N_out)",
)
