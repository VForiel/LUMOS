r"""Fraunhofer propagation with matrix Fourier transforms (REQ-PROP-2, step P4.3).

Far field
---------
At infinite distance the field in the direction of direction cosines :math:`(\alpha, \beta)` is the Fourier
transform of the pupil field at the spatial frequencies :math:`(\alpha/\lambda, \beta/\lambda)` (Goodman,
*Introduction to Fourier Optics*, §4.2):

.. math:: U_\infty(\alpha, \beta) = \frac{1}{i\lambda} \iint U(x, y)\,
          e^{-2\pi i (\alpha x + \beta y)/\lambda}\, dx\, dy ,

normalised so that :math:`\iint |U_\infty|^2 d\alpha\, d\beta` is the power (Parseval).

Finite distance and focal plane
-------------------------------
At a distance :math:`z` in the Fraunhofer regime (Fresnel number :math:`a^2/\lambda z \ll 1`), or in the focal
plane of a thin lens of focal length :math:`z` placed in the input plane (where the lens cancels the input
quadratic phase exactly),

.. math:: U(x, y) = \frac{e^{ikz}}{i\lambda z} e^{ik(x^2 + y^2)/2z}
          \iint U(x', y')\, e^{-2\pi i (x x' + y y')/\lambda z}\, dx'\, dy' .

The transforms are evaluated as matrix Fourier transforms (Soummer et al. 2007, Opt. Express 15, 15935):
two matrix products with :math:`e^{-2\pi i u x}` matrices, so the output sampling and centre are arbitrary and
the cost is :math:`O(N^2 M + N M^2)` for :math:`N^2` input and :math:`M^2` output pixels.
"""

from __future__ import annotations

import math
from typing import Any

from lumos.backend._core import backend_of
from lumos.fields.grid import Grid
from lumos.propagation._common import Method, axis_coordinates, mft

__all__ = ["METHOD", "fraunhofer"]


def fraunhofer(field: Any, grid_in: Grid, grid_out: Grid, wavelength: float, z: float | None, xp: Any) -> Any:
    """Far field (``z is None``, angular output) or Fraunhofer field at distance ``z`` (spatial output)."""
    xo = axis_coordinates(grid_out.nx, grid_out.dx_si, grid_out.x0_si)
    yo = axis_coordinates(grid_out.ny, grid_out.dy_si, grid_out.y0_si)
    if z is None:
        if grid_out.plane != "angular":
            raise ValueError("the far field is computed on an angular grid")
        return mft(field, xo / wavelength, yo / wavelength, grid_in, -1.0, xp) / (1j * wavelength)
    if grid_out.plane != "spatial":
        raise ValueError("a finite distance needs a spatial output grid")
    k = 2 * math.pi / wavelength
    spectrum = mft(field, xo / (wavelength * z), yo / (wavelength * z), grid_in, -1.0, xp)
    xx, yy = grid_out.coords(backend=backend_of(field))
    chirp = xp.exp(1j * (k * z + k * (xx**2 + yy**2) / (2 * z)))
    return chirp * spectrum / (1j * wavelength * z)


METHOD = Method(
    name="fraunhofer",
    kernel=fraunhofer,
    any_output_grid=True,
    far_field=True,
    finite_distance=True,
    description="Fraunhofer far field / focal plane by matrix Fourier transform (any output sampling)",
)
