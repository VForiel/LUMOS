"""Symbolic optics (phase 10, REQ-SYM-*).

- :mod:`~lumos.symbolic.symbols`: canonical variables ``x, y, theta_x, theta_y, lam, t``;
- :class:`SymbolicField`: closed-form expressions of fields with their parameters;
- :mod:`~lumos.symbolic.evaluate`: evaluation on grids (``field.to_wavefront(grid)``, ``to_wavecube``);
- :mod:`~lumos.symbolic.closed_forms`: Gaussian beams through ABCD systems, tabulated transforms, fallback.
"""

from lumos.symbolic import closed_forms, evaluate
from lumos.symbolic.closed_forms import gaussian_beam, plane_wave, rectangular_pupil
from lumos.symbolic.evaluate import to_wavecube, to_wavefront
from lumos.symbolic.field import PLANES, SymbolicField
from lumos.symbolic.symbols import CANONICAL, lam, r, rho, t, theta_x, theta_y, x, y

__all__ = [
    "CANONICAL",
    "PLANES",
    "SymbolicField",
    "closed_forms",
    "evaluate",
    "gaussian_beam",
    "lam",
    "plane_wave",
    "r",
    "rectangular_pupil",
    "rho",
    "t",
    "theta_x",
    "theta_y",
    "to_wavecube",
    "to_wavefront",
    "x",
    "y",
]
