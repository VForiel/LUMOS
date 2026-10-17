"""Symbolic optics (phase 10, REQ-SYM-*).

- :mod:`~lumos.symbolic.symbols`: canonical variables ``x, y, theta_x, theta_y, lam, t``;
- :class:`SymbolicField`: closed-form expressions of fields with their parameters;
- :mod:`~lumos.symbolic.evaluate`: evaluation on grids (``field.to_wavefront(grid)``, ``to_wavecube``).
"""

from lumos.symbolic import evaluate
from lumos.symbolic.evaluate import to_wavecube, to_wavefront
from lumos.symbolic.field import PLANES, SymbolicField
from lumos.symbolic.symbols import CANONICAL, lam, r, rho, t, theta_x, theta_y, x, y

__all__ = [
    "CANONICAL",
    "PLANES",
    "SymbolicField",
    "evaluate",
    "lam",
    "r",
    "rho",
    "t",
    "theta_x",
    "theta_y",
    "to_wavecube",
    "to_wavefront",
    "x",
    "y",
]
