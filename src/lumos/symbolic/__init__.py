"""Symbolic optics (phase 10, REQ-SYM-*).

- :mod:`~lumos.symbolic.symbols`: canonical variables ``x, y, theta_x, theta_y, lam, t``;
- :class:`SymbolicField`: closed-form expressions of fields with their parameters.
"""

from lumos.symbolic.field import PLANES, SymbolicField
from lumos.symbolic.symbols import CANONICAL, lam, r, rho, t, theta_x, theta_y, x, y

__all__ = ["CANONICAL", "PLANES", "SymbolicField", "lam", "r", "rho", "t", "theta_x", "theta_y", "x", "y"]
