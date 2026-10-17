"""Canonical SymPy variables of symbolic fields (step P10.1, REQ-SYM-1).

Coordinates and wavelengths are in unscaled SI units inside expressions: ``x`` and ``y`` in metres (spatial
planes), ``theta_x`` and ``theta_y`` in radians (angular planes), ``lam`` (λ) in metres, ``t`` in seconds.
They are real (λ positive), which lets SymPy simplify moduli and phases. ``r`` and ``rho`` are the radii of
the spatial and angular planes.
"""

from __future__ import annotations

import sympy as sp

__all__ = ["CANONICAL", "lam", "r", "rho", "t", "theta_x", "theta_y", "x", "y"]

x = sp.Symbol("x", real=True)
y = sp.Symbol("y", real=True)
theta_x = sp.Symbol("theta_x", real=True)
theta_y = sp.Symbol("theta_y", real=True)
lam = sp.Symbol("lambda", positive=True)
t = sp.Symbol("t", real=True)

#: Radius in a spatial plane.
r = sp.sqrt(x**2 + y**2)
#: Radius in an angular plane.
rho = sp.sqrt(theta_x**2 + theta_y**2)

#: Canonical variables of each plane.
CANONICAL: dict[str, frozenset[sp.Symbol]] = {
    "spatial": frozenset({x, y, lam, t}),
    "angular": frozenset({theta_x, theta_y, lam, t}),
}
