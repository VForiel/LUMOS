"""Symbolic optics (phase 10, requirements REQ-SYM).

- :mod:`~lumos.symbolic.symbols`: canonical variables ``x, y, theta_x, theta_y, lam, t``;
- :class:`SymbolicField`: closed-form expressions of fields with their parameters;
- :mod:`~lumos.symbolic.evaluate`: evaluation on grids (``field.to_wavefront(grid)``, ``to_wavecube``);
- :mod:`~lumos.symbolic.closed_forms`: Gaussian beams through ABCD systems, tabulated transforms, fallback;
- :mod:`~lumos.symbolic.reconstruct`: from grids to expressions (projections, fits, regressions);
- :class:`SymbolicSource` and :mod:`~lumos.symbolic.integration`: symbolic fields in contexts, converted to
  numbers automatically where needed (``to_numeric``); :mod:`~lumos.symbolic.serialize`: safe JSON form.
"""

from lumos.symbolic import closed_forms, evaluate, integration, reconstruct, serialize
from lumos.symbolic.closed_forms import gaussian_beam, plane_wave, rectangular_pupil
from lumos.symbolic.evaluate import to_wavecube, to_wavefront
from lumos.symbolic.field import PLANES, SymbolicField
from lumos.symbolic.integration import to_numeric
from lumos.symbolic.source import SymbolicSource
from lumos.symbolic.symbols import CANONICAL, lam, r, rho, t, theta_x, theta_y, x, y

__all__ = [
    "CANONICAL",
    "PLANES",
    "SymbolicField",
    "SymbolicSource",
    "closed_forms",
    "evaluate",
    "gaussian_beam",
    "integration",
    "lam",
    "plane_wave",
    "r",
    "reconstruct",
    "rectangular_pupil",
    "rho",
    "serialize",
    "t",
    "theta_x",
    "theta_y",
    "to_numeric",
    "to_wavecube",
    "to_wavefront",
    "x",
    "y",
]
