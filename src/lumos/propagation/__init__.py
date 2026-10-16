"""Numerical propagation of optical fields between arbitrary planes and grids (REQ-PROP-*).

- :func:`propagate`: free-space propagation over a distance, or to the far field;
- :func:`focus`: focal plane of a thin lens;
- :func:`methods`: the available methods; ``method="auto"`` lets the planner choose and log one.
"""

from lumos.propagation import rs
from lumos.propagation.api import focus, methods, propagate, register_method

register_method(rs.METHOD)

__all__ = ["focus", "methods", "propagate", "register_method"]
