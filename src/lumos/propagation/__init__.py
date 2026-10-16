"""Numerical propagation of optical fields between arbitrary planes and grids (REQ-PROP-*).

- :func:`propagate`: free-space propagation over a distance, or to the far field;
- :func:`focus`: focal plane of a thin lens;
- :func:`methods`: the available methods; ``method="auto"`` lets the planner choose and log one.
"""

from lumos.propagation import asm, fraunhofer, fresnel, rs
from lumos.propagation.api import focus, methods, propagate, register_method

register_method(rs.METHOD)
register_method(fraunhofer.METHOD)
register_method(fresnel.FRESNEL_TF)
register_method(fresnel.FRESNEL_IR)
register_method(fresnel.FRESNEL_MFT)
register_method(asm.METHOD)

__all__ = ["focus", "methods", "propagate", "register_method"]
