"""Discrete representations of optical fields and detected data (REQ-SIG-*).

- :class:`Grid`: rectangular sampling of a spatial or angular plane.
- :class:`Wavefront`: coherent monochromatic field sampled on a grid.
"""

from typing import Any

from lumos.fields.grid import Grid, Plane
from lumos.fields.wavefront import Wavefront

__all__ = ["Grid", "Plane", "Wavefront", "pytree_classes"]


def pytree_classes() -> tuple[type[Any], ...]:
    """Signal classes registered as JAX pytrees (each defines ``_tree_flatten`` and ``_tree_unflatten``)."""
    return (Wavefront,)
