"""Discrete representations of optical fields and detected data (REQ-SIG-*).

- :class:`Grid`: rectangular sampling of a spatial or angular plane.
- :class:`Wavefront`: coherent monochromatic field sampled on a grid, scalar or polarised.
- :class:`Wavecube`: stack of wavefronts on a spectral axis, integrated incoherently.
- :class:`IncoherentBundle`: mutually incoherent modes (point sources, natural light...).
- :class:`RadianceField`: spectral radiance maps of incoherent scenes.
- :mod:`~lumos.fields.jones`: Jones vectors and matrices, Stokes parameters.
- :func:`spectral_samples`: quadrature of a spectral band.
"""

from typing import Any

from lumos.fields import jones
from lumos.fields.grid import Grid, Plane
from lumos.fields.incoherent import IncoherentBundle
from lumos.fields.radiance import RadianceField
from lumos.fields.spectral import spectral_samples
from lumos.fields.wavecube import Wavecube
from lumos.fields.wavefront import Wavefront

__all__ = [
    "Grid",
    "IncoherentBundle",
    "Plane",
    "RadianceField",
    "Wavecube",
    "Wavefront",
    "jones",
    "pytree_classes",
    "spectral_samples",
]


def pytree_classes() -> tuple[type[Any], ...]:
    """Signal classes registered as JAX pytrees (each defines ``_tree_flatten`` and ``_tree_unflatten``)."""
    return (Wavefront, Wavecube, IncoherentBundle, RadianceField)
