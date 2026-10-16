"""Discrete representations of optical fields and detected data (``REQ-SIG-*``).

- :class:`Grid`: rectangular sampling of a spatial or angular plane.
- :class:`Wavefront`: coherent monochromatic field sampled on a grid, scalar or polarised.
- :class:`Wavecube`: stack of wavefronts on a spectral axis, integrated incoherently.
- :class:`IncoherentBundle`: mutually incoherent modes (point sources, natural light...).
- :class:`RadianceField`: spectral radiance maps of incoherent scenes.
- :class:`ModalField`: modal amplitudes of guided channels; :mod:`~lumos.fields.smatrix` scattering matrices.
- :class:`SourceList`: unresolved sources at infinity with their photon spectra.
- :class:`DataProduct`: detected data with axes, mask, uncertainties; FITS, NPZ, HDF5.
- :mod:`~lumos.fields.jones`: Jones vectors and matrices, Stokes parameters.
- :func:`spectral_samples`: quadrature of a spectral band.
"""

from typing import Any

from lumos.fields import jones, smatrix
from lumos.fields.dataproduct import DataProduct
from lumos.fields.grid import Grid, Plane
from lumos.fields.incoherent import IncoherentBundle
from lumos.fields.modal import ModalField
from lumos.fields.radiance import RadianceField
from lumos.fields.sources import PHOTON_FLUX_DENSITY, SourceList
from lumos.fields.spectral import spectral_samples
from lumos.fields.wavecube import Wavecube
from lumos.fields.wavefront import Wavefront

__all__ = [
    "PHOTON_FLUX_DENSITY",
    "DataProduct",
    "Grid",
    "IncoherentBundle",
    "ModalField",
    "Plane",
    "RadianceField",
    "SourceList",
    "Wavecube",
    "Wavefront",
    "jones",
    "pytree_classes",
    "smatrix",
    "spectral_samples",
]


def pytree_classes() -> tuple[type[Any], ...]:
    """Signal classes registered as JAX pytrees (each defines ``_tree_flatten`` and ``_tree_unflatten``)."""
    return (Wavefront, Wavecube, IncoherentBundle, RadianceField, ModalField)
