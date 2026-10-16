"""Pure data-processing functions shared by plots, processing blocks and the service layer.

The processing *elements* (blocks of a graph with :class:`~lumos.fields.DataProduct` ports) live in
:mod:`lumos.elements.processing`; the functions here work on plain arrays and grids.
"""

from lumos.processing.profiles import radial_profile

__all__ = ["radial_profile"]
