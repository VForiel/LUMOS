"""Data processing: reduction, photometry, interferometry, detection statistics and calibration (phase 11).

Functions work on :class:`~lumos.fields.DataProduct` objects (or plain arrays and grids); the processing
*elements* defined next to them wrap the functions with ``DataProduct`` ports, so that pipelines are built in
the same graph as the optics (REQ-PIPE-1).

- :mod:`~lumos.processing.reduction`: dark, flat field, stacking, cropping, rebinning, resampling;
- :mod:`~lumos.processing.photometry`: aperture photometry, CCD signal-to-noise ratio, contrast curves,
  radial profiles;
- :mod:`~lumos.processing.interferometry`: null depths, kernel observables, transmission maps;
- :func:`radial_profile`: azimuthal averages of maps sampled on a grid.
"""

from lumos.processing import interferometry, photometry, reduction
from lumos.processing.profiles import radial_profile

__all__ = ["interferometry", "photometry", "radial_profile", "reduction"]
