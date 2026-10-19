"""Data processing: reduction, photometry, interferometry, detection statistics and calibration (phase 11).

Functions work on :class:`~lumos.fields.DataProduct` objects (or plain arrays and grids); the processing
*elements* defined next to them wrap the functions with ``DataProduct`` ports, so that pipelines are built in
the same graph as the optics (REQ-PIPE-1).

- :mod:`~lumos.processing.reduction`: dark, flat field, stacking, cropping, rebinning, resampling;
- :mod:`~lumos.processing.photometry`: aperture photometry, CCD signal-to-noise ratio, contrast curves,
  radial profiles;
- :mod:`~lumos.processing.interferometry`: null depths, kernel observables, transmission maps;
- :mod:`~lumos.processing.detection`: test statistics, ROC curves, contrast limits, detection capability;
- :mod:`~lumos.processing.calibration`: phase fitting, slanted-edge MTF, PSF measurements;
- :func:`script`: user functions as elements with declared ports (ADR-0013);
- :class:`Recorder`: a block whose signal is recorded at every point of a study (P14.20);
- :func:`radial_profile`: azimuthal averages of maps sampled on a grid.
"""

from lumos.processing import calibration, detection, interferometry, photometry, reduction
from lumos.processing.profiles import radial_profile
from lumos.processing.recording import Recorder
from lumos.processing.script import script

__all__ = [
    "Recorder",
    "calibration",
    "detection",
    "interferometry",
    "photometry",
    "radial_profile",
    "reduction",
    "script",
]
