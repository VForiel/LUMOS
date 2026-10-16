"""Results of LUMOS runs: figures, data files and archives (phase 7, REQ-RES-*).

- :func:`save_figure`, :func:`save_data`, :func:`load_data`: figures as ``.png`` and ``.plt`` (PltEdit) with
  the data they show in ``.npz`` files;
- :func:`new_archive`, :class:`RunArchive`: dated archives of runs, next to the executed script or notebook.
"""

from lumos.results.archive import RunArchive, default_archive_root, new_archive
from lumos.results.figures import FIGURE_FORMATS, load_data, save_data, save_figure

__all__ = [
    "FIGURE_FORMATS",
    "RunArchive",
    "default_archive_root",
    "load_data",
    "new_archive",
    "save_data",
    "save_figure",
]
