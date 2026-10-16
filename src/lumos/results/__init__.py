"""Results of LUMOS runs: figures, data files and archives (phase 7, REQ-RES-*).

- :func:`save_figure`, :func:`save_data`, :func:`load_data`: figures as ``.png`` and ``.plt`` (PltEdit) with
  the data they show in ``.npz`` files;
"""

from lumos.results.figures import FIGURE_FORMATS, load_data, save_data, save_figure

__all__ = ["FIGURE_FORMATS", "load_data", "save_data", "save_figure"]
