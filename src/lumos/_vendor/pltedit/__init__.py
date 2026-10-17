"""PltEdit: save and load matplotlib figures as ``.plt`` files (vendored from VFSciTools, MIT license).

The ``.plt`` format is a NumPy archive holding the serialised figure and metadata (creation date, Python,
matplotlib and PltEdit versions). Loading unpickles the figure: open only trusted files (ADR-0010).
"""

from lumos._vendor.pltedit._io import load, save
from lumos._vendor.pltedit._style import set_style
from lumos._vendor.pltedit._version import __version__

__all__ = ["__version__", "load", "save", "set_style"]
