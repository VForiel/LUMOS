"""Third-party code shipped with LUMOS.

Copies of the parts of VFSciTools (https://github.com/VForiel/VFSciTools, MIT license, same author, commit
``b0f862e``) that LUMOS uses, so that LUMOS installs without it while VFSciTools is not on PyPI:

- :mod:`lumos._vendor.pltedit`: ``save`` and ``load`` of PltEdit ``.plt`` figures (same file format, the
  files open in the PltEdit application);
- :mod:`lumos._vendor.vfst_archive`: dated run archives (``vfst.archive.new``).

The files are kept as they are upstream (only their imports are changed); update them by copying again.
"""
