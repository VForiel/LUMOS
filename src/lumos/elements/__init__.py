"""Library of elements (§5.9).

Sources, apertures, optics, telescopes, atmosphere, coronagraphs, guided and integrated optics,
detectors.

Importing this package registers every built-in element type in :data:`lumos.core.registry.registry`, so
that ``.lumos`` files can refer to them by their stable identifiers (``lumos.<family>.<Name>``).
"""

from lumos.elements import sources

__all__ = ["sources"]
