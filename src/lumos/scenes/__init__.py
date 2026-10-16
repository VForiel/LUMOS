"""Scenes: descriptions of what is observed (phase 8, REQ-SCN-*).

- :class:`Scene`: a group of components in a celestial, terrestrial or laboratory frame, with the outputs
  ``sources`` (point sources) and ``radiance`` (diffuse radiance on the angular grid of the scene);
- :mod:`~lumos.scenes.diffuse`: parametric diffuse components (disks, Gaussians, Sérsic profiles, inclined
  rings, backgrounds, zodiacal light);
- :class:`~lumos.scenes.mapped.MappedComponent`: radiance maps from FITS files, arrays and images.

Importing this package registers its element types for ``.lumos`` files.
"""

from lumos.scenes import diffuse, mapped
from lumos.scenes.mapped import MappedComponent
from lumos.scenes.scene import FRAMES, RADIANCE_UNIT, Scene, SceneCombiner, scene_grid

__all__ = [
    "FRAMES",
    "RADIANCE_UNIT",
    "MappedComponent",
    "Scene",
    "SceneCombiner",
    "diffuse",
    "mapped",
    "scene_grid",
]
