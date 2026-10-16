"""Scenes: descriptions of what is observed (phase 8, REQ-SCN-*).

- :class:`Scene`: a group of components in a celestial, terrestrial or laboratory frame, with the outputs
  ``sources`` (point sources) and ``radiance`` (diffuse radiance on the angular grid of the scene).
"""

from lumos.scenes.scene import FRAMES, RADIANCE_UNIT, Scene, SceneCombiner, scene_grid

__all__ = ["FRAMES", "RADIANCE_UNIT", "Scene", "SceneCombiner", "scene_grid"]
