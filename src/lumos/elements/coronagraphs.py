r"""Coronagraphs (§5.9, step P6.6): four-quadrant phase mask, vortex and Lyot coronagraphs.

A coronagraph element receives the field in a pupil plane, where a pupil of diameter ``pupil_diameter`` is
centred on the coordinate origin. It focuses the field, applies its focal-plane mask :math:`M(\mathbf{u})`
(coordinates in :math:`\lambda/D` at the wavelength of each channel), returns to the next pupil plane and
applies a circular Lyot stop of diameter ``lyot_stop`` × ``pupil_diameter``. The output is the field just
after the Lyot stop, on the input grid (:func:`lumos.physics.coronagraphy.lyot_plane`).

Masks
-----
- :class:`VortexCoronagraph`: :math:`M = e^{il\theta}` of topological charge :math:`l`; even charges reject
  all the light of an on-axis source in an unobstructed circular pupil (Mawet et al. 2005; Foo et al. 2005).
- :class:`FourQuadrantPhaseMask`: phase 0 in the quadrants where :math:`u_x u_y > 0` and :math:`\pi` where
  :math:`u_x u_y < 0` (Rouan et al. 2000), with the same perfect rejection.
- :class:`LyotCoronagraph`: opaque disk of radius ``mask_radius`` :math:`\lambda_0/D`, a physical size that
  becomes :math:`m_0\lambda_0/\lambda` in units of :math:`\lambda/D`; its edge is anti-aliased.

Phase masks are achromatic by default (e.g. geometric-phase vector vortices). With ``achromatic=False``, the
phase delays are designed for ``reference_wavelength`` and scale as :math:`\lambda_0/\lambda`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from astropy import units as u

from lumos.backend._core import get
from lumos.core.element import Element, Values
from lumos.core.param import Param, Setting
from lumos.core.port import Input, Output
from lumos.core.registry import register
from lumos.core.signal import OpticalSignal
from lumos.elements.optics import _coherent
from lumos.physics import apertures as shapes
from lumos.physics.coronagraphy import lyot_plane
from lumos.propagation.api import _map_channels

__all__ = ["Coronagraph", "FourQuadrantPhaseMask", "LyotCoronagraph", "VortexCoronagraph"]

Mask = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _sampling(ux: np.ndarray) -> float:
    return abs(float(ux[0, 1] - ux[0, 0])) if ux.shape[-1] > 1 else 1.0


class Coronagraph(Element):
    """Base class: focal-plane mask between two pupil planes, followed by a Lyot stop."""

    pupil_diameter = Param(unit=u.m, default=1 * u.m, bounds=(0 * u.m, None), inclusive=(False, True))
    lyot_stop = Param(default=0.95, bounds=(0, None), inclusive=(False, True), doc="in pupil diameters")
    reference_wavelength = Param(
        unit=u.nm,
        default=1000 * u.nm,
        bounds=(0 * u.m, None),
        inclusive=(False, True),
        doc="design wavelength",
    )
    in_ = Input(OpticalSignal)
    out = Output(OpticalSignal)

    @staticmethod
    def _mask(values: Values, wavelength: float) -> Mask:
        raise NotImplementedError

    @classmethod
    def _transfer(cls, values: Values, inputs: Any) -> dict[str, Any]:
        signal = _coherent(inputs["in"], cls.__name__)
        grid = signal.grid
        if grid.plane != "spatial":
            raise ValueError(f"{cls.__name__} needs a pupil field on a spatial grid")
        backend = get(signal.backend)
        diameter = values.pupil_diameter
        x = np.asarray(grid.axes(backend="numpy")[0], dtype=float) / diameter
        y = np.asarray(grid.axes(backend="numpy")[1], dtype=float) / diameter
        stop = shapes.disk(grid, values.lyot_stop * diameter, supersample=8)

        def apply(field: Any, wavelength: float) -> Any:
            lyot = lyot_plane(backend.to_numpy(field), x, y, cls._mask(values, wavelength))
            return backend.asarray(lyot * stop)

        return {"out": _map_channels(signal, grid, apply)}


def _phase_scale(values: Values, wavelength: float) -> float:
    return 1.0 if values.achromatic else values.reference_wavelength / wavelength


@register("lumos.coronagraphs.VortexCoronagraph")
class VortexCoronagraph(Coronagraph):
    r"""Vortex phase mask :math:`e^{il\theta}` followed by a Lyot stop."""

    charge = Setting(int, default=2, doc="topological charge l")
    achromatic = Setting(bool, default=True, doc="False: phase designed for the reference wavelength")

    @staticmethod
    def _mask(values: Values, wavelength: float) -> Mask:
        scale = _phase_scale(values, wavelength)
        charge = int(values.charge)

        def mask(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
            centre = (ux == 0) & (uy == 0)
            return np.where(centre, 0.0, np.exp(1j * charge * scale * np.arctan2(uy, ux)))

        return mask

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the field after the vortex mask and the Lyot stop."""
        return VortexCoronagraph._transfer(values, inputs)


@register("lumos.coronagraphs.FourQuadrantPhaseMask")
class FourQuadrantPhaseMask(Coronagraph):
    """Four-quadrant phase mask (π phase in two opposite quadrants) followed by a Lyot stop."""

    achromatic = Setting(bool, default=True, doc="False: π step designed for the reference wavelength")

    @staticmethod
    def _mask(values: Values, wavelength: float) -> Mask:
        step = np.exp(1j * math.pi * _phase_scale(values, wavelength))

        def mask(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
            sign = np.sign(ux * uy)  # 0 on the quadrant edges: the mean of both sides
            return np.where(sign > 0, 1.0, np.where(sign < 0, step, (1 + step) / 2))

        return mask

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the field after the quadrant mask and the Lyot stop."""
        return FourQuadrantPhaseMask._transfer(values, inputs)


@register("lumos.coronagraphs.LyotCoronagraph")
class LyotCoronagraph(Coronagraph):
    """Opaque focal-plane disk (classical Lyot coronagraph) followed by a Lyot stop."""

    mask_radius = Param(default=3.0, bounds=(0, None), doc="radius in units of reference_wavelength / D")

    @staticmethod
    def _mask(values: Values, wavelength: float) -> Mask:
        radius = values.mask_radius * values.reference_wavelength / wavelength

        def mask(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
            return np.clip(0.5 + (np.hypot(ux, uy) - radius) / _sampling(ux), 0.0, 1.0)

        return mask

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the field after the occulting disk and the Lyot stop."""
        return LyotCoronagraph._transfer(values, inputs)
