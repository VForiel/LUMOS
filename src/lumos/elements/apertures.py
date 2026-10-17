"""Apertures and pupils (§5.9, step P6.2).

An aperture element multiplies the optical signal it receives by its amplitude transmission, computed on the
grid of the signal with anti-aliased edges (:mod:`lumos.physics.apertures`): wavefronts, wavecubes and
incoherent bundles are multiplied by the mask, radiance fields by its square (a power transmission).

Presets give approximate geometric models of telescope pupils:

- :func:`vlt`: 8.2 m primary, 1.116 m central obstruction, four 5 cm spider vanes (ESO VLT User Manual);
- :func:`jwst`: 18 hexagonal segments of 1.32 m flat to flat with 7 mm gaps (Gardner et al. 2023), without the
  secondary supports;
- :func:`elt`: 798 hexagonal segments of 1.4 m corner to corner with 4 mm gaps between radii 4.75 m and
  19.65 m (39.3 m tip to tip; ESO ELT Construction Proposal 2011), six 50 cm spider vanes.
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
from lumos.fields.grid import Grid
from lumos.fields.radiance import RadianceField
from lumos.physics import apertures as shapes

__all__ = ["Aperture", "CircularAperture", "HexSegmentedAperture", "PolygonAperture", "elt", "jwst", "vlt"]


def _apply_mask(signal: Any, mask: np.ndarray) -> Any:
    if isinstance(signal, RadianceField):
        return signal * get(signal.backend).asarray(mask**2)
    if not hasattr(signal, "grid") or not hasattr(signal, "field"):
        raise TypeError(f"an aperture cannot act on a {type(signal).__name__}")
    return signal * get(signal.backend).asarray(mask)


def _transfer(mask_function: Callable[[Values, Grid], np.ndarray]) -> staticmethod:
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        signal = inputs["in"]
        return {"out": _apply_mask(signal, mask_function(values, signal.grid))}

    transfer.__doc__ = "Multiply the incoming signal by the amplitude transmission of the aperture."
    return staticmethod(transfer)


def _with_spiders(values: Values, grid: Grid, mask: np.ndarray) -> np.ndarray:
    if values.n_spiders and values.spider_width > 0:
        vanes = shapes.spiders(
            grid,
            values.n_spiders,
            values.spider_width,
            rotation=values.spider_rotation,
            supersample=values.supersample,
        )
        mask = mask * (1 - vanes)
    return mask


class Aperture(Element):
    """Base class of apertures: a mask applied to the incoming optical signal."""

    in_ = Input(OpticalSignal)
    out = Output(OpticalSignal)

    @staticmethod
    def _mask(values: Values, grid: Grid) -> np.ndarray:
        raise NotImplementedError

    def mask(self, grid: Grid) -> np.ndarray:
        """Amplitude transmission of the aperture sampled on ``grid`` (NumPy array, values in [0, 1])."""
        return type(self)._mask(self.values(), grid)


def _circular(values: Values, grid: Grid) -> np.ndarray:
    mask = shapes.annulus(grid, values.diameter, values.obstruction, supersample=values.supersample)
    return _with_spiders(values, grid, mask)


@register("lumos.apertures.CircularAperture")
class CircularAperture(Aperture):
    """Circular pupil with an optional central obstruction and radial spider vanes."""

    diameter = Param(unit=u.m, default=1 * u.m, bounds=(0 * u.m, None), inclusive=(False, True))
    obstruction = Param(unit=u.m, default=0 * u.m, bounds=(0 * u.m, None), doc="central obstruction diameter")
    spider_width = Param(unit=u.m, default=0 * u.m, bounds=(0 * u.m, None))
    spider_rotation = Param(unit=u.deg, default=0 * u.deg)
    n_spiders = Setting(int, default=0)
    supersample = Setting(int, default=4, doc="sub-pixels per pixel side for anti-aliasing")

    _mask = staticmethod(_circular)
    transfer = _transfer(_circular)

    @staticmethod
    def transfer_symbolic(values: Values, inputs: Any) -> dict[str, Any]:
        """Multiply a symbolic field by the indicator of the annulus (no closed form with spiders)."""
        import sympy as sp

        from lumos.symbolic.symbols import r

        if values.n_spiders and values.spider_width > 0:
            raise NotImplementedError("circular apertures with spiders have no symbolic transfer")
        outer = sp.Float(values.diameter / 2)
        mask = sp.Piecewise((1, r <= outer), (0, True))
        if values.obstruction > 0:
            inner = sp.Float(values.obstruction / 2)
            mask = sp.Piecewise((1, (r <= outer) & (r >= inner)), (0, True))
        return {"out": inputs["in"].multiply(mask)}


def _polygon(values: Values, grid: Grid) -> np.ndarray:
    return shapes.regular_polygon(
        grid, values.n_sides, values.circumradius, rotation=values.rotation, supersample=values.supersample
    )


@register("lumos.apertures.PolygonAperture")
class PolygonAperture(Aperture):
    """Regular polygon (e.g. a hexagonal segment) inscribed in a circle."""

    circumradius = Param(unit=u.m, default=0.5 * u.m, bounds=(0 * u.m, None), inclusive=(False, True))
    rotation = Param(unit=u.deg, default=0 * u.deg, doc="angle of a vertex from +x")
    n_sides = Setting(int, default=6)
    supersample = Setting(int, default=4)

    _mask = staticmethod(_polygon)
    transfer = _transfer(_polygon)


def _segments(values: Values, grid: Grid) -> tuple[np.ndarray, int]:
    mask, count = shapes.hex_segments(
        grid,
        flat_to_flat=values.flat_to_flat,
        gap=values.gap,
        rings=values.rings,
        missing_center=values.missing_center,
        inner_radius=values.inner_radius,
        outer_radius=values.outer_radius,
        supersample=values.supersample,
    )
    return _with_spiders(values, grid, mask), count


def _segmented(values: Values, grid: Grid) -> np.ndarray:
    return _segments(values, grid)[0]


@register("lumos.apertures.HexSegmentedAperture")
class HexSegmentedAperture(Aperture):
    """Primary made of hexagonal segments on a hexagonal lattice, ring by ring around the centre."""

    flat_to_flat = Param(unit=u.m, default=1 * u.m, bounds=(0 * u.m, None), inclusive=(False, True))
    gap = Param(unit=u.mm, default=5 * u.mm, bounds=(0 * u.m, None))
    inner_radius = Param(unit=u.m, default=0 * u.m, bounds=(0 * u.m, None), doc="0: no inner limit")
    outer_radius = Param(unit=u.m, default=0 * u.m, bounds=(0 * u.m, None), doc="0: no outer limit")
    spider_width = Param(unit=u.m, default=0 * u.m, bounds=(0 * u.m, None))
    spider_rotation = Param(unit=u.deg, default=0 * u.deg)
    rings = Setting(int, default=2)
    missing_center = Setting(bool, default=False)
    n_spiders = Setting(int, default=0)
    supersample = Setting(int, default=4)

    _mask = staticmethod(_segmented)
    transfer = _transfer(_segmented)

    def segment_count(self, grid: Grid) -> int:
        """Return the number of segments of the primary (independent of the sampling ``grid``)."""
        values = self.values()
        return len(
            shapes.hex_centers(
                values.flat_to_flat + values.gap,
                values.rings,
                missing_center=values.missing_center,
                inner_radius=values.inner_radius,
                outer_radius=values.outer_radius,
                segment_circumradius=values.flat_to_flat / math.sqrt(3),
            )
        )

    @property
    def diameter(self) -> u.Quantity:
        """Tip-to-tip diameter of the primary."""
        values = self.values()
        if values.outer_radius > 0:
            return 2 * values.outer_radius * u.m
        centers = shapes.hex_centers(
            values.flat_to_flat + values.gap, values.rings, missing_center=values.missing_center
        )
        reach = max(math.hypot(x, y) for x, y in centers) + values.flat_to_flat / math.sqrt(3)
        return 2 * reach * u.m


def vlt(name: str = "VLT") -> CircularAperture:
    """Approximate pupil of a VLT unit telescope."""
    return CircularAperture(
        name=name,
        diameter=8.2 * u.m,
        obstruction=1.116 * u.m,
        n_spiders=4,
        spider_width=5 * u.cm,
        spider_rotation=45 * u.deg,
    )


def jwst(name: str = "JWST") -> HexSegmentedAperture:
    """Approximate primary of the James Webb Space Telescope (18 segments)."""
    return HexSegmentedAperture(
        name=name, flat_to_flat=1.32 * u.m, gap=7 * u.mm, rings=2, missing_center=True
    )


def elt(name: str = "ELT") -> HexSegmentedAperture:
    """Approximate primary of the Extremely Large Telescope (798 segments)."""
    return HexSegmentedAperture(
        name=name,
        flat_to_flat=1.4 * math.sqrt(3) / 2 * u.m,
        gap=4 * u.mm,
        rings=22,
        inner_radius=4.75 * u.m,
        outer_radius=19.65 * u.m,
        n_spiders=6,
        spider_width=50 * u.cm,
        spider_rotation=30 * u.deg,
    )
