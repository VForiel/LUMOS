"""Tests for apertures and pupils (step P6.2).

Requirements: §5.9 (parametric pupils: disks, polygons, obstruction, spiders, segmented primaries; presets
JWST, VLT, ELT), plan P6.2 (primitives, presets, anti-aliasing).

Independent references (analytic areas):

- disk ``π R²``, annulus ``π (R² - r²)``;
- regular hexagon of flat-to-flat width f: ``(√3/2) f²``;
- n radial vanes of width w between radii r and R: ``n w (R - r)`` (vanes do not overlap outside r);
- supersampled (grey) masks converge to the exact area: the error of an s × s supersampling decreases with s;
- VLT unit telescope: primary 8.2 m, central obstruction 1.116 m (ESO, VLT User Manual);
- JWST: 18 hexagonal segments of 1.32 m flat to flat, collecting area 25.4 m² (Gardner et al. 2023, PASP 135,
  068001; the model here has no secondary support, hence a slightly larger area);
- ELT: 39.3 m primary made of 798 segments (ESO, ELT Construction Proposal, 2011).
"""

import math

import numpy as np
import pytest
from astropy import units as u

from lumos import Context
from lumos.elements.apertures import (
    CircularAperture,
    HexSegmentedAperture,
    PolygonAperture,
    elt,
    jwst,
    vlt,
)
from lumos.fields import Grid, IncoherentBundle, Wavecube, Wavefront
from lumos.physics import apertures as ap


def area(mask, grid):
    return float(np.sum(mask)) * grid.dx_si * grid.dy_si


GRID = Grid(160, dx=1 * u.cm)  # 1.6 m


# --- Primitives --------------------------------------------------------------------------------


def test_disk_area_converges_with_supersampling():
    errors = []
    for s in (1, 2, 4, 8):
        mask = ap.disk(GRID, 1.0, supersample=s)
        errors.append(abs(area(mask, GRID) - math.pi * 0.25))
    assert errors[-1] < 1e-4 * math.pi * 0.25
    assert errors[3] < errors[0]


def test_masks_are_between_zero_and_one():
    mask = ap.disk(GRID, 1.0, supersample=4)
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0
    assert 0 < np.count_nonzero((mask > 0) & (mask < 1))  # grey edge pixels


def test_annulus_area():
    mask = ap.annulus(GRID, 1.4, 0.4, supersample=8)
    assert area(mask, GRID) == pytest.approx(math.pi * (0.7**2 - 0.2**2), rel=2e-4)


def test_hexagon_area():
    mask = ap.regular_polygon(GRID, 6, circumradius=0.6, supersample=8)
    flat = 0.6 * math.sqrt(3)
    assert area(mask, GRID) == pytest.approx(math.sqrt(3) / 2 * flat**2, rel=5e-4)


def test_spider_vanes_area():
    clear = ap.annulus(GRID, 1.4, 0.4, supersample=8)
    vanes = ap.spiders(GRID, n=4, width=0.03, supersample=8)
    blocked = area(clear * (1 - vanes), GRID)
    expected = math.pi * (0.7**2 - 0.2**2) - 4 * 0.03 * (0.7 - 0.2)
    assert blocked == pytest.approx(expected, rel=2e-3)


def test_masks_are_centred():
    odd = Grid(101, dx=1 * u.cm)
    mask = ap.annulus(odd, 0.8, 0.2, supersample=4) * (1 - ap.spiders(odd, n=4, width=0.02, supersample=4))
    np.testing.assert_allclose(mask, mask[::-1, ::-1], atol=1e-12)


def test_segmented_hexagons():
    grid = Grid(512, dx=1 * u.cm)
    mask, count = ap.hex_segments(
        grid, flat_to_flat=1.0, gap=0.02, rings=2, missing_center=True, supersample=4
    )
    assert count == 18
    assert area(mask, grid) == pytest.approx(18 * math.sqrt(3) / 2 * 1.0**2, rel=2e-3)


# --- Presets -----------------------------------------------------------------------------------


def test_vlt_preset():
    pupil = vlt()
    assert pupil.diameter == 8.2 * u.m
    assert pupil.obstruction == 1.116 * u.m
    grid = Grid(512, dx=8.4 / 512 * u.m)
    measured = area(pupil.mask(grid), grid)
    width = pupil.spider_width.to_value(u.m)
    expected = math.pi / 4 * (8.2**2 - 1.116**2) - 4 * width * (8.2 - 1.116) / 2
    assert measured == pytest.approx(expected, rel=2e-3)


def test_jwst_preset():
    pupil = jwst()
    grid = Grid(512, dx=7.0 / 512 * u.m)
    measured = area(pupil.mask(grid), grid)
    assert pupil.segment_count(grid) == 18
    assert measured == pytest.approx(18 * math.sqrt(3) / 2 * 1.32**2, rel=3e-3)
    assert measured == pytest.approx(25.4, rel=0.1)


def test_elt_preset():
    pupil = elt()
    assert pupil.diameter.to_value(u.m) == pytest.approx(39.3, rel=0.01)
    grid = Grid(1024, dx=40.0 / 1024 * u.m)
    mask = pupil.mask(grid)
    assert pupil.segment_count(grid) == pytest.approx(798, rel=0.03)
    outer = math.pi / 4 * 39.3**2
    assert 0.6 * outer < area(mask, grid) < outer


# --- Elements on fields --------------------------------------------------------------------------


def test_aperture_transmits_the_masked_power():
    wf = Wavefront.plane(GRID, 1 * u.um, power=1 * u.W)
    ctx = Context()
    pupil = ctx.add(CircularAperture(name="pupil", diameter=1 * u.m))
    mask = pupil.mask(GRID)
    out = pupil.apply(**{"in": wf})["out"]
    expected = float(np.sum(mask**2)) / GRID.size  # amplitude transmission: power scales as |t|²
    assert out.power.to_value(u.W) == pytest.approx(expected, rel=1e-12)


def test_apertures_are_elements_acting_on_every_signal():
    aperture = PolygonAperture(n_sides=6, circumradius=0.6 * u.m)
    mask = aperture.mask(GRID)
    wf = Wavefront.plane(GRID, 1 * u.um, power=1 * u.W)
    np.testing.assert_allclose(
        np.asarray(aperture.apply(**{"in": wf})["out"].field), np.asarray(wf.field) * mask
    )
    cube = Wavecube.plane(GRID, [1.0, 2.0] * u.um, power=1 * u.W)
    out = aperture.apply(**{"in": cube})["out"]
    np.testing.assert_allclose(np.asarray(out.field)[1], np.asarray(cube.field)[1] * mask)
    bundle = IncoherentBundle.point_sources(
        GRID, 1 * u.um, theta_x=[0, 1e-6] * u.rad, theta_y=[0, 0] * u.rad, power=[1, 1] * u.W
    )
    out = aperture.apply(**{"in": bundle})["out"]
    np.testing.assert_allclose(np.asarray(out.field)[1], np.asarray(bundle.field)[1] * mask)


def test_segmented_aperture_element():
    aperture = HexSegmentedAperture(flat_to_flat=0.3 * u.m, gap=1 * u.cm, rings=2)
    mask = aperture.mask(GRID)
    assert aperture.segment_count(GRID) == 19
    assert area(mask, GRID) == pytest.approx(19 * math.sqrt(3) / 2 * 0.3**2, rel=5e-3)
