"""Tests for scenes and their frames (step P8.1, REQ-SCN-1, REQ-SCN-2, REQ-SCN-7).

Requirements: a ``Scene`` is an ``ElementGroup`` of scene components placed in a frame: celestial (ICRS,
objects at infinity, angular coordinates), terrestrial (latitude, longitude, altitude) or laboratory (finite
distance, coordinates in metres). Point components give a ``SourceList``; diffuse components give radiance
maps, combined on the angular grid of the scene.

Independent references:

- radiance is invariant along a ray in vacuum (Born & Wolf, *Principles of Optics*, §4.8.1; McCluney,
  *Introduction to Radiometry and Photometry*, §2.8): a laboratory radiance map ``L(x, y)`` at distance ``d``
  is seen as ``L(θ d)`` on the sky of the instrument, with pixels of ``dx / d`` radians (small angles);
- the flux ``Σ L ΔΩ`` of a map resampled onto a grid covering it is conserved (definition of a conservative
  resampling);
- astropy's ``SkyCoord`` gives the ICRS coordinates of the pointing.
"""

import math

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

import lumos.io
from lumos import Context
from lumos.core.element import Element
from lumos.core.group import ElementGroup
from lumos.core.param import Param, Setting
from lumos.core.port import Output
from lumos.elements.sources import PlanetarySystem, PointSource
from lumos.fields import Grid, RadianceField, SourceList
from lumos.scenes import FRAMES, Scene, SceneCombiner

RADIANCE = u.W / u.m**2 / u.sr / u.m


class FlatPatch(Element):
    """Test component: a uniform square patch of radiance on its own grid (not registered)."""

    observation_dependent = True
    radiance = Param(unit=RADIANCE, default=1 * RADIANCE)
    step = Param(unit=u.mas, default=10 * u.mas)
    n = Setting(int, default=8)
    out = Output(RadianceField)

    @staticmethod
    def transfer(values, inputs):
        grid = Grid(values.n, dx=values.step * u.rad)
        lam = np.asarray(values.wavelengths)
        data = np.full((len(lam), values.n, values.n), values.radiance)
        return {"out": RadianceField(data * RADIANCE, grid, lam * u.m)}


class LabPatch(Element):
    """Test component: a uniform patch on a spatial grid (laboratory frame)."""

    observation_dependent = True
    out = Output(RadianceField)

    @staticmethod
    def transfer(values, inputs):
        grid = Grid(4, dx=1 * u.mm)
        lam = np.asarray(values.wavelengths)
        data = np.full((len(lam), 4, 4), 2.0)
        return {"out": RadianceField(data * RADIANCE, grid, lam * u.m)}


def band(ctx, n=2):
    ctx.observation.band = (1.0 * u.um, 2.0 * u.um)
    ctx.observation.n_wavelengths = n
    return ctx


# --- Frames -------------------------------------------------------------------------------------------------


def test_frames():
    assert set(FRAMES) == {"celestial", "terrestrial", "laboratory"}


def test_scene_is_an_element_group_with_two_outputs():
    scene = Scene(name="sky")
    assert isinstance(scene, ElementGroup)
    assert set(scene.instance_ports()) == {"sources", "radiance"}
    assert scene.instance_ports()["sources"].signal_type is SourceList
    assert scene.instance_ports()["radiance"].signal_type is RadianceField


def test_unknown_frame_is_refused():
    with pytest.raises(ValueError):
        Scene(frame="galactic-ish")


def test_celestial_pointing_is_icrs():
    scene = Scene(frame="celestial", ra=83.82 * u.deg, dec=-5.39 * u.deg)
    pointing = scene.pointing
    assert isinstance(pointing, SkyCoord)
    assert pointing.frame.name == "icrs"
    assert u.Quantity(pointing.ra).to_value(u.deg) == pytest.approx(83.82)
    assert u.Quantity(pointing.dec).to_value(u.deg) == pytest.approx(-5.39)


def test_terrestrial_frame_parameters():
    scene = Scene(frame="terrestrial", latitude=45 * u.deg, longitude=5 * u.deg, altitude=1 * u.km)
    assert scene.frame == "terrestrial"
    assert scene.location.lat.to_value(u.deg) == pytest.approx(45)
    assert scene.location.height.to_value(u.m) == pytest.approx(1000)


def test_pointing_is_only_defined_for_celestial_scenes():
    with pytest.raises(ValueError, match="celestial"):
        _ = Scene(frame="laboratory").pointing


def test_parameters_need_units():
    with pytest.raises((TypeError, u.UnitsError)):
        Scene(frame="laboratory", distance=2.0)


# --- Point components ---------------------------------------------------------------------------------------


def test_point_components_are_combined():
    ctx = band(Context(seed=1))
    scene = ctx.add(Scene(name="sky"))
    scene.add(PointSource(name="a", flux_density=1 * u.Jy))
    scene.add(PointSource(name="b", flux_density=2 * u.Jy, theta_x=1 * u.mas))
    sources = ctx.run(["sky.sources"])["sky.sources"]
    assert sources.n_sources == 2
    np.testing.assert_allclose(sources.theta_x.to_value(u.mas), [0, 1])
    flux = sources.flux.value
    np.testing.assert_allclose(flux[1] / flux[0], 2.0, rtol=1e-12)


def test_scene_without_components_is_dark():
    ctx = band(Context(seed=1))
    ctx.add(Scene(name="sky", n_pixels=16))
    result = ctx.run(["sky.sources", "sky.radiance"])
    assert result["sky.sources"].n_sources == 0
    radiance = result["sky.radiance"]
    assert radiance.grid.shape == (16, 16)
    assert float(np.max(np.abs(np.asarray(radiance.data)))) == 0.0


def test_scene_moves_with_the_clock():
    ctx = band(Context(seed=1), n=1)
    scene = ctx.add(Scene(name="sky"))
    scene.add(PlanetarySystem(name="system", inclination=0 * u.deg))
    before = ctx.run(["sky.sources"])["sky.sources"].theta_x[1]
    ctx.wait(90 * u.day)
    after = ctx.run(["sky.sources"])["sky.sources"].theta_x[1]
    assert not math.isclose(before.to_value(u.rad), after.to_value(u.rad))


# --- Diffuse components -------------------------------------------------------------------------------------


def test_diffuse_components_are_resampled_on_the_scene_grid_conserving_flux():
    ctx = band(Context(seed=1))
    scene = ctx.add(Scene(name="sky", n_pixels=32, pixel_scale=5 * u.mas))
    scene.add(FlatPatch(name="patch", step=10 * u.mas, n=8))
    radiance = ctx.run(["sky.radiance"])["sky.radiance"]
    assert radiance.grid.plane == "angular"
    assert radiance.grid.shape == (32, 32)
    assert radiance.grid.dx.to_value(u.mas) == pytest.approx(5)
    expected = 1.0 * (8 * 10 * u.mas).to_value(u.rad) ** 2  # W m-2 m-1
    np.testing.assert_allclose(radiance.integrated().to_value(u.W / u.m**3), expected, rtol=1e-10)


def test_diffuse_components_add_up():
    ctx = band(Context(seed=1))
    scene = ctx.add(Scene(name="sky", n_pixels=16, pixel_scale=10 * u.mas))
    scene.add(FlatPatch(name="p1", radiance=1 * RADIANCE))
    scene.add(FlatPatch(name="p2", radiance=3 * RADIANCE))
    radiance = ctx.run(["sky.radiance"])["sky.radiance"]
    data = np.asarray(radiance.data)
    assert data.max() == pytest.approx(4.0)


def test_laboratory_maps_keep_their_radiance_at_a_distance():
    ctx = band(Context(seed=1))
    d = 2.0
    pixel = 1e-3 / d  # rad
    scene = ctx.add(
        Scene(name="bench", frame="laboratory", distance=d * u.m, n_pixels=8, pixel_scale=pixel * u.rad)
    )
    scene.add(LabPatch(name="patch"))
    radiance = ctx.run(["bench.radiance"])["bench.radiance"]
    data = np.asarray(radiance.data)[0]
    assert data.max() == pytest.approx(2.0, rel=1e-12)
    assert np.count_nonzero(data > 1.0) == 16
    expected = 2.0 * 16 * pixel**2
    np.testing.assert_allclose(radiance.integrated().to_value(u.W / u.m**3), expected, rtol=1e-10)


# --- Serialisation ------------------------------------------------------------------------------------------


def test_scene_round_trip_through_lumos(tmp_path):
    ctx = band(Context(seed=1))
    scene = ctx.add(Scene(name="sky", ra=10 * u.deg, dec=20 * u.deg, n_pixels=16))
    scene.add(PointSource(name="star", flux_density=3 * u.Jy))
    path = tmp_path / "scene.lumos"
    lumos.io.save(ctx, path)
    loaded = lumos.io.load(path)
    again = loaded["sky"]
    assert isinstance(again, Scene)
    assert again.pointing.ra.to_value(u.deg) == pytest.approx(10)
    sources = loaded.run(["sky.sources"])["sky.sources"]
    original = ctx.run(["sky.sources"])["sky.sources"]
    np.testing.assert_allclose(sources.flux.value, original.flux.value)


def test_scene_combiner_is_registered():
    from lumos.core.registry import registry

    assert registry.type_id(SceneCombiner) == "lumos.scenes.SceneCombiner"
