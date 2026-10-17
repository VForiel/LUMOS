"""Tests for name resolution and the catalogue constructors (step P9.8, REQ-CAT-1, REQ-CAT-6).

Requirements: ``lumos.catalogs.resolve("<name>")`` identifies the kind of object and returns a typed record;
direct constructors ``Star.from_catalog("Vega")``, ``Planet.from_catalog("HD 209458 b")``,
``Scene.from_catalog("TRAPPIST-1")``, ``MappedComponent.from_survey("M42", ...)`` (acceptance of v0.3:
``Scene.from_catalog("TRAPPIST-1")`` usable in a context).

Independent references: the recorded catalogue answers (see ``test_providers.py``); for the planet, the
fluxes of the planet of a ``PlanetarySystem`` with the same parameters (element validated in step P6.1).
"""

import numpy as np
import pytest
from astropy import units as u

from lumos import Context
from lumos.catalogs import resolve
from lumos.catalogs.records import ExoplanetRecord, ExtendedObjectRecord, StarRecord
from lumos.elements.sources import Planet, PlanetarySystem, Star
from lumos.scenes import Scene


def band(ctx, low=1.0, high=2.0, n=3):
    ctx.observation.band = (low * u.um, high * u.um)
    ctx.observation.n_wavelengths = n
    return ctx


# --- resolve -----------------------------------------------------------------------------------------------


def test_resolve_a_star(replay):
    record = resolve("Vega")
    assert isinstance(record, StarRecord)
    assert record.name == "* alf Lyr"
    assert record.spectral_type == "A0V"


def test_resolve_a_planet(replay):
    record = resolve("HD 209458 b")
    assert isinstance(record, ExoplanetRecord)
    assert record.host == "HD 209458"


def test_resolve_an_extended_object(replay):
    record = resolve("M42")
    assert isinstance(record, ExtendedObjectRecord)
    assert record.object_type == "HII"
    assert record.ra.to_value(u.deg) == pytest.approx(83.8201)
    assert "NGC  1976" in record.identifiers


# --- Star ---------------------------------------------------------------------------------------------------


def test_star_from_catalog(replay):
    star = Star.from_catalog("Vega")
    assert star.name == "Vega"
    assert star.temperature.to_value(u.K) == pytest.approx(9375)
    assert star.band == "V"
    assert star.magnitude == pytest.approx(0.03)
    assert star.distance.to_value(u.pc) == pytest.approx(1000 / 130.23)


# --- Planet -------------------------------------------------------------------------------------------------


def test_planet_from_catalog(replay):
    planet = Planet.from_catalog("HD 209458 b")
    assert planet.name == "HD 209458 b"
    assert planet.star_temperature.to_value(u.K) == pytest.approx(6091)
    assert planet.star_radius.to_value(u.R_sun) == pytest.approx(1.19)
    assert planet.star_mass.to_value(u.M_sun) == pytest.approx(1.23)
    assert planet.distance.to_value(u.pc) == pytest.approx(48.3016)
    assert planet.planet_radius.to_value(u.R_earth) == pytest.approx(15.58051)
    assert planet.semi_major_axis.to_value(u.au) == pytest.approx(0.04707)
    assert planet.inclination.to_value(u.deg) == pytest.approx(86.71)


def test_planet_matches_the_planet_of_a_planetary_system():
    values = {
        "star_temperature": 6000 * u.K,
        "star_radius": 1.1 * u.R_sun,
        "star_mass": 1.05 * u.M_sun,
        "distance": 20 * u.pc,
        "planet_radius": 1.3 * u.R_jup,
        "semi_major_axis": 0.05 * u.au,
        "eccentricity": 0.1,
        "inclination": 60 * u.deg,
        "argument_periapsis": 30 * u.deg,
        "longitude_node": 10 * u.deg,
        "mean_anomaly": 45 * u.deg,
        "albedo": 0.2,
    }
    ctx = band(Context(seed=1))
    ctx.add(Planet(name="planet", **values))
    ctx.add(PlanetarySystem(name="system", **values))
    for _ in range(2):
        result = ctx.run(["planet.out", "system.out"])
        alone, pair = result["planet.out"], result["system.out"]
        assert alone.n_sources == 1
        np.testing.assert_allclose(alone.flux.value[0], pair.flux.value[1], rtol=1e-12)
        assert alone.theta_x[0].to_value(u.rad) == pytest.approx(pair.theta_x[1].to_value(u.rad), rel=1e-12)
        assert alone.theta_y[0].to_value(u.rad) == pytest.approx(pair.theta_y[1].to_value(u.rad), rel=1e-12)
        ctx.wait(1 * u.day)


# --- Scene --------------------------------------------------------------------------------------------------


def test_scene_from_catalog_trappist_1(replay):
    scene = Scene.from_catalog("TRAPPIST-1")
    assert isinstance(scene, Scene)
    assert scene.frame == "celestial"
    assert scene.pointing.ra.to_value(u.deg) == pytest.approx(346.62236872857875)
    assert scene.pointing.dec.to_value(u.deg) == pytest.approx(-5.041399250518334)
    names = sorted(scene.components)
    assert len(names) == 8
    ctx = band(Context(seed=1))
    ctx.add(scene)
    sources = ctx.run([f"{ctx.id_of(scene)}.sources"])[f"{ctx.id_of(scene)}.sources"]
    assert sources.n_sources == 8
    star_flux = sources.flux.value[0]
    assert np.all(sources.flux.value[1:] < 1e-2 * star_flux)


def test_scene_round_trip_after_catalog(replay, tmp_path):
    import lumos.io

    ctx = band(Context(seed=1))
    ctx.add(Scene.from_catalog("TRAPPIST-1"))
    path = tmp_path / "trappist.lumos"
    lumos.io.save(ctx, path)
    again = lumos.io.load(path)
    key = f"{next(iter(again.elements))}.sources"
    np.testing.assert_allclose(again.run([key])[key].flux.value, ctx.run([key])[key].flux.value, rtol=1e-12)
