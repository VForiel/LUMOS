"""Tests for the Solar System providers (step P9.4, REQ-CAT-2).

Requirements: JPL Horizons ephemerides and the reference solar spectrum.

Independent references (recorded responses in ``tests/fixtures/catalogs``):

- JPL Horizons (Giorgini et al. 1996), Mars seen from the geocentre on 2026-01-01 00:00 UT: RA 283.48904°,
  Dec -23.75166°, apparent diameter 3.884890″, heliocentric distance r = 1.428708512639 au, distance
  Δ = 2.41069876736921 au, phase angle (Sun-Target-Observer) 1.5830°;
- solar reference spectrum (CALSPEC ``sun_reference_stis_002``, Bohlin, Dickinson & Calzetti 2001, AJ 122,
  2118), 1195 Å to 2.7 µm at 1 au: its integral is the total solar irradiance 1360.8 W m⁻² (Kopp & Lean 2011,
  GRL 38, L01706) times the fraction of the solar flux in that range, here taken from a 5772 K blackbody
  (the Sun departs from it by a few percent: tolerance 2 %).
"""

import math

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time
from scipy.integrate import quad

from lumos.catalogs.records import SolarSystemRecord
from lumos.catalogs.solar_system import Horizons, solar_spectrum


def test_horizons_ephemeris_of_mars(replay):
    records = Horizons().ephemeris(
        "Mars",
        start=Time("2026-01-01 00:00", scale="utc"),
        stop=Time("2026-01-02 00:00", scale="utc"),
        step="1 d",
    )
    assert len(records) == 2
    mars = records[0]
    assert isinstance(mars, SolarSystemRecord)
    assert mars.name == "Mars"
    assert mars.ra.to_value(u.deg) == pytest.approx(283.48904)
    assert mars.dec.to_value(u.deg) == pytest.approx(-23.75166)
    assert mars.angular_diameter.to_value(u.arcsec) == pytest.approx(3.884890)
    assert mars.heliocentric_distance.to_value(u.au) == pytest.approx(1.428708512639)
    assert mars.delta.to_value(u.au) == pytest.approx(2.41069876736921)
    assert mars.phase_angle.to_value(u.deg) == pytest.approx(1.5830)
    assert mars.provenance("delta")[0] == "jpl_horizons"
    assert Time(mars.epoch.to_value(u.day), format="jd") == Time("2026-01-01 00:00", scale="utc")
    assert records[1].ra.to_value(u.deg) == pytest.approx(284.32308)


def test_horizons_body_identifiers(replay):
    request = Horizons()
    assert request.command("Mars") == "499"
    assert request.command("Jupiter") == "599"
    assert request.command("Moon") == "301"
    assert request.command("1P/Halley") == "1P/Halley"


def planck_fraction(low, high, temperature=5772.0):
    h, c, k = 6.62607015e-34, 299_792_458.0, 1.380649e-23

    def planck(lam):
        return 2 * h * c**2 / lam**5 / math.expm1(h * c / (lam * k * temperature))

    total = quad(planck, 1e-8, 1e-3, limit=500, points=[5e-7, 1e-6, 5e-6])[0]
    return quad(planck, low, high, limit=500)[0] / total


def test_solar_spectrum_integrates_to_the_solar_constant(replay):
    sun = solar_spectrum()
    lam = sun.wavelengths.to_value(u.m)
    flux = sun.flux.to_value(u.W / u.m**2 / u.m)
    irradiance = np.trapezoid(flux, lam)
    expected = 1360.8 * planck_fraction(lam.min(), lam.max())
    assert irradiance == pytest.approx(expected, rel=2e-2)
    assert sun.source == "calspec"
