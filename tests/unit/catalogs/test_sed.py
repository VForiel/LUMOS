"""Tests for the fusion and completion of spectral energy distributions (step P9.7, REQ-CAT-4).

Requirements: fusion of multi-source SEDs with documented priority rules; optional physical completion,
flagged as estimated.

Independent references:

- Planck's law with the CODATA constants (``scipy.constants``): ``F_ν = Ω B_ν(T)`` for a blackbody of solid
  angle Ω;
- Vega: V = 0.03 and K = 0.13 mag (SIMBAD) with the zero points 3636 Jy (V) and 640 Jy (K) of Bessell,
  Castelli & Plez (1998, A&A 333, 231, table A2): 3536 Jy and 568 Jy. A blackbody fitted to its photometry
  between 0.4 and 2.5 µm reproduces them within 10 %; its temperature is not the effective temperature (the
  Balmer jump and the Paschen continuum of an A star are not those of a blackbody), so only the photometry
  is compared;
- Vega: effective temperature 9602 ± 180 K and mean radius between the polar (2.36 R☉) and equatorial
  (2.82 R☉) radii (Yoon et al. 2010, ApJ 708, 71); with the measured temperature, the solid angle fitted to
  the photometry gives a radius in this range within ±10 %.
"""

import math

import numpy as np
import pytest
from astropy import units as u
from scipy import constants as const

from lumos.catalogs.records import Measurement, StarRecord
from lumos.catalogs.sed import DEFAULT_PRIORITIES, complete_star, fit_blackbody, fuse
from lumos.catalogs.vizier import PhotometricPoints, VizierPhotometry


def planck_nu_jy(lam_m, temperature, omega):
    nu = const.c / lam_m
    b = 2 * const.h * nu**3 / const.c**2 / np.expm1(const.h * nu / (const.k * temperature))
    return omega * b / 1e-26


FILTERS = {
    "U": 0.36e-6,
    "B": 0.44e-6,
    "V": 0.55e-6,
    "R": 0.64e-6,
    "I": 0.79e-6,
    "J": 1.22e-6,
    "H": 1.63e-6,
    "K": 2.19e-6,
}


def synthetic(temperature=6000.0, omega=1e-16):
    lam, flux, err, filters, catalogues = [], [], [], [], []
    for name, wavelength in FILTERS.items():
        value = planck_nu_jy(wavelength, temperature, omega)
        for factor, sigma, catalogue in (
            (1.0, 0.01, "J/good"),
            (1.2, 0.30, "J/poor"),
            (0.5, np.nan, "J/none"),
        ):
            lam.append(wavelength)
            flux.append(value * factor)
            err.append(value * sigma)
            filters.append(f"Johnson:{name}")
            catalogues.append(catalogue)
    return PhotometricPoints(
        np.array(lam) * u.m, np.array(flux) * u.Jy, np.array(err) * u.Jy, filters, catalogues
    )


def test_fusion_keeps_one_point_per_filter_with_the_smallest_relative_uncertainty():
    sed = fuse(synthetic())
    assert len(sed) == len(FILTERS)
    assert set(sed.sources) == {"J/good"}
    expected = [planck_nu_jy(lam, 6000.0, 1e-16) for lam in FILTERS.values()]
    np.testing.assert_allclose(sed.flux_density.to_value(u.Jy), expected, rtol=1e-12)
    assert np.all(np.diff(sed.wavelengths.to_value(u.m)) > 0)
    assert not np.any(sed.estimated)


def test_priority_catalogues_come_first():
    points = PhotometricPoints(
        [1.22, 1.22] * u.um,
        [10.0, 20.0] * u.Jy,
        [0.1, np.nan] * u.Jy,
        ["2MASS:J", "2MASS:J"],
        ["J/other", "II/246/out"],
    )
    sed = fuse(points)
    assert sed.flux_density.to_value(u.Jy)[0] == 20.0
    assert any(prefix.startswith("II/246") for prefix in DEFAULT_PRIORITIES)


def test_median_without_uncertainties():
    points = PhotometricPoints(
        [0.55, 0.55, 0.55] * u.um,
        [1.0, 2.0, 9.0] * u.Jy,
        [np.nan] * 3 * u.Jy,
        ["Johnson:V"] * 3,
        ["A", "B", "C"],
    )
    sed = fuse(points)
    assert sed.flux_density.to_value(u.Jy)[0] == 2.0


def test_blackbody_fit_recovers_temperature_and_solid_angle():
    temperature, omega = fit_blackbody(fuse(synthetic(5000.0, 3e-17)))
    assert temperature.to_value(u.K) == pytest.approx(5000.0, rel=1e-3)
    assert omega.to_value(u.sr) == pytest.approx(3e-17, rel=5e-3)


def test_completion_is_flagged_as_estimated():
    sed = fuse(synthetic())
    extra = [5.0, 10.0] * u.um
    completed = sed.complete(extra)
    assert len(completed) == len(sed) + 2
    estimated = completed.estimated
    assert estimated.sum() == 2
    added = completed.flux_density.to_value(u.Jy)[estimated]
    expected = [planck_nu_jy(lam, 6000.0, 1e-16) for lam in (5e-6, 10e-6)]
    np.testing.assert_allclose(added, expected, rtol=5e-3)
    assert all(source == "lumos" for source, flag in zip(completed.sources, estimated, strict=True) if flag)


def test_vega_blackbody_fit_reproduces_the_photometry(replay):
    sed = fuse(VizierPhotometry().sed("Vega"))
    temperature, omega = fit_blackbody(sed, wavelength_range=(0.4 * u.um, 2.5 * u.um))
    model = planck_nu_jy(np.array([0.545e-6, 2.19e-6]), temperature.to_value(u.K), omega.to_value(u.sr))
    np.testing.assert_allclose(model, [3536.0, 568.0], rtol=0.10)


def test_complete_star_estimates_temperature_and_radius(replay):
    vega = StarRecord("Vega", parallax=Measurement(130.23 * u.mas, 0.36 * u.mas, source="simbad"))
    sed = fuse(VizierPhotometry().sed("Vega"))
    completed = complete_star(vega, sed, wavelength_range=(0.4 * u.um, 2.5 * u.um))
    assert completed["teff"].estimated
    assert completed["radius"].estimated
    assert completed["distance"].estimated
    assert math.isfinite(completed.radius.to_value(u.R_sun))
    assert "teff" not in vega  # the original record is unchanged


def test_complete_star_keeps_measured_values(replay):
    vega = StarRecord(
        "Vega",
        parallax=Measurement(130.23 * u.mas, source="simbad"),
        teff=Measurement(9602 * u.K, source="literature"),
    )
    sed = fuse(VizierPhotometry().sed("Vega"))
    completed = complete_star(vega, sed, wavelength_range=(0.4 * u.um, 2.5 * u.um))
    assert completed["teff"].estimated is False
    assert completed.teff == 9602 * u.K
    radius = completed.radius.to_value(u.R_sun)  # solid angle fitted at the measured temperature
    assert 2.36 * 0.9 < radius < 2.82 * 1.1
