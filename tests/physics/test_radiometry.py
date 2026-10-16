"""Tests for radiometry (step P8.2, REQ-SCN-5).

Requirements: spectral radiance, irradiance, flux collected by a pupil, étendue, photon ↔ energy conversions,
magnitude systems and bands.

Independent references:

- Stefan-Boltzmann law: the exitance of a blackbody, ``∫ π B_λ dλ``, is ``σ T⁴`` (σ from astropy.constants);
- Wien's displacement law: ``λ_max T = b = 2.897771955e-3 m K`` (CODATA 2018);
- photon energy ``h c / λ`` (astropy.constants);
- the solid angle of a cone of half-angle α is ``2π (1 - cos α)``; the étendue of a pupil of area A
  receiving a cone of half-angle α from a Lambertian source is ``π A sin² α`` (Welford & Winston, *High
  Collection Nonimaging Optics*, §2.6; Born & Wolf §4.8);
- the exitance of a Lambertian surface of radiance L is ``π L``; a uniform disk of radiance L subtending an
  angular radius θ produces the irradiance ``π L sin² θ`` on a surface facing it (McCluney, *Introduction to
  Radiometry and Photometry*, §2.8);
- AB magnitudes: ``m = -2.5 log10(F_ν / 3631 Jy)`` (Oke & Gunn 1983, ApJ 266, 713).
"""

import math

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u

from lumos.physics import radiometry as rad

SPECTRAL_RADIANCE = u.W / u.m**2 / u.sr / u.m


@pytest.mark.parametrize("temperature", [300.0, 5772.0, 10000.0])
def test_stefan_boltzmann(temperature):
    peak = 2.897771955e-3 / temperature
    lam = np.geomspace(0.05 * peak, 200 * peak, 200_000) * u.m
    exitance = rad.lambertian_exitance(rad.blackbody_radiance(temperature * u.K, lam))
    total = rad.integrate_spectrum(exitance, lam)
    expected = const.sigma_sb * (temperature * u.K) ** 4
    assert total.to_value(u.W / u.m**2) == pytest.approx(expected.to_value(u.W / u.m**2), rel=1e-4)


def test_wien_displacement():
    temperature = 5772.0
    lam = np.linspace(400, 600, 20001) * u.nm
    radiance = rad.blackbody_radiance(temperature * u.K, lam)
    peak = lam[np.argmax(radiance.value)]
    assert peak.to_value(u.m) * temperature == pytest.approx(2.897771955e-3, rel=1e-4)


def test_blackbody_radiance_units():
    radiance = rad.blackbody_radiance(5000 * u.K, 1 * u.um)
    assert radiance.unit.is_equivalent(SPECTRAL_RADIANCE)


def test_photon_energy():
    energy = rad.photon_energy(500 * u.nm)
    expected = const.h * const.c / (500 * u.nm)
    assert energy.to_value(u.J) == pytest.approx(expected.to_value(u.J), rel=1e-12)


def test_photon_and_energy_conversions():
    lam = 633 * u.nm
    photons = rad.to_photons(1 * u.W, lam)
    assert photons.unit.is_equivalent(u.photon / u.s)
    expected = 1 / (const.h * const.c / lam).to_value(u.J)
    assert photons.to_value(u.photon / u.s) == pytest.approx(expected, rel=1e-12)
    back = rad.to_energy(photons, lam)
    assert back.to_value(u.W) == pytest.approx(1.0, rel=1e-12)


def test_conversions_of_spectral_quantities():
    lam = np.array([0.5, 1.0]) * u.um
    radiance = np.array([1.0, 2.0]) * SPECTRAL_RADIANCE
    photons = rad.to_photons(radiance, lam)
    expected = radiance.value / (const.h * const.c / lam).to_value(u.J)
    np.testing.assert_allclose(photons.to_value(u.photon / u.s / u.m**2 / u.sr / u.m), expected, rtol=1e-12)


def test_cone_solid_angle():
    assert rad.cone_solid_angle(90 * u.deg).to_value(u.sr) == pytest.approx(2 * math.pi)
    alpha = 0.1
    assert rad.cone_solid_angle(alpha * u.rad).to_value(u.sr) == pytest.approx(
        2 * math.pi * (1 - math.cos(alpha))
    )


def test_etendue():
    area, alpha = 2.0, 0.05
    assert rad.etendue(area * u.m**2, alpha * u.rad).to_value(u.m**2 * u.sr) == pytest.approx(
        math.pi * area * math.sin(alpha) ** 2
    )


def test_lambertian_exitance():
    assert rad.lambertian_exitance(2 * u.W / u.m**2 / u.sr).to_value(u.W / u.m**2) == pytest.approx(
        2 * math.pi
    )


def test_irradiance_of_a_uniform_disk():
    theta = 0.2
    radiance = 3 * u.W / u.m**2 / u.sr
    assert rad.disk_irradiance(radiance, theta * u.rad).to_value(u.W / u.m**2) == pytest.approx(
        3 * math.pi * math.sin(theta) ** 2
    )


def test_power_collected_by_a_pupil():
    radiance = 2 * u.W / u.m**2 / u.sr
    power = rad.pupil_power(radiance, area=0.5 * u.m**2, solid_angle=1e-6 * u.sr)
    assert power.to_value(u.W) == pytest.approx(1e-6)


def test_ab_magnitudes_round_trip():
    assert rad.flux_from_ab_magnitude(0.0).to_value(u.Jy) == pytest.approx(3631)
    assert rad.flux_from_ab_magnitude(5.0).to_value(u.Jy) == pytest.approx(3631 / 100)
    assert rad.ab_magnitude(36.31 * u.Jy) == pytest.approx(5.0)


def test_band_averaged_flux_of_a_flat_spectrum():
    lam = np.linspace(1.0, 2.0, 101) * u.um
    flux = np.full(101, 7.0) * u.W / u.m**2 / u.um
    assert rad.integrate_spectrum(flux, lam).to_value(u.W / u.m**2) == pytest.approx(7.0)


def test_integrate_refuses_mismatched_shapes():
    with pytest.raises(ValueError, match="shape"):
        rad.integrate_spectrum(np.ones(3) * u.W / u.m**2 / u.um, np.linspace(1, 2, 4) * u.um)
