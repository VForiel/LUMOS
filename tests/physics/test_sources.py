"""Tests for point and laboratory sources and their physics (step P6.1).

Requirements: REQ-SCN-2 (point components with arbitrary spectra, position and motion; Star, Planet and
planetary systems with orbital geometry), REQ-SCN-5 (radiometry: spectral radiance, photons ↔ energy,
magnitude systems and bands), §5.9 (laboratory sources: Gaussian laser, incoherent lamp).

Independent references:

- Planck's law is compared with ``astropy.modeling.models.BlackBody``; its integral is the Stefan-Boltzmann
  law ``π ∫ B_λ dλ = σ T⁴`` (σ from ``astropy.constants``);
- AB magnitudes: ``m_AB = -2.5 log10(F_ν / 3631 Jy)`` (Oke & Gunn 1983, ApJ 266, 713);
- Vega zero points of the Johnson-Cousins-Glass system: V 3636 Jy, J 1589 Jy, K 640 Jy (Bessell, Castelli &
  Plez 1998, A&A 333, 231, table A2);
- a star of radius R at distance d has the flux density ``F_λ = π B_λ(T) (R/d)²`` (Rybicki & Lightman §1.4);
- Kepler's third law ``P = 2π sqrt(a³ / G M)``: 1 au around 1 solar mass gives 365.25 days (within 0.1 %);
  the Kepler equation ``E - e sin E = M`` is checked as an identity on its solution;
- Lambert phase function ``Φ(α) = [sin α + (π - α) cos α] / π`` (Sobolev 1975; Seager, *Exoplanet
  Atmospheres*, eq. 3.32): Φ(0) = 1, Φ(π/2) = 1/π;
- a Lambertian emitter of radiance L and area A emits ``P = π L A`` (Lambert's cosine law);
- a Gaussian laser beam carries its power (Siegman §16.1).
"""

import math

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u
from astropy.modeling.models import BlackBody
from hypothesis import given
from hypothesis import strategies as st

from lumos import Context
from lumos.elements.sources import GaussianLaser, IncoherentLamp, PlanetarySystem, PointSource, Star
from lumos.fields import RadianceField, SourceList, Wavefront
from lumos.physics.blackbody import planck_lambda, planck_photon_lambda
from lumos.physics.orbits import kepler_position, orbital_period, solve_kepler
from lumos.physics.photometry import (
    BANDS,
    ab_magnitude,
    flux_from_magnitude,
    lambert_phase,
    magnitude_from_flux,
)

PHOTON_FLUX = u.photon / u.s / u.m**2 / u.m


def band_context(low=1.0, high=2.0, n=5):
    ctx = Context(seed=1)
    ctx.observation.band = (low * u.um, high * u.um)
    ctx.observation.n_wavelengths = n
    return ctx


# --- Blackbody ---------------------------------------------------------------------------------


@given(temperature=st.floats(300, 30000), wavelength_um=st.floats(0.2, 20))
def test_planck_matches_astropy(temperature, wavelength_um):
    lam = wavelength_um * u.um
    ours = planck_lambda(lam, temperature * u.K)
    reference = BlackBody(temperature=temperature * u.K)(lam).to(
        u.W / u.m**2 / u.sr / u.m, equivalencies=u.spectral_density(lam)
    )
    assert ours.to_value(u.W / u.m**2 / u.sr / u.m) == pytest.approx(reference.value, rel=1e-10)


def test_planck_integrates_to_stefan_boltzmann():
    temperature = 5772 * u.K
    lam = np.geomspace(0.02, 500, 200_000) * u.um
    radiance = planck_lambda(lam, temperature)
    integral = np.trapezoid(radiance.to_value(u.W / u.m**2 / u.sr / u.m), lam.to_value(u.m))
    sigma_t4 = (const.sigma_sb * temperature**4).to_value(u.W / u.m**2)
    assert math.pi * integral == pytest.approx(sigma_t4, rel=1e-4)


def test_photon_radiance_is_energy_radiance_over_photon_energy():
    lam, temperature = 1.6 * u.um, 3000 * u.K
    energy = planck_lambda(lam, temperature).to_value(u.W / u.m**2 / u.sr / u.m)
    photons = planck_photon_lambda(lam, temperature).to_value(u.photon / u.s / u.m**2 / u.sr / u.m)
    assert photons == pytest.approx(energy * 1.6e-6 / (const.h.value * const.c.value), rel=1e-12)


# --- Photometry --------------------------------------------------------------------------------


def test_ab_magnitude_definition():
    assert ab_magnitude(3631 * u.Jy) == pytest.approx(0.0, abs=1e-12)
    assert ab_magnitude(363.1 * u.Jy) == pytest.approx(2.5, rel=1e-12)


@pytest.mark.parametrize(("band", "zero_point"), [("V", 3636), ("J", 1589), ("K", 640)])
def test_vega_zero_points_of_bessell_1998(band, zero_point):
    assert BANDS[band].zero_point.to_value(u.Jy) == pytest.approx(zero_point)
    flux = flux_from_magnitude(0.0, band)
    assert flux.to_value(u.Jy) == pytest.approx(zero_point)
    assert magnitude_from_flux(flux * 10 ** (-0.4 * 7.5), band) == pytest.approx(7.5, rel=1e-12)


def test_lambert_phase_function():
    assert lambert_phase(0 * u.deg) == pytest.approx(1.0)
    assert lambert_phase(90 * u.deg) == pytest.approx(1 / math.pi)
    assert lambert_phase(180 * u.deg) == pytest.approx(0.0, abs=1e-15)


# --- Orbits ------------------------------------------------------------------------------------


def test_kepler_third_law_for_the_earth():
    period = orbital_period(1 * u.au, 1 * u.M_sun)
    assert period.to_value(u.day) == pytest.approx(365.25, rel=1e-3)


@given(e=st.floats(0, 0.95), mean_anomaly=st.floats(-10, 10))
def test_kepler_equation_is_solved(e, mean_anomaly):
    eccentric = solve_kepler(mean_anomaly, e)
    assert eccentric - e * math.sin(eccentric) == pytest.approx(mean_anomaly, abs=1e-10)


def test_circular_face_on_orbit_keeps_its_separation():
    period = 2.0
    for t in np.linspace(0, 2, 9):
        x, y = kepler_position(t, period=period, a=1.0, e=0.0, inclination=0.0)
        assert math.hypot(x, y) == pytest.approx(1.0, rel=1e-12)
    x, y = kepler_position(0.5, period=period, a=1.0, e=0.0, inclination=0.0)  # a quarter of the orbit
    assert (x, y) == pytest.approx((0.0, 1.0), abs=1e-12)


def test_edge_on_orbit_projects_on_a_line():
    x, y = kepler_position(0.5, period=2.0, a=1.0, e=0.0, inclination=math.pi / 2)
    assert y == pytest.approx(0.0, abs=1e-12)
    assert x == pytest.approx(0.0, abs=1e-12)  # quadrature seen edge-on: behind or in front of the star


# --- Source list signal ------------------------------------------------------------------------


def test_source_list_photon_rate():
    lam = [1.0, 2.0] * u.um
    sources = SourceList(
        theta_x=[0, 1] * u.arcsec,
        theta_y=[0, 0] * u.arcsec,
        wavelengths=lam,
        bandwidths=[0.5, 0.5] * u.um,
        flux=np.array([[1.0, 2.0], [3.0, 4.0]]) * PHOTON_FLUX,
    )
    assert sources.n_sources == 2
    np.testing.assert_allclose(sources.photon_rate(2 * u.m**2).to_value(u.photon / u.s), [3.0e-6, 7.0e-6])


def test_source_list_shapes_are_checked():
    with pytest.raises(ValueError):
        SourceList(
            theta_x=[0] * u.rad,
            theta_y=[0] * u.rad,
            wavelengths=[1, 2] * u.um,
            bandwidths=[1, 1] * u.um,
            flux=np.ones((1, 3)) * PHOTON_FLUX,
        )


# --- Source elements ---------------------------------------------------------------------------


def test_point_source_with_a_flat_spectrum_in_jansky():
    ctx = band_context(1.0, 2.0, 4)
    src = ctx.add(PointSource(name="src", flux_density=10 * u.Jy, theta_x=2 * u.mas, theta_y=-1 * u.mas))
    sources = ctx.run(["src.out"])["src.out"]
    lam = sources.wavelengths
    expected = (10 * u.Jy).to(PHOTON_FLUX, equivalencies=u.spectral_density(lam))
    np.testing.assert_allclose(sources.flux[0].to_value(PHOTON_FLUX), expected.value, rtol=1e-12)
    assert sources.theta_x[0].to_value(u.mas) == pytest.approx(2.0)
    np.testing.assert_allclose(sources.bandwidths.to_value(u.um), 0.25)
    assert src.observation_dependent


def test_star_from_radius_and_distance():
    ctx = band_context(0.5, 0.6, 3)
    ctx.add(Star(name="sun", temperature=5772 * u.K, radius=1 * u.R_sun, distance=10 * u.pc))
    sources = ctx.run(["sun.out"])["sun.out"]
    lam = sources.wavelengths
    expected = (
        math.pi * u.sr * BlackBody(temperature=5772 * u.K)(lam) * (const.R_sun / (10 * u.pc)) ** 2
    ).to(PHOTON_FLUX, equivalencies=u.spectral_density(lam))
    np.testing.assert_allclose(sources.flux[0].to_value(PHOTON_FLUX), expected.value, rtol=1e-9)


def test_star_scaled_to_a_magnitude():
    # Convention: F_nu(lambda_eff) = zero point × 10^(-0.4 m); lambda_eff(V) = 0.545 um (Bessell et al. 1998).
    ctx = band_context(0.5449, 0.5451, 1)  # one sample, exactly at 0.545 um
    ctx.add(Star(name="star", temperature=9600 * u.K, magnitude=5.0, band="V"))
    sources = ctx.run(["star.out"])["star.out"]
    assert sources.wavelengths[0].to_value(u.um) == pytest.approx(0.545)
    flux_nu = sources.flux[0, 0].to(u.Jy, equivalencies=u.spectral_density(sources.wavelengths[0]))
    assert flux_nu.value == pytest.approx(3636 * 10 ** (-0.4 * 5.0), rel=1e-9)
    assert BANDS["V"].effective_wavelength.to_value(u.um) == pytest.approx(0.545)


def test_planetary_system_geometry_and_reflected_light():
    ctx = band_context(0.5, 0.6, 2)
    system = ctx.add(
        PlanetarySystem(
            name="sys",
            star_temperature=5772 * u.K,
            star_radius=1 * u.R_sun,
            star_mass=1 * u.M_sun,
            distance=10 * u.pc,
            planet_radius=1 * u.R_earth,
            semi_major_axis=1 * u.au,
            albedo=0.3,
            inclination=0 * u.deg,
        )
    )
    sources = ctx.run(["sys.out"])["sys.out"]
    assert sources.n_sources == 2
    separation = math.hypot(sources.theta_x[1].to_value(u.rad), sources.theta_y[1].to_value(u.rad))
    assert separation == pytest.approx((1 * u.au / (10 * u.pc)).decompose().value, rel=1e-9)
    # face-on circular orbit: phase angle 90°, reflected contrast A_g Φ(90°) (R_p / a)²
    contrast = sources.flux[1] / sources.flux[0]
    expected = 0.3 / math.pi * float((const.R_earth / const.au).decompose()) ** 2
    np.testing.assert_allclose(contrast.decompose().value, expected, rtol=1e-6)
    assert system.time_dependent


def test_planet_moves_with_time():
    ctx = band_context(0.5, 0.6, 1)
    ctx.add(
        PlanetarySystem(
            name="sys",
            star_temperature=5772 * u.K,
            star_radius=1 * u.R_sun,
            star_mass=1 * u.M_sun,
            distance=10 * u.pc,
            planet_radius=1 * u.R_earth,
            semi_major_axis=1 * u.au,
            albedo=0.3,
            inclination=0 * u.deg,
        )
    )
    first = ctx.run(["sys.out"])["sys.out"]
    period = orbital_period(1 * u.au, 1 * u.M_sun)
    ctx.wait(period / 4)
    second = ctx.run(["sys.out"])["sys.out"]
    p1 = np.array([first.theta_x[1].value, first.theta_y[1].value])
    p2 = np.array([second.theta_x[1].value, second.theta_y[1].value])
    assert np.dot(p1, p2) == pytest.approx(0.0, abs=1e-6 * np.dot(p1, p1))  # a quarter of a turn


def test_gaussian_laser_emits_a_wavefront_with_its_power():
    ctx = Context()
    ctx.add(
        GaussianLaser(
            name="laser",
            wavelength=633 * u.nm,
            power=2 * u.mW,
            waist=50 * u.um,
            pixel_scale=5 * u.um,
            n_pixels=128,
        )
    )
    wf = ctx.run(["laser.out"])["laser.out"]
    assert isinstance(wf, Wavefront)
    assert wf.power.to_value(u.mW) == pytest.approx(2.0, rel=1e-9)
    assert wf.wavelength == 633 * u.nm


def test_incoherent_lamp_follows_lambert():
    ctx = band_context(0.5, 0.7, 2)
    ctx.add(IncoherentLamp(name="lamp", power=1 * u.W, diameter=1 * u.mm, pixel_scale=20 * u.um, n_pixels=64))
    radiance = ctx.run(["lamp.out"])["lamp.out"]
    assert isinstance(radiance, RadianceField)
    # π ∫ L dA dλ = P (Lambertian emitter, flat spectrum over the band)
    integrated = radiance.integrated()  # Σ L ΔA per wavelength, W sr⁻¹ m⁻¹
    bandwidths = np.full(2, 0.1e-6)
    total = math.pi * np.sum(integrated.to_value(u.W / u.sr / u.m) * bandwidths)
    assert total == pytest.approx(1.0, rel=2e-2)  # disk digitisation on 50 pixels across
