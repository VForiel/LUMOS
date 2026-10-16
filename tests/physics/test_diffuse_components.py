"""Tests for the parametric diffuse scene components (step P8.3, REQ-SCN-3).

Requirements: uniform disk, Gaussian, Sérsic profile, inclined ring or disk (debris disks, exo-zodi), local
zodiacal light and uniform background, each with its spectrum.

Independent references:

- a spectral flux density per unit frequency becomes ``F_λ = F_ν c / λ²`` per unit wavelength (astropy's
  ``spectral_density`` equivalency); a surface brightness in Jy arcsec⁻² is converted the same way per
  steradian;
- blackbody spectra: astropy's ``BlackBody`` model gives ``B_ν(T)``;
- a Gaussian of full width at half maximum ``FWHM`` has ``σ = FWHM / (2 √(2 ln 2))``;
- the effective radius of a Sérsic profile encloses half of the light (Sérsic 1963; Graham & Driver 2005,
  PASA 22, 118, §2.1), whatever the index;
- a thin circular ring of inclination ``i`` projects onto an ellipse of axis ratio ``cos i`` (geometry); its
  major axis is at the position angle, measured from North (+y) towards East (+x) (IAU convention);
- a uniform disk of radius R holding the flux F has the surface brightness ``F / (π R²)``.
"""

import math

import numpy as np
import pytest
from astropy import units as u
from astropy.modeling.models import BlackBody

import lumos.io
from lumos import Context
from lumos.fields import RadianceField
from lumos.scenes import Scene
from lumos.scenes.diffuse import (
    GaussianComponent,
    InclinedRing,
    SersicComponent,
    UniformBackground,
    UniformDisk,
    ZodiacalLight,
)

RADIANCE = u.W / u.m**2 / u.sr / u.m
F_LAMBDA = u.W / u.m**2 / u.m


def context(n=3, low=1.0, high=2.0):
    ctx = Context(seed=1)
    ctx.observation.band = (low * u.um, high * u.um)
    ctx.observation.n_wavelengths = n
    return ctx


def evaluate(component, **band):
    ctx = context(**band)
    ctx.add(component)
    field = ctx.run([f"{ctx.id_of(component)}.out"])[f"{ctx.id_of(component)}.out"]
    assert isinstance(field, RadianceField)
    return field


def f_lambda(fnu, wavelengths):
    return fnu.to(F_LAMBDA, equivalencies=u.spectral_density(wavelengths))


# --- Uniform disk ----------------------------------------------------------------------------------------


def test_uniform_disk_total_flux():
    disk = UniformDisk(
        name="disk", radius=50 * u.mas, flux_density=2 * u.Jy, n_pixels=64, pixel_scale=2 * u.mas
    )
    field = evaluate(disk)
    expected = f_lambda(2 * u.Jy, field.wavelengths)
    np.testing.assert_allclose(field.integrated().to_value(F_LAMBDA), expected.value, rtol=1e-3)


def test_uniform_disk_surface_brightness():
    radius = 50 * u.mas
    disk = UniformDisk(name="disk", radius=radius, flux_density=2 * u.Jy, n_pixels=64, pixel_scale=2 * u.mas)
    field = evaluate(disk)
    centre = field.radiance[:, 32, 32]
    area = math.pi * radius.to_value(u.rad) ** 2
    expected = f_lambda(2 * u.Jy, field.wavelengths).value / area
    np.testing.assert_allclose(centre.to_value(RADIANCE), expected, rtol=1e-3)


def test_offset_moves_the_centroid():
    disk = UniformDisk(
        name="disk", radius=10 * u.mas, x0=20 * u.mas, y0=-10 * u.mas, n_pixels=64, pixel_scale=2 * u.mas
    )
    field = evaluate(disk)
    image = field.radiance[0].value
    x, y = (np.asarray(c) for c in field.grid.coords(backend="numpy"))
    cx = (image * x).sum() / image.sum()
    cy = (image * y).sum() / image.sum()
    assert cx == pytest.approx((20 * u.mas).to_value(u.rad), rel=1e-3)
    assert cy == pytest.approx((-10 * u.mas).to_value(u.rad), rel=1e-3)


# --- Spectra ------------------------------------------------------------------------------------------------


def test_blackbody_spectrum_shape():
    temperature = 3000 * u.K
    disk = UniformDisk(
        name="disk",
        radius=20 * u.mas,
        flux_density=1 * u.Jy,
        reference_wavelength=1.5 * u.um,
        spectrum="blackbody",
        temperature=temperature,
        n_pixels=32,
        pixel_scale=2 * u.mas,
    )
    field = evaluate(disk)
    fnu = field.integrated().to(u.Jy, equivalencies=u.spectral_density(field.wavelengths))
    bb = BlackBody(temperature)
    expected = bb(field.wavelengths) / bb(1.5 * u.um)
    np.testing.assert_allclose(fnu.to_value(u.Jy), expected.decompose().value, rtol=2e-3)


def test_flat_spectrum_is_constant_in_f_nu():
    disk = UniformDisk(
        name="disk", radius=20 * u.mas, flux_density=5 * u.Jy, n_pixels=32, pixel_scale=2 * u.mas
    )
    field = evaluate(disk)
    fnu = field.integrated().to(u.Jy, equivalencies=u.spectral_density(field.wavelengths))
    np.testing.assert_allclose(fnu.to_value(u.Jy), 5.0, rtol=1e-3)


# --- Gaussian -----------------------------------------------------------------------------------------------


def test_gaussian_width_and_flux():
    fwhm = 30 * u.mas
    blob = GaussianComponent(name="g", fwhm=fwhm, flux_density=1 * u.Jy, n_pixels=128, pixel_scale=1 * u.mas)
    field = evaluate(blob, n=1)
    image = field.radiance[0].value
    row = image[64, 64:]  # from the centre (index N // 2) towards +x, one value per 1 mas pixel
    x = np.arange(row.size) * 1.0
    half = np.interp(row[0] / 2, row[::-1], x[::-1])  # half-maximum crossing, linear interpolation
    assert 2 * half == pytest.approx(30.0, rel=2e-2)
    expected = f_lambda(1 * u.Jy, field.wavelengths)
    np.testing.assert_allclose(field.integrated().to_value(F_LAMBDA), expected.value, rtol=1e-3)


# --- Sérsic -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("index", [1.0, 4.0])
def test_sersic_effective_radius_holds_half_the_light(index):
    re = 20 * u.mas
    galaxy = SersicComponent(
        name="galaxy",
        effective_radius=re,
        index=index,
        flux_density=1 * u.Jy,
        n_pixels=512,
        pixel_scale=0.5 * u.mas,
    )
    field = evaluate(galaxy, n=1)
    image = field.radiance[0].value
    x, y = (np.asarray(c) for c in field.grid.coords(backend="numpy"))
    inside = np.hypot(x, y) <= re.to_value(u.rad)
    total = f_lambda(1 * u.Jy, field.wavelengths)[0].value
    pixel = field.grid.dx_si * field.grid.dy_si
    fraction = image[inside].sum() * pixel / total
    assert fraction == pytest.approx(0.5, abs=0.02)


# --- Inclined ring ------------------------------------------------------------------------------------------


def extents(image, grid, level=0.05):
    x, y = (np.asarray(c) for c in grid.coords(backend="numpy"))
    mask = image > level * image.max()
    return np.ptp(x[mask]), np.ptp(y[mask])


@pytest.mark.parametrize("inclination", [0.0, 60.0])
def test_inclined_ring_projects_with_cos_i(inclination):
    ring = InclinedRing(
        name="ring",
        inner_radius=40 * u.mas,
        outer_radius=50 * u.mas,
        inclination=inclination * u.deg,
        flux_density=1 * u.Jy,
        n_pixels=256,
        pixel_scale=0.5 * u.mas,
    )
    field = evaluate(ring, n=1)
    width, height = extents(field.radiance[0].value, field.grid)
    assert width / height == pytest.approx(math.cos(math.radians(inclination)), abs=0.03)
    expected = f_lambda(1 * u.Jy, field.wavelengths)
    np.testing.assert_allclose(field.integrated().to_value(F_LAMBDA), expected.value, rtol=1e-3)


def test_position_angle_rotates_the_ring():
    ring = InclinedRing(
        name="ring",
        inner_radius=40 * u.mas,
        outer_radius=50 * u.mas,
        inclination=60 * u.deg,
        position_angle=90 * u.deg,
        n_pixels=256,
        pixel_scale=0.5 * u.mas,
    )
    field = evaluate(ring, n=1)
    width, height = extents(field.radiance[0].value, field.grid)
    assert height / width == pytest.approx(0.5, abs=0.03)


# --- Backgrounds --------------------------------------------------------------------------------------------


def test_uniform_background_surface_brightness():
    background = UniformBackground(
        name="sky", surface_brightness=2e-3 * u.Jy / u.arcsec**2, n_pixels=16, pixel_scale=10 * u.mas
    )
    field = evaluate(background)
    values = field.radiance.to_value(RADIANCE)
    sb = (2e-3 * u.Jy / u.arcsec**2).to(u.Jy / u.sr)
    expected = (sb * u.sr).to(F_LAMBDA, equivalencies=u.spectral_density(field.wavelengths)).value
    for k in range(len(expected)):
        np.testing.assert_allclose(values[k], expected[k], rtol=1e-10)


def test_zodiacal_light_matches_its_reference_brightness():
    zodi = ZodiacalLight(
        name="zodi",
        surface_brightness=1e-4 * u.Jy / u.arcsec**2,
        reference_wavelength=1.5 * u.um,
        n_pixels=8,
        pixel_scale=10 * u.mas,
    )
    field = evaluate(zodi, n=3, low=1.0, high=2.0)  # channels at 1.1667, 1.5, 1.8333 µm
    values = field.radiance.to(
        u.Jy / u.sr, equivalencies=u.spectral_density(field.wavelengths[:, None, None])
    )
    middle = values[1].to_value(u.Jy / u.arcsec**2)
    np.testing.assert_allclose(middle, 1e-4, rtol=1e-6)
    assert np.all(values.value > 0)
    assert np.ptp(values[0].value) == 0.0


# --- Integration --------------------------------------------------------------------------------------------


def test_components_feed_a_scene_and_round_trip(tmp_path):
    ctx = context()
    scene = ctx.add(Scene(name="sky", n_pixels=64, pixel_scale=2 * u.mas))
    scene.add(
        UniformDisk(name="disk", radius=20 * u.mas, flux_density=1 * u.Jy, n_pixels=32, pixel_scale=2 * u.mas)
    )
    scene.add(GaussianComponent(name="blob", fwhm=10 * u.mas, flux_density=1 * u.Jy, x0=30 * u.mas))
    total = ctx.run(["sky.radiance"])["sky.radiance"].integrated().to_value(F_LAMBDA)
    path = tmp_path / "scene.lumos"
    lumos.io.save(ctx, path)
    again = lumos.io.load(path).run(["sky.radiance"])["sky.radiance"].integrated().to_value(F_LAMBDA)
    np.testing.assert_allclose(again, total, rtol=1e-12)
    expected = 2 * f_lambda(1 * u.Jy, ctx.observation.wavelengths()[0]).value
    np.testing.assert_allclose(total, expected, rtol=1e-2)
