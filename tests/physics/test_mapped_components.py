"""Tests for mapped scene components (step P8.4, REQ-SCN-4).

Requirements: import radiance maps (FITS with WCS, hyperspectral cubes, calibrated PNG/TIFF images) with the
conversion of radiometric units and a resampling that conserves the flux.

Independent references:

- astropy.io.fits and astropy.wcs write the test files (FITS standard 4.0: ``CDELTi``, ``CUNITi``,
  ``CTYPEi``, ``BUNIT``); the pixel solid angle of a small pixel is ``|CDELT1 CDELT2|``;
- unit conversions: astropy's ``spectral_density`` equivalency (``F_λ = F_ν c / λ²``);
- sky orientation (FITS WCS paper II, Calabretta & Greisen 2002, §5): with ``CDELT1 < 0`` the right
  ascension, i.e. East, decreases with the column index; LUMOS draws East towards +x (§9 and
  :mod:`lumos.elements.telescopes`), so the first column of such an image is the easternmost one;
- linear interpolation between the planes of a cube reproduces a spectrum linear in wavelength exactly;
- matplotlib's ``imread`` reads the reference values of PNG images.
"""

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits

import lumos.io
from lumos import Context
from lumos.fields import RadianceField
from lumos.scenes import Scene
from lumos.scenes.mapped import MappedComponent

RADIANCE = u.W / u.m**2 / u.sr / u.m
F_LAMBDA = u.W / u.m**2 / u.m


def context(n=2, low=1.0, high=2.0):
    ctx = Context(seed=1)
    ctx.observation.band = (low * u.um, high * u.um)
    ctx.observation.n_wavelengths = n
    return ctx


def run(component, ctx=None):
    ctx = ctx or context()
    ctx.add(component)
    key = f"{ctx.id_of(component)}.out"
    return ctx.run([key])[key], ctx


def write_image(path, data, *, cdelt_mas=(-2.0, 2.0), bunit="Jy/pixel"):
    header = fits.Header()
    header["CTYPE1"], header["CTYPE2"] = "RA---TAN", "DEC--TAN"
    header["CUNIT1"], header["CUNIT2"] = "deg", "deg"
    header["CDELT1"] = cdelt_mas[0] / 3.6e6
    header["CDELT2"] = cdelt_mas[1] / 3.6e6
    header["CRPIX1"], header["CRPIX2"] = (data.shape[1] + 1) / 2, (data.shape[0] + 1) / 2
    header["CRVAL1"], header["CRVAL2"] = 83.8, -5.4
    header["BUNIT"] = bunit
    fits.PrimaryHDU(data, header).writeto(path)
    return path


# --- FITS images --------------------------------------------------------------------------------------------


def test_fits_image_in_jy_per_pixel_conserves_the_flux(tmp_path):
    data = np.arange(1.0, 13.0).reshape(3, 4)
    path = write_image(tmp_path / "map.fits", data)
    field, _ = run(MappedComponent.from_fits(path, name="map", reference_wavelength=1 * u.um))
    assert isinstance(field, RadianceField)
    total = data.sum() * u.Jy
    expected = total.to(F_LAMBDA, equivalencies=u.spectral_density(field.wavelengths))
    np.testing.assert_allclose(field.integrated().to_value(F_LAMBDA), expected.value, rtol=1e-10)


def test_fits_pixel_scale_comes_from_the_wcs(tmp_path):
    path = write_image(tmp_path / "map.fits", np.ones((4, 6)), cdelt_mas=(-3.0, 2.0))
    field, _ = run(MappedComponent.from_fits(path, name="map"))
    assert field.grid.shape == (4, 6)
    assert field.grid.dx.to_value(u.mas) == pytest.approx(3.0)
    assert field.grid.dy.to_value(u.mas) == pytest.approx(2.0)


def test_surface_brightness_units_are_converted(tmp_path):
    data = np.full((4, 4), 5.0)
    path = write_image(tmp_path / "map.fits", data, bunit="MJy/sr")
    field, _ = run(MappedComponent.from_fits(path, name="map"))
    expected = (5.0 * u.MJy / u.sr * u.sr).to(F_LAMBDA, equivalencies=u.spectral_density(field.wavelengths))
    np.testing.assert_allclose(field.radiance[:, 0, 0].to_value(RADIANCE), expected.value, rtol=1e-10)


def test_east_is_towards_positive_x(tmp_path):
    data = np.zeros((5, 5))
    data[2, 0] = 1.0  # first column: easternmost pixel when CDELT1 < 0
    path = write_image(tmp_path / "map.fits", data, cdelt_mas=(-1.0, 1.0))
    field, _ = run(MappedComponent.from_fits(path, name="map"))
    image = field.radiance[0].value
    row, column = np.unravel_index(np.argmax(image), image.shape)
    assert (row, column) == (2, 4)


def test_north_up_is_kept(tmp_path):
    data = np.zeros((5, 5))
    data[4, 2] = 1.0  # last row: northernmost pixel when CDELT2 > 0
    path = write_image(tmp_path / "map.fits", data, cdelt_mas=(-1.0, 1.0))
    field, _ = run(MappedComponent.from_fits(path, name="map"))
    image = field.radiance[0].value
    assert np.unravel_index(np.argmax(image), image.shape) == (4, 2)


def test_unknown_or_missing_units_are_refused(tmp_path):
    path = write_image(tmp_path / "map.fits", np.ones((2, 2)), bunit="K")
    with pytest.raises(u.UnitsError):
        MappedComponent.from_fits(path, name="map")


# --- Cubes --------------------------------------------------------------------------------------------------


def test_spectral_cube_is_interpolated_to_the_channels(tmp_path):
    lam = np.array([0.5, 1.0, 1.5, 2.0, 2.5])  # µm
    cube = (lam[:, None, None] * 2.0 + 1.0) * np.ones((5, 3, 3))  # Jy/pixel, linear in λ
    header = fits.Header()
    header["CTYPE1"], header["CTYPE2"], header["CTYPE3"] = "RA---TAN", "DEC--TAN", "WAVE"
    header["CUNIT1"], header["CUNIT2"], header["CUNIT3"] = "deg", "deg", "um"
    header["CDELT1"], header["CDELT2"], header["CDELT3"] = -1 / 3.6e6, 1 / 3.6e6, 0.5
    header["CRPIX1"], header["CRPIX2"], header["CRPIX3"] = 2, 2, 1
    header["CRVAL1"], header["CRVAL2"], header["CRVAL3"] = 10.0, 20.0, 0.5
    header["BUNIT"] = "Jy/pixel"
    path = tmp_path / "cube.fits"
    fits.PrimaryHDU(cube, header).writeto(path)
    field, ctx = run(MappedComponent.from_fits(path, name="cube"), context(n=2, low=1.0, high=2.0))
    channels = field.wavelengths.to_value(u.um)  # 1.25 and 1.75 µm
    expected_jy = (channels * 2.0 + 1.0) * 9
    expected = (expected_jy * u.Jy).to(F_LAMBDA, equivalencies=u.spectral_density(field.wavelengths))
    np.testing.assert_allclose(field.integrated().to_value(F_LAMBDA), expected.value, rtol=1e-10)


# --- Arrays and images --------------------------------------------------------------------------------------


def test_from_array_with_a_pixel_scale():
    data = np.ones((4, 4)) * 2.0 * u.Jy / u.arcsec**2
    component = MappedComponent.from_array(data, pixel_scale=0.5 * u.arcsec, name="arr")
    field, _ = run(component)
    expected = (2.0 * u.Jy / u.arcsec**2 * 16 * (0.5 * u.arcsec) ** 2).to(
        F_LAMBDA, equivalencies=u.spectral_density(field.wavelengths)
    )
    np.testing.assert_allclose(field.integrated().to_value(F_LAMBDA), expected.value, rtol=1e-10)


def test_from_array_refuses_non_radiometric_units():
    with pytest.raises(u.UnitsError):
        MappedComponent.from_array(np.ones((2, 2)) * u.m, pixel_scale=1 * u.mas)


def test_calibrated_png(tmp_path):
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    gray = np.linspace(0.0, 1.0, 12).reshape(3, 4)
    path = tmp_path / "map.png"
    plt.imsave(path, gray, cmap="gray", vmin=0, vmax=1, origin="lower")
    reference = mpimg.imread(path)[..., :3].mean(axis=-1)[::-1]  # rows from the bottom (origin lower)
    component = MappedComponent.from_image(
        path, pixel_scale=1 * u.mas, calibration=3 * u.Jy / u.sr, name="png"
    )
    field, _ = run(component)
    values = field.radiance.to(
        u.Jy / u.sr, equivalencies=u.spectral_density(field.wavelengths[:, None, None])
    )
    np.testing.assert_allclose(values[0].value, 3 * reference, rtol=1e-6, atol=1e-12)


# --- Scenes and files ---------------------------------------------------------------------------------------


def test_resampling_in_a_scene_conserves_the_flux(tmp_path):
    rng = np.random.default_rng(3)
    data = rng.uniform(0, 1, (10, 10))
    path = write_image(tmp_path / "map.fits", data, cdelt_mas=(-3.0, 3.0))
    ctx = context()
    scene = ctx.add(Scene(name="sky", n_pixels=40, pixel_scale=1.7 * u.mas))
    scene.add(MappedComponent.from_fits(path, name="map"))
    radiance = ctx.run(["sky.radiance"])["sky.radiance"]
    expected = (data.sum() * u.Jy).to(F_LAMBDA, equivalencies=u.spectral_density(radiance.wavelengths))
    np.testing.assert_allclose(radiance.integrated().to_value(F_LAMBDA), expected.value, rtol=1e-10)


def test_mapped_component_round_trip(tmp_path):
    data = np.arange(16.0).reshape(4, 4)
    path = write_image(tmp_path / "map.fits", data)
    ctx = context()
    ctx.add(MappedComponent.from_fits(path, name="map"))
    target = tmp_path / "ctx.lumos"
    lumos.io.save(ctx, target)
    loaded = lumos.io.load(target)
    a = ctx.run(["map.out"])["map.out"].radiance
    b = loaded.run(["map.out"])["map.out"].radiance
    np.testing.assert_allclose(b.value, a.value, rtol=1e-12)
