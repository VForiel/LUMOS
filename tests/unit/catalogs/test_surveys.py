"""Tests for images of diffuse objects turned into scene components (step P9.6, REQ-CAT-3, REQ-SCN-4).

Requirements: retrieve images (SkyView, HiPS through hips2fits) convertible into mapped scene components.

Independent references (recorded responses in ``tests/fixtures/catalogs``, read here with astropy):

- SkyView (McGlynn et al. 1998), DSS image of M42: 64 × 64 pixels of 0.0078125° (28.125″), no ``BUNIT``
  (photographic densities, uncalibrated);
- hips2fits (CDS, Fernique et al. 2015), ``CDS/P/DSS2/red``: 64 × 64 pixels of 0.0078125496°;
- survey images are uncalibrated: a calibration (brightness of one data unit) or a total flux density is
  required, and East is turned towards +x (``CDELT1 < 0`` in both files).
"""

import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits

from lumos import Context
from lumos.catalogs.surveys import Hips2Fits, SkyView, survey_component
from lumos.scenes import MappedComponent

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "catalogs"


def fixture_image(name):
    content = base64.b64decode(json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))["content"])
    with fits.open(io.BytesIO(content)) as hdul:
        return np.asarray(hdul[0].data, dtype=float), hdul[0].header


def evaluate(component):
    ctx = Context(seed=1)
    ctx.observation.band = (0.6 * u.um, 0.7 * u.um)
    ctx.add(component)
    return ctx.run([f"{ctx.id_of(component)}.out"])[f"{ctx.id_of(component)}.out"]


def test_skyview_image(replay):
    hdu = SkyView().image("M42", survey="DSS", pixels=64, size=0.5 * u.deg)
    data, header = fixture_image("skyview_m42")
    np.testing.assert_array_equal(np.asarray(hdu.data, dtype=float), data)
    assert hdu.header["CDELT2"] == header["CDELT2"]


def test_hips2fits_image(replay):
    hdu = Hips2Fits().image(
        "CDS/P/DSS2/red", ra=83.8221 * u.deg, dec=-5.3911 * u.deg, fov=0.5 * u.deg, pixels=64
    )
    data, _ = fixture_image("hips2fits_m42")
    np.testing.assert_array_equal(np.asarray(hdu.data, dtype=float), data)


def test_calibrated_survey_component(replay):
    hdu = SkyView().image("M42", survey="DSS", pixels=64, size=0.5 * u.deg)
    component = survey_component(
        hdu, name="m42", calibration=1e-3 * u.MJy / u.sr, reference_wavelength=0.65 * u.um
    )
    assert isinstance(component, MappedComponent)
    field = evaluate(component)
    data, header = fixture_image("skyview_m42")
    assert field.grid.dx.to_value(u.arcsec) == pytest.approx(abs(header["CDELT1"]) * 3600)
    sigma = field.radiance[0].to(u.MJy / u.sr, equivalencies=u.spectral_density(field.wavelengths[0])).value
    np.testing.assert_allclose(sigma, 1e-3 * data[:, ::-1], rtol=1e-10)  # East towards +x


def test_survey_component_normalised_to_a_total_flux(replay):
    hdu = Hips2Fits().image(
        "CDS/P/DSS2/red", ra=83.8221 * u.deg, dec=-5.3911 * u.deg, fov=0.5 * u.deg, pixels=64
    )
    component = survey_component(hdu, name="m42", flux_density=10 * u.Jy, reference_wavelength=0.65 * u.um)
    field = evaluate(component)
    total = field.integrated()[0].to(u.Jy, equivalencies=u.spectral_density(field.wavelengths[0]))
    assert total.to_value(u.Jy) == pytest.approx(10.0, rel=1e-10)


def test_survey_component_needs_a_calibration(replay):
    hdu = SkyView().image("M42", survey="DSS", pixels=64, size=0.5 * u.deg)
    with pytest.raises(ValueError, match="calibration"):
        survey_component(hdu, name="m42")


def test_mapped_component_from_survey(replay):
    component = MappedComponent.from_survey(
        "M42", survey="DSS", pixels=64, size=0.5 * u.deg, flux_density=5 * u.Jy
    )
    field = evaluate(component)
    assert field.grid.shape == (64, 64)
    total = field.integrated()[0].to(u.Jy, equivalencies=u.spectral_density(field.wavelengths[0]))
    assert total.to_value(u.Jy) == pytest.approx(5.0, rel=1e-10)
