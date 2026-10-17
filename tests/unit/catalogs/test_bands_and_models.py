"""Tests for photometric bands and model spectra (step P9.5, REQ-CAT-2).

Requirements: the SVO Filter Profile Service and libraries of model spectra (Pickles...).

Independent references (recorded responses in ``tests/fixtures/catalogs``):

- SVO Filter Profile Service (Rodrigo & Solano 2020), ``Generic/Johnson.V``: 15 points from 4600 to 7400 Å;
  the service gives ``WavelengthMean`` = 5537.16 Å (its definition: ∫λT dλ / ∫T dλ) and a Vega zero point of
  3617.50 Jy;
- Pickles (1998, PASP 110, 863) UVKLIB library, STScI copy: the A0V spectrum is ``pickles_uk_9`` and covers
  1150 Å to 2.5 µm; its shape between 4000 and 7000 Å matches that of Vega (A0V, CALSPEC) within 10 %.
"""

import numpy as np
import pytest
from astropy import units as u

from lumos.catalogs.calspec import Calspec
from lumos.catalogs.models import PicklesLibrary
from lumos.catalogs.spectra import FilterCurve
from lumos.catalogs.svo import SvoFilterService


def test_svo_filter_curve(replay):
    curve = SvoFilterService().filter("Generic/Johnson.V")
    assert isinstance(curve, FilterCurve)
    assert len(curve.wavelengths) == 15
    assert curve.wavelengths[0].to_value(u.AA) == pytest.approx(4600.0)
    assert curve.effective_wavelength().to_value(u.AA) == pytest.approx(5537.16, rel=5e-3)
    assert curve.zero_point.to_value(u.Jy) == pytest.approx(3617.50, rel=1e-5)
    assert curve.source == "svo"


def test_filter_transmission_is_interpolated(replay):
    curve = SvoFilterService().filter("Generic/Johnson.V")
    assert curve.at(4000 * u.AA) == 0.0
    assert 0 < curve.at(5500 * u.AA) <= 1


def test_pickles_spectrum_by_spectral_type(replay):
    library = PicklesLibrary()
    assert library.file_of("A0V") == "pickles_uk_9"
    spectrum = library.spectrum("A0V")
    assert spectrum.wavelengths[0].to_value(u.AA) == pytest.approx(1150.0)
    assert spectrum.wavelengths[-1].to_value(u.AA) == pytest.approx(25000.0)
    assert spectrum.source == "pickles"


def test_pickles_a0v_looks_like_vega(replay):
    model = PicklesLibrary().spectrum("A0V")
    vega = Calspec().spectrum("alpha_lyr")
    lam = np.linspace(4000, 7000, 31) * u.AA
    shape_model = model.at(lam).value / model.at(5556 * u.AA).value
    shape_vega = vega.at(lam).value / vega.at(5556 * u.AA).value
    np.testing.assert_allclose(shape_model, shape_vega, rtol=0.10)


def test_unknown_spectral_type(replay):
    from lumos.catalogs import CatalogError

    with pytest.raises(CatalogError, match="Q9"):
        PicklesLibrary().file_of("Q9V")
