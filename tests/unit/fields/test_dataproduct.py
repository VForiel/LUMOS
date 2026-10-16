"""Tests for lumos.fields.DataProduct and its files (step P3.8).

Requirements: REQ-SIG-9 (detected data: image, cube, time series or scalars, with units such as ADU, e⁻ or
photons, physical axes, mask, uncertainties and acquisition metadata; export to FITS, NPZ or HDF5) and the
plan of P3.8 (FITS round trip), NF on security (no pickle when loading).

Independent references:

- the FITS standard (Pence et al. 2010, A&A 524, A42): ``BUNIT`` holds the data unit, ``NAXIS1`` is the
  fastest-varying axis (the last NumPy axis) and a linear axis is described by ``CRPIX``, ``CRVAL``,
  ``CDELT`` and ``CUNIT`` with 1-based pixel indices; files are read back with astropy.io.fits and
  astropy.wcs, independent of LUMOS;
- independent uncertainties add in quadrature for sums and differences, and scale with ``|c|`` for a
  product by a constant (JCGM 100:2008, *GUM*, §5.1; Taylor, *An Introduction to Error Analysis*, §3.6).
"""

import importlib.util
import json

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

from lumos.core.signal import DataSignal
from lumos.fields import DataProduct, Grid

HAS_H5PY = importlib.util.find_spec("h5py") is not None
RNG = np.random.default_rng(0)


def image(**kwargs):
    data = RNG.random((4, 6)) * u.electron
    return DataProduct(
        data,
        axes={"y": np.arange(4) * 2.0 * u.um, "x": (np.arange(6) - 3) * 1.5 * u.um},
        **kwargs,
    )


def test_data_product_is_a_data_signal():
    assert issubclass(DataProduct, DataSignal)


# --- Construction ------------------------------------------------------------------------------


def test_basic_attributes():
    dp = image()
    assert dp.shape == (4, 6)
    assert dp.unit == u.electron
    assert dp.axis_names == ("y", "x")
    assert dp.axes["x"].unit == u.um
    assert dp.mask is None
    assert dp.uncertainty is None
    assert isinstance(dp.data, u.Quantity)


def test_data_must_be_a_quantity():
    with pytest.raises(TypeError):
        DataProduct(np.ones(3))
    assert DataProduct(np.ones(3) * u.dimensionless_unscaled).unit == u.dimensionless_unscaled


def test_axes_must_match_the_data():
    with pytest.raises(ValueError, match="axis"):
        DataProduct(np.ones((4, 6)) * u.adu, axes={"y": np.arange(4) * u.m})  # missing x
    with pytest.raises(ValueError, match="axis"):
        DataProduct(np.ones((4, 6)) * u.adu, axes={"y": np.arange(5) * u.m, "x": np.arange(6) * u.m})


def test_default_axes_are_pixel_indices():
    dp = DataProduct(np.ones((2, 3)) * u.adu)
    assert dp.axis_names == ("axis0", "axis1")
    np.testing.assert_array_equal(dp.axes["axis1"].to_value(u.pix), [0, 1, 2])


def test_mask_and_uncertainty_are_validated():
    with pytest.raises(ValueError, match="mask"):
        image(mask=np.zeros((3, 6), dtype=bool))
    with pytest.raises(ValueError, match="uncertainty"):
        image(uncertainty=np.ones((4, 5)) * u.electron)
    with pytest.raises(u.UnitsError):
        image(uncertainty=np.ones((4, 6)) * u.s)


def test_uncertainty_is_converted_to_the_data_unit():
    dp = DataProduct(np.ones(3) * u.mW, uncertainty=np.full(3, 2.0) * u.uW)
    np.testing.assert_allclose(dp.uncertainty.to_value(u.mW), 2e-3)
    assert dp.uncertainty.unit == u.mW


def test_from_grid_uses_the_grid_coordinates():
    grid = Grid(6, 4, dx=1.5 * u.um, dy=2 * u.um, y0=3 * u.um)
    dp = DataProduct.from_grid(np.ones((4, 6)) * u.adu, grid)
    assert dp.axis_names == ("y", "x")
    np.testing.assert_allclose(dp.axes["x"].to_value(u.um), grid.x.to_value(u.um))
    np.testing.assert_allclose(dp.axes["y"].to_value(u.um), grid.y.to_value(u.um))


def test_from_grid_with_leading_axes():
    grid = Grid(3, 2, dx=1 * u.arcsec)
    lam = [1.0, 2.0, 3.0, 4.0] * u.um
    dp = DataProduct.from_grid(np.ones((4, 2, 3)) * u.photon, grid, leading_axes={"wavelength": lam})
    assert dp.axis_names == ("wavelength", "y", "x")


# --- Uncertainty propagation -------------------------------------------------------------------


def test_sum_adds_uncertainties_in_quadrature():
    a = DataProduct(np.full(3, 10.0) * u.electron, uncertainty=np.full(3, 3.0) * u.electron)
    b = DataProduct(np.full(3, 5.0) * u.electron, uncertainty=np.full(3, 4.0) * u.electron)
    for total in (a + b, a - b):
        np.testing.assert_allclose(total.uncertainty.to_value(u.electron), 5.0)
    np.testing.assert_allclose((a - b).data.to_value(u.electron), 5.0)


def test_scaling_scales_the_uncertainty():
    a = DataProduct(np.full(3, 10.0) * u.electron, uncertainty=np.full(3, 3.0) * u.electron)
    gain = 2.5 * u.adu / u.electron
    scaled = a * gain
    assert scaled.unit.is_equivalent(u.adu)
    np.testing.assert_allclose(scaled.uncertainty.to_value(u.adu), 7.5)
    np.testing.assert_allclose((a * -2).uncertainty.to_value(u.electron), 6.0)


def test_masks_are_combined():
    m1 = np.array([True, False, False])
    m2 = np.array([False, False, True])
    a = DataProduct(np.ones(3) * u.adu, mask=m1)
    b = DataProduct(np.ones(3) * u.adu, mask=m2)
    np.testing.assert_array_equal((a + b).mask, [True, False, True])


def test_sum_requires_matching_axes():
    with pytest.raises(ValueError, match="axes"):
        image() + DataProduct(np.ones((4, 6)) * u.electron)


# --- FITS ----------------------------------------------------------------------------------------


def test_fits_round_trip(tmp_path):
    mask = RNG.random((4, 6)) > 0.7
    unc = RNG.random((4, 6)) * u.electron
    meta = {"exposure_time": 2.5 * u.s, "instrument": "LL-CAM", "gain": 1.2, "frames": 3}
    dp = image(mask=mask, uncertainty=unc, meta=meta)
    path = tmp_path / "image.fits"
    dp.to_fits(path)
    back = DataProduct.from_fits(path)
    np.testing.assert_array_equal(back.data.value, dp.data.value)
    assert back.unit == u.electron
    np.testing.assert_array_equal(back.mask, mask)
    np.testing.assert_array_equal(back.uncertainty.value, unc.value)
    assert back.axis_names == ("y", "x")
    np.testing.assert_allclose(back.axes["x"].to_value(u.um), dp.axes["x"].to_value(u.um))
    assert back.meta["exposure_time"] == 2.5 * u.s
    assert back.meta["instrument"] == "LL-CAM"
    assert back.meta["frames"] == 3


def test_fits_file_follows_the_standard(tmp_path):
    dp = image()
    path = tmp_path / "image.fits"
    dp.to_fits(path)
    with fits.open(path) as hdul:
        header = hdul[0].header
        assert header["BUNIT"] == "electron"
        assert header["NAXIS1"] == 6  # fastest axis = last NumPy axis (x)
        assert header["NAXIS2"] == 4
        np.testing.assert_array_equal(hdul[0].data, dp.data.value)
        wcs = WCS(header)
        world = wcs.pixel_to_world_values(np.arange(6), np.zeros(6))
    # Linear axes keep their CUNIT in astropy.wcs (only celestial and spectral axes are converted to SI).
    x_unit, y_unit = (u.Unit(unit) for unit in wcs.world_axis_units)
    np.testing.assert_allclose(world[0], dp.axes["x"].to_value(x_unit), rtol=1e-12)
    np.testing.assert_allclose(world[1], dp.axes["y"][0].to_value(y_unit), atol=1e-18)


def test_fits_with_irregular_axis_round_trips(tmp_path):
    t = np.array([0.0, 1.0, 3.0, 7.0]) * u.s  # not linear: stored as a table, no WCS
    dp = DataProduct(np.arange(4.0) * u.adu, axes={"time": t})
    path = tmp_path / "series.fits"
    dp.to_fits(path)
    back = DataProduct.from_fits(path)
    np.testing.assert_array_equal(back.axes["time"].to_value(u.s), t.value)


# --- NPZ and HDF5 ------------------------------------------------------------------------------


def test_npz_round_trip_without_pickle(tmp_path):
    dp = image(mask=RNG.random((4, 6)) > 0.5, uncertainty=np.ones((4, 6)) * u.electron, meta={"t": 1 * u.s})
    path = tmp_path / "image.npz"
    dp.to_npz(path)
    with np.load(path, allow_pickle=False) as archive:  # raises if any object array was stored
        assert all(archive[key].dtype.kind != "O" for key in archive.files)
    back = DataProduct.from_npz(path)
    np.testing.assert_array_equal(back.data.value, dp.data.value)
    np.testing.assert_array_equal(back.mask, dp.mask)
    assert back.meta["t"] == 1 * u.s
    assert back.unit == u.electron


@pytest.mark.skipif(not HAS_H5PY, reason="h5py is not installed")
def test_hdf5_round_trip(tmp_path):
    dp = image(uncertainty=np.ones((4, 6)) * u.electron, meta={"note": "flat"})
    path = tmp_path / "image.h5"
    dp.to_hdf5(path)
    back = DataProduct.from_hdf5(path)
    np.testing.assert_array_equal(back.data.value, dp.data.value)
    np.testing.assert_array_equal(back.uncertainty.value, dp.uncertainty.value)
    assert back.meta["note"] == "flat"
    assert back.axis_names == ("y", "x")


def test_json_round_trip():
    dp = image(mask=np.zeros((4, 6), dtype=bool), meta={"t": 3 * u.ms})
    back = DataProduct.from_json(json.loads(json.dumps(dp.to_json())))
    np.testing.assert_array_equal(back.data.value, dp.data.value)
    assert back.meta["t"] == 3 * u.ms
