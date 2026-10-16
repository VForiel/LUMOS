"""Tests for lumos.fields.RadianceField (step P3.6).

Requirements: REQ-SIG-6 (incoherent description of a scene: spectral radiance ``L(x, y, λ [, Stokes])`` on an
angular or spatial grid, with radiometric units) and §9 of the specs (spectral radiance in W m⁻² sr⁻¹ m⁻¹ or
photons s⁻¹ m⁻² sr⁻¹ m⁻¹; scene maps conserve the flux when resampled).

Independent references:

- per-frequency and per-wavelength densities: ``L_λ = L_ν c / λ²`` (Rybicki & Lightman, *Radiative Processes
  in Astrophysics*, §1.3); 1 Jy = 1e-26 W m⁻² Hz⁻¹ (IAU); at λ = 1 µm, 1 Jy/sr is
  ``1e-26 × 299792458 / 1e-12 = 2.99792458e-6`` W m⁻² sr⁻¹ m⁻¹;
- photon radiance ``L_ph = L λ / (h c)`` with the exact 2019 SI values;
- the flux collected from a radiance map on an angular grid is ``Σ L ΔΩ`` per unit area and wavelength;
- flux-conserving resampling of piecewise-constant maps: the value of a new pixel is the area-weighted mean of
  the old pixels it overlaps (e.g. 2×2 binning gives the block means), so ``Σ L ΔA`` is preserved whenever the
  new grid covers the old one.
"""

import importlib.util
import json
import math

import numpy as np
import pytest
from astropy import units as u
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from lumos import backend
from lumos.core.signal import OpticalSignal
from lumos.fields import Grid, RadianceField

HAS_JAX = importlib.util.find_spec("jax") is not None
BACKENDS = [
    "numpy",
    pytest.param("jax", marks=pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")),
]

H = 6.62607015e-34
C = 299_792_458.0
SPECTRAL_RADIANCE = u.W / u.m**2 / u.sr / u.m
SKY = Grid(8, 6, dx=10 * u.mas)
WAVELENGTHS = [1.0, 2.0] * u.um


def test_radiance_field_is_an_optical_signal():
    assert issubclass(RadianceField, OpticalSignal)


def test_basic_attributes():
    data = np.ones((2, 6, 8))
    rf = RadianceField(data * SPECTRAL_RADIANCE, SKY, WAVELENGTHS)
    assert rf.shape == (2, 6, 8)
    assert rf.n_wavelengths == 2
    assert not rf.has_stokes
    assert rf.unit == SPECTRAL_RADIANCE
    np.testing.assert_allclose(rf.wavelengths.to_value(u.um), [1.0, 2.0])
    assert rf.radiance.unit == SPECTRAL_RADIANCE
    assert not isinstance(rf.data, u.Quantity)


def test_stokes_axis():
    rf = RadianceField(np.ones((2, 4, 6, 8)) * SPECTRAL_RADIANCE, SKY, WAVELENGTHS)
    assert rf.has_stokes
    assert rf.shape == (2, 4, 6, 8)


def test_shape_must_match():
    with pytest.raises(ValueError, match="shape"):
        RadianceField(np.ones((3, 6, 8)) * SPECTRAL_RADIANCE, SKY, WAVELENGTHS)
    with pytest.raises(ValueError, match="shape"):
        RadianceField(np.ones((2, 3, 6, 8)) * SPECTRAL_RADIANCE, SKY, WAVELENGTHS)


def test_data_must_be_a_spectral_radiance():
    with pytest.raises(TypeError):
        RadianceField(np.ones((2, 6, 8)), SKY, WAVELENGTHS)
    with pytest.raises(u.UnitsError):
        RadianceField(np.ones((2, 6, 8)) * u.W / u.m**2, SKY, WAVELENGTHS)


@pytest.mark.parametrize(
    "unit",
    [
        SPECTRAL_RADIANCE,
        u.photon / u.s / u.m**2 / u.sr / u.m,
        u.Jy / u.sr,
        u.erg / u.s / u.cm**2 / u.sr / u.AA,
        u.W / u.m**2 / u.sr / u.Hz,
    ],
)
def test_accepted_radiance_units(unit):
    assert RadianceField(np.ones((2, 6, 8)) * unit, SKY, WAVELENGTHS).unit == unit


# --- Unit conversions --------------------------------------------------------------------------


def test_jansky_per_steradian_to_per_wavelength():
    rf = RadianceField(np.ones((2, 6, 8)) * u.Jy / u.sr, SKY, WAVELENGTHS)
    converted = rf.to(SPECTRAL_RADIANCE)
    values = converted.radiance.to_value(SPECTRAL_RADIANCE)
    assert values[0, 0, 0] == pytest.approx(2.99792458e-6, rel=1e-12)  # 1 µm
    assert values[1, 0, 0] == pytest.approx(2.99792458e-6 / 4, rel=1e-12)  # 2 µm: 1/λ²


def test_energy_to_photon_radiance():
    rf = RadianceField(np.full((2, 6, 8), 3.0) * SPECTRAL_RADIANCE, SKY, WAVELENGTHS)
    photons = rf.to(u.photon / u.s / u.m**2 / u.sr / u.m).radiance.value
    np.testing.assert_allclose(photons[0], 3.0 * 1e-6 / (H * C), rtol=1e-12)
    np.testing.assert_allclose(photons[1], 3.0 * 2e-6 / (H * C), rtol=1e-12)


def test_unit_round_trip():
    data = np.random.default_rng(1).random((2, 6, 8))
    rf = RadianceField(data * SPECTRAL_RADIANCE, SKY, WAVELENGTHS)
    back = rf.to(u.Jy / u.sr).to(u.photon / u.s / u.cm**2 / u.sr / u.nm).to(SPECTRAL_RADIANCE)
    np.testing.assert_allclose(back.radiance.value, data, rtol=1e-12)


# --- Integrated quantities ---------------------------------------------------------------------


def test_spectral_irradiance_on_an_angular_grid():
    rf = RadianceField(np.full((2, 6, 8), 2.0) * SPECTRAL_RADIANCE, SKY, WAVELENGTHS)
    domega = ((10 * u.mas).to_value(u.rad)) ** 2
    e = rf.integrated()
    assert e.unit.is_equivalent(u.W / u.m**2 / u.m)
    np.testing.assert_allclose(e.to_value(u.W / u.m**2 / u.m), 2.0 * 48 * domega, rtol=1e-12)


def test_spatial_grid_integrates_over_area():
    lab = Grid(4, dx=1 * u.mm)
    rf = RadianceField(np.ones((2, 4, 4)) * SPECTRAL_RADIANCE, lab, WAVELENGTHS)
    i = rf.integrated()
    assert i.unit.is_equivalent(u.W / u.sr / u.m)
    np.testing.assert_allclose(i.to_value(u.W / u.sr / u.m), 16 * 1e-6, rtol=1e-12)


def test_integrated_uses_s0_with_stokes():
    data = np.zeros((2, 4, 6, 8))
    data[:, 0] = 1.0
    data[:, 1] = 0.5
    rf = RadianceField(data * SPECTRAL_RADIANCE, SKY, WAVELENGTHS)
    domega = ((10 * u.mas).to_value(u.rad)) ** 2
    np.testing.assert_allclose(rf.integrated().to_value(u.W / u.m**2 / u.m), 48 * domega, rtol=1e-12)


# --- Flux-conserving resampling ------------------------------------------------------------------


def test_two_by_two_binning_gives_block_means():
    fine = Grid(4, dx=1 * u.mas)
    data = np.arange(16.0).reshape(1, 4, 4)
    rf = RadianceField(data * SPECTRAL_RADIANCE, fine, [1.0] * u.um)
    # fine centres -2..1 (edges -2.5..1.5); coarse centres (i - 1)·2 + 0.5 = -1.5, 0.5 cover 2×2 blocks
    coarse = rf.resample(Grid(2, dx=2 * u.mas, x0=0.5 * u.mas, y0=0.5 * u.mas))
    expected = np.array(
        [[(0 + 1 + 4 + 5) / 4, (2 + 3 + 6 + 7) / 4], [(8 + 9 + 12 + 13) / 4, (10 + 11 + 14 + 15) / 4]]
    )
    np.testing.assert_allclose(backend.to_numpy(coarse.data[0]), expected, rtol=1e-12)


def test_half_pixel_shift_by_hand():
    old = Grid(2, 1, dx=1 * u.mas)  # pixel centres x = -1, 0 mas; edges -1.5, -0.5, 0.5
    rf = RadianceField(np.array([[[2.0, 4.0]]]) * SPECTRAL_RADIANCE, old, [1.0] * u.um)
    new = rf.resample(Grid(1, 1, dx=1 * u.mas, x0=-0.5 * u.mas))  # one pixel covering [-1, 0]
    assert float(backend.to_numpy(new.data)[0, 0, 0]) == pytest.approx(3.0, rel=1e-12)


@given(
    data=hnp.arrays(np.float64, (1, 6, 8), elements=st.floats(0, 1e3, allow_nan=False)),
    factor=st.sampled_from([0.5, 0.7, 1.3, 2.0, 3.1]),
)
def test_resampling_conserves_the_flux_when_the_new_grid_covers_the_old(data, factor):
    rf = RadianceField(data * SPECTRAL_RADIANCE, SKY, [1.0] * u.um)
    dx = 10 * factor
    n = math.ceil(80 / dx) + 2  # covers the 80 mas × 60 mas map with margin
    new = rf.resample(Grid(n, dx=dx * u.mas, x0=0.3 * u.mas, y0=-0.2 * u.mas))
    before = rf.integrated().to_value(u.W / u.m**2 / u.m)
    after = new.integrated().to_value(u.W / u.m**2 / u.m)
    np.testing.assert_allclose(after, before, rtol=1e-10, atol=1e-30)


def test_uniform_map_stays_uniform_inside():
    rf = RadianceField(np.full((1, 6, 8), 5.0) * SPECTRAL_RADIANCE, SKY, [1.0] * u.um)
    new = rf.resample(Grid(5, 3, dx=7 * u.mas))  # strictly inside the old map
    np.testing.assert_allclose(backend.to_numpy(new.data), 5.0, rtol=1e-12)


def test_resampling_requires_the_same_plane():
    rf = RadianceField(np.ones((1, 6, 8)) * SPECTRAL_RADIANCE, SKY, [1.0] * u.um)
    with pytest.raises(ValueError, match="plane"):
        rf.resample(Grid(4, dx=1 * u.mm))


# --- Backends and serialisation ----------------------------------------------------------------


@pytest.mark.parametrize("name", BACKENDS)
def test_integrated_flux_is_invariant_under_backend_change(name):
    data = np.random.default_rng(2).random((2, 6, 8))
    rf = RadianceField(data * SPECTRAL_RADIANCE, SKY, WAVELENGTHS)
    moved = rf.to_backend(name)
    assert moved.backend == name
    np.testing.assert_allclose(moved.integrated().value, rf.integrated().value, rtol=1e-12)


def test_json_round_trip():
    data = np.random.default_rng(3).random((2, 4, 6, 8))
    rf = RadianceField(data * u.Jy / u.sr, SKY, WAVELENGTHS)
    back = RadianceField.from_json(json.loads(json.dumps(rf.to_json())))
    assert back.unit == u.Jy / u.sr
    assert back.grid == SKY
    np.testing.assert_array_equal(backend.to_numpy(back.data), data)
