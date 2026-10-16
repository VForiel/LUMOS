"""Tests for lumos.fields.Wavecube and spectral sampling (step P3.4).

Requirements: REQ-SIG-4 (stack of wavefronts on a spectral axis, shape ``(nλ, [2,] ny, nx)``, spectral
weights, grid common to every wavelength).

Independent references:

- fields at different wavelengths do not interfere in the time-averaged intensity: the detected intensity of
  a polychromatic field is the (weighted) sum of the monochromatic intensities (Born & Wolf, *Principles of
  Optics*, §10.3, quasi-monochromatic and polychromatic fields);
- the n-point Gauss-Legendre rule integrates polynomials of degree ≤ 2n - 1 exactly (Abramowitz & Stegun,
  *Handbook of Mathematical Functions*, 25.4.29); the composite midpoint rule with n panels of width h
  integrates λ² over a band of width Δ with the error ``n h³ / 12 = Δ³ / (12 n²)``;
- photon energy hc/λ with the exact 2019 SI values of h and c;
- plane-wave phase ``k x sin θ`` with ``k = 2π/λ`` for each wavelength (Born & Wolf, §1.4).
"""

import importlib.util
import json
import math

import numpy as np
import pytest
from astropy import units as u
from hypothesis import given
from hypothesis import strategies as st

from lumos import backend
from lumos.core.signal import OpticalSignal
from lumos.fields import Grid, Wavecube, Wavefront, jones, spectral_samples

HAS_JAX = importlib.util.find_spec("jax") is not None
BACKENDS = [
    "numpy",
    pytest.param("jax", marks=pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")),
]

H = 6.62607015e-34
C = 299_792_458.0

GRID = Grid(8, 6, dx=1 * u.um)
WAVELENGTHS = [1.0, 1.5, 2.0] * u.um


def cube(powers=(1.0, 2.0, 3.0), wavelengths=WAVELENGTHS, **kwargs):
    return Wavecube.plane(GRID, wavelengths, power=np.asarray(powers) * u.W, **kwargs)


# --- Construction ------------------------------------------------------------------------------


def test_wavecube_is_an_optical_signal():
    assert issubclass(Wavecube, OpticalSignal)


def test_scalar_cube_shape_and_attributes():
    wc = cube()
    assert wc.shape == (3, 6, 8)
    assert wc.n_wavelengths == 3
    assert not wc.polarized
    np.testing.assert_allclose(wc.wavelengths.to_value(u.um), [1.0, 1.5, 2.0])
    np.testing.assert_allclose(wc.wavelengths_si, [1e-6, 1.5e-6, 2e-6], rtol=1e-15)
    np.testing.assert_array_equal(wc.weights, [1.0, 1.0, 1.0])


def test_polarized_cube_shape():
    wc = cube(polarization="H")
    assert wc.shape == (3, 2, 6, 8)
    assert wc.polarized


def test_field_must_match_wavelengths_and_grid():
    with pytest.raises(ValueError, match="shape"):
        Wavecube(np.ones((2, 6, 8)), GRID, WAVELENGTHS)
    with pytest.raises(ValueError, match="shape"):
        Wavecube(np.ones((3, 8, 6)), GRID, WAVELENGTHS)


def test_weights_must_match_wavelengths():
    with pytest.raises(ValueError, match="weights"):
        Wavecube(np.ones((3, 6, 8)), GRID, WAVELENGTHS, weights=[1.0, 2.0])


def test_wavelengths_must_be_positive_lengths():
    with pytest.raises(TypeError):
        Wavecube(np.ones((3, 6, 8)), GRID, [1e-6, 2e-6, 3e-6])
    with pytest.raises(ValueError):
        Wavecube(np.ones((3, 6, 8)), GRID, [1.0, -1.0, 2.0] * u.um)


def test_from_wavefronts_and_indexing_round_trip():
    wfs = [Wavefront.plane(GRID, lam, power=p * u.W) for lam, p in zip(WAVELENGTHS, (1, 2, 3), strict=True)]
    wc = Wavecube.from_wavefronts(wfs, weights=[0.5, 1.0, 0.5])
    np.testing.assert_array_equal(wc.weights, [0.5, 1.0, 0.5])
    for k, wf in enumerate(wfs):
        assert isinstance(wc[k], Wavefront)
        assert wc[k].wavelength == wf.wavelength
        np.testing.assert_array_equal(backend.to_numpy(wc[k].field), backend.to_numpy(wf.field))
    assert len(list(wc.wavefronts())) == 3


def test_from_wavefronts_requires_a_common_grid():
    a = Wavefront.plane(GRID, 1 * u.um, power=1 * u.W)
    b = Wavefront.plane(Grid(8, 6, dx=2 * u.um), 2 * u.um, power=1 * u.W)
    with pytest.raises(ValueError, match="grid"):
        Wavecube.from_wavefronts([a, b])


# --- Powers and spectral integration -----------------------------------------------------------


def test_power_per_channel():
    np.testing.assert_allclose(cube().power.to_value(u.W), [1.0, 2.0, 3.0], rtol=1e-14)


def test_channels_add_incoherently():
    # Same amplitude and phase at two wavelengths: no interference term, the powers add (2, not 4).
    wc = cube(powers=(1.0, 1.0), wavelengths=[1.0, 1.1] * u.um)
    assert wc.total_power.to_value(u.W) == pytest.approx(2.0, rel=1e-14)
    np.testing.assert_allclose(wc.integrated_intensity, wc[0].intensity + wc[1].intensity, rtol=1e-14)


def test_weights_enter_the_integration():
    wc = cube(weights=[0.5, 0.25, 2.0])
    assert wc.total_power.to_value(u.W) == pytest.approx(0.5 + 0.5 + 6.0, rel=1e-14)
    expected = 0.5 * wc[0].intensity + 0.25 * wc[1].intensity + 2.0 * wc[2].intensity
    np.testing.assert_allclose(wc.integrated_intensity, expected, rtol=1e-14)


def test_polarized_integrated_intensity_has_the_grid_shape():
    assert cube(polarization="D").integrated_intensity.shape == GRID.shape


# --- Spectral sampling ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_gauss_legendre_integrates_polynomials_exactly(n):
    center, width = 1.6, 0.4  # um
    lam, w = spectral_samples(center * u.um, width * u.um, n, rule="gauss")
    x = lam.to_value(u.um)
    a, b = center - width / 2, center + width / 2
    assert np.sum(w) == pytest.approx(1.0, rel=1e-14)
    for degree in range(2 * n):
        exact = (b ** (degree + 1) - a ** (degree + 1)) / (degree + 1) / width
        assert np.sum(w * x**degree) == pytest.approx(exact, rel=1e-12)


@given(n=st.integers(1, 50))
def test_midpoint_rule_error_on_a_quadratic(n):
    center, width = 2.0, 0.5
    lam, w = spectral_samples(center * u.um, width * u.um, n, rule="midpoint")
    x = lam.to_value(u.um)
    a, b = center - width / 2, center + width / 2
    exact = (b**3 - a**3) / 3 / width
    # composite midpoint underestimates ∫λ² by Δ³ / (12 n²); divided by Δ for a band average
    assert np.sum(w * x**2) == pytest.approx(exact - width**2 / (12 * n**2), rel=1e-12)
    np.testing.assert_allclose(np.diff(x), width / n, rtol=1e-10)


def test_spectral_samples_validate_their_inputs():
    with pytest.raises(ValueError):
        spectral_samples(1 * u.um, 0.2 * u.um, 0)
    with pytest.raises(ValueError):
        spectral_samples(1 * u.um, 3 * u.um, 3)  # band would reach negative wavelengths
    with pytest.raises(ValueError):
        spectral_samples(1 * u.um, 0.2 * u.um, 3, rule="simpson")


# --- Operations --------------------------------------------------------------------------------


def test_photon_rate_per_channel():
    wc = cube().with_power_unit(u.photon / u.s)
    lam = np.array([1.0, 1.5, 2.0]) * 1e-6
    np.testing.assert_allclose(
        wc.power.to_value(u.photon / u.s), np.array([1, 2, 3]) * lam / (H * C), rtol=1e-12
    )


@given(theta=st.floats(-0.2, 0.2))
def test_tilt_is_chromatic(theta):
    wc = cube().tilt(theta * u.rad, 0 * u.rad)
    for k, lam in enumerate([1e-6, 1.5e-6, 2e-6]):
        ratio = backend.to_numpy(wc.field[k, :, 1:] / wc.field[k, :, :-1])
        np.testing.assert_allclose(ratio, np.exp(2j * math.pi / lam * 1e-6 * math.sin(theta)), rtol=1e-9)


def test_chromatic_mask_multiplies_each_channel():
    mask = np.zeros((3, *GRID.shape))
    mask[1] = 1.0
    out = cube() * mask
    np.testing.assert_allclose(out.power.to_value(u.W), [0.0, 2.0, 0.0], atol=1e-14)


def test_achromatic_mask_broadcasts_over_channels():
    mask = np.zeros(GRID.shape)
    mask[:3] = 1.0  # half of the rows
    np.testing.assert_allclose((cube() * mask).power.to_value(u.W), [0.5, 1.0, 1.5], rtol=1e-14)


def test_jones_matrices_act_on_polarized_cubes():
    out = jones.apply(jones.polarizer(60 * u.deg), cube(polarization="H"))
    np.testing.assert_allclose(out.power.to_value(u.W), np.array([1.0, 2.0, 3.0]) * 0.25, rtol=1e-12)
    s = jones.stokes(cube(polarization="R"))
    assert s.shape == (3, 4, *GRID.shape)
    np.testing.assert_allclose(s[:, 3] / s[:, 0], 1.0, rtol=1e-12)


# --- Backends and serialisation ----------------------------------------------------------------


@pytest.mark.parametrize("name", BACKENDS)
def test_total_power_is_invariant_under_backend_change(name):
    wc = cube(weights=[0.2, 0.3, 0.5]).tilt(0.1 * u.rad, 0.05 * u.rad)
    moved = wc.to_backend(name)
    assert moved.backend == name
    assert moved.total_power.to_value(u.W) == pytest.approx(wc.total_power.to_value(u.W), rel=1e-12)


def test_json_round_trip():
    wc = cube(weights=[0.2, 0.3, 0.5], polarization="L").tilt(0.1 * u.rad, 0 * u.rad)
    back = Wavecube.from_json(json.loads(json.dumps(wc.to_json())))
    np.testing.assert_array_equal(backend.to_numpy(back.field), backend.to_numpy(wc.field))
    np.testing.assert_array_equal(back.weights, wc.weights)
    assert back.grid == GRID
    assert np.all(back.wavelengths == wc.wavelengths)
    assert back.total_power == wc.total_power


@pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")
def test_wavecube_is_a_jax_pytree():
    import jax

    wc = cube().to_backend("jax")
    out = jax.jit(lambda c: c * 2.0)(wc)
    assert isinstance(out, Wavecube)
    np.testing.assert_allclose(out.power.to_value(u.W), [4.0, 8.0, 12.0], rtol=1e-12)
