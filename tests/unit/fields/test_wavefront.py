"""Tests for lumos.fields.Wavefront (step P3.2).

Requirements: REQ-SIG-2 (complex field sampled on a Grid at one wavelength; ``intensity``, ``amplitude``,
``phase``, ``power``; backend array without units; normalisation in W or photons/s as metadata) and §9 of the
specs (``U ∝ e^{+ikz}``; ``Σ |U|² dx dy`` is the power).

Independent references:

- uniform field of amplitude A on ``nx·ny`` pixels of area ``dx·dy``: ``P = A² nx ny dx dy``;
- Gaussian beam ``U = sqrt(2P/(πw²)) exp(-r²/w²)``: its intensity integrates to P (Siegman, *Lasers*, 1986,
  §16.1, eq. 16.5); with w = 8 pixels the Riemann sum equals the integral to better than 1e-12 (Poisson
  summation: the aliasing error is ~exp(-(π w / dx)²));
- photon energy ``E = hc/λ`` with the exact SI values of h and c (2019 SI): 1 W at 1 µm is
  ``1e-6 / (6.62607015e-34 × 299792458) = 5.0341165675427096e18`` photons/s;
- two-beam interference: equal in-phase fields give 4× the power of one, opposite phases give 0;
- a plane wave travelling at angle θ to z in the x-z plane has the phase ``k x sin θ`` (Born & Wolf,
  *Principles of Optics*, §1.4), so adjacent pixels differ by ``k dx sin θ``.
"""

import importlib.util
import json
import math

import numpy as np
import pytest
from astropy import units as u
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import lumos
from lumos import backend
from lumos.core.signal import OpticalSignal
from lumos.fields import Grid, Wavefront

HAS_JAX = importlib.util.find_spec("jax") is not None
BACKENDS = [
    "numpy",
    pytest.param("jax", marks=pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")),
]
needs_jax = pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")

H = 6.62607015e-34  # J s, exact (2019 SI)
C = 299_792_458.0  # m/s, exact

GRID = Grid(16, 8, dx=2 * u.um, dy=3 * u.um)
WAVELENGTH = 1 * u.um


@pytest.fixture
def restore_precision():
    previous = lumos.config.precision
    yield
    lumos.config.precision = previous


complex_fields = hnp.arrays(
    np.complex128,
    (8, 16),
    elements=st.complex_numbers(max_magnitude=1e3, allow_nan=False, allow_infinity=False),
)


# --- Construction ------------------------------------------------------------------------------


def test_wavefront_is_an_optical_signal():
    assert issubclass(Wavefront, OpticalSignal)


def test_basic_attributes():
    field = np.ones(GRID.shape)
    wf = Wavefront(field, GRID, WAVELENGTH)
    assert wf.grid is GRID
    assert wf.wavelength == 1 * u.um
    assert wf.wavelength_si == pytest.approx(1e-6, rel=1e-15)
    assert wf.wavenumber_si == pytest.approx(2 * math.pi / 1e-6, rel=1e-15)
    assert wf.power_unit == u.W
    assert wf.shape == (8, 16)


def test_field_is_stored_as_complex_backend_array_without_units():
    wf = Wavefront(np.ones(GRID.shape), GRID, WAVELENGTH)
    assert not isinstance(wf.field, u.Quantity)
    assert wf.field.dtype == np.complex128
    assert wf.backend == backend.current().name  # NumPy input is copied into the current backend


def test_field_follows_the_precision(restore_precision):
    lumos.config.precision = "single"
    assert Wavefront(np.ones(GRID.shape), GRID, WAVELENGTH).field.dtype == np.complex64


@pytest.mark.parametrize("name", BACKENDS)
def test_field_is_created_in_the_current_backend(name):
    with backend.use(name):
        wf = Wavefront(np.ones(GRID.shape), GRID, WAVELENGTH)
    assert wf.backend == name
    assert backend.backend_of(wf.field) == name


def test_shape_must_match_the_grid():
    with pytest.raises(ValueError, match="shape"):
        Wavefront(np.ones((16, 8)), GRID, WAVELENGTH)


def test_wavelength_must_be_a_positive_length():
    with pytest.raises(TypeError):
        Wavefront(np.ones(GRID.shape), GRID, 1e-6)
    with pytest.raises(u.UnitsError):
        Wavefront(np.ones(GRID.shape), GRID, 1 * u.s)
    with pytest.raises(ValueError):
        Wavefront(np.ones(GRID.shape), GRID, -1 * u.um)


def test_wavelength_accepts_a_frequency():
    wf = Wavefront(np.ones(GRID.shape), GRID, C * u.Hz)
    assert wf.wavelength_si == pytest.approx(1.0, rel=1e-15)


def test_power_unit_must_be_watt_or_photon_rate():
    assert Wavefront(np.ones(GRID.shape), GRID, WAVELENGTH, power_unit=u.photon / u.s).power_unit == (
        u.photon / u.s
    )
    with pytest.raises(u.UnitsError):
        Wavefront(np.ones(GRID.shape), GRID, WAVELENGTH, power_unit=u.m)


# --- Derived quantities ------------------------------------------------------------------------


@given(field=complex_fields)
def test_intensity_amplitude_phase_are_consistent(field):
    wf = Wavefront(field, GRID, WAVELENGTH)
    np.testing.assert_allclose(wf.intensity, np.abs(field) ** 2, rtol=1e-12, atol=1e-300)
    np.testing.assert_allclose(wf.amplitude, np.abs(field), rtol=1e-12, atol=1e-300)
    rebuilt = wf.amplitude * np.exp(1j * wf.phase)
    np.testing.assert_allclose(rebuilt, field, rtol=1e-12, atol=1e-9)


def test_phase_is_in_minus_pi_pi():
    wf = Wavefront(np.full(GRID.shape, -1.0 + 0j), GRID, WAVELENGTH)
    assert np.all(np.abs(wf.phase) <= math.pi)


def test_power_of_a_uniform_field():
    amplitude = 3.0  # sqrt(W) / m
    wf = Wavefront(np.full(GRID.shape, amplitude), GRID, WAVELENGTH)
    expected = amplitude**2 * 16 * 8 * 2e-6 * 3e-6
    assert wf.power.unit == u.W
    assert wf.power.to_value(u.W) == pytest.approx(expected, rel=1e-14)


def test_intensity_unit_is_power_per_area_or_per_solid_angle():
    assert Wavefront(np.ones(GRID.shape), GRID, WAVELENGTH).intensity_unit.is_equivalent(u.W / u.m**2)
    angular = Grid(4, dx=1 * u.arcsec)
    wf = Wavefront(np.ones(angular.shape), angular, WAVELENGTH, power_unit=u.photon / u.s)
    assert wf.intensity_unit.is_equivalent(u.photon / u.s / u.sr)
    # Power of a uniform field on an angular grid: |U|² × N × dΩ
    dtheta = (1 * u.arcsec).to_value(u.rad)
    assert wf.power.to_value(u.photon / u.s) == pytest.approx(16 * dtheta**2, rel=1e-12)


def test_gaussian_beam_carries_the_requested_power():
    grid = Grid(128, dx=1 * u.um)  # w = 8 px, half-width = 8 w
    wf = Wavefront.gaussian(grid, WAVELENGTH, waist=8 * u.um, power=2 * u.mW)
    assert wf.power.to_value(u.mW) == pytest.approx(2.0, rel=1e-12)
    # Peak intensity 2P / (π w²)
    iy, ix = grid.center_index
    assert wf.intensity[iy, ix] == pytest.approx(2 * 2e-3 / (math.pi * 64e-12), rel=1e-12)


def test_plane_wave_carries_the_requested_power():
    wf = Wavefront.plane(GRID, WAVELENGTH, power=5 * u.W)
    assert wf.power.to_value(u.W) == pytest.approx(5.0, rel=1e-14)
    np.testing.assert_allclose(wf.phase, 0.0)


@given(field=complex_fields, target=st.floats(1e-6, 1e6))
def test_normalized_sets_the_power(field, target):
    wf = Wavefront(np.abs(field) + 1.0, GRID, WAVELENGTH)  # never zero
    out = wf.normalized(target * u.W)
    assert out.power.to_value(u.W) == pytest.approx(target, rel=1e-10)


def test_normalizing_a_zero_field_is_refused():
    with pytest.raises(ValueError, match="zero"):
        Wavefront(np.zeros(GRID.shape), GRID, WAVELENGTH).normalized(1 * u.W)


# --- Power unit conversion (E = hc/λ) ----------------------------------------------------------


def test_watt_to_photon_rate_uses_photon_energy():
    wf = Wavefront.plane(GRID, 1 * u.um, power=1 * u.W)
    photons = wf.with_power_unit(u.photon / u.s)
    assert photons.power_unit == u.photon / u.s
    assert photons.power.to_value(u.photon / u.s) == pytest.approx(1e-6 / (H * C), rel=1e-12)
    assert photons.power.to_value(u.photon / u.s) == pytest.approx(5.0341165675427096e18, rel=1e-12)


def test_power_unit_round_trip():
    wf = Wavefront.plane(GRID, 550 * u.nm, power=3 * u.mW)
    back = wf.with_power_unit(u.photon / u.s).with_power_unit(u.W)
    assert back.power.to_value(u.mW) == pytest.approx(3.0, rel=1e-12)


# --- Operations --------------------------------------------------------------------------------


def test_multiplying_by_a_transmission_mask():
    wf = Wavefront.plane(GRID, WAVELENGTH, power=1 * u.W)
    mask = np.zeros(GRID.shape)
    mask[:, :8] = 1.0  # half of the pixels
    out = wf * mask
    assert isinstance(out, Wavefront)
    assert out.power.to_value(u.W) == pytest.approx(0.5, rel=1e-14)
    assert (mask * wf).power.to_value(u.W) == pytest.approx(0.5, rel=1e-14)


def test_multiplying_by_a_scalar_scales_the_power_quadratically():
    wf = Wavefront.plane(GRID, WAVELENGTH, power=1 * u.W)
    assert (wf * 3).power.to_value(u.W) == pytest.approx(9.0, rel=1e-14)
    assert (wf * 1j).power.to_value(u.W) == pytest.approx(1.0, rel=1e-14)


def test_coherent_sum_interferes():
    wf = Wavefront.plane(GRID, WAVELENGTH, power=1 * u.W)
    assert (wf + wf).power.to_value(u.W) == pytest.approx(4.0, rel=1e-14)
    assert (wf + wf * -1).power.to_value(u.W) == pytest.approx(0.0, abs=1e-30)


def test_coherent_sum_needs_the_same_grid_wavelength_and_unit():
    wf = Wavefront.plane(GRID, WAVELENGTH, power=1 * u.W)
    with pytest.raises(ValueError, match="grid"):
        wf + Wavefront.plane(Grid(16, 8, dx=1 * u.um), WAVELENGTH, power=1 * u.W)
    with pytest.raises(ValueError, match="wavelength"):
        wf + Wavefront.plane(GRID, 2 * u.um, power=1 * u.W)
    with pytest.raises(ValueError, match="unit"):
        wf + wf.with_power_unit(u.photon / u.s)


@given(theta=st.floats(-0.3, 0.3))
def test_tilt_adds_the_plane_wave_phase_gradient(theta):
    wf = Wavefront.plane(GRID, WAVELENGTH, power=1 * u.W).tilt(theta * u.rad, 0 * u.rad)
    k = 2 * math.pi / 1e-6
    step = np.exp(1j * k * 2e-6 * math.sin(theta))
    ratio = wf.field[:, 1:] / wf.field[:, :-1]
    np.testing.assert_allclose(ratio, step, rtol=1e-9)
    np.testing.assert_allclose(wf.field[:, 0] / wf.field[0, 0], 1.0, rtol=1e-9)  # no y gradient
    assert wf.power.to_value(u.W) == pytest.approx(1.0, rel=1e-12)  # a pure phase


def test_tilt_phase_is_zero_at_the_grid_centre():
    wf = Wavefront.plane(GRID, WAVELENGTH, power=1 * u.W).tilt(0.01 * u.rad, -0.02 * u.rad)
    iy, ix = GRID.center_index
    assert np.angle(wf.field[iy, ix]) == pytest.approx(0.0, abs=1e-12)


def test_tilt_requires_a_spatial_grid():
    angular = Grid(4, dx=1 * u.arcsec)
    with pytest.raises(ValueError, match="spatial"):
        Wavefront(np.ones(angular.shape), angular, WAVELENGTH).tilt(1 * u.arcsec, 0 * u.arcsec)


# --- Backends and serialisation ----------------------------------------------------------------


@pytest.mark.parametrize("name", BACKENDS)
# the first example of a new shape compiles the JAX operations (about 0.2 s), which is not what is tested
@settings(deadline=None)
@given(field=complex_fields)
def test_power_is_invariant_under_backend_change(name, field):
    wf = Wavefront(field, GRID, WAVELENGTH)
    moved = wf.to_backend(name)
    assert moved.backend == name
    assert moved.power.to_value(u.W) == pytest.approx(wf.power.to_value(u.W), rel=1e-12, abs=1e-300)


@given(field=complex_fields)
def test_power_is_invariant_under_serialisation(field):
    wf = Wavefront(field, GRID, 633 * u.nm, power_unit=u.photon / u.s)
    back = Wavefront.from_json(json.loads(json.dumps(wf.to_json())))
    np.testing.assert_array_equal(backend.to_numpy(back.field), field)
    assert back.grid == GRID
    assert back.wavelength == 633 * u.nm
    assert back.power_unit == u.photon / u.s
    assert back.power == wf.power


@needs_jax
def test_wavefront_is_a_jax_pytree():
    import jax
    import jax.numpy as jnp

    wf = Wavefront.plane(GRID, WAVELENGTH, power=1 * u.W).to_backend("jax")

    @jax.jit
    def attenuate(w):
        return w * 0.5

    out = attenuate(wf)
    assert isinstance(out, Wavefront)
    assert out.grid == GRID
    assert out.power.to_value(u.W) == pytest.approx(0.25, rel=1e-12)

    def power(scale):
        return jnp.sum(jnp.abs((wf * scale).field) ** 2) * GRID.dx_si * GRID.dy_si

    # d(s² P)/ds = 2 s P with P = 1 W
    assert float(jax.grad(power)(3.0)) == pytest.approx(6.0, rel=1e-12)
