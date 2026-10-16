"""Tests for lumos.fields.IncoherentBundle (step P3.5).

Requirements: REQ-SIG-5 (set of mutually incoherent modes: point sources, samples of an extended scene,
partial coherence or polarisation modes; propagation vectorised over the mode axis; summed **in intensity** at
detection) and the plan of P3.5 (mode axis, intensity sum, phase ramps for off-axis sources).

Independent references:

- a point source at infinity in the direction (θx, θy) illuminates the pupil with the plane wave
  ``exp[ik(x sin θx + y sin θy)]`` (Born & Wolf, *Principles of Optics*, §1.4);
- the intensities of mutually incoherent fields add, with no interference term (Born & Wolf §10.3; Goodman,
  *Statistical Optics*, §5.2); two coherent plane waves of equal amplitude A at ±θ give the fringes
  ``4 A² cos²(k x sin θ)`` (Hecht, *Optics*, §9.3);
- natural (unpolarised) light is an equal incoherent mixture of two orthogonal polarisations: an ideal
  polariser transmits half of it whatever its angle, and its degree of polarisation is 0 (Hecht §8.2,
  Born & Wolf §10.8).
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
from lumos.fields import Grid, IncoherentBundle, Wavecube, Wavefront, jones

HAS_JAX = importlib.util.find_spec("jax") is not None
BACKENDS = [
    "numpy",
    pytest.param("jax", marks=pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")),
]

GRID = Grid(32, 16, dx=1 * u.um)
WAVELENGTH = 1 * u.um
K = 2 * math.pi / 1e-6


def test_bundle_is_an_optical_signal():
    assert issubclass(IncoherentBundle, OpticalSignal)


# --- Point sources and phase ramps -------------------------------------------------------------


@given(tx=st.floats(-0.1, 0.1), ty=st.floats(-0.1, 0.1))
def test_off_axis_point_source_has_the_plane_wave_phase_ramp(tx, ty):
    bundle = IncoherentBundle.point_sources(
        GRID, WAVELENGTH, theta_x=[tx] * u.rad, theta_y=[ty] * u.rad, power=[1.0] * u.W
    )
    field = backend.to_numpy(bundle.field[0])
    xx, yy = GRID.coords(backend="numpy")
    expected = np.exp(1j * K * (xx * math.sin(tx) + yy * math.sin(ty)))
    np.testing.assert_allclose(field / field[GRID.center_index], expected, rtol=1e-9, atol=1e-9)


def test_point_sources_carry_their_powers():
    bundle = IncoherentBundle.point_sources(
        GRID,
        WAVELENGTH,
        theta_x=[0, 0.01, -0.02] * u.rad,
        theta_y=[0, 0, 0.01] * u.rad,
        power=[1, 2, 3] * u.W,
    )
    assert bundle.n_modes == 3
    np.testing.assert_allclose(bundle.power.to_value(u.W), [1, 2, 3], rtol=1e-12)
    assert bundle.total_power.to_value(u.W) == pytest.approx(6.0, rel=1e-12)


def test_point_source_angles_accept_any_angle_unit():
    a = IncoherentBundle.point_sources(
        GRID, WAVELENGTH, theta_x=[1] * u.arcsec, theta_y=[0] * u.arcsec, power=[1] * u.W
    )
    b = IncoherentBundle.point_sources(
        GRID, WAVELENGTH, theta_x=[math.pi / 648000] * u.rad, theta_y=[0] * u.rad, power=[1] * u.W
    )
    np.testing.assert_allclose(backend.to_numpy(a.field), backend.to_numpy(b.field), rtol=1e-12)


def test_point_source_arrays_must_have_the_same_length():
    with pytest.raises(ValueError):
        IncoherentBundle.point_sources(
            GRID, WAVELENGTH, theta_x=[0, 1] * u.rad, theta_y=[0] * u.rad, power=[1, 1] * u.W
        )


# --- Incoherent sum ------------------------------------------------------------------------------


def test_incoherent_modes_add_in_intensity_without_fringes():
    theta = 0.05
    bundle = IncoherentBundle.point_sources(
        GRID, WAVELENGTH, theta_x=[theta, -theta] * u.rad, theta_y=[0, 0] * u.rad, power=[1, 1] * u.W
    )
    intensity = backend.to_numpy(bundle.intensity)
    a2 = 1.0 / (GRID.size * 1e-12)  # |A|² of a 1 W plane wave on the grid
    np.testing.assert_allclose(intensity, 2 * a2, rtol=1e-12)


def test_the_same_modes_added_coherently_make_fringes():
    theta = 0.05
    bundle = IncoherentBundle.point_sources(
        GRID, WAVELENGTH, theta_x=[theta, -theta] * u.rad, theta_y=[0, 0] * u.rad, power=[1, 1] * u.W
    )
    coherent = bundle[0] + bundle[1]
    xx, _ = GRID.coords(backend="numpy")
    a2 = 1.0 / (GRID.size * 1e-12)
    np.testing.assert_allclose(
        coherent.intensity, 4 * a2 * np.cos(K * xx * math.sin(theta)) ** 2, rtol=1e-9, atol=1e-9 * a2
    )


def test_mode_weights_scale_the_sum():
    wfs = [Wavefront.plane(GRID, WAVELENGTH, power=1 * u.W), Wavefront.plane(GRID, WAVELENGTH, power=2 * u.W)]
    bundle = IncoherentBundle.from_signals(wfs, weights=[0.5, 3.0])
    assert bundle.total_power.to_value(u.W) == pytest.approx(0.5 + 6.0, rel=1e-14)
    expected = 0.5 * wfs[0].intensity + 3.0 * wfs[1].intensity
    np.testing.assert_allclose(bundle.intensity, expected, rtol=1e-14)
    np.testing.assert_allclose(bundle.mode_intensity[1], wfs[1].intensity, rtol=1e-14)


def test_indexing_returns_the_member_signal():
    wfs = [Wavefront.gaussian(GRID, WAVELENGTH, waist=4 * u.um, power=p * u.W) for p in (1, 2)]
    bundle = IncoherentBundle.from_signals(wfs)
    assert isinstance(bundle[1], Wavefront)
    np.testing.assert_array_equal(backend.to_numpy(bundle[1].field), backend.to_numpy(wfs[1].field))
    assert len(list(bundle.members())) == 2


def test_members_must_be_compatible():
    a = Wavefront.plane(GRID, WAVELENGTH, power=1 * u.W)
    with pytest.raises(ValueError, match="grid"):
        IncoherentBundle.from_signals(
            [a, Wavefront.plane(Grid(32, 16, dx=2 * u.um), WAVELENGTH, power=1 * u.W)]
        )
    with pytest.raises(ValueError, match="wavelength"):
        IncoherentBundle.from_signals([a, Wavefront.plane(GRID, 2 * u.um, power=1 * u.W)])
    with pytest.raises(ValueError):
        IncoherentBundle.from_signals([])


def test_mask_applies_to_every_mode():
    bundle = IncoherentBundle.point_sources(
        GRID, WAVELENGTH, theta_x=[0, 0.01] * u.rad, theta_y=[0, 0] * u.rad, power=[1, 1] * u.W
    )
    mask = np.zeros(GRID.shape)
    mask[:, :16] = 1.0
    np.testing.assert_allclose((bundle * mask).power.to_value(u.W), [0.5, 0.5], rtol=1e-12)


# --- Polychromatic modes -------------------------------------------------------------------------


def test_polychromatic_point_sources():
    lam = [1.0, 2.0] * u.um
    bundle = IncoherentBundle.point_sources(
        GRID,
        lam,
        theta_x=[0.01, 0.0] * u.rad,
        theta_y=[0, 0] * u.rad,
        power=[1, 1] * u.W,
        weights=[0.25, 0.75],
    )
    assert bundle.shape == (2, 2, *GRID.shape)
    assert isinstance(bundle[0], Wavecube)
    # Each mode carries 1 W per channel; integrated with the spectral weights.
    assert bundle.total_power.to_value(u.W) == pytest.approx(2 * (0.25 + 0.75), rel=1e-12)
    # The ramp of mode 0 is chromatic: phase step k_λ dx sin θ.
    field = backend.to_numpy(bundle.field[0, 1])
    np.testing.assert_allclose(
        field[:, 1:] / field[:, :-1], np.exp(1j * math.pi / 1e-6 * 1e-6 * math.sin(0.01)), rtol=1e-9
    )


# --- Unpolarised light ---------------------------------------------------------------------------


def test_unpolarized_light_is_two_orthogonal_incoherent_modes():
    bundle = IncoherentBundle.unpolarized(Wavefront.plane(GRID, WAVELENGTH, power=2 * u.W))
    assert bundle.n_modes == 2
    assert bundle.polarized
    assert bundle.total_power.to_value(u.W) == pytest.approx(2.0, rel=1e-14)
    s = backend.to_numpy(jones.stokes(bundle))
    assert s.shape == (4, *GRID.shape)
    assert jones.degree_of_polarization(s[:, 3, 5]) == pytest.approx(0.0, abs=1e-12)


@given(theta=st.floats(-math.pi, math.pi))
def test_polarizer_transmits_half_of_unpolarized_light(theta):
    bundle = IncoherentBundle.unpolarized(Wavefront.plane(GRID, WAVELENGTH, power=1 * u.W))
    out = jones.apply(jones.polarizer(theta * u.rad), bundle)
    assert out.total_power.to_value(u.W) == pytest.approx(0.5, rel=1e-12)


# --- Backends and serialisation ----------------------------------------------------------------


@pytest.mark.parametrize("name", BACKENDS)
def test_total_power_is_invariant_under_backend_change(name):
    bundle = IncoherentBundle.point_sources(
        GRID, WAVELENGTH, theta_x=[0, 0.01] * u.rad, theta_y=[0.02, 0] * u.rad, power=[1, 3] * u.W
    )
    moved = bundle.to_backend(name)
    assert moved.backend == name
    assert moved.total_power.to_value(u.W) == pytest.approx(4.0, rel=1e-12)


def test_json_round_trip():
    bundle = IncoherentBundle.point_sources(
        GRID,
        [1.0, 1.5] * u.um,
        theta_x=[0, 0.01] * u.rad,
        theta_y=[0.02, 0] * u.rad,
        power=[1, 3] * u.W,
        mode_weights=[2.0, 0.5],
    )
    back = IncoherentBundle.from_json(json.loads(json.dumps(bundle.to_json())))
    np.testing.assert_array_equal(backend.to_numpy(back.field), backend.to_numpy(bundle.field))
    np.testing.assert_array_equal(back.mode_weights, [2.0, 0.5])
    assert back.total_power == bundle.total_power


@pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")
def test_bundle_is_a_jax_pytree():
    import jax

    bundle = IncoherentBundle.unpolarized(Wavefront.plane(GRID, WAVELENGTH, power=1 * u.W)).to_backend("jax")
    out = jax.jit(lambda b: b * 0.5)(bundle)
    assert isinstance(out, IncoherentBundle)
    assert out.total_power.to_value(u.W) == pytest.approx(0.25, rel=1e-12)
