"""Tests for polarisation: polarised Wavefronts, Jones matrices and Stokes parameters (step P3.3).

Requirements: REQ-SIG-3 (Jones components ``(Ex, Ey)`` of shape ``(2, ny, nx)``; polarising elements act by
uniform or spatially varying Jones matrices) and §9 of the specs (laboratory (x, y) basis, Jones matrices,
Stokes for incoherent descriptions).

Independent references (Hecht, *Optics*, 5th ed., §8; Born & Wolf, *Principles of Optics*, §1.4.2):

- Malus's law: a polariser at angle θ to a linear polarisation transmits ``cos² θ`` of the power;
- crossed polarisers transmit 0; polarisers at 0°, 45°, 90° in sequence transmit ``cos⁴ 45° = 1/4``;
- a half-wave plate with its fast axis at θ turns a linear polarisation at 0° into one at 2θ;
- two quarter-wave plates with parallel axes make a half-wave plate;
- a quarter-wave plate at 45° turns a linear polarisation into a circular one (``|S3| = S0``);
- retarders and rotators are unitary, polarisers are idempotent projectors;
- a fully polarised state satisfies ``S0² = S1² + S2² + S3²``;
- Stokes vectors of the basis states: H ``(1, 1, 0, 0)``, V ``(1, -1, 0, 0)``, D ``(1, 0, 1, 0)``,
  A ``(1, 0, -1, 0)``, R ``(1, 0, 0, 1)``, L ``(1, 0, 0, -1)`` with ``S3 = 2 Im(Ex* Ey)`` and, with the
  ``e^{-iωt}`` convention of §9, R = (1, i)/√2 (clockwise rotation seen by an observer facing the source).
"""

import importlib.util
import math

import numpy as np
import pytest
from astropy import units as u
from hypothesis import given
from hypothesis import strategies as st

from lumos import backend
from lumos.fields import Grid, Wavefront, jones

HAS_JAX = importlib.util.find_spec("jax") is not None
BACKENDS = [
    "numpy",
    pytest.param("jax", marks=pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")),
]

GRID = Grid(8, 6, dx=1 * u.um)
WAVELENGTH = 1 * u.um
angles = st.floats(-math.pi, math.pi, allow_nan=False)


def polarized(state, power=1 * u.W):
    return Wavefront.plane(GRID, WAVELENGTH, power=power, polarization=state)


def power_w(wf):
    return wf.power.to_value(u.W)


# --- Polarised wavefronts ----------------------------------------------------------------------


def test_polarized_field_has_two_components():
    wf = polarized(jones.vector("H"))
    assert wf.polarized
    assert wf.shape == (2, 6, 8)
    assert wf.intensity.shape == (6, 8)
    assert power_w(wf) == pytest.approx(1.0, rel=1e-14)


def test_scalar_field_is_not_polarized():
    assert not Wavefront.plane(GRID, WAVELENGTH, power=1 * u.W).polarized


def test_polarized_field_from_array():
    field = np.zeros((2, *GRID.shape), dtype=complex)
    field[1] = 2.0
    wf = Wavefront(field, GRID, WAVELENGTH)
    assert wf.polarized
    assert power_w(wf) == pytest.approx(4.0 * GRID.size * 1e-12, rel=1e-14)


def test_bad_number_of_components_is_refused():
    with pytest.raises(ValueError, match="shape"):
        Wavefront(np.ones((3, *GRID.shape)), GRID, WAVELENGTH)


def test_with_polarization_turns_a_scalar_field_into_a_polarized_one():
    scalar = Wavefront.gaussian(Grid(64, dx=1 * u.um), WAVELENGTH, waist=8 * u.um, power=1 * u.W)
    wf = scalar.with_polarization(jones.vector("D"))
    np.testing.assert_allclose(wf.field[0], scalar.field / math.sqrt(2), rtol=1e-14)
    np.testing.assert_allclose(wf.field[1], scalar.field / math.sqrt(2), rtol=1e-14)
    assert power_w(wf) == pytest.approx(power_w(scalar), rel=1e-14)


def test_polarization_vector_is_normalised():
    wf = Wavefront.plane(GRID, WAVELENGTH, power=1 * u.W, polarization=np.array([3.0, 4.0j]))
    assert power_w(wf) == pytest.approx(1.0, rel=1e-14)


def test_jones_matrix_on_a_scalar_field_is_refused():
    with pytest.raises(ValueError, match="polari"):
        jones.apply(jones.polarizer(0 * u.deg), Wavefront.plane(GRID, WAVELENGTH, power=1 * u.W))


def test_adding_polarized_and_scalar_fields_is_refused():
    with pytest.raises(ValueError):
        polarized(jones.vector("H")) + Wavefront.plane(GRID, WAVELENGTH, power=1 * u.W)


# --- Basis states and Stokes parameters --------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("H", (1, 1, 0, 0)),
        ("V", (1, -1, 0, 0)),
        ("D", (1, 0, 1, 0)),
        ("A", (1, 0, -1, 0)),
        ("R", (1, 0, 0, 1)),
        ("L", (1, 0, 0, -1)),
    ],
)
def test_stokes_of_basis_states(name, expected):
    s = jones.stokes(polarized(jones.vector(name)))
    assert s.shape == (4, *GRID.shape)
    s0 = s[0, 0, 0]
    np.testing.assert_allclose(s[:, 2, 3] / s0, expected, atol=1e-14)


def test_right_circular_vector_convention():
    np.testing.assert_allclose(jones.vector("R"), np.array([1, 1j]) / math.sqrt(2))


def test_stokes_s0_is_the_intensity():
    wf = polarized(jones.linear(0.3 * u.rad), power=2 * u.W)
    np.testing.assert_allclose(jones.stokes(wf)[0], wf.intensity, rtol=1e-14)


@given(a=st.complex_numbers(max_magnitude=10), b=st.complex_numbers(max_magnitude=10))
def test_fully_polarized_states_lie_on_the_poincare_sphere(a, b):
    if abs(a) + abs(b) < 1e-3:
        return
    s = jones.stokes(polarized(np.array([a, b])))[:, 0, 0]
    assert s[0] ** 2 == pytest.approx(s[1] ** 2 + s[2] ** 2 + s[3] ** 2, rel=1e-10)
    assert jones.degree_of_polarization(s) == pytest.approx(1.0, rel=1e-10)


def test_linear_state_at_angle():
    np.testing.assert_allclose(jones.linear(30 * u.deg), [math.cos(math.pi / 6), math.sin(math.pi / 6)])


def test_unknown_state_name_is_refused():
    with pytest.raises(ValueError):
        jones.vector("X")


# --- Jones matrices ----------------------------------------------------------------------------


@given(theta=angles)
def test_malus_law(theta):
    out = jones.apply(jones.polarizer(theta * u.rad), polarized(jones.vector("H")))
    assert power_w(out) == pytest.approx(math.cos(theta) ** 2, abs=1e-12)


def test_crossed_polarizers_extinguish():
    wf = jones.apply(jones.polarizer(90 * u.deg), jones.apply(jones.polarizer(0 * u.deg), polarized("H")))
    assert power_w(wf) == pytest.approx(0.0, abs=1e-28)


def test_three_polarizers_transmit_a_quarter():
    wf = polarized(jones.vector("H"))
    for angle in (0, 45, 90):
        wf = jones.apply(jones.polarizer(angle * u.deg), wf)
    assert power_w(wf) == pytest.approx(0.25, rel=1e-12)


@given(theta=angles)
def test_half_wave_plate_rotates_linear_polarisation_by_twice_its_angle(theta):
    out = jones.apply(jones.half_wave_plate(theta * u.rad), polarized(jones.vector("H")))
    # Analyse with a polariser at 2θ: everything passes; at 2θ + 90°: nothing passes.
    along = jones.apply(jones.polarizer(2 * theta * u.rad), out)
    across = jones.apply(jones.polarizer((2 * theta + math.pi / 2) * u.rad), out)
    assert power_w(along) == pytest.approx(1.0, rel=1e-12)
    assert power_w(across) == pytest.approx(0.0, abs=1e-12)


@given(theta=angles)
def test_two_quarter_wave_plates_make_a_half_wave_plate(theta):
    q = jones.quarter_wave_plate(theta * u.rad)
    h = jones.half_wave_plate(theta * u.rad)
    np.testing.assert_allclose(q @ q, h, atol=1e-14)


def test_quarter_wave_plate_at_45_degrees_makes_circular_light():
    out = jones.apply(jones.quarter_wave_plate(45 * u.deg), polarized(jones.vector("H")))
    s = jones.stokes(out)[:, 0, 0]
    assert abs(s[3]) == pytest.approx(s[0], rel=1e-12)
    assert s[1] == pytest.approx(0.0, abs=1e-12 * s[0])
    assert s[2] == pytest.approx(0.0, abs=1e-12 * s[0])


@given(
    retardance=angles,
    theta=angles,
)
def test_retarders_and_rotators_are_unitary(retardance, theta):
    for matrix in (jones.retarder(retardance * u.rad, theta * u.rad), jones.rotator(theta * u.rad)):
        np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(2), atol=1e-14)


@given(theta=angles)
def test_polarizer_is_an_idempotent_hermitian_projector(theta):
    p = jones.polarizer(theta * u.rad)
    np.testing.assert_allclose(p @ p, p, atol=1e-14)
    np.testing.assert_allclose(p.conj().T, p, atol=1e-14)


def test_rotator_rotates_linear_polarisation():
    out = jones.apply(jones.rotator(30 * u.deg), polarized(jones.vector("H")))
    np.testing.assert_allclose(
        out.field[:, 0, 0] / out.field[0, 0, 0] * math.cos(math.pi / 6), jones.linear(30 * u.deg), atol=1e-14
    )


def test_matrix_angles_require_quantities():
    with pytest.raises(TypeError):
        jones.polarizer(0.5)


def test_spatially_varying_matrix():
    # Left half: horizontal polariser; right half: vertical polariser, on diagonal light.
    matrix = np.zeros((2, 2, *GRID.shape), dtype=complex)
    matrix[:, :, :, :4] = jones.polarizer(0 * u.deg)[:, :, None, None]
    matrix[:, :, :, 4:] = jones.polarizer(90 * u.deg)[:, :, None, None]
    out = jones.apply(matrix, polarized(jones.vector("D")))
    assert power_w(out) == pytest.approx(0.5, rel=1e-12)
    assert np.all(np.abs(backend.to_numpy(out.field[1, :, :4])) < 1e-15)
    assert np.all(np.abs(backend.to_numpy(out.field[0, :, 4:])) < 1e-15)


def test_matrix_of_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="shape"):
        jones.apply(np.eye(3), polarized(jones.vector("H")))


@pytest.mark.parametrize("name", BACKENDS)
def test_jones_matrices_apply_on_every_backend(name):
    wf = polarized(jones.vector("H")).to_backend(name)
    out = jones.apply(jones.polarizer(60 * u.deg), wf)
    assert out.backend == name
    assert power_w(out) == pytest.approx(0.25, rel=1e-12)
