"""Tests for lumos.fields.ModalField and scattering matrices (step P3.7).

Requirements: REQ-SIG-7 (complex amplitudes per waveguide or mode, vectorised over wavelengths and incoherent
modes; photonic components act by scattering matrices, possibly wavelength dependent).

Independent references:

- lossless components have unitary scattering matrices, ``S^† S = 1`` (Pozar, *Microwave Engineering*,
  4th ed., §4.3), so they conserve the total power;
- symmetric directional coupler of power coupling ratio κ²: ``[[t, iκ], [iκ, t]]`` with ``t = sqrt(1 - κ²)``
  (Okamoto, *Fundamentals of Optical Waveguides*, 2nd ed., §4.4); the 50/50 coupler is
  ``[[1, i], [i, 1]] / √2``;
- Mach-Zehnder interferometer made of two 50/50 couplers with a phase φ on the first arm: the output powers
  are ``sin²(φ/2)`` and ``cos²(φ/2)`` for light entering port 0 (direct product of the matrices above);
- mutually incoherent inputs add in power, with no interference term.
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
from lumos.fields import ModalField, smatrix

HAS_JAX = importlib.util.find_spec("jax") is not None
BACKENDS = [
    "numpy",
    pytest.param("jax", marks=pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")),
]
LAM = [1.55] * u.um


def single(amplitudes, wavelengths=LAM, **kwargs):
    return ModalField(np.asarray(amplitudes, dtype=complex)[None, :], wavelengths, **kwargs)


def test_modal_field_is_an_optical_signal():
    assert issubclass(ModalField, OpticalSignal)


# --- Construction ------------------------------------------------------------------------------


def test_shapes_and_attributes():
    mf = ModalField(np.ones((2, 3)), [1.5, 1.6] * u.um)
    assert mf.n_wavelengths == 2
    assert mf.n_modes == 3
    assert mf.n_sources == 1
    assert mf.shape == (2, 3)
    assert mf.power_unit == u.W


def test_incoherent_mode_axis():
    mf = ModalField(np.ones((4, 2, 3)), [1.5, 1.6] * u.um)
    assert mf.n_sources == 4
    assert mf.shape == (4, 2, 3)


def test_shape_must_match_wavelengths():
    with pytest.raises(ValueError, match="shape"):
        ModalField(np.ones((3, 2)), [1.5, 1.6] * u.um)


def test_mode_labels():
    mf = single([1, 0, 0], labels=["a", "b", "c"])
    assert mf.labels == ("a", "b", "c")
    with pytest.raises(ValueError):
        single([1, 0], labels=["a"])


# --- Powers --------------------------------------------------------------------------------------


def test_mode_power_is_the_squared_modulus():
    mf = single([1.0, 2j, 3 - 4j])
    np.testing.assert_allclose(mf.mode_power.to_value(u.W), [[1.0, 4.0, 25.0]], rtol=1e-14)
    np.testing.assert_allclose(mf.total_power.to_value(u.W), [30.0], rtol=1e-14)


def test_incoherent_modes_add_in_power():
    # Two incoherent inputs with opposite phases: no cancellation.
    mf = ModalField(np.array([[[1.0, 0.0]], [[-1.0, 0.0]]]), LAM)
    np.testing.assert_allclose(mf.mode_power.to_value(u.W), [[2.0, 0.0]], rtol=1e-14)


def test_spectral_weights_enter_the_detected_power():
    mf = ModalField(np.array([[1.0, 0.0], [0.0, 2.0]]), [1.5, 1.6] * u.um, weights=[0.5, 0.25])
    np.testing.assert_allclose(mf.detected_power.to_value(u.W), [0.5, 1.0], rtol=1e-14)


# --- Scattering matrices ---------------------------------------------------------------------------


def test_50_50_coupler_matrix_and_unitarity():
    c = smatrix.coupler(0.5)
    np.testing.assert_allclose(c, np.array([[1, 1j], [1j, 1]]) / math.sqrt(2), atol=1e-15)
    np.testing.assert_allclose(c.conj().T @ c, np.eye(2), atol=1e-15)
    assert smatrix.is_unitary(c)


@given(ratio=st.floats(0, 1))
def test_couplers_are_unitary_and_split_as_requested(ratio):
    c = smatrix.coupler(ratio)
    assert smatrix.is_unitary(c)
    out = single([1, 0]).apply(c)
    np.testing.assert_allclose(out.mode_power.to_value(u.W)[0], [1 - ratio, ratio], atol=1e-14)


def test_coupler_ratio_is_validated():
    with pytest.raises(ValueError):
        smatrix.coupler(1.5)


@given(phi=st.floats(-2 * math.pi, 2 * math.pi))
def test_mach_zehnder_output_powers(phi):
    c = smatrix.coupler(0.5)
    mzi = c @ smatrix.phase_shifter([phi * u.rad, 0 * u.rad]) @ c
    out = single([1, 0]).apply(mzi).mode_power.to_value(u.W)[0]
    np.testing.assert_allclose(out, [math.sin(phi / 2) ** 2, math.cos(phi / 2) ** 2], atol=1e-14)


def test_phase_shifter_is_diagonal():
    ps = smatrix.phase_shifter([math.pi / 2 * u.rad, 0 * u.rad])
    np.testing.assert_allclose(ps, np.diag([1j, 1]), atol=1e-15)


@given(
    real=hnp.arrays(np.float64, (4, 4), elements=st.floats(-1, 1)),
    imag=hnp.arrays(np.float64, (4, 4), elements=st.floats(-1, 1)),
    a=hnp.arrays(np.complex128, (3, 4), elements=st.complex_numbers(max_magnitude=10)),
)
def test_unitary_matrices_conserve_the_power(real, imag, a):
    q, _ = np.linalg.qr(real + 1j * imag + 3 * np.eye(4))  # well-conditioned, so Q is unitary
    mf = ModalField(a, [1.0, 1.1, 1.2] * u.um)
    out = mf.apply(q)
    np.testing.assert_allclose(out.total_power.value, mf.total_power.value, rtol=1e-10, atol=1e-10)


def test_wavelength_dependent_matrix():
    s = np.stack([smatrix.coupler(0.0), smatrix.coupler(1.0)])  # bar at λ0, cross at λ1
    mf = ModalField(np.array([[1.0, 0.0], [1.0, 0.0]]), [1.5, 1.6] * u.um)
    out = mf.apply(s).mode_power.to_value(u.W)
    np.testing.assert_allclose(out, [[1.0, 0.0], [0.0, 1.0]], atol=1e-14)


def test_non_square_matrix_changes_the_number_of_modes():
    splitter = np.array([[1.0], [1.0]]) / math.sqrt(2)  # 1 -> 2 outputs (lossless in this direction)
    out = single([2.0]).apply(splitter, labels=["left", "right"])
    assert out.n_modes == 2
    assert out.labels == ("left", "right")
    np.testing.assert_allclose(out.mode_power.to_value(u.W), [[2.0, 2.0]], rtol=1e-14)


def test_matrix_shape_is_validated():
    with pytest.raises(ValueError, match="shape"):
        single([1, 0]).apply(np.eye(3))
    with pytest.raises(ValueError, match="shape"):
        ModalField(np.ones((2, 2)), [1.5, 1.6] * u.um).apply(np.ones((3, 2, 2)))


def test_matrices_apply_to_every_incoherent_mode():
    mf = ModalField(np.array([[[1.0, 0.0]], [[0.0, 1.0]]]), LAM)
    out = mf.apply(smatrix.coupler(0.5))
    np.testing.assert_allclose(out.mode_power.to_value(u.W), [[1.0, 1.0]], rtol=1e-14)


# --- Backends and serialisation ----------------------------------------------------------------


@pytest.mark.parametrize("name", BACKENDS)
def test_apply_on_every_backend(name):
    mf = single([1, 0]).to_backend(name)
    out = mf.apply(smatrix.coupler(0.3))
    assert out.backend == name
    np.testing.assert_allclose(out.mode_power.to_value(u.W)[0], [0.7, 0.3], atol=1e-14)


def test_json_round_trip():
    mf = ModalField(np.array([[[1.0, 2j]], [[0.5, -1.0]]]), LAM, labels=["a", "b"], power_unit=u.photon / u.s)
    back = ModalField.from_json(json.loads(json.dumps(mf.to_json())))
    np.testing.assert_array_equal(backend.to_numpy(back.amplitudes), backend.to_numpy(mf.amplitudes))
    assert back.labels == ("a", "b")
    assert back.power_unit == u.photon / u.s


@pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")
def test_gradient_through_a_mach_zehnder():
    import jax
    import jax.numpy as jnp

    c = jnp.asarray(smatrix.coupler(0.5))
    mf = single([1, 0]).to_backend("jax")

    def bar_power(phi):
        ps = jnp.diag(jnp.stack([jnp.exp(1j * phi), jnp.asarray(1.0 + 0j)]))
        return jnp.abs(mf.apply(c @ ps @ c).amplitudes[0, 0]) ** 2

    # d/dφ sin²(φ/2) = sin(φ)/2
    phi = 0.7
    assert float(jax.grad(bar_power)(phi)) == pytest.approx(math.sin(phi) / 2, rel=1e-10)


def test_a_modal_field_names_the_source_list_it_describes():
    """Lanes interfere only if they describe the same sources (step P15.6, ADR-0023)."""
    mf = single([1, 0], sources="abc")
    assert mf.sources == "abc"
    assert mf.apply(smatrix.coupler(0.5)).sources == "abc"
    assert (mf * 2).sources == "abc"
    assert single([1, 0]).sources is None
    back = ModalField.from_json(json.loads(json.dumps(mf.to_json())))
    assert back.sources == "abc"
