"""A photonic component given by its transfer matrix (step P14.19, review of 19/10/2026; §5.9, REQ-SIG-7).

Requirement: a simplified MMI whose complex transfer matrix ``S_ij = A_ij exp(i φ_ij)`` (output ``i``, input
``j``) is typed by the user as amplitudes and phases at a given wavelength. It acts on the channels listed in
``channels`` (all by default), may have more or fewer outputs than inputs, is the same at every wavelength
(``dispersion = "achromatic"``) or has phases that are optical paths (``"optical_path"``: ``φ λ0 / λ``, as the
thermo-optic phase shifter). A passive component cannot amplify light: a matrix whose largest singular value
exceeds 1 is refused, at validation and at run time.

Independent references:

- a lossless (unitary) matrix conserves the power, for any input (Pozar, *Microwave Engineering*, §4.3);
- the ideal 3-dB coupler ``(1/√2) [[1, i], [i, 1]]`` sends half of the power of one input to each output, the
  cross output in quadrature (Soldano & Pennings 1995, J. Lightwave Technol. 13, 615), as the
  ``DirectionalCoupler`` of ratio 0.5;
- a 2 → 1 combiner ``(1/√2) [1, 1]`` gives ``2P`` for two inputs of power ``P`` in phase and 0 in antiphase;
- the amplitude transmission ``t`` multiplies the power by ``t²``.
"""

import math

import numpy as np
import pytest
from astropy import units as u
from hypothesis import given, settings
from hypothesis import strategies as st

from lumos import Context
from lumos.elements.photonics import DirectionalCoupler, TransferMatrix
from lumos.fields import ModalField

LAM = 1.55e-6


def modal(amplitudes, wavelengths=(LAM,)):
    a = np.asarray(amplitudes, dtype=complex)
    if a.ndim == 1:
        a = np.tile(a, (len(wavelengths), 1))
    return ModalField(a, np.asarray(wavelengths) * u.m)


def run(element, field):
    return np.asarray(element.apply(**{"in": field})["out"].amplitudes)


def component(matrix, **options):
    matrix = np.asarray(matrix, dtype=complex)
    return TransferMatrix(amplitudes=np.abs(matrix), phases=np.angle(matrix) * u.rad, **options)


def test_the_default_is_an_ideal_three_db_coupler():
    element = TransferMatrix()
    out = run(element, modal([1.0, 0.0]))[0]
    assert np.abs(out) ** 2 == pytest.approx([0.5, 0.5], abs=1e-12)
    assert np.angle(out[1] / out[0]) == pytest.approx(math.pi / 2, abs=1e-12)
    rng = np.random.default_rng(4)
    a = rng.normal(size=2) + 1j * rng.normal(size=2)
    coupler = DirectionalCoupler(ratio=0.5)
    np.testing.assert_allclose(np.abs(run(element, modal(a))), np.abs(run(coupler, modal(a))), atol=1e-12)


@settings(max_examples=40, deadline=None)
@given(n=st.integers(1, 5), seed=st.integers(0, 2**32 - 1))
def test_a_unitary_matrix_conserves_the_power(n, seed):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    unitary = q * (np.diag(r) / np.abs(np.diag(r)))
    a = rng.normal(size=n) + 1j * rng.normal(size=n)
    out = run(component(unitary), modal(a))[0]
    assert np.sum(np.abs(out) ** 2) == pytest.approx(np.sum(np.abs(a) ** 2), rel=1e-9)
    np.testing.assert_allclose(out, unitary @ a, atol=1e-9)


def test_a_lossy_matrix_on_selected_channels():
    element = component([[0.5]], channels=[1])
    out = run(element, modal([1.0, 2.0, 3.0]))[0]
    assert abs(out[1]) ** 2 == pytest.approx(0.25 * 4.0)
    assert out[0] == 1.0
    assert out[2] == 3.0


@pytest.mark.parametrize(("phase", "power"), [(0.0, 2.0), (math.pi, 0.0)])
def test_a_combiner_has_one_output_for_two_inputs(phase, power):
    element = component([[1 / math.sqrt(2), 1 / math.sqrt(2)]])
    out = run(element, modal([1.0, np.exp(1j * phase)]))
    assert out.shape == (1, 1)
    assert abs(out[0, 0]) ** 2 == pytest.approx(power, abs=1e-12)


def test_phases_given_as_optical_paths_scale_with_the_wavelength():
    lams = (1.55e-6, 3.1e-6)
    achromatic = component([[np.exp(1.2j)]], wavelength=1.55 * u.um)
    optical = component([[np.exp(1.2j)]], wavelength=1.55 * u.um, dispersion="optical_path")
    np.testing.assert_allclose(run(achromatic, modal([1.0], lams))[:, 0], np.exp(1.2j), atol=1e-12)
    np.testing.assert_allclose(
        run(optical, modal([1.0], lams))[:, 0], np.exp(1.2j * np.array([1.0, 0.5])), atol=1e-12
    )


def test_a_matrix_that_amplifies_is_refused():
    element = TransferMatrix(amplitudes=[[1.2]], phases=[[0.0]] * u.rad)
    assert any("amplif" in problem for problem in element.problems())
    with pytest.raises(ValueError, match="amplif"):
        run(element, modal([1.0]))


def test_amplitudes_and_phases_must_have_the_same_shape():
    element = TransferMatrix(amplitudes=[[1.0, 0.0], [0.0, 1.0]], phases=[[0.0]] * u.rad)
    assert any("shape" in problem for problem in element.problems())
    with pytest.raises(ValueError, match="shape"):
        run(element, modal([1.0, 1.0]))


def test_the_number_of_inputs_must_match_the_channels():
    element = TransferMatrix()  # 2 inputs
    with pytest.raises(ValueError, match="channel"):
        run(element, modal([1.0, 1.0, 1.0]))


def test_it_survives_a_lumos_round_trip():
    import lumos.io

    ctx = Context()
    ctx.add(component([[0.6, 0.8j], [0.8j, 0.6]], wavelength=1.3 * u.um, name="mmi"))
    loaded = lumos.io.from_document(lumos.io.to_document(ctx))
    element = loaded["mmi"]
    assert isinstance(element, TransferMatrix)
    np.testing.assert_allclose(np.asarray(element.amplitudes), [[0.6, 0.8], [0.8, 0.6]])
    assert element.wavelength == 1.3 * u.um
