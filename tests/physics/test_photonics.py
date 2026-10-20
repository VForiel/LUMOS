"""Tests for integrated photonics: components, MMI mode solver and recombiner chips (step P6.8).

Requirements: §5.9 (waveguide, thermo-optic phase shifter, Y-splitter, directional coupler, MMI with a mode
solver, crossing; chip = ElementGroup of S-matrices; Bracewell and kernel-nuller presets after Martinache &
Ireland 2018), REQ-SIG-7 (modal amplitudes and scattering matrices), UC1.

Independent references:

- a waveguide of effective index n, length L and loss α multiplies the amplitude by
  ``10^(-α L / 20) exp(2iπ n L / λ)``; a thermo-optic shifter adds an optical path, so its phase scales as
  ``φ0 λ0 / λ``;
- lossless components conserve the power (unitary S, Pozar §4.3); a symmetric Y-junction splits the power
  in halves and combines two inputs as ``(a + b) / √2``, radiating the antisymmetric part (Okamoto §4);
- TE modes of a symmetric slab: ``u tan u = w`` (even) and ``-u cot u = w`` (odd), ``u² + w² = V²``
  (Marcuse, *Theory of Dielectric Optical Waveguides*, §1.3), solved independently with ``brentq``; a
  second-order finite-difference solver with 5 nm steps must match within 1e-5;
- an N × N multimode interference coupler of length ``3 L_π / N`` with ports at ``W_e (2i + 1 - N) / 2N``
  images its inputs into N outputs of power 1/N, the cross output of a 2 × 2 coupler being in quadrature
  with the bar output (Soldano & Pennings 1995, J. Lightwave Technol. 13, 615; Bachmann et al. 1994, Appl.
  Opt. 33, 3905); self-imaging is paraxial, so a well-designed coupler (ports narrower than W/N) deviates by
  a few percent: 2 % of 1/N for 2 × 2 and 5 % for 4 × 4, with excess losses below 2 % and 5 %;
- a Bracewell nuller (Bracewell 1978, Nature 274, 780) with inputs ``√P e^{iφ_j}`` gives ``2P sin²(Δφ/2)`` on
  its dark output and ``2P cos²(Δφ/2)`` on its bright output;
- a kernel nuller (Martinache & Ireland 2018, A&A 619, A87): a real 4 × 4 Hadamard nuller (one bright, three
  dark outputs) followed by a sensing stage combining the dark outputs pairwise in quadrature; on-axis light
  goes to the bright output only, and the kernels (differences of the paired outputs) are free of
  second-order terms in the piston errors and odd under a point reflection of the source position.
"""

import math

import numpy as np
import pytest
from astropy import units as u
from scipy.optimize import brentq

from lumos.core.element import Element
from lumos.core.group import ElementGroup
from lumos.elements.photonics import (
    KERNEL_PAIRS,
    MMI,
    Crossing,
    DirectionalCoupler,
    PhaseShifter,
    TransferMatrix,
    Waveguide,
    YJunction,
    bracewell,
    kernel_nuller,
    kernels,
)
from lumos.fields import Bus, ModalField
from lumos.physics.photonics import slab_modes

LAM = 1.55e-6


def modal(amplitudes, wavelengths=(LAM,)):
    """A guided bus of single-mode lanes, one per column of ``amplitudes`` (ADR-0023)."""
    a = np.asarray(amplitudes, dtype=complex)
    if a.ndim == 1:
        a = np.tile(a, (len(wavelengths), 1))
    return Bus.guided(a, np.asarray(wavelengths) * u.m)


def run(element, field):
    """Amplitudes of all the modes of all the output lanes, ``(nλ, Σ modes)``."""
    return np.asarray(element.apply(**{"in": field})["out"].modal.amplitudes)


# --- Components -----------------------------------------------------------------------------------------


def test_waveguide_phase_and_loss_on_selected_channels():
    guide = Waveguide(length=2 * u.cm, effective_index=1.5, loss=0.5 * u.dB / u.cm, channels=[1])
    lams = (1.5e-6, 1.6e-6)
    out = run(guide, modal([1.0, 1.0, 1.0], lams))
    for k, lam in enumerate(lams):
        expected = 10 ** (-1.0 / 20) * np.exp(2j * math.pi * 1.5 * 0.02 / lam)
        assert out[k, 1] == pytest.approx(expected, rel=1e-9)
        assert out[k, 0] == out[k, 2] == 1.0


def test_thermo_optic_phase_scales_with_the_inverse_wavelength():
    shifter = PhaseShifter(phase=1.2 * u.rad, reference_wavelength=1.55 * u.um, channels=[0])
    lams = (1.55e-6, 2.0e-6)
    out = run(shifter, modal([1.0, 1.0], lams))
    np.testing.assert_allclose(out[:, 0], np.exp(1j * 1.2 * 1.55e-6 / np.array(lams)), rtol=1e-12)
    np.testing.assert_allclose(out[:, 1], 1.0)


@pytest.mark.parametrize("ratio", [0.0, 0.3, 0.5, 0.9])
def test_directional_coupler_splits_and_conserves_power(ratio):
    coupler = DirectionalCoupler(ratio=ratio, channels=[0, 2])
    out = run(coupler, modal([1.0, 0.5, 0.0]))[0]
    assert abs(out[0]) ** 2 == pytest.approx(1 - ratio, abs=1e-12)
    assert abs(out[2]) ** 2 == pytest.approx(ratio, abs=1e-12)
    assert out[1] == 0.5
    rng = np.random.default_rng(1)
    a = rng.normal(size=3) + 1j * rng.normal(size=3)
    assert np.sum(np.abs(run(coupler, modal(a))) ** 2) == pytest.approx(np.sum(np.abs(a) ** 2), rel=1e-12)


def test_y_junction_split_inserts_a_channel():
    out = run(YJunction(mode="split", channels=[1]), modal([1.0, 2.0, 3.0]))[0]
    np.testing.assert_allclose(out, [1.0, 2 / math.sqrt(2), 2 / math.sqrt(2), 3.0])


@pytest.mark.parametrize(("b", "power"), [(1.0, 2.0), (-1.0, 0.0), (1j, 1.0)])
def test_y_junction_combines_symmetric_light(b, power):
    out = run(YJunction(mode="combine", channels=[0, 1]), modal([1.0, b, 5.0]))[0]
    assert len(out) == 2
    assert abs(out[0]) ** 2 == pytest.approx(power, abs=1e-12)
    assert out[1] == 5.0


def test_crossing_crosstalk():
    out = run(Crossing(crosstalk=0.01, channels=[0, 1]), modal([1.0, 0.0]))[0]
    assert abs(out[1]) ** 2 == pytest.approx(0.99, abs=1e-12)
    assert abs(out[0]) ** 2 == pytest.approx(0.01, abs=1e-12)


# --- Mode solver and MMI -------------------------------------------------------------------------------


def analytic_slab_indices(width, n_core, n_clad, lam):
    k = 2 * math.pi / lam
    v = k * width / 2 * math.sqrt(n_core**2 - n_clad**2)
    roots = []
    for j in range(100):
        low, high = j * math.pi / 2 + 1e-12, min((j + 1) * math.pi / 2, v) - 1e-12
        if low >= v:
            break

        def f(x, j=j):
            w = math.sqrt(v * v - x * x)
            return x * math.tan(x) - w if j % 2 == 0 else -x / math.tan(x) - w

        if f(low) * f(high) < 0:
            roots.append(brentq(f, low, high, xtol=1e-15))
    return np.sqrt(n_core**2 - (np.array(roots) / (width / 2) / k) ** 2)


def test_slab_mode_solver_matches_the_analytic_modes():
    x = np.linspace(-10e-6, 10e-6, 4001)
    index = np.where(np.abs(x) <= 2e-6, 1.6, 1.45)
    modes = slab_modes(x, index, LAM, n_cladding=1.45)
    expected = analytic_slab_indices(4e-6, 1.6, 1.45, LAM)
    assert len(modes.beta) == len(expected)
    np.testing.assert_allclose(modes.beta / (2 * math.pi / LAM), expected, rtol=1e-5)
    np.testing.assert_allclose(np.sum(modes.fields**2, axis=1) * modes.dx, 1.0, rtol=1e-12)


def mmi(n, width, port_width):
    return MMI(
        n_ports=n,
        width=width * u.um,
        port_width=port_width * u.um,
        n_core=1.6,
        n_cladding=1.45,
        reference_wavelength=LAM * u.m,
        channels=list(range(n)),
    )


def test_two_by_two_mmi_is_a_quadrature_three_db_coupler():
    s = np.column_stack([run(mmi(2, 10, 3), modal(np.eye(2)[j]))[0] for j in range(2)])
    power = np.abs(s) ** 2
    np.testing.assert_allclose(power, 0.5, atol=0.01)
    assert np.all(1 - power.sum(axis=0) < 0.02)
    assert abs(np.angle(s[1, 0] / s[0, 0])) == pytest.approx(math.pi / 2, abs=0.02)


def test_four_by_four_mmi_divides_the_power_in_four():
    s = np.column_stack([run(mmi(4, 16, 3.5), modal(np.eye(4)[j]))[0] for j in range(4)])
    power = np.abs(s) ** 2
    np.testing.assert_allclose(power, 0.25, atol=0.0125)
    assert np.all(1 - power.sum(axis=0) < 0.05)


# --- Chips ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dphi", [0.0, 0.3, 1.0, math.pi / 2, math.pi])
def test_bracewell_outputs(dphi):
    chip = bracewell()
    assert isinstance(chip, ElementGroup)
    p = 2.0
    out = run(chip, modal([math.sqrt(p), math.sqrt(p) * np.exp(1j * dphi)]))[0]
    assert abs(out[0]) ** 2 == pytest.approx(2 * p * math.sin(dphi / 2) ** 2, abs=1e-12)
    assert abs(out[1]) ** 2 == pytest.approx(2 * p * math.cos(dphi / 2) ** 2, abs=1e-12)


def kernel_outputs(phases):
    return np.abs(run(kernel_nuller(), modal(np.exp(1j * np.asarray(phases))))[0]) ** 2


def test_kernel_nuller_sends_on_axis_light_to_the_bright_output():
    power = kernel_outputs(np.zeros(4))
    assert power.shape == (7,)
    assert power[0] == pytest.approx(4.0, rel=1e-12)
    np.testing.assert_allclose(power[1:], 0.0, atol=1e-24)


def test_kernel_nuller_is_lossless():
    rng = np.random.default_rng(3)
    a = rng.normal(size=4) + 1j * rng.normal(size=4)
    out = run(kernel_nuller(), modal(a))[0]
    assert np.sum(np.abs(out) ** 2) == pytest.approx(np.sum(np.abs(a) ** 2), rel=1e-12)


def test_kernels_are_free_of_second_order_piston_errors():
    rng = np.random.default_rng(4)
    for _ in range(5):
        pistons = rng.normal(scale=1e-3, size=4)
        power = kernel_outputs(pistons)
        assert power[1:].max() > 1e-8  # dark outputs see the pistons to second order
        assert np.max(np.abs(kernels(power))) < 1e-2 * power[1:].max()


def test_kernels_are_odd_under_point_reflection():
    phases = np.array([0.0, 0.4, -0.7, 1.1])
    k_plus = kernels(kernel_outputs(phases))
    k_minus = kernels(kernel_outputs(-phases))
    assert np.max(np.abs(k_plus)) > 1e-2
    np.testing.assert_allclose(k_plus, -k_minus, atol=1e-12)
    assert len(KERNEL_PAIRS) == 3


def test_kernel_nuller_chip_survives_a_lumos_round_trip():
    chip = kernel_nuller()
    loaded = Element.from_lumos(chip.to_lumos())
    a = modal(np.exp(1j * np.array([0.1, 0.2, -0.3, 0.05])))
    np.testing.assert_allclose(run(loaded, a), run(chip, a), atol=1e-14)


# --- Lanes of a bus (step P15.6, ADR-0023) ----------------------------------------------------------------
# A bus groups guides separated in space; ``channels`` lists lanes. A mixing component applies its matrix
# to all the modes of its selected lanes, and only lanes describing the same sources can interfere
# (docs/specs/pistes_evolution.md §1.2).


def test_channels_designate_lanes_of_a_bus():
    bus = Bus([ModalField(np.ones((1, 2)), [LAM] * u.m), ModalField(np.ones((1, 1)), [LAM] * u.m)])
    out = Waveguide(length=1 * u.mm, effective_index=1.5, channels=[1]).apply(**{"in": bus})["out"]
    assert out.mode_counts == (2, 1)
    np.testing.assert_allclose(np.asarray(out[0].amplitudes), 1.0)
    phase = np.exp(2j * math.pi * 1.5 * 1e-3 / LAM)
    np.testing.assert_allclose(np.asarray(out[1].amplitudes), phase, rtol=1e-12)


def test_a_mixing_component_acts_on_every_mode_of_its_lanes():
    rng = np.random.default_rng(5)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    element = TransferMatrix(amplitudes=np.abs(q), phases=np.angle(q) * u.rad)
    a = rng.normal(size=3) + 1j * rng.normal(size=3)
    bus = Bus([ModalField(a[None, :2], [LAM] * u.m), ModalField(a[None, 2:], [LAM] * u.m)])
    out = element.apply(**{"in": bus})["out"]
    assert out.mode_counts == (2, 1)
    np.testing.assert_allclose(np.asarray(out.modal.amplitudes)[0], q @ a, atol=1e-12)


def test_lanes_of_different_sources_do_not_interfere():
    lanes = [ModalField(np.ones((1, 1)), [LAM] * u.m, sources=name) for name in ("star", "other star")]
    with pytest.raises(ValueError, match="sources"):
        DirectionalCoupler(ratio=0.5).apply(**{"in": Bus(lanes)})
    # a diagonal component acts lane by lane and accepts them
    out = PhaseShifter(phase=1 * u.rad, achromatic=True).apply(**{"in": Bus(lanes)})["out"]
    np.testing.assert_allclose([complex(np.asarray(s.amplitudes)[0, 0]) for s in out], np.exp(1j))


def test_a_splitter_inserts_a_named_lane():
    bus = Bus.guided(np.array([[1.0, 2.0, 3.0]]), [LAM] * u.m, names=["a", "b", "c"])
    out = YJunction(mode="split", channels=[1]).apply(**{"in": bus})["out"]
    assert out.names == ("a", "b.0", "b.1", "c")
    assert out.mode_counts == (1, 1, 1, 1)


def test_a_single_lane_is_a_bus_of_one_lane():
    out = Waveguide(length=0 * u.mm).apply(**{"in": ModalField(np.ones((1, 1)), [LAM] * u.m)})["out"]
    assert isinstance(out, Bus)
    assert len(out) == 1
