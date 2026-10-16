"""Tests for guided optics: LP modes, injection and fibres (step P6.7).

Requirements: §5.9 (single-mode and multimode fibres with LP modes, injection and extraction), REQ-SIG-7
(modal amplitudes), REQ-EL-8 (transfers on every coherent signal).

Independent references:

- LP modes of a weakly guiding step-index fibre (Gloge 1971, Appl. Opt. 10, 2252): the characteristic equation
  ``U J_{l-1}(U) / J_l(U) = -W K_{l-1}(W) / K_l(W)`` with ``U² + W² = V²``, solved independently here with
  SciPy's ``brentq``; ``n_eff² = n_cl² + b (n_co² - n_cl²)`` with ``b = W² / V²``;
- cutoffs: LP_0m at the zeros ``j_{1, m-1}`` of J1 (LP01 has none), LP_lm (l ≥ 1) at the zeros ``j_{l-1, m}``
  of J_{l-1} (Gloge 1971), from ``scipy.special.jn_zeros``;
- LP01: ``b ≈ (1.1428 - 0.9960 / V)²`` within 0.2 % for 1.5 ≤ V ≤ 2.5 (Rudolph & Neumann 1976), and the
  Gaussian of best overlap has the waist ``w / a = 0.65 + 1.619 V^(-3/2) + 2.879 V^(-6)`` within about 1 %
  (Marcuse 1977, Bell Syst. Tech. J. 56, 703);
- coupling of Gaussian beams of waists w1 and w2 offset by d:
  ``(2 w1 w2 / (w1² + w2²))² exp(-2 d² / (w1² + w2²))`` (Joyce & DeLoach 1984, Appl. Opt. 23, 4187);
- the Airy pattern of an unobstructed pupil couples at most 81.45 % of its power into a Gaussian mode
  (Shaklan & Roddier 1988, Appl. Opt. 27, 2334); the Airy amplitude ``2 J1(x) / x``, ``x = π r / (λ F)``,
  carries the power ``4 (λ F)² / π``;
- a fibre of attenuation α (dB/km) and length L transmits ``10^(-α L / 10)`` of the power.
"""

import math

import numpy as np
import pytest
from astropy import units as u
from scipy.optimize import brentq
from scipy.special import j1, jn_zeros, jv, kv

from lumos.elements.fibers import Fiber
from lumos.fields import Grid, IncoherentBundle, ModalField, Wavecube, Wavefront
from lumos.physics.fibers import lp_modes, lp_radial

LAM = 1.55e-6


def v_number(a, na, lam=LAM):
    return 2 * math.pi * a / lam * na


def fiber_for_v(v, a=4e-6, **kw):
    return Fiber(core_radius=a * u.m, numerical_aperture=v * LAM / (2 * math.pi * a), **kw)


def independent_b(v, ell, m):
    """m-th root U of Gloge's characteristic equation for LP_lm, multiplied by J_l(U) K_l(W) (no poles)."""

    def g(x):
        w = math.sqrt(v * v - x * x)
        return x * jv(ell - 1, x) * kv(ell, w) + w * kv(ell - 1, w) * jv(ell, x)

    xs = np.linspace(1e-6, v - 1e-9, 4001)
    values = np.array([g(x) for x in xs])
    roots = [
        brentq(g, xs[i], xs[i + 1], xtol=1e-15)
        for i in range(len(xs) - 1)
        if np.sign(values[i]) != np.sign(values[i + 1])
    ]
    x = roots[m - 1]
    return 1 - x * x / (v * v)


# --- Modes --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("v", [1.5, 1.8, 2.0, 2.2, 2.4])
def test_fundamental_mode_follows_rudolph_neumann(v):
    (mode,) = lp_modes(v)
    assert (mode.ell, mode.m) == (0, 1)
    assert mode.b == pytest.approx((1.1428 - 0.9960 / v) ** 2, rel=2e-3)


@pytest.mark.parametrize(("v", "ell", "m"), [(2.2, 0, 1), (4.0, 1, 1), (4.0, 0, 2), (6.0, 2, 1), (6.0, 1, 2)])
def test_propagation_constants_solve_the_characteristic_equation(v, ell, m):
    modes = {(mode.ell, mode.m): mode for mode in lp_modes(v)}
    assert modes[(ell, m)].b == pytest.approx(independent_b(v, ell, m), rel=1e-9)


def cutoff(ell, m):
    if ell == 0:
        return 0.0 if m == 1 else float(jn_zeros(1, m - 1)[-1])
    return float(jn_zeros(ell - 1, m)[-1])


@pytest.mark.parametrize("v", [1.0, 2.3, 2.5, 3.5, 3.9, 5.0, 5.6, 7.5])
def test_guided_modes_are_those_above_cutoff(v):
    expected = {(ell, m) for ell in range(12) for m in range(1, 6) if v > cutoff(ell, m)}
    assert {(mode.ell, mode.m) for mode in lp_modes(v)} == expected


@pytest.mark.parametrize("v", [1.8, 2.0, 2.2, 2.4])
def test_best_gaussian_width_follows_marcuse(v):
    (mode,) = lp_modes(v)
    rho = np.linspace(0, 8, 80001)
    psi = lp_radial(mode, rho)
    widths = np.linspace(0.8, 2.5, 1701)

    def overlap(w):
        g = np.exp(-(rho**2) / w**2)
        return np.trapezoid(psi * g * rho, rho) ** 2 / np.trapezoid(g**2 * rho, rho)

    best = widths[int(np.argmax([overlap(w) for w in widths]))]
    assert best == pytest.approx(0.65 + 1.619 * v**-1.5 + 2.879 * v**-6, rel=1e-2)


def test_mode_fields_are_orthonormal_on_a_grid():
    fiber = fiber_for_v(6.0, a=5e-6)
    grid = Grid(240, dx=0.2 * u.um)
    fields = fiber.mode_fields(grid, LAM * u.m)
    assert len(fields) == len(fiber.mode_labels(LAM * u.m))
    flat = fields.reshape(len(fields), -1)
    gram = flat.conj() @ flat.T * grid.dx_si * grid.dy_si
    np.testing.assert_allclose(gram, np.eye(len(fields)), atol=1e-3)


def test_effective_index_lies_between_cladding_and_core():
    fiber = fiber_for_v(2.2, cladding_index=1.444)
    n_core = math.sqrt(1.444**2 + fiber.numerical_aperture.value**2)
    (mode,) = lp_modes(2.2)
    expected = math.sqrt(1.444**2 + mode.b * (n_core**2 - 1.444**2))
    assert fiber.effective_indices(LAM * u.m)[0] == pytest.approx(expected, rel=1e-12)
    assert 1.444 < expected < n_core


# --- Injection ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(("w_beam", "offset"), [(5e-6, 0.0), (3e-6, 0.0), (8e-6, 2e-6), (5e-6, 4e-6)])
def test_gaussian_beam_coupling(w_beam, offset):
    w_mode = 5e-6
    fiber = Fiber(mode_model="gaussian", mode_field_diameter=2 * w_mode * u.m, length=0 * u.m)
    grid = Grid(256, dx=0.25 * u.um, x0=-offset * u.m)
    beam = Wavefront.gaussian(grid, LAM * u.m, waist=w_beam * u.m, power=1 * u.W)
    modes = fiber.apply(**{"in": beam})["modes"]
    assert isinstance(modes, ModalField)
    s = w_beam**2 + w_mode**2
    expected = (2 * w_beam * w_mode / s) ** 2 * math.exp(-2 * offset**2 / s)
    assert modes.total_power.to_value(u.W) == pytest.approx(expected, rel=1e-6)


def test_airy_pattern_couples_at_most_81_percent():
    lam_f = 10e-6  # λ F
    grid = Grid(512, dx=lam_f / 16 * u.m)
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    x = math.pi * np.hypot(xx, yy) / lam_f
    with np.errstate(invalid="ignore", divide="ignore"):
        amplitude = np.where(x == 0, 1.0, 2 * j1(x) / x)
    airy = Wavefront(amplitude, grid, LAM * u.m)
    total = 4 * lam_f**2 / math.pi
    efficiencies = []
    for w in np.linspace(0.6, 0.8, 41) * lam_f * 2:
        fiber = Fiber(mode_model="gaussian", mode_field_diameter=w * u.m, length=0 * u.m)
        efficiencies.append(fiber.apply(**{"in": airy})["modes"].total_power.to_value(u.W) / total)
    assert max(efficiencies) == pytest.approx(0.8145, abs=3e-3)


# --- Propagation and extraction -------------------------------------------------------------------------


def test_attenuation():
    fiber = Fiber(
        mode_model="gaussian", mode_field_diameter=10 * u.um, length=2 * u.km, attenuation=3 * u.dB / u.km
    )
    grid = Grid(128, dx=0.5 * u.um)
    beam = Wavefront.gaussian(grid, LAM * u.m, waist=5 * u.um, power=1 * u.W)
    power = fiber.apply(**{"in": beam})["modes"].total_power.to_value(u.W)
    assert power == pytest.approx(10 ** (-0.6), rel=1e-6)


def test_exit_field_carries_the_guided_power_and_reinjects_identically():
    fiber = fiber_for_v(4.5, a=5e-6, length=1 * u.m)
    grid = Grid(200, dx=0.25 * u.um)
    beam = Wavefront.gaussian(grid, LAM * u.m, waist=4 * u.um, power=1 * u.W).tilt(0.02 * u.rad, 0.01 * u.rad)
    outputs = fiber.apply(**{"in": beam})
    modes, exit_field = outputs["modes"], outputs["out"]
    assert isinstance(exit_field, Wavefront)
    assert exit_field.power.to_value(u.W) == pytest.approx(modes.total_power.to_value(u.W), rel=1e-3)
    again = fiber_for_v(4.5, a=5e-6, length=0 * u.m).apply(**{"in": exit_field})["modes"]
    np.testing.assert_allclose(
        np.abs(np.asarray(again.amplitudes)), np.abs(np.asarray(modes.amplitudes)), atol=2e-3
    )


def test_relative_phase_follows_the_propagation_constants():
    fiber_a = fiber_for_v(4.5, a=5e-6, length=0 * u.m)
    fiber_b = fiber_for_v(4.5, a=5e-6, length=1 * u.mm)
    grid = Grid(200, dx=0.25 * u.um)
    beam = Wavefront.gaussian(grid, LAM * u.m, waist=4 * u.um, power=1 * u.W).tilt(0.02 * u.rad, 0 * u.rad)
    a = np.asarray(fiber_a.apply(**{"in": beam})["modes"].amplitudes)[0]
    b = np.asarray(fiber_b.apply(**{"in": beam})["modes"].amplitudes)[0]
    beta = 2 * math.pi / LAM * np.asarray(fiber_a.effective_indices(LAM * u.m))
    guided = np.abs(a) > 1e-3 * np.abs(a).max()
    np.testing.assert_allclose(np.exp(1j * beta[guided] * 1e-3), (b / a)[guided], atol=1e-9)


def test_modes_cut_off_at_long_wavelengths_carry_no_light():
    fiber = fiber_for_v(3.0, a=5e-6)  # V = 3.0 at 1.55 um: LP01 and LP11; single-mode at 2.2 um (V = 2.11)
    grid = Grid(200, dx=0.25 * u.um)
    cube = Wavecube(
        np.stack(
            [
                Wavefront.gaussian(grid, lam * u.um, waist=4 * u.um, power=1 * u.W)
                .tilt(0.03 * u.rad, 0 * u.rad)
                .field
                for lam in (1.55, 2.2)
            ]
        ),
        grid,
        [1.55, 2.2] * u.um,
    )
    modes = fiber.apply(**{"in": cube})["modes"]
    power = np.abs(np.asarray(modes.amplitudes)) ** 2
    assert power.shape == (2, 3)
    assert power[0, 1:].sum() > 1e-3
    np.testing.assert_array_equal(power[1, 1:], 0.0)


def test_bundle_modes_stay_incoherent():
    fiber = Fiber(mode_model="gaussian", mode_field_diameter=10 * u.um, length=0 * u.m)
    grid = Grid(128, dx=0.5 * u.um)
    bundle = IncoherentBundle.point_sources(
        grid, LAM * u.m, theta_x=[0, 0.05] * u.rad, theta_y=[0, 0] * u.rad, power=[1, 1] * u.W
    )
    modes = fiber.apply(**{"in": bundle})["modes"]
    assert np.asarray(modes.amplitudes).shape == (2, 1, 1)
