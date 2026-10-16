"""Tests for the turbulent atmosphere, adaptive optics, deformable mirrors and phase plates (step P6.5).

Requirements: §5.9 (multi-layer von Kármán atmosphere with frozen flow, time-dependent and chromatic;
deformable mirror with Zernike modes and actuators; phase plates), REQ-EL-8 (transfers on every coherent
signal), REQ-CTX-6 (reproducible random streams).

Independent references:

- von Kármán phase power spectral density ``Φ(f) = 0.023 r0^(-5/3) (f² + 1/L0²)^(-11/6)`` (Noll 1976, JOSA 66,
  207, eq. 18 for L0 → ∞; Roddier 1981, Progress in Optics 19, §3.5.2);
- residual phase variance of Kolmogorov turbulence over a pupil of diameter D after removing the first J
  Zernike modes, ``Δ_J (D/r0)^(5/3)`` rad², with Δ_3 = 0.134 and Δ_10 = 0.0401 (Noll 1976, Table IV);
- layers of fractional Cn² weights w_i combine as ``r0^(-5/3) = Σ r0_i^(-5/3)`` with ``r0_i = r0 w_i^(-3/5)``
  (Roddier 1981);
- frozen flow (Taylor's hypothesis): ``φ(x, t) = φ(x - v t, 0)``;
- the optical path difference of dry-air turbulence is achromatic, so ``λ φ(λ)`` does not depend on λ
  (Roddier 1981, r0 ∝ λ^(6/5));
- a mirror deformed by a surface s changes the optical path by ``2 s`` (reflection);
- Gaussian influence functions with an inter-actuator coupling c are worth c at the neighbouring actuators.
"""

import math

import numpy as np
import pytest
from astropy import units as u

from lumos import Context
from lumos.elements.atmosphere import Atmosphere, DeformableMirror
from lumos.elements.optics import PhasePlate
from lumos.fields import Grid, Wavecube, Wavefront
from lumos.physics.turbulence import von_karman_psd
from lumos.physics.zernike import zernike_basis

D = 1.0
N = 64
GRID = Grid(N, dx=D / N * u.m)


def pupil_mask(grid=GRID, diameter=D):
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    return np.hypot(xx, yy) <= diameter / 2


# --- Power spectrum --------------------------------------------------------------------------------


def test_von_karman_psd():
    f = np.array([0.1, 1.0, 10.0]) / u.m
    r0, l0 = 0.1, 20.0
    expected = 0.023 * r0 ** (-5 / 3) * (f.to_value(1 / u.m) ** 2 + 1 / l0**2) ** (-11 / 6)
    psd = von_karman_psd(f, r0=r0 * u.m, outer_scale=l0 * u.m)
    np.testing.assert_allclose(psd.to_value(u.m**2), expected, rtol=1e-12)
    kolmogorov = von_karman_psd(f, r0=r0 * u.m, outer_scale=np.inf * u.m)
    np.testing.assert_allclose(kolmogorov.to_value(u.m**2), 0.023 * r0 ** (-5 / 3) * f.value ** (-11 / 3))


# --- Statistics of the screens ------------------------------------------------------------------------


def residual_variance(corrected_modes, n_screens=200):
    """Mean residual phase variance (rad² at 500 nm) over the pupil, D/r0 = 4, two layers."""
    mask = pupil_mask()
    variances = []
    for seed in range(n_screens):
        ctx = Context(seed=seed)
        atm = ctx.add(
            Atmosphere(
                name="atm",
                r0=0.25 * u.m,
                reference_wavelength=500 * u.nm,
                outer_scale=1e5 * u.m,
                weights=[0.6, 0.4],
                wind_speed=[10, 20] * u.m / u.s,
                wind_direction=[0, 90] * u.deg,
                corrected_modes=corrected_modes,
                pupil_diameter=D * u.m,
            )
        )
        phase = 2 * math.pi * atm.opd(GRID).to_value(u.m) / 500e-9
        values = phase[mask]
        variances.append(np.mean((values - values.mean()) ** 2))
    return float(np.mean(variances))


@pytest.mark.parametrize(("modes", "delta"), [(3, 0.134), (10, 0.0401)])
def test_residual_variance_after_correction_follows_noll(modes, delta):
    # 200 independent screens: the statistical error on the mean is about 3 %; 10 % is ~3 sigma
    assert residual_variance(modes) == pytest.approx(delta * 4 ** (5 / 3), rel=0.10)


def test_correction_removes_the_projection_on_the_corrected_modes():
    ctx = Context(seed=5)
    atm = ctx.add(Atmosphere(name="atm", r0=0.2 * u.m, corrected_modes=6, pupil_diameter=D * u.m))
    opd = atm.opd(GRID).to_value(u.m)
    mask = pupil_mask()
    basis = zernike_basis(GRID, D * u.m, 6)[:, mask]
    coefficients, *_ = np.linalg.lstsq(basis.T, opd[mask], rcond=None)
    assert np.max(np.abs(coefficients)) < 1e-9 * np.std(opd[mask])


def test_same_seed_same_screen_other_seed_other_screen():
    def screen(seed):
        ctx = Context(seed=seed)
        return ctx.add(Atmosphere(name="atm", r0=0.2 * u.m)).opd(GRID).to_value(u.m)

    np.testing.assert_array_equal(screen(1), screen(1))
    assert not np.allclose(screen(1), screen(2))


# --- Time and wavelength ------------------------------------------------------------------------------


@pytest.mark.parametrize(("direction", "axis"), [(0, 1), (90, 0)])
def test_frozen_flow(direction, axis):
    ctx = Context(seed=3)
    speed = 10.0
    atm = ctx.add(
        Atmosphere(
            name="atm", r0=0.2 * u.m, wind_speed=[speed] * u.m / u.s, wind_direction=[direction] * u.deg
        )
    )
    before = atm.opd(GRID).to_value(u.m)
    ctx.wait(3 * GRID.dx_si / speed * u.s)
    after = atm.opd(GRID).to_value(u.m)
    shifted = np.roll(before, 3, axis=axis)
    region = (slice(None), slice(3, None)) if axis == 1 else (slice(3, None), slice(None))
    np.testing.assert_allclose(after[region], shifted[region], atol=1e-6 * np.std(before))


def test_optical_path_is_achromatic():
    ctx = Context(seed=8)
    atm = ctx.add(Atmosphere(name="atm", r0=5 * u.m, pupil_diameter=D * u.m))
    cube = Wavecube.plane(GRID, [1.0, 2.0] * u.um, power=1 * u.W)
    out = atm.apply(**{"in": cube})["out"]
    ratio = np.asarray(out.field) / np.asarray(cube.field)
    np.testing.assert_allclose(np.abs(ratio), 1.0, atol=1e-12)
    opd = atm.opd(GRID).to_value(u.m)
    for k, lam in enumerate([1e-6, 2e-6]):
        np.testing.assert_allclose(np.angle(ratio[k]) * lam / (2 * math.pi), opd, atol=1e-15)


# --- Deformable mirror ----------------------------------------------------------------------------------


def dm_grid():
    pitch = 1 / 7
    return Grid(64, dx=pitch / 4 * u.m)


def test_poked_actuator_and_coupling():
    dm = DeformableMirror(n_actuators=8, diameter=1 * u.m, coupling=0.15)
    commands = np.zeros(64)
    k = 3 * 8 + 4
    commands[k] = 50.0
    dm.commands = commands * u.nm
    grid = dm_grid()
    opd = dm.opd(grid).to_value(u.nm)
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    positions = dm.actuator_positions().to_value(u.m)
    pitch = 1 / 7

    def at(x, y):
        index = np.unravel_index(np.argmin(np.hypot(xx - x, yy - y)), xx.shape)
        assert math.hypot(xx[index] - x, yy[index] - y) < 1e-12
        return opd[index]

    x0, y0 = positions[k]
    assert at(x0, y0) == pytest.approx(100.0, rel=1e-12)  # 2 × 50 nm
    assert at(x0 + pitch, y0) == pytest.approx(100.0 * 0.15, rel=1e-12)
    assert at(x0, y0 - pitch) == pytest.approx(100.0 * 0.15, rel=1e-12)


def test_fit_recovers_the_commands():
    rng = np.random.default_rng(0)
    dm = DeformableMirror(n_actuators=6, diameter=1 * u.m, coupling=0.2)
    commands = rng.normal(size=36) * 30 * u.nm
    dm.commands = commands
    grid = Grid(48, dx=1.2 / 48 * u.m)
    fitted = DeformableMirror(n_actuators=6, diameter=1 * u.m, coupling=0.2).fit(dm.opd(grid), grid)
    np.testing.assert_allclose(fitted.to_value(u.nm), commands.to_value(u.nm), rtol=1e-6, atol=1e-6)


def test_mirror_applies_its_optical_path():
    dm = DeformableMirror(n_actuators=4, diameter=1 * u.m)
    dm.commands = np.linspace(-40, 40, 16) * u.nm
    grid = Grid(32, dx=1.2 / 32 * u.m)
    wf = Wavefront.plane(grid, 1 * u.um, power=1 * u.W)
    out = dm.apply(**{"in": wf})["out"]
    expected = np.exp(2j * math.pi * dm.opd(grid).to_value(u.m) / 1e-6)
    np.testing.assert_allclose(np.asarray(out.field) / np.asarray(wf.field), expected, atol=1e-12)


# --- Phase plate ----------------------------------------------------------------------------------------


def test_zernike_phase_plate():
    plate = PhasePlate(diameter=1 * u.m, coefficients=[0, 0, 0, 100] * u.nm)
    grid = Grid(65, dx=1 / 64 * u.m)
    opd = plate.opd(grid).to_value(u.nm)
    center = (32, 32)
    assert opd[center] == pytest.approx(-math.sqrt(3) * 100, rel=1e-12)
    assert opd[32, 64] == pytest.approx(math.sqrt(3) * 100, rel=1e-12)  # ρ = 1 at x = 0.5 m
    wf = Wavefront.plane(grid, 500 * u.nm, power=1 * u.W)
    out = plate.apply(**{"in": wf})["out"]
    phase = np.angle(np.asarray(out.field)[center] / np.asarray(wf.field)[center])
    assert phase == pytest.approx(
        math.remainder(2 * math.pi * -math.sqrt(3) * 100 / 500, 2 * math.pi), abs=1e-12
    )
