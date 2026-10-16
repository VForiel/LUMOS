"""Tests for the propagation planner (step P4.6).

Requirements: REQ-PROP-4 (automatic choice of the method from sampling criteria and estimated cost, logged
and overridable) and the acceptance criterion of phase 4: the planner chooses a valid method in 100 % of the
cases.

Validity is judged against independent analytic solutions, not against the planner's own criteria:

- paraxial Gaussian beam (Siegman, *Lasers*, §17.1) with non-paraxial corrections ``(k w0)^-2 < 1e-3``;
- exact first Rayleigh-Sommerfeld on-axis field of a circular aperture,
  ``U0 [e^{ikz} - z / sqrt(z² + a²) e^{ik sqrt(z² + a²)}]`` (Goodman, problem 3-12).
"""

import logging
import math

import numpy as np
from astropy import units as u
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lumos import backend
from lumos.fields import Grid, Wavefront
from lumos.propagation import methods, plan, propagate

LAM = 1e-6
K = 2 * math.pi / LAM


def gaussian(grid, w0, z):
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    zr = math.pi * w0**2 / LAM
    r2 = xx**2 + yy**2
    w = w0 * math.sqrt(1 + (z / zr) ** 2)
    curvature = 0.0 if z == 0 else 1 / (z * (1 + (zr / z) ** 2))
    return (
        math.sqrt(2 / math.pi)
        / w
        * np.exp(-r2 / w**2)
        * np.exp(1j * (K * z + K * r2 * curvature / 2 - math.atan2(z, zr)))
    )


# --- Plan objects and logging --------------------------------------------------------------------


def test_plan_describes_the_choice():
    grid = Grid(128, dx=2e-6 * u.m)
    wf = Wavefront(gaussian(grid, 20e-6, 0.0), grid, LAM * u.m)
    p = plan(wf, 1 * u.mm)
    assert p.method in methods()
    assert isinstance(p.reason, str)
    assert p.reason
    assert p.cost > 0
    assert p.output_grid == grid


def test_far_field_uses_fraunhofer():
    grid = Grid(64, dx=1e-6 * u.m)
    wf = Wavefront(np.ones(grid.shape), grid, LAM * u.m)
    assert plan(wf, None, output_grid=Grid(32, dx=1e-3 * u.rad)).method == "fraunhofer"


def test_back_propagation_uses_the_angular_spectrum():
    grid = Grid(64, dx=2e-6 * u.m)
    wf = Wavefront(gaussian(grid, 15e-6, 0.0), grid, LAM * u.m)
    assert plan(wf, -0.5 * u.mm).method == "asm"


def test_the_choice_is_logged(caplog):
    grid = Grid(64, dx=2e-6 * u.m)
    wf = Wavefront(gaussian(grid, 15e-6, 0.0), grid, LAM * u.m)
    with caplog.at_level(logging.INFO, logger="lumos.propagation"):
        propagate(wf, 0.3 * u.mm)
    chosen = plan(wf, 0.3 * u.mm).method
    assert any(chosen in record.getMessage() for record in caplog.records)


def test_explicit_method_overrides_the_planner(caplog):
    grid = Grid(32, dx=2e-6 * u.m)
    wf = Wavefront(gaussian(grid, 10e-6, 0.0), grid, LAM * u.m)
    with caplog.at_level(logging.INFO, logger="lumos.propagation"):
        propagate(wf, 0.1 * u.mm, method="rs")
    assert not any("planner" in record.getMessage() for record in caplog.records)


def test_expensive_direct_sums_are_avoided_when_another_method_is_valid():
    grid = Grid(1024, dx=1e-6 * u.m)
    wf = Wavefront(gaussian(grid, 30e-6, 0.0), grid, LAM * u.m)
    assert plan(wf, 2 * u.mm).method != "rs"


def test_a_valid_method_is_reported_for_a_point_on_axis():
    grid = Grid(256, dx=1e-6 * u.m)
    wf = Wavefront(np.ones(grid.shape), grid, LAM * u.m)
    p = plan(wf, 1 * u.mm, output_grid=Grid(1, dx=1e-6 * u.m))
    assert methods()[p.method].any_output_grid


# --- 100 % valid choices on analytic cases ------------------------------------------------------


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.sampled_from([64, 96, 128]),
    samples_per_waist=st.floats(4.0, 12.0),
    z_over_zr=st.floats(0.05, 3.0),
    other_grid=st.booleans(),
    shift=st.floats(-1.0, 1.0),
    scale=st.floats(0.5, 2.0),
)
def test_gaussian_beams_are_always_propagated_accurately(
    n, samples_per_waist, z_over_zr, other_grid, shift, scale
):
    dx = 2e-6
    w0 = samples_per_waist * dx
    zr = math.pi * w0**2 / LAM
    z = z_over_zr * zr
    w = w0 * math.sqrt(1 + z_over_zr**2)
    grid = Grid(n, dx=dx * u.m)
    if 3 * w > n * dx / 2:  # the beam must fit in the window at the output plane
        return
    wf = Wavefront(gaussian(grid, w0, 0.0), grid, LAM * u.m)
    if other_grid:
        out_grid = Grid(24, 18, dx=scale * w / 6 * u.m, x0=shift * w * u.m, y0=-0.5 * shift * w * u.m)
    else:
        out_grid = grid
    out = propagate(wf, z * u.m, output_grid=out_grid)
    expected = gaussian(out_grid, w0, z)
    np.testing.assert_allclose(backend.to_numpy(out.field), expected, atol=2e-3 * math.sqrt(2 / math.pi) / w)


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(fresnel_number=st.floats(0.3, 6.0))
def test_aperture_on_axis_is_always_accurate(fresnel_number):
    grid = Grid(384, dx=1e-6 * u.m)
    a = 40e-6
    z = a**2 / (LAM * fresnel_number)
    x, y = (np.asarray(c) for c in grid.axes(backend="numpy"))
    offsets = ((np.arange(16) + 0.5) / 16 - 0.5) * 1e-6
    coverage = sum(
        ((x[None, :] + ox) ** 2 + (y[:, None] + oy) ** 2) <= a**2 for oy in offsets for ox in offsets
    )
    wf = Wavefront(coverage / 256.0, grid, LAM * u.m)
    out = propagate(wf, z * u.m, output_grid=Grid(1, dx=1e-6 * u.m))
    r = math.sqrt(z**2 + a**2)
    expected = np.exp(1j * K * z) - z / r * np.exp(1j * K * r)
    # The pixel-coverage map is the disk convolved with a pixel, so its diffraction is the exact one times
    # sinc(f dx): edge waves reach the axis at f = a / (λ z), attenuated by ≈ (π a dx / (λ z))² / 6. The rest
    # (edge digitisation, quadrature) stays below 1 % of U0.
    pixel_attenuation = (math.pi * a * 1e-6 / (LAM * z)) ** 2 / 6
    assert abs(backend.to_numpy(out.field)[0, 0] - expected) < 0.01 + pixel_attenuation
