"""Tests for closed-form symbolic propagation (step P10.4, REQ-SYM-4, REQ-SYM-5).

Requirements: Gaussian, Hermite-Gauss and Laguerre-Gauss beams through ABCD systems; tabulated transforms
(circ → Airy, rect → sinc) at the focus of a lens; explicit, logged numerical fallback otherwise.

Independent references (computed in the tests from the textbook formulas):

- Gaussian beams (Siegman, *Lasers*, ch. 17 and 20; Kogelnik & Li 1966, Appl. Opt. 5, 1550): complex beam
  parameter ``1/q = 1/R + iλ/(πw²)`` (``e^{+ikz}`` convention), ``q → (Aq + B)/(Cq + D)``, power ``P``,
  intensity ``2P/(πw²) exp(-2r²/w²)``, phase curvature ``k r²/(2R)``; Gouy phase ``atan(z/z_R)`` per mode
  order ``n + m + 1``;
- LG₀¹ intensity ∝ (r/w)² exp(-2r²/w²), maximal at ``r = w/√2``;
- focal plane of a lens of focal length f behind a uniform pupil of amplitude A (Born & Wolf §8.5,
  Goodman §5.2): circular pupil of diameter D: ``I = I₀ [2J₁(v)/v]²`` with ``v = πDr/(λf)`` and
  ``I₀ = A² (πD²/4)² / (λf)²``; rectangle a × b: ``I = A²(ab)²/(λf)² sinc²(ax/λf) sinc²(by/λf)``.
"""

import cmath
import logging
import math

import numpy as np
import pytest
import sympy as sp
from astropy import units as u
from scipy.special import j1

from lumos.elements.apertures import CircularAperture
from lumos.elements.optics import FreeSpace, ThinLens
from lumos.fields import Grid, Wavefront
from lumos.symbolic import SymbolicField, x, y
from lumos.symbolic.closed_forms import GaussianBeamForm, gaussian_beam, plane_wave, rectangular_pupil
from lumos.symbolic.evaluate import to_wavefront

LAM = 1.0e-6
W0 = 0.5e-3
ZR = math.pi * W0**2 / LAM
P = 2e-3


def evaluate(field, points):
    """Values of the field at ``points`` (list of (x, y) in metres)."""
    expression = field.substituted()
    function = sp.lambdify((x, y), expression.subs(sp.Symbol("lambda", positive=True), LAM), "scipy")
    return np.array([complex(function(px, py)) for px, py in points])


def abcd(q, a, b, c, d):
    return (a * q + b) / (c * q + d)


def beam_parameters(q):
    inverse = 1 / q
    radius = math.inf if inverse.real == 0 else 1 / inverse.real
    width = math.sqrt(LAM / (math.pi * inverse.imag))
    return width, radius


POINTS = [(0.0, 0.0), (0.2e-3, 0.0), (0.0, 0.5e-3), (0.4e-3, 0.3e-3), (0.9e-3, -0.6e-3)]


# --- Gaussian beams -----------------------------------------------------------------------------------------


def test_gaussian_beam_at_its_waist():
    field = gaussian_beam(waist=W0 * u.m, wavelength=LAM * u.m, power=P * u.W)
    assert isinstance(field.form, GaussianBeamForm)
    grid = Grid(256, dx=10 * u.um)
    assert to_wavefront(field, grid).power.to_value(u.W) == pytest.approx(P, rel=1e-9)
    values = evaluate(field, POINTS)
    r2 = np.array([px**2 + py**2 for px, py in POINTS])
    np.testing.assert_allclose(
        np.abs(values) ** 2, 2 * P / (math.pi * W0**2) * np.exp(-2 * r2 / W0**2), rtol=1e-10
    )


def test_free_space_propagation_of_a_gaussian_beam():
    d = 0.7
    field = gaussian_beam(waist=W0 * u.m, wavelength=LAM * u.m, power=P * u.W)
    out = FreeSpace(distance=d * u.m).apply_symbolic(**{"in": field})["out"]
    assert isinstance(out, SymbolicField)
    width, radius = beam_parameters(abcd(complex(0, -ZR), 1, d, 0, 1))
    assert width == pytest.approx(W0 * math.sqrt(1 + (d / ZR) ** 2), rel=1e-12)  # Siegman eq. 17.4
    values = evaluate(out, POINTS)
    r2 = np.array([px**2 + py**2 for px, py in POINTS])
    np.testing.assert_allclose(
        np.abs(values) ** 2, 2 * P / (math.pi * width**2) * np.exp(-2 * r2 / width**2), rtol=1e-6
    )
    k = 2 * math.pi / LAM
    phase = np.angle(values[1] / values[0])
    expected = (k * POINTS[1][0] ** 2 / (2 * radius) + math.pi) % (2 * math.pi) - math.pi
    assert phase == pytest.approx(expected, abs=1e-6)


def test_gaussian_beam_through_a_lens_and_free_space():
    f, d = 0.25, 0.4
    field = gaussian_beam(waist=W0 * u.m, wavelength=LAM * u.m, power=P * u.W)
    after_lens = ThinLens(focal_length=f * u.m).apply_symbolic(**{"in": field})["out"]
    out = FreeSpace(distance=d * u.m).apply_symbolic(**{"in": after_lens})["out"]
    q = abcd(abcd(complex(0, -ZR), 1, 0, -1 / f, 1), 1, d, 0, 1)
    width, radius = beam_parameters(q)
    values = evaluate(out, POINTS)
    r2 = np.array([px**2 + py**2 for px, py in POINTS])
    np.testing.assert_allclose(
        np.abs(values) ** 2, 2 * P / (math.pi * width**2) * np.exp(-2 * r2 / width**2), rtol=1e-6
    )
    k = 2 * math.pi / LAM
    sample = (0.05e-3, 0.0)
    phase = cmath.phase(evaluate(out, [sample])[0] / values[0])
    expected = (k * sample[0] ** 2 / (2 * radius) + math.pi) % (2 * math.pi) - math.pi
    assert phase == pytest.approx(expected, abs=1e-6)


def test_hermite_gauss_gouy_phase():
    d = 0.9
    fundamental = FreeSpace(distance=d * u.m).apply_symbolic(
        **{"in": gaussian_beam(waist=W0 * u.m, wavelength=LAM * u.m, power=P * u.W)}
    )["out"]
    hg10 = FreeSpace(distance=d * u.m).apply_symbolic(
        **{"in": gaussian_beam(waist=W0 * u.m, wavelength=LAM * u.m, power=P * u.W, indices=(1, 0))}
    )["out"]
    point = (0.3e-3, 0.1e-3)
    ratio = evaluate(hg10, [point])[0] / evaluate(fundamental, [point])[0]
    assert cmath.phase(ratio) == pytest.approx(-math.atan(d / ZR), abs=1e-9)
    width = W0 * math.sqrt(1 + (d / ZR) ** 2)
    assert abs(ratio) == pytest.approx(2 * point[0] / width, rel=1e-9)  # H1(√2 x/w)/√2 = 2x/w


def test_laguerre_gauss_ring():
    field = gaussian_beam(waist=W0 * u.m, wavelength=LAM * u.m, power=P * u.W, family="lg", indices=(0, 1))
    radii = np.linspace(0.01, 2.0, 2000) * W0
    intensity = np.abs(evaluate(field, [(r, 0.0) for r in radii])) ** 2
    assert radii[np.argmax(intensity)] == pytest.approx(W0 / math.sqrt(2), rel=2e-3)
    assert to_wavefront(field, Grid(256, dx=10 * u.um)).power.to_value(u.W) == pytest.approx(P, rel=1e-6)


# --- Tabulated transforms ----------------------------------------------------------------------------------


F = 0.5
D = 2e-3
A = 3.0  # amplitude of the plane wave, sqrt(W)/m


def focal_field(pupil):
    lens = ThinLens(focal_length=F * u.m).apply_symbolic(**{"in": pupil})["out"]
    return FreeSpace(distance=F * u.m).apply_symbolic(**{"in": lens})["out"]


def test_circular_pupil_gives_an_airy_pattern():
    wave = plane_wave(amplitude=A * u.W**0.5 / u.m, wavelength=LAM * u.m)
    pupil = CircularAperture(diameter=D * u.m).apply_symbolic(**{"in": wave})["out"]
    focus = focal_field(pupil)
    assert isinstance(focus, SymbolicField)
    assert focus.expression.has(sp.besselj)
    radii = np.array([1e-7, 50e-6, 120e-6, 305e-6, 600e-6])
    intensity = np.abs(evaluate(focus, [(r, 0.0) for r in radii])) ** 2
    v = math.pi * D * radii / (LAM * F)
    i0 = A**2 * (math.pi * D**2 / 4) ** 2 / (LAM * F) ** 2
    np.testing.assert_allclose(intensity, i0 * (2 * j1(v) / v) ** 2, rtol=1e-6)


def test_rectangular_pupil_gives_a_sinc_pattern():
    a, b = 1e-3, 2e-3
    pupil = rectangular_pupil(
        width=a * u.m, height=b * u.m, amplitude=A * u.W**0.5 / u.m, wavelength=LAM * u.m
    )
    focus = focal_field(pupil)
    points = [(0.0, 0.0), (100e-6, 0.0), (0.0, 130e-6), (230e-6, -70e-6)]
    intensity = np.abs(evaluate(focus, points)) ** 2
    expected = [
        A**2
        * (a * b) ** 2
        / (LAM * F) ** 2
        * np.sinc(a * px / (LAM * F)) ** 2
        * np.sinc(b * py / (LAM * F)) ** 2
        for px, py in points
    ]
    np.testing.assert_allclose(intensity, expected, rtol=1e-6)


# --- Fallback -----------------------------------------------------------------------------------------------


def test_numerical_fallback_is_logged(caplog):
    a = sp.Symbol("a", positive=True)
    grid = Grid(64, dx=20 * u.um)
    field = SymbolicField(
        sp.exp(-sp.Abs(x) / a - sp.Abs(y) / a), parameters={a: 0.2 * u.mm}, wavelength=LAM * u.m, grid=grid
    )
    with caplog.at_level(logging.INFO, logger="lumos"):
        out = FreeSpace(distance=50 * u.mm).apply_symbolic(**{"in": field})["out"]
    assert isinstance(out, Wavefront)
    assert any("numerical" in record.getMessage() for record in caplog.records)
    numeric = FreeSpace(distance=50 * u.mm).apply(**{"in": to_wavefront(field, grid)})["out"]
    np.testing.assert_allclose(np.asarray(out.field), np.asarray(numeric.field), rtol=1e-12, atol=1e-15)


def test_fallback_needs_a_grid():
    a = sp.Symbol("a", positive=True)
    field = SymbolicField(sp.exp(-sp.Abs(x) / a), parameters={a: 0.2 * u.mm}, wavelength=LAM * u.m)
    with pytest.raises(ValueError, match="grid"):
        FreeSpace(distance=50 * u.mm).apply_symbolic(**{"in": field})
