"""Tests for reconstructions from grids to expressions (step P10.5, REQ-SYM-3).

Requirements: modal projections (Zernike, Hermite-Gauss, Laguerre-Gauss, Fourier) controlled by the residual
error, fits of parametric models (Gaussian, complex q, Airy), sparse regression, optional PySR; every
reconstruction returns the expression and an error metric. Test of the plan: injected Zernike coefficients are
recovered.

Independent references:

- Zernike modes of Noll (1976), as implemented and validated in ``lumos.physics.zernike`` (step P6.5), used to
  inject known coefficients;
- Hermite-Gauss modes from ``scipy.special.eval_hermite`` with the textbook normalisation;
- a Gaussian beam of waist w, centre (x₀, y₀) and radius of curvature R: ``exp(-r²/w²) exp(ik r²/2R)``;
- the Airy pattern ``[2J₁(v)/v]²`` from ``scipy.special.j1``;
- a polynomial field with two terms, recovered term by term.
"""

import math

import numpy as np
import pytest
import sympy as sp
from astropy import units as u
from scipy.special import eval_hermite, j1

from lumos.fields import Grid, Wavefront
from lumos.physics.zernike import zernike_basis
from lumos.symbolic import SymbolicField, x, y
from lumos.symbolic.evaluate import to_wavefront
from lumos.symbolic.reconstruct import (
    Reconstruction,
    fit_airy,
    fit_gaussian,
    project_fourier,
    project_hermite_gauss,
    project_zernike,
    sparse_regression,
    symbolic_regression,
)

LAM = 1e-6


def test_injected_zernike_coefficients_are_recovered():
    grid = Grid(128, dx=1 / 120 * u.m)
    coefficients = np.array([0.0, 0.1, -0.05, 0.2, 0.0, 0.03, -0.07, 0.0, 0.0, 0.04])  # radians, Noll 1..10
    basis = zernike_basis(grid, 1 * u.m, len(coefficients))
    phase = np.tensordot(coefficients, basis, axes=1)
    pupil = (basis[0] != 0).astype(float)
    field = Wavefront(pupil * np.exp(1j * phase), grid, LAM * u.m)
    result = project_zernike(field, diameter=1 * u.m, n_modes=10)
    assert isinstance(result, Reconstruction)
    np.testing.assert_allclose(result.coefficients[1:], coefficients[1:], atol=1e-10)
    assert result.residual < 1e-10
    assert isinstance(result.field, SymbolicField)
    evaluated = np.angle(np.asarray(to_wavefront(result.field, grid).field)) * pupil
    np.testing.assert_allclose(evaluated, np.angle(np.exp(1j * phase)) * pupil, atol=1e-8)


def hg(n, m, w, xx, yy):
    norm = math.sqrt(2 / math.pi) / (w * math.sqrt(2 ** (n + m) * math.factorial(n) * math.factorial(m)))
    return (
        norm
        * eval_hermite(n, math.sqrt(2) * xx / w)
        * eval_hermite(m, math.sqrt(2) * yy / w)
        * np.exp(-(xx**2 + yy**2) / w**2)
    )


def test_hermite_gauss_projection():
    grid = Grid(128, dx=20 * u.um)
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    w = 0.3e-3
    data = hg(0, 0, w, xx, yy) + 0.3 * hg(1, 0, w, xx, yy) + 0.2j * hg(0, 1, w, xx, yy)
    result = project_hermite_gauss(Wavefront(data, grid, LAM * u.m), waist=w * u.m, max_order=3)
    assert result.coefficients[(0, 0)] == pytest.approx(1.0, abs=1e-8)
    assert result.coefficients[(1, 0)] == pytest.approx(0.3, abs=1e-8)
    assert result.coefficients[(0, 1)] == pytest.approx(0.2j, abs=1e-8)
    assert result.residual < 1e-8
    assert result.method == "hermite-gauss"


def test_fourier_projection_of_a_cosine():
    grid = Grid(64, dx=1 * u.um)
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    period = 16e-6
    data = 1.0 + 0.5 * np.cos(2 * math.pi * xx / period)
    result = project_fourier(Wavefront(data.astype(complex), grid, LAM * u.m), n_terms=3)
    assert result.residual < 1e-10
    values = np.asarray(to_wavefront(result.field, grid).field)
    np.testing.assert_allclose(values, data, atol=1e-10)


def test_gaussian_fit_with_curvature():
    grid = Grid(128, dx=10 * u.um)
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    w, x0, y0, radius = 0.25e-3, 40e-6, -25e-6, 0.8
    k = 2 * math.pi / LAM
    r2 = (xx - x0) ** 2 + (yy - y0) ** 2
    data = 2.0 * np.exp(-r2 / w**2) * np.exp(1j * k * r2 / (2 * radius))
    result = fit_gaussian(Wavefront(data, grid, LAM * u.m))
    parameters = result.coefficients
    assert parameters["waist"].to_value(u.m) == pytest.approx(w, rel=1e-6)
    assert parameters["x0"].to_value(u.m) == pytest.approx(x0, abs=1e-9)
    assert parameters["y0"].to_value(u.m) == pytest.approx(y0, abs=1e-9)
    assert parameters["radius"].to_value(u.m) == pytest.approx(radius, rel=1e-4)
    assert result.residual < 1e-6


def test_airy_fit():
    grid = Grid(96, dx=0.5 * u.um)
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    scale, x0 = 4e-6, 1.5e-6  # λf/D and centre
    v = math.pi * np.hypot(xx - x0, yy) / scale
    with np.errstate(invalid="ignore", divide="ignore"):
        pattern = np.where(v > 0, (2 * j1(v) / np.where(v > 0, v, 1)) ** 2, 1.0)
    result = fit_airy(pattern * 3.0, grid)
    assert result.coefficients["scale"].to_value(u.m) == pytest.approx(scale, rel=1e-5)
    assert result.coefficients["x0"].to_value(u.m) == pytest.approx(x0, abs=1e-9)
    assert result.coefficients["peak"] == pytest.approx(3.0, rel=1e-5)
    assert result.residual < 1e-6


def test_sparse_regression_finds_the_terms():
    grid = Grid(48, dx=0.1 * u.mm)
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    scale = 1e-3
    data = 2.0 * (xx / scale) ** 2 + 0.5 * (yy / scale)
    result = sparse_regression(
        Wavefront(data.astype(complex), grid, LAM * u.m), degree=3, length_scale=scale * u.m
    )
    terms = {str(key): value for key, value in result.coefficients.items()}
    assert set(terms) == {"X**2", "Y"}
    assert terms["X**2"] == pytest.approx(2.0, abs=1e-8)
    assert terms["Y"] == pytest.approx(0.5, abs=1e-8)
    assert result.residual < 1e-10
    assert result.field.expression.free_symbols <= {x, y}


def test_pysr_is_optional():
    pytest.importorskip("pysr", reason="PySR installed: nothing to check about its absence")


def test_symbolic_regression_without_pysr():
    try:
        import pysr  # noqa: F401

        pytest.skip("PySR is installed")
    except ImportError:
        pass
    grid = Grid(8, dx=1 * u.um)
    with pytest.raises(ImportError, match="pysr"):
        symbolic_regression(Wavefront(np.ones((8, 8), complex), grid, LAM * u.m))


def test_reconstructions_report_their_error():
    grid = Grid(64, dx=20 * u.um)
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    data = np.exp(-(np.abs(xx) + np.abs(yy)) / 0.2e-3).astype(complex)  # not a Gaussian
    result = project_hermite_gauss(Wavefront(data, grid, LAM * u.m), waist=0.2e-3 * u.m, max_order=2)
    assert 1e-3 < result.residual < 1
    gaussian = SymbolicField(sp.exp(-(x**2 + y**2) / sp.Float(0.2e-3) ** 2), wavelength=LAM * u.m)
    exact = project_hermite_gauss(
        to_wavefront(gaussian, grid), waist=0.2e-3 / math.sqrt(2) * math.sqrt(2) * u.m, max_order=0
    )
    assert exact.residual < 1e-8
