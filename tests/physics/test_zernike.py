"""Tests for Zernike polynomials (step P6.5: phase plates, deformable mirrors and adaptive optics correction).

Reference: R. J. Noll, "Zernike polynomials and atmospheric turbulence", JOSA 66, 207 (1976): ordering of the
modes (j → n, m), normalisation ``(1/π) ∫ Z_i Z_j = δ_ij`` over the unit disk, and the explicit polynomials of
his Table I for j = 1 … 11.
"""

import math

import numpy as np
import pytest
from astropy import units as u
from hypothesis import given, settings
from hypothesis import strategies as st

from lumos.fields import Grid
from lumos.physics.zernike import noll_indices, zernike, zernike_basis

NOLL_TABLE = {
    1: lambda r, t: np.ones_like(r),
    2: lambda r, t: 2 * r * np.cos(t),
    3: lambda r, t: 2 * r * np.sin(t),
    4: lambda r, t: math.sqrt(3) * (2 * r**2 - 1),
    5: lambda r, t: math.sqrt(6) * r**2 * np.sin(2 * t),
    6: lambda r, t: math.sqrt(6) * r**2 * np.cos(2 * t),
    7: lambda r, t: math.sqrt(8) * (3 * r**3 - 2 * r) * np.sin(t),
    8: lambda r, t: math.sqrt(8) * (3 * r**3 - 2 * r) * np.cos(t),
    9: lambda r, t: math.sqrt(8) * r**3 * np.sin(3 * t),
    10: lambda r, t: math.sqrt(8) * r**3 * np.cos(3 * t),
    11: lambda r, t: math.sqrt(5) * (6 * r**4 - 6 * r**2 + 1),
}


@pytest.mark.parametrize(
    ("j", "n", "m"),
    [
        (1, 0, 0),
        (2, 1, 1),
        (3, 1, -1),
        (4, 2, 0),
        (5, 2, -2),
        (6, 2, 2),
        (7, 3, -1),
        (8, 3, 1),
        (9, 3, -3),
        (10, 3, 3),
        (11, 4, 0),
        (12, 4, 2),
        (13, 4, -2),
        (14, 4, 4),
        (15, 4, -4),
        (22, 6, 0),
    ],
)
def test_noll_indices(j, n, m):
    assert noll_indices(j) == (n, m)


@pytest.mark.parametrize("j", sorted(NOLL_TABLE))
def test_polynomials_match_noll_table(j):
    rng = np.random.default_rng(j)
    r = rng.uniform(0, 1, 50)
    t = rng.uniform(-math.pi, math.pi, 50)
    np.testing.assert_allclose(zernike(j, r, t), NOLL_TABLE[j](r, t), atol=1e-12)


@settings(max_examples=20, deadline=None)
@given(st.integers(1, 28), st.integers(1, 28))
def test_orthonormality_on_the_unit_disk(i, j):
    # Gauss-Legendre in r (r dr) and uniform trapezoid in θ are exact for these polynomials
    nodes, weights = np.polynomial.legendre.leggauss(20)
    r = (nodes + 1) / 2
    w_r = weights / 2 * r
    t = np.linspace(0, 2 * math.pi, 64, endpoint=False)
    rr, tt = np.meshgrid(r, t, indexing="ij")
    integral = np.sum(w_r[:, None] * zernike(i, rr, tt) * zernike(j, rr, tt)) * (2 * math.pi / 64) / math.pi
    assert integral == pytest.approx(1.0 if i == j else 0.0, abs=1e-10)


def test_basis_on_a_grid_is_zero_outside_the_pupil():
    grid = Grid(64, dx=1 / 60 * u.m)
    basis = zernike_basis(grid, 1 * u.m, 6)
    assert basis.shape == (6, 64, 64)
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    outside = np.hypot(xx, yy) > 0.5
    assert np.all(basis[:, outside] == 0)
    inside = ~outside
    rho, theta = np.hypot(xx, yy)[inside] / 0.5, np.arctan2(yy, xx)[inside]
    np.testing.assert_allclose(basis[3][inside], NOLL_TABLE[4](rho, theta), atol=1e-12)
