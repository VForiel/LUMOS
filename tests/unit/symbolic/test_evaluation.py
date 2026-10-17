"""Tests for the evaluation of symbolic fields on grids (step P10.2, REQ-SYM-2).

Requirements: evaluation on a ``Grid`` (→ ``Wavefront``) or on a grid and a spectral axis (→ ``Wavecube``)
when the expression depends on λ; compilation with ``lambdify`` for the backend, cached; anti-aliasing of
discontinuities by supersampling and integration over each pixel.

Independent references: NumPy evaluations of the same formulas at the pixel centres
``(i - N//2) dx + x0`` (§9); the power of ``exp(-r²/w²)`` is ``π w² / 2``; the area of a disk is ``π R²``.
"""

import math

import numpy as np
import pytest
import sympy as sp
from astropy import units as u

from lumos.backend import available, to_numpy, use
from lumos.fields import Grid, Wavecube, Wavefront
from lumos.symbolic import SymbolicField, lam, r, theta_x, x, y
from lumos.symbolic.evaluate import cache_info, clear_cache, to_wavecube, to_wavefront

W = sp.Symbol("w", positive=True)
R = sp.Symbol("R", positive=True)


def gaussian(width=0.4 * u.mm):
    return SymbolicField(sp.exp(-(x**2 + y**2) / W**2), parameters={W: width}, wavelength=633 * u.nm)


def test_gaussian_values_at_pixel_centres():
    grid = Grid(32, 24, dx=50 * u.um, x0=0.1 * u.mm)
    wavefront = to_wavefront(gaussian(), grid)
    assert isinstance(wavefront, Wavefront)
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    expected = np.exp(-(xx**2 + yy**2) / 0.4e-3**2)
    np.testing.assert_allclose(np.asarray(to_numpy(wavefront.field)), expected, rtol=1e-12, atol=1e-15)
    assert wavefront.wavelength == 633 * u.nm


def test_power_of_a_gaussian():
    grid = Grid(256, dx=10 * u.um)
    power = to_wavefront(gaussian(), grid).power.to_value(u.W)
    assert power == pytest.approx(math.pi * 0.4e-3**2 / 2, rel=1e-10)


def test_supersampling_anti_aliases_a_disk():
    disk = SymbolicField(sp.Heaviside(R - r), parameters={R: 0.5 * u.mm}, wavelength=1 * u.um)
    grid = Grid(64, dx=23 * u.um)
    exact = math.pi * 0.5e-3**2
    errors = []
    for supersample in (1, 8):
        field = to_wavefront(disk, grid, supersample=supersample)
        area = float(np.sum(np.asarray(to_numpy(field.field)).real)) * grid.dx_si * grid.dy_si
        errors.append(abs(area - exact) / exact)
    assert errors[1] < 1e-3
    assert errors[1] < errors[0]


def test_chromatic_expression_gives_a_wavecube():
    field = SymbolicField(sp.exp(sp.I * 2 * sp.pi * x / lam))
    grid = Grid(16, dx=1 * u.um)
    lam_values = [1.0, 2.0] * u.um
    cube = to_wavecube(field, grid, lam_values)
    assert isinstance(cube, Wavecube)
    xx, _ = (np.asarray(c) for c in grid.coords(backend="numpy"))
    values = np.asarray(to_numpy(cube.field))
    for k, wavelength in enumerate([1e-6, 2e-6]):
        np.testing.assert_allclose(values[k], np.exp(2j * np.pi * xx / wavelength), rtol=1e-12)


def test_achromatic_field_on_several_wavelengths():
    cube = to_wavecube(gaussian(), Grid(8, dx=0.1 * u.mm), [1.0, 1.5] * u.um)
    values = np.asarray(to_numpy(cube.field))
    np.testing.assert_allclose(values[0], values[1])


def test_chromatic_field_needs_a_wavelength_for_a_wavefront():
    field = SymbolicField(sp.exp(sp.I * 2 * sp.pi * x / lam))
    with pytest.raises(ValueError, match="wavelength"):
        to_wavefront(field, Grid(4, dx=1 * u.um))
    wavefront = to_wavefront(field, Grid(4, dx=1 * u.um), wavelength=2 * u.um)
    assert wavefront.wavelength == 2 * u.um


def test_constant_expression_fills_the_grid():
    field = SymbolicField(sp.Integer(2) + 0 * x, wavelength=1 * u.um)
    values = np.asarray(to_numpy(to_wavefront(field, Grid(5, 3, dx=1 * u.um)).field))
    assert values.shape == (3, 5)
    np.testing.assert_array_equal(values, 2.0)


def test_planes_must_match():
    angular = SymbolicField(sp.cos(theta_x), plane="angular", wavelength=1 * u.um)
    with pytest.raises(ValueError, match="plane"):
        to_wavefront(angular, Grid(4, dx=1 * u.um))
    values = to_wavefront(angular, Grid(4, dx=1 * u.mrad))
    assert values.grid.plane == "angular"


def test_compiled_functions_are_cached():
    clear_cache()
    grid = Grid(8, dx=0.1 * u.mm)
    to_wavefront(gaussian(), grid)
    first = cache_info()
    to_wavefront(gaussian(0.5 * u.mm), grid)  # same expression, other parameter value
    second = cache_info()
    assert second.misses == first.misses
    assert second.hits == first.hits + 1


@pytest.mark.skipif("jax" not in available(), reason="JAX is not installed")
def test_jax_backend_agrees_with_numpy():
    grid = Grid(16, dx=50 * u.um)
    reference = np.asarray(to_numpy(to_wavefront(gaussian(), grid).field))
    with use("jax"):
        field = to_wavefront(gaussian(), grid)
        assert field.backend == "jax"
        np.testing.assert_allclose(np.asarray(to_numpy(field.field)), reference, rtol=1e-12)


def test_methods_on_the_field():
    grid = Grid(8, dx=0.1 * u.mm)
    np.testing.assert_array_equal(
        np.asarray(to_numpy(gaussian().to_wavefront(grid).field)),
        np.asarray(to_numpy(to_wavefront(gaussian(), grid).field)),
    )
