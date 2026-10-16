"""Tests for lumos.fields.Grid (step P3.1).

Requirements: REQ-SIG-1 (rectangular 2D sampling, ``nx != ny`` allowed, steps ``dx, dy``, shiftable centre,
spatial plane in metres or angular plane in radians, single centring convention) and §9 of the specs:

- arrays are indexed ``[..., y, x]``;
- the central pixel is index ``N // 2`` for even and odd ``N``; ``x_i = (i - N//2)·dx + x0``;
- angular planes map to the focal plane with ``x = f·θ``.

Reference values are written by hand from these formulas, or come from NumPy's independent FFT frequency
ordering (``fftshift(fftfreq(N, 1/N))`` lists ``i - N//2`` for every ``N``), so that a centred DFT and a
LUMOS grid agree on which pixel is the origin.
"""

import importlib.util
import json
import math

import numpy as np
import pytest
from astropy import units as u
from hypothesis import given
from hypothesis import strategies as st

import lumos
from lumos import backend
from lumos.fields import Grid

HAS_JAX = importlib.util.find_spec("jax") is not None
BACKENDS = [
    "numpy",
    pytest.param("jax", marks=pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")),
]

sizes = st.integers(min_value=1, max_value=64)
steps = st.floats(min_value=1e-9, max_value=1e3, allow_nan=False, allow_infinity=False)
offsets = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)


@pytest.fixture
def restore_precision():
    previous = lumos.config.precision
    yield
    lumos.config.precision = previous


# --- Construction and validation ---------------------------------------------------------------


def test_square_grid_from_single_size():
    grid = Grid(8, dx=1 * u.um)
    assert (grid.nx, grid.ny) == (8, 8)
    assert grid.shape == (8, 8)
    assert grid.size == 64
    assert grid.dy == grid.dx == 1 * u.um


def test_rectangular_grid_shape_is_ny_nx():
    grid = Grid(6, 4, dx=2 * u.mm, dy=3 * u.mm)
    assert grid.shape == (4, 6)
    assert grid.dx == 2 * u.mm
    assert grid.dy == 3 * u.mm


def test_length_step_gives_spatial_plane():
    grid = Grid(4, dx=10 * u.um)
    assert grid.plane == "spatial"
    assert grid.dx_si == pytest.approx(1e-5, rel=1e-15)


def test_angle_step_gives_angular_plane():
    grid = Grid(4, dx=1 * u.mas)
    assert grid.plane == "angular"
    # 1 mas = 1e-3 / 3600 degree = pi / (180 * 3600 * 1000) rad
    assert grid.dx_si == pytest.approx(math.pi / 648_000_000, rel=1e-14)


def test_user_unit_is_preserved():
    grid = Grid(4, dx=5 * u.mm, x0=1 * u.cm)
    assert grid.dx.unit == u.mm
    assert grid.unit == u.mm
    assert grid.x0.unit == u.mm
    assert grid.x0.value == pytest.approx(10.0)


@pytest.mark.parametrize("bad", [0, -3, 2.5, True])
def test_invalid_sizes_are_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        Grid(bad, dx=1 * u.m)


def test_raw_float_step_is_rejected():
    with pytest.raises(TypeError):
        Grid(4, dx=1e-6)


@pytest.mark.parametrize("bad", [0 * u.m, -1 * u.m, np.inf * u.m, np.nan * u.m])
def test_non_positive_or_non_finite_steps_are_rejected(bad):
    with pytest.raises(ValueError):
        Grid(4, dx=bad)


def test_step_that_is_neither_length_nor_angle_is_rejected():
    with pytest.raises(u.UnitsError):
        Grid(4, dx=1 * u.s)


def test_mixed_planes_are_rejected():
    with pytest.raises(u.UnitsError):
        Grid(4, dx=1 * u.um, dy=1 * u.arcsec)
    with pytest.raises(u.UnitsError):
        Grid(4, dx=1 * u.um, x0=1 * u.arcsec)


def test_from_extent_divides_the_width_by_the_number_of_pixels():
    grid = Grid.from_extent(100, 50, width=1 * u.cm, height=2 * u.cm)
    assert grid.dx.to_value(u.m) == pytest.approx(1e-4)
    assert grid.dy.to_value(u.m) == pytest.approx(4e-4)
    assert grid.extent[0].to_value(u.cm) == pytest.approx(1.0)
    assert grid.extent[1].to_value(u.cm) == pytest.approx(2.0)


# --- Centring convention (§9) ------------------------------------------------------------------


def test_even_axis_coordinates_by_hand():
    grid = Grid(4, dx=1 * u.m)
    np.testing.assert_array_equal(grid.x.to_value(u.m), [-2.0, -1.0, 0.0, 1.0])


def test_odd_axis_coordinates_by_hand():
    grid = Grid(5, dx=1 * u.m)
    np.testing.assert_array_equal(grid.x.to_value(u.m), [-2.0, -1.0, 0.0, 1.0, 2.0])


def test_shifted_rectangular_coordinates_by_hand():
    grid = Grid(3, 2, dx=2 * u.mm, dy=0.5 * u.mm, x0=10 * u.mm, y0=-1 * u.mm)
    np.testing.assert_allclose(grid.x.to_value(u.mm), [8.0, 10.0, 12.0])
    np.testing.assert_allclose(grid.y.to_value(u.mm), [-1.5, -1.0])  # (j - 1)·0.5 - 1


def test_center_index_is_n_over_two():
    assert Grid(4, 7, dx=1 * u.m).center_index == (3, 2)  # (iy, ix)


@given(n=sizes)
def test_axis_matches_numpy_centred_fft_ordering(n):
    expected = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / n))
    np.testing.assert_allclose(Grid(n, dx=1 * u.m).x.to_value(u.m), expected, atol=1e-12)


@given(nx=sizes, ny=sizes, dx=steps, dy=steps, x0=offsets, y0=offsets)
def test_centre_pixel_sits_at_the_centre(nx, ny, dx, dy, x0, y0):
    grid = Grid(nx, ny, dx=dx * u.m, dy=dy * u.m, x0=x0 * u.m, y0=y0 * u.m)
    xx, yy = grid.coords(backend="numpy")
    iy, ix = grid.center_index
    assert xx[iy, ix] == pytest.approx(x0, abs=1e-9 * max(1.0, abs(x0)))
    assert yy[iy, ix] == pytest.approx(y0, abs=1e-9 * max(1.0, abs(y0)))


def test_bounds_include_half_pixels():
    grid = Grid(4, 3, dx=1 * u.m, dy=2 * u.m)
    xmin, xmax, ymin, ymax = (b.to_value(u.m) for b in grid.bounds)
    assert (xmin, xmax) == (-2.5, 1.5)
    assert (ymin, ymax) == (-3.0, 3.0)


def test_extent_and_pixel_area():
    grid = Grid(10, 20, dx=1 * u.um, dy=2 * u.um)
    assert grid.extent[0].to_value(u.um) == pytest.approx(10.0)
    assert grid.extent[1].to_value(u.um) == pytest.approx(40.0)
    assert grid.pixel_area.to_value(u.um**2) == pytest.approx(2.0)


def test_angular_pixel_area_converts_to_steradian():
    grid = Grid(4, dx=1 * u.rad)
    assert grid.pixel_area.to_value(u.sr) == pytest.approx(1.0)


# --- Coordinate arrays ---------------------------------------------------------------------------


def test_coords_use_y_x_indexing():
    grid = Grid(3, 2, dx=1 * u.m)
    xx, yy = grid.coords(backend="numpy")
    assert xx.shape == yy.shape == (2, 3)
    np.testing.assert_array_equal(xx, [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    np.testing.assert_array_equal(yy, [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0]])


def test_axes_are_si_one_dimensional_arrays():
    grid = Grid(4, 2, dx=1 * u.mm)
    x, y = grid.axes(backend="numpy")
    np.testing.assert_allclose(x, [-2e-3, -1e-3, 0.0, 1e-3])
    np.testing.assert_allclose(y, [-1e-3, 0.0])


def test_polar_coordinates_by_hand():
    grid = Grid(3, dx=1 * u.m)
    r, theta = grid.polar(backend="numpy")
    assert r[1, 1] == 0.0
    assert r[2, 2] == pytest.approx(math.sqrt(2.0))
    assert theta[1, 2] == pytest.approx(0.0)  # +x axis
    assert theta[2, 1] == pytest.approx(math.pi / 2)  # +y axis (row index increases with y)


def test_coordinates_are_lazy():
    # 1e5 x 1e5 float64 coordinates would need 160 GB: building the grid must not allocate them.
    grid = Grid(100_000, dx=1 * u.um)
    assert grid.shape == (100_000, 100_000)


def test_coords_are_cached():
    grid = Grid(16, dx=1 * u.m)
    assert grid.coords(backend="numpy")[0] is grid.coords(backend="numpy")[0]


@pytest.mark.parametrize("name", BACKENDS)
def test_coords_follow_the_current_backend(name):
    grid = Grid(5, 3, dx=1 * u.um)
    with backend.use(name):
        xx, _ = grid.coords()
    assert backend.backend_of(xx) == name
    np.testing.assert_allclose(backend.to_numpy(xx), grid.coords(backend="numpy")[0])


def test_coords_follow_the_precision(restore_precision):
    grid = Grid(4, dx=1 * u.m)
    lumos.config.precision = "single"
    assert grid.coords(backend="numpy")[0].dtype == np.float32
    lumos.config.precision = "double"
    assert grid.coords(backend="numpy")[0].dtype == np.float64


@given(nx=sizes, ny=sizes, dx=steps, shift=offsets, data=st.data())
def test_index_inverts_coordinates(nx, ny, dx, shift, data):
    # The centre is expressed in pixels: an offset of 1e3 m on a 1 nm grid cannot be inverted to 1e-6 pixel
    # in double precision (cancellation), which is a property of floats and not of the grid.
    grid = Grid(nx, ny, dx=dx * u.m, x0=shift * dx * u.m)
    ix = data.draw(st.integers(0, nx - 1))
    iy = data.draw(st.integers(0, ny - 1))
    fx, fy = grid.index(grid.x[ix], grid.y[iy])
    assert fx == pytest.approx(ix, abs=1e-6)
    assert fy == pytest.approx(iy, abs=1e-6)


# --- Plane conversion (x = f·θ) ---------------------------------------------------------------


def test_angular_to_spatial_uses_x_equals_f_theta():
    grid = Grid(8, 4, dx=1 * u.urad, dy=2 * u.urad, x0=3 * u.urad)
    focal = Grid(8, 4, dx=1 * u.um, dy=2 * u.um, x0=3 * u.um)
    assert grid.to_spatial(1 * u.m) == focal


def test_spatial_to_angular_round_trip():
    grid = Grid(8, dx=5 * u.um, y0=-2 * u.um)
    back = grid.to_angular(2 * u.m).to_spatial(2 * u.m)
    assert back.dx_si == pytest.approx(grid.dx_si)
    assert back.y0_si == pytest.approx(grid.y0_si)
    assert back.plane == "spatial"


def test_plane_conversion_requires_the_matching_plane():
    with pytest.raises(ValueError):
        Grid(4, dx=1 * u.m).to_spatial(1 * u.m)
    with pytest.raises(ValueError):
        Grid(4, dx=1 * u.rad).to_angular(1 * u.m)


# --- Value semantics and serialisation -----------------------------------------------------------


def test_equality_is_by_value_across_units():
    assert Grid(4, dx=1 * u.mm) == Grid(4, dx=1e-3 * u.m)
    assert hash(Grid(4, dx=1 * u.mm)) == hash(Grid(4, dx=1e-3 * u.m))
    assert Grid(4, dx=1 * u.mm) != Grid(5, dx=1 * u.mm)
    assert Grid(4, dx=1 * u.mm) != Grid(4, dx=1 * u.mm, x0=1 * u.mm)


def test_grid_is_immutable():
    grid = Grid(4, dx=1 * u.m)
    with pytest.raises(AttributeError):
        grid.nx = 5


def test_with_center_shifts_only_the_centre():
    grid = Grid(4, dx=1 * u.m).with_center(1 * u.m, 2 * u.m)
    assert (grid.x0, grid.y0) == (1 * u.m, 2 * u.m)
    assert grid.shape == (4, 4)


@given(nx=sizes, ny=sizes, dx=steps, dy=steps, x0=offsets, angular=st.booleans())
def test_json_round_trip(nx, ny, dx, dy, x0, angular):
    unit = u.arcsec if angular else u.um
    grid = Grid(nx, ny, dx=dx * unit, dy=dy * unit, x0=x0 * unit)
    data = json.loads(json.dumps(grid.to_json()))
    assert Grid.from_json(data) == grid
    assert Grid.from_json(data).unit == unit
