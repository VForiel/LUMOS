"""Tests for the basic plots of lumos.plotting (step P3.9).

Requirements: plan P3.9 (physical axes, phase drawn with a cyclic colour map) and §9 of the specs (arrays
``[..., y, x]`` drawn with the origin at the bottom left, x to the right and y upwards).

References: the image extent must be the outer pixel edges of the grid (centres at ``(i - N//2)·dx + x0``,
§9); matplotlib's cyclic colour maps are ``twilight``, ``twilight_shifted`` and ``hsv`` (matplotlib
documentation, *Choosing Colormaps*, "Cyclic"), and the phase is wrapped to ``[-π, π]``.
"""

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u
from matplotlib.colors import LogNorm

from lumos import plotting
from lumos.fields import DataProduct, Grid, RadianceField, Wavefront, jones

CYCLIC = {"twilight", "twilight_shifted", "hsv"}
GRID = Grid(8, 6, dx=2 * u.um, x0=1 * u.um)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def gaussian():
    return Wavefront.gaussian(GRID, 1 * u.um, waist=4 * u.um, power=1 * u.mW).tilt(0.05 * u.rad, 0 * u.rad)


def only_image(ax):
    assert len(ax.images) == 1
    return ax.images[0]


def test_intensity_plot_uses_physical_axes_and_lower_origin():
    ax = plotting.plot_intensity(gaussian())
    im = only_image(ax)
    xmin, xmax, ymin, ymax = (b.to_value(u.um) for b in GRID.bounds)
    np.testing.assert_allclose(im.get_extent(), [xmin, xmax, ymin, ymax])
    assert im.origin == "lower"
    assert "µm" in ax.get_xlabel() or "um" in ax.get_xlabel()
    assert ax.get_xlabel().startswith("x")
    assert ax.get_ylabel().startswith("y")


def test_intensity_plot_shows_the_intensity():
    wf = gaussian()
    im = only_image(plotting.plot_intensity(wf))
    np.testing.assert_allclose(np.asarray(im.get_array()), np.asarray(wf.intensity), rtol=1e-12)


def test_intensity_colorbar_has_the_intensity_unit():
    ax = plotting.plot_intensity(gaussian())
    colorbar = only_image(ax).colorbar
    assert colorbar is not None
    assert "W" in colorbar.ax.get_ylabel()
    assert "m" in colorbar.ax.get_ylabel()


def test_log_scale():
    im = only_image(plotting.plot_intensity(gaussian(), log=True))
    assert isinstance(im.norm, LogNorm)


def test_phase_plot_uses_a_cyclic_colour_map_over_minus_pi_pi():
    wf = gaussian()
    im = only_image(plotting.plot_phase(wf))
    assert im.get_cmap().name in CYCLIC
    assert im.norm.vmin == pytest.approx(-math.pi)
    assert im.norm.vmax == pytest.approx(math.pi)
    np.testing.assert_allclose(np.ma.getdata(im.get_array()), np.asarray(wf.phase), atol=1e-12)


def test_phase_is_masked_where_the_amplitude_is_negligible():
    im = only_image(plotting.plot_phase(gaussian(), threshold=1e-3))
    data = im.get_array()
    amplitude = np.asarray(gaussian().amplitude)
    assert np.all(np.ma.getmaskarray(data) == (amplitude < 1e-3 * amplitude.max()))


def test_plot_field_draws_intensity_and_phase_side_by_side():
    fig = plotting.plot_field(gaussian())
    assert len([ax for ax in fig.axes if ax.images]) == 2


def test_polarized_field_plots_each_component():
    wf = Wavefront.plane(GRID, 1 * u.um, power=1 * u.W, polarization="D")
    fig = plotting.plot_field(wf)
    assert len([ax for ax in fig.axes if ax.images]) == 4  # (|Ex|², arg Ex, |Ey|², arg Ey)
    out = jones.apply(jones.polarizer(0 * u.deg), wf)
    assert plotting.plot_intensity(out) is not None


def test_existing_axes_are_used():
    _fig, ax = plt.subplots()
    assert plotting.plot_intensity(gaussian(), ax=ax) is ax


def test_angular_grid_axes_are_labelled_in_angles():
    sky = Grid(8, dx=5 * u.mas)
    rf = RadianceField(np.ones((1, 8, 8)) * u.Jy / u.sr, sky, [1.0] * u.um)
    ax = plotting.plot_radiance(rf)
    assert "mas" in ax.get_xlabel()
    np.testing.assert_allclose(only_image(ax).get_extent(), [b.to_value(u.mas) for b in sky.bounds])


def test_data_product_image_uses_its_axes():
    dp = DataProduct.from_grid(np.arange(48.0).reshape(6, 8) * u.adu, GRID)
    ax = plotting.plot_data(dp)
    im = only_image(ax)
    assert im.origin == "lower"
    np.testing.assert_allclose(im.get_extent(), [b.to_value(u.um) for b in GRID.bounds])
    assert "adu" in im.colorbar.ax.get_ylabel().lower()


def test_one_dimensional_data_product_is_a_line_plot():
    t = np.linspace(0, 1, 5) * u.s
    dp = DataProduct(np.arange(5.0) * u.W, axes={"time": t})
    ax = plotting.plot_data(dp)
    assert len(ax.lines) == 1
    np.testing.assert_allclose(ax.lines[0].get_xdata(), t.value)
    assert "time" in ax.get_xlabel()
    assert "W" in ax.get_ylabel()
