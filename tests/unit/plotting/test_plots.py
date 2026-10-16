"""Tests for the result plots of lumos.plotting (step P7.2).

Requirements: plan P7.2 (intensity and phase, PSF, radial profile, spectra, transmission maps, chip outputs,
context graph), REQ-RES-3 (quantities and units on the axes, labelled colour bars), REQ-CTX-11 (diagram of
the graph as SVG or PNG).

Independent references:

- a PSF normalised to its peak has a maximum of exactly 1; in an angular focal plane, ``λ/D`` units are the
  angles divided by ``λ/D`` (definition);
- the radial profile of a map whose value is the index ``floor(r / w)`` of the ring of width ``w`` holding
  each pixel is exactly ``k`` in ring ``k`` (construction), with ``r`` measured from pixel centres
  ``(i - N//2)·dx + x0`` (§9 of the specifications);
- the radial profile of a Gaussian ``exp(-r²/w²)`` is that function, averaged over each ring;
- the SVG format starts with an XML declaration or an ``<svg`` element (W3C SVG 1.1, §1.3), PNG files with the
  signature ``\\x89PNG\\r\\n\\x1a\\n`` (ISO/IEC 15948, §5.2).
"""

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u
from matplotlib.patches import FancyArrowPatch

from lumos import Context, plotting
from lumos.elements.detectors import PowerMeter
from lumos.elements.optics import ThinLens
from lumos.elements.sources import GaussianLaser
from lumos.fields import PHOTON_FLUX_DENSITY, DataProduct, Grid, SourceList, Wavefront
from lumos.processing import radial_profile


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def airy_like(grid, wavelength=1 * u.um):
    x, y = (np.asarray(c) for c in grid.coords(backend="numpy"))
    field = np.exp(-(x**2 + y**2) / (4 * grid.dx_si) ** 2).astype(complex) * 3.0
    return Wavefront(field, grid, wavelength)


# --- Radial profile ---------------------------------------------------------------------------------------


def test_radial_profile_of_ring_indices_is_exact():
    grid = Grid(64, 48, dx=1 * u.um, x0=2 * u.um)
    x, y = (np.asarray(c) for c in grid.coords(backend="numpy"))
    r = np.hypot(x - 2e-6, y)
    width = 3e-6
    image = np.floor(r / width)
    radii, profile = radial_profile(image, grid, center=(2 * u.um, 0 * u.um), bin_width=width * u.m)
    k = np.arange(len(profile))
    np.testing.assert_allclose(profile, k)
    np.testing.assert_allclose(radii.to_value(u.m), (k + 0.5) * width)


def test_radial_profile_default_center_and_bin_width():
    grid = Grid(33, dx=2 * u.um)
    x, y = (np.asarray(c) for c in grid.coords(backend="numpy"))
    image = np.floor(np.hypot(x, y) / 2e-6)
    radii, profile = radial_profile(image, grid)
    np.testing.assert_allclose(profile, np.arange(len(profile)))
    assert radii.unit.is_equivalent(u.m)


def test_radial_profile_of_a_gaussian():
    grid = Grid(256, dx=0.1 * u.um)
    x, y = (np.asarray(c) for c in grid.coords(backend="numpy"))
    w = 3e-6
    image = np.exp(-(x**2 + y**2) / w**2)
    radii, profile = radial_profile(image, grid, bin_width=0.1 * u.um)
    r = radii.to_value(u.m)
    keep = r < 2 * w
    np.testing.assert_allclose(profile[keep], np.exp(-(r[keep] ** 2) / w**2), atol=2e-2)


def test_radial_profile_refuses_mismatched_shapes():
    with pytest.raises(ValueError, match="shape"):
        radial_profile(np.zeros((4, 5)), Grid(5, 5, dx=1 * u.um))


# --- PSF ----------------------------------------------------------------------------------------------------


def test_psf_is_normalised_to_its_peak():
    grid = Grid(32, dx=1 * u.um)
    ax = plotting.plot_psf(airy_like(grid))
    image = np.asarray(ax.images[0].get_array())
    assert image.max() == pytest.approx(1.0, rel=1e-12)


def test_psf_in_lambda_over_d_units_on_an_angular_grid():
    lam, d = 1e-6, 2.0
    grid = Grid(32, dx=0.25 * lam / d * u.rad)
    ax = plotting.plot_psf(airy_like(grid, lam * u.m), diameter=d * u.m)
    xmin, xmax, ymin, ymax = ax.images[0].get_extent()
    bounds = [b.to_value(u.rad) / (lam / d) for b in grid.bounds]
    np.testing.assert_allclose([xmin, xmax, ymin, ymax], bounds, rtol=1e-12)
    assert "λ/D" in ax.get_xlabel()


def test_psf_has_a_labelled_colour_bar():
    grid = Grid(16, dx=1 * u.um)
    ax = plotting.plot_psf(airy_like(grid))
    assert ax.images[0].colorbar.ax.get_ylabel() != ""


# --- Radial profile plot ------------------------------------------------------------------------------------


def test_radial_profile_plot_shows_the_profile():
    grid = Grid(64, dx=1 * u.um)
    wf = airy_like(grid)
    ax = plotting.plot_radial_profile(wf, log=False)
    radii, profile = radial_profile(np.asarray(wf.intensity), grid)
    line = ax.get_lines()[0]
    np.testing.assert_allclose(line.get_xdata(), radii.to_value(grid.unit))
    np.testing.assert_allclose(line.get_ydata(), profile)
    assert ax.get_xlabel().startswith("r")


# --- Spectra ------------------------------------------------------------------------------------------------


def test_spectrum_of_a_source_list_has_one_line_per_source():
    lam = np.array([1.0, 1.5, 2.0]) * u.um
    flux = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) * PHOTON_FLUX_DENSITY
    sources = SourceList([0, 1e-6] * u.rad, [0, 0] * u.rad, lam, [0.1, 0.1, 0.1] * u.um, flux)
    ax = plotting.plot_spectrum(sources)
    lines = ax.get_lines()
    assert len(lines) == 2
    for line, row in zip(lines, flux.value, strict=True):
        np.testing.assert_allclose(line.get_xdata(), [1.0, 1.5, 2.0])
        np.testing.assert_allclose(line.get_ydata(), row, rtol=1e-12)
    assert "µm" in ax.get_xlabel()


def test_spectrum_of_a_data_product_with_a_wavelength_axis():
    product = DataProduct([1.0, 2.0, 4.0] * u.W, axes={"wavelength": [1.0, 2.0, 3.0] * u.um})
    ax = plotting.plot_spectrum(product)
    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), [1.0, 2.0, 4.0])
    assert "W" in ax.get_ylabel()


def test_spectrum_refuses_signals_without_spectrum():
    with pytest.raises(TypeError):
        plotting.plot_spectrum(np.zeros(3))


# --- Transmission maps and chip outputs ---------------------------------------------------------------------


def test_transmission_map_shows_the_data():
    data = np.linspace(0, 1, 12).reshape(3, 4)
    product = DataProduct(
        data * u.dimensionless_unscaled,
        axes={"theta_y": [0, 1, 2] * u.mas, "theta_x": [0, 1, 2, 3] * u.mas},
    )
    ax = plotting.plot_transmission_map(product)
    np.testing.assert_allclose(np.asarray(ax.images[0].get_array()), data)
    assert "mas" in ax.get_xlabel()
    assert ax.images[0].colorbar.ax.get_ylabel() == "transmission"


def test_chip_outputs_bar_heights():
    product = DataProduct([1e-9, 2e-9, 0.5e-9] * u.W, axes={"channel": [0, 1, 2] * u.dimensionless_unscaled})
    ax = plotting.plot_chip_outputs(product, labels=["bright", "dark 1", "dark 2"])
    heights = [p.get_height() for p in ax.patches]
    np.testing.assert_allclose(heights, [1e-9, 2e-9, 0.5e-9])
    assert [t.get_text() for t in ax.get_xticklabels()] == ["bright", "dark 1", "dark 2"]
    assert "W" in ax.get_ylabel()


# --- Context graph ------------------------------------------------------------------------------------------


def chain():
    ctx = Context(seed=1)
    laser = ctx.add(GaussianLaser(name="laser"))
    lens = ctx.add(ThinLens(name="lens"))
    meter = ctx.add(PowerMeter(name="meter"))
    meter2 = ctx.add(PowerMeter(name="meter2"))
    laser >> lens >> meter
    ctx.connect(lens.port("out"), meter2.port("in"))
    return ctx


def test_context_graph_shows_every_element_and_connection():
    ctx = chain()
    ax = plotting.plot_context(ctx)
    texts = " ".join(t.get_text() for t in ax.texts)
    for identifier in ("laser", "lens", "meter", "meter2"):
        assert identifier in texts
    arrows = [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
    assert len(arrows) == len(ctx.graph.connections) == 3


def test_context_graph_places_elements_left_to_right_in_topological_order():
    ctx = chain()
    ax = plotting.plot_context(ctx)
    position = {t.get_text().split("\n")[0]: t.get_position()[0] for t in ax.texts}
    assert position["laser"] < position["lens"] < position["meter"]
    assert math.isclose(position["meter"], position["meter2"])


@pytest.mark.parametrize(("suffix", "signature"), [(".svg", (b"<?xml", b"<svg")), (".png", (b"\x89PNG",))])
def test_context_diagram_files(tmp_path, suffix, signature):
    path = tmp_path / f"graph{suffix}"
    plotting.save_context_diagram(chain(), path)
    assert path.read_bytes().startswith(signature)
