"""Tests for the figure style of LUMOS (step P7.1, REQ-RES-3).

Requirements (REQ-RES-3): quantities and units on the axes, labelled colour bars, perceptual colour maps
(cyclic for the phase), homogeneous style, sizes suited to publications.

Independent references:

- matplotlib documentation, *Choosing Colormaps*: the perceptually uniform sequential maps are ``viridis``,
  ``plasma``, ``inferno``, ``magma`` and ``cividis``; the cyclic maps are ``twilight``, ``twilight_shifted``
  and ``hsv``; ``RdBu``, ``coolwarm``... are diverging;
- A&A author guide (*Preparing figures*): a column is 88 mm wide, a full page 180 mm (17 cm text width plus
  margins rounded); raster figures need at least 300 dpi;
- unit strings: astropy's ``"unicode"`` format (``u.um`` → ``"µm"`` with the micro sign).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from astropy import units as u

from lumos.plotting import style

PERCEPTUAL = {"viridis", "plasma", "inferno", "magma", "cividis"}
CYCLIC = {"twilight", "twilight_shifted", "hsv"}
DIVERGING = {"RdBu", "RdBu_r", "coolwarm", "bwr", "seismic", "PuOr", "PuOr_r", "BrBG", "RdYlBu_r"}
MM = 1 / 25.4


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_colour_maps_by_role():
    assert style.CMAPS["intensity"] in PERCEPTUAL
    assert style.CMAPS["radiance"] in PERCEPTUAL
    assert style.CMAPS["data"] in PERCEPTUAL
    assert style.CMAPS["phase"] in CYCLIC
    assert style.CMAPS["diverging"] in DIVERGING


@pytest.mark.parametrize(
    ("unit", "text"),
    [(u.um, "µm"), (u.rad, "rad"), (u.photon / u.s, "ph s⁻¹"), (u.Jy, "Jy"), (u.electron, "e⁻")],
)
def test_unit_label_uses_unicode(unit, text):
    assert style.unit_label(unit) == text


def test_unit_label_of_dimensionless_quantities():
    assert style.unit_label(u.dimensionless_unscaled) == "dimensionless"


def test_axis_label_adds_the_unit_in_parentheses():
    assert style.axis_label("x", u.mm) == "x (mm)"
    assert style.axis_label("wavelength", u.um) == "wavelength (µm)"


def test_axis_label_without_unit_for_dimensionless_quantities():
    assert style.axis_label("contrast", u.dimensionless_unscaled) == "contrast"
    assert style.axis_label("contrast", None) == "contrast"


def test_axis_label_accepts_a_quantity():
    assert style.axis_label("x", 3 * u.mm) == "x (mm)"


@pytest.mark.parametrize(("size", "width_mm"), [("single", 88), ("double", 180)])
def test_publication_figure_widths(size, width_mm):
    fig, _ = style.figure(size=size)
    assert fig.get_figwidth() == pytest.approx(width_mm * MM, rel=1e-6)


def test_figure_aspect_and_grid_of_axes():
    fig, axes = style.figure(size="double", nrows=2, ncols=3, aspect=0.5)
    assert axes.shape == (2, 3)
    assert fig.get_figheight() == pytest.approx(0.5 * fig.get_figwidth(), rel=1e-6)


def test_unknown_figure_size_is_refused():
    with pytest.raises(ValueError, match="size"):
        style.figure(size="poster-a0")


def test_style_context_sets_publication_settings_and_restores_them():
    before = dict(plt.rcParams)
    with style.context():
        assert plt.rcParams["savefig.dpi"] >= 300
        assert plt.rcParams["image.origin"] == "lower"
        assert plt.rcParams["image.cmap"] in PERCEPTUAL
    assert plt.rcParams["savefig.dpi"] == before["savefig.dpi"]
    assert plt.rcParams["image.origin"] == before["image.origin"]


def test_style_contains_only_valid_matplotlib_settings():
    for key in style.STYLE:
        assert key in matplotlib.rcParams


def test_figure_uses_the_style():
    with style.context():
        fig, _ = style.figure()
        assert fig.dpi == plt.rcParams["figure.dpi"]
