"""Tests for the export of figures and their data (step P7.3, REQ-RES-4, REQ-RES-5).

Requirements: every figure is exported as ``.png`` and ``.plt`` (``pltedit.save``), and the data behind it
is saved (``.npz``). Acceptance criterion of phase 7: the ``.plt`` files reload with ``pltedit.load``.

Independent references:

- PNG files start with the signature ``\\x89PNG\\r\\n\\x1a\\n`` (ISO/IEC 15948, §5.2) and matplotlib's
  ``imread`` reads them back as an ``(height, width, 4)`` array;
- ``pltedit.load`` (VFSciTools) is the reader of the ``.plt`` format;
- ``.npz`` files are read with ``numpy.load(..., allow_pickle=False)``: no pickled object is needed.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u

from lumos._vendor import pltedit
from lumos.results import load_data, save_data, save_figure


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def line_figure():
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0, 2.0], [3.0, 1.0, 4.0])
    ax.set_xlabel("x (mm)")
    return fig


def test_save_figure_writes_png_and_plt(tmp_path):
    paths = save_figure(line_figure(), tmp_path / "profile")
    assert sorted(p.name for p in paths) == ["profile.plt", "profile.png"]
    png = tmp_path / "profile.png"
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    image = mpimg.imread(png)
    assert image.ndim == 3


def test_plt_files_reload_with_pltedit(tmp_path):
    save_figure(line_figure(), tmp_path / "profile")
    figure = pltedit.load(tmp_path / "profile.plt")
    line = figure.axes[0].get_lines()[0]
    np.testing.assert_array_equal(line.get_ydata(), [3.0, 1.0, 4.0])
    assert figure.axes[0].get_xlabel() == "x (mm)"


def test_save_figure_accepts_axes(tmp_path):
    fig = line_figure()
    save_figure(fig.axes[0], tmp_path / "from_axes")
    assert (tmp_path / "from_axes.png").exists()


def test_save_figure_with_data(tmp_path):
    data = {"radius": [1.0, 2.0] * u.um, "profile": np.array([0.5, 0.25])}
    paths = save_figure(line_figure(), tmp_path / "fig", data=data)
    assert tmp_path / "fig.npz" in paths
    with np.load(tmp_path / "fig.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["profile"], [0.5, 0.25])
        np.testing.assert_array_equal(archive["radius"], [1.0, 2.0])


def test_save_figure_selected_formats(tmp_path):
    paths = save_figure(line_figure(), tmp_path / "only", formats=("png", "pdf"))
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"]
    assert not (tmp_path / "only.plt").exists()


def test_unknown_format_is_refused(tmp_path):
    with pytest.raises(ValueError, match="format"):
        save_figure(line_figure(), tmp_path / "x", formats=("docx",))


def test_data_round_trip_keeps_units(tmp_path):
    data = {
        "wavelength": np.array([1.0, 1.5]) * u.um,
        "flux": np.array([[1.0, 2.0]]) * u.Jy,
        "counts": np.arange(3),
        "phase": np.exp(1j * np.array([0.1, 0.2])),
    }
    path = save_data(tmp_path / "values", data)
    assert path.suffix == ".npz"
    loaded = load_data(path)
    assert set(loaded) == set(data)
    assert loaded["wavelength"].unit == u.um
    np.testing.assert_array_equal(loaded["wavelength"].value, [1.0, 1.5])
    assert loaded["flux"].unit == u.Jy
    np.testing.assert_array_equal(loaded["counts"], np.arange(3))
    assert not isinstance(loaded["counts"], u.Quantity)
    np.testing.assert_array_equal(loaded["phase"], data["phase"])


def test_data_names_must_be_identifiers(tmp_path):
    with pytest.raises(ValueError, match="name"):
        save_data(tmp_path / "bad", {"__unit__x": np.zeros(2)})
