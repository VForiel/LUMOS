"""Homogeneous, publication-ready figure style (step P7.1, REQ-RES-3).

Every LUMOS figure follows the same conventions:

- **colour maps by role** (:data:`CMAPS`): perceptually uniform sequential maps for intensities, radiances and
  data (equal steps of value look like equal steps of colour, and the maps stay readable in grey scale and
  for colour-blind readers), a **cyclic** map for phases (``-π`` and ``π`` share a colour, so the wrap-around
  is invisible), a diverging map for signed quantities (residuals, kernels);
- **axis labels with units**, written ``"quantity (unit)"`` with astropy's unicode unit format
  (:func:`axis_label`);
- **publication sizes** (:func:`figure`): one column (88 mm) or the full page width (180 mm) of the
  A&A layout, close to that of most journals;
- **matplotlib settings** (:data:`STYLE`): readable font sizes at these widths, 300 dpi raster output and
  images drawn with the origin at the bottom left (§9 of the specifications).

Use :func:`context` to apply the settings to a block, or :func:`apply` to apply them for the whole session::

    from lumos.plotting import style

    with style.context():
        fig, ax = style.figure("single")
        ax.set_xlabel(style.axis_label("x", u.um))
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from astropy import units as u

__all__ = ["CMAPS", "FIGURE_WIDTHS", "STYLE", "apply", "axis_label", "context", "figure", "unit_label"]

#: Colour map of each role.
CMAPS: dict[str, str] = {
    "intensity": "inferno",
    "radiance": "magma",
    "data": "viridis",
    "phase": "twilight",
    "diverging": "RdBu_r",
}

_MM = 1 / 25.4

#: Figure widths in inches: one column and the full page width of the A&A layout.
FIGURE_WIDTHS: dict[str, float] = {"single": 88 * _MM, "double": 180 * _MM}

#: matplotlib settings of the LUMOS style.
STYLE: dict[str, Any] = {
    "figure.dpi": 100,
    "figure.constrained_layout.use": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.grid": False,
    "image.origin": "lower",
    "image.cmap": CMAPS["data"],
    "image.interpolation": "nearest",
}


def unit_label(unit: Any) -> str:
    """Return the unicode representation of ``unit`` (``"dimensionless"`` for dimensionless quantities)."""
    text = u.Unit(unit).to_string("unicode")
    return text if text else "dimensionless"


def axis_label(name: str, unit: Any = None) -> str:
    """Return the label ``"name (unit)"`` of an axis; dimensionless quantities get their name only.

    Parameters
    ----------
    name
        Name of the quantity, e.g. ``"x"`` or ``"wavelength"``.
    unit
        Unit, quantity (its unit is used) or ``None``.
    """
    if unit is None:
        return name
    if isinstance(unit, u.Quantity):
        unit = unit.unit
    unit = u.Unit(unit)
    if unit == u.dimensionless_unscaled:
        return name
    return f"{name} ({unit_label(unit)})"


@contextmanager
def context() -> Iterator[None]:
    """Apply the LUMOS style inside a ``with`` block (the previous settings are restored afterwards)."""
    import matplotlib as mpl

    with mpl.rc_context(STYLE):
        yield


def apply() -> None:
    """Apply the LUMOS style for the rest of the session."""
    import matplotlib as mpl

    mpl.rcParams.update(STYLE)


def figure(
    size: str = "single", *, nrows: int = 1, ncols: int = 1, aspect: float = 0.75, **kwargs: Any
) -> tuple[Any, Any]:
    """Create a figure of a publication width and its axes.

    Parameters
    ----------
    size
        ``"single"`` (one column, 88 mm) or ``"double"`` (full page width, 180 mm).
    nrows, ncols
        Grid of axes (a single ``Axes`` is returned for 1 × 1, an array otherwise).
    aspect
        Height of the figure divided by its width.
    **kwargs
        Passed to :func:`matplotlib.pyplot.subplots`.
    """
    import matplotlib.pyplot as plt

    if size not in FIGURE_WIDTHS:
        raise ValueError(f"unknown figure size {size!r}; use one of {', '.join(FIGURE_WIDTHS)}")
    width = FIGURE_WIDTHS[size]
    return plt.subplots(nrows, ncols, figsize=(width, aspect * width), **kwargs)
