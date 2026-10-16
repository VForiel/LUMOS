"""Basic plots of fields and data (step P3.9).

Every map is drawn with ``origin="lower"`` and an extent equal to the outer pixel edges of its grid, so the
axes show physical coordinates with x to the right and y upwards (§9 of the specifications). Phases are drawn
with the cyclic ``twilight`` colour map over ``[-π, π]``, so that the wrap-around at ±π is invisible.

Each function accepts an existing matplotlib ``Axes`` and returns the axes it drew on (or the figure for
multi-panel plots).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from astropy import units as u

from lumos.backend import to_numpy
from lumos.fields import DataProduct, Grid, IncoherentBundle, RadianceField, Wavecube, Wavefront

__all__ = ["PHASE_CMAP", "plot_data", "plot_field", "plot_intensity", "plot_phase", "plot_radiance"]

#: Cyclic colour map used for phases.
PHASE_CMAP = "twilight"


def _axes(ax: Any) -> Any:
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    return ax


def _unit_label(unit: Any) -> str:
    text = u.Unit(unit).to_string()
    return text if text else "dimensionless"


def _image(ax: Any, data: np.ndarray, grid: Grid, *, colorbar_label: str | None, **kwargs: Any) -> Any:
    unit = grid.unit
    extent = [b.to_value(unit) for b in grid.bounds]
    im = ax.imshow(data, origin="lower", extent=extent, **kwargs)
    ax.set_xlabel(f"x ({_unit_label(unit)})")
    ax.set_ylabel(f"y ({_unit_label(unit)})")
    if colorbar_label is not None:
        colorbar = ax.figure.colorbar(im, ax=ax)
        colorbar.set_label(colorbar_label)
    return im


def _intensity_of(signal: Any) -> tuple[np.ndarray, Any]:
    if isinstance(signal, Wavecube):
        return np.asarray(to_numpy(signal.integrated_intensity)), signal.intensity_unit
    if isinstance(signal, Wavefront | IncoherentBundle):
        return np.asarray(to_numpy(signal.intensity)), signal.intensity_unit
    raise TypeError(f"cannot plot the intensity of a {type(signal).__name__}")


def plot_intensity(
    signal: Any, *, ax: Any = None, log: bool = False, colorbar: bool = True, **kwargs: Any
) -> Any:
    """Draw the intensity of a wavefront, wavecube (spectrally integrated) or incoherent bundle.

    Parameters
    ----------
    signal
        :class:`~lumos.fields.Wavefront`, :class:`~lumos.fields.Wavecube` or
        :class:`~lumos.fields.IncoherentBundle`.
    ax
        Axes to draw on (a new figure by default).
    log
        Logarithmic colour scale.
    colorbar
        Add a colour bar labelled with the intensity unit.
    **kwargs
        Passed to ``imshow``.
    """
    from matplotlib.colors import LogNorm

    ax = _axes(ax)
    data, unit = _intensity_of(signal)
    if log:
        positive = data[data > 0]
        vmin = float(positive.min()) if positive.size else 1e-300
        kwargs.setdefault("norm", LogNorm(vmin=max(vmin, float(data.max()) * 1e-12), vmax=float(data.max())))
    kwargs.setdefault("cmap", "inferno")
    _image(ax, data, signal.grid, colorbar_label=_unit_label(unit) if colorbar else None, **kwargs)
    return ax


def _phase_image(
    ax: Any, field: np.ndarray, grid: Grid, *, threshold: float | None, colorbar: bool, **kwargs: Any
) -> Any:
    phase = np.angle(field)
    amplitude = np.abs(field)
    if threshold is not None:
        phase = np.ma.masked_where(amplitude < threshold * amplitude.max(), phase)
    kwargs.setdefault("cmap", PHASE_CMAP)
    kwargs.setdefault("vmin", -math.pi)
    kwargs.setdefault("vmax", math.pi)
    return _image(ax, phase, grid, colorbar_label="phase (rad)" if colorbar else None, **kwargs)


def plot_phase(
    signal: Wavefront,
    *,
    ax: Any = None,
    component: int = 0,
    threshold: float | None = None,
    colorbar: bool = True,
    **kwargs: Any,
) -> Any:
    """Draw the phase of a wavefront in ``[-π, π]`` with a cyclic colour map.

    Parameters
    ----------
    signal
        Scalar or polarised wavefront (``component`` selects Ex = 0 or Ey = 1).
    threshold
        Mask the phase where the amplitude is below ``threshold × max amplitude``.
    """
    ax = _axes(ax)
    field = np.asarray(to_numpy(signal.field))
    if signal.polarized:
        field = field[component]
    _phase_image(ax, field, signal.grid, threshold=threshold, colorbar=colorbar, **kwargs)
    return ax


def plot_field(signal: Wavefront, *, threshold: float | None = 1e-3) -> Any:
    """Draw intensity and phase side by side (one row per Jones component for a polarised field)."""
    import matplotlib.pyplot as plt

    field = np.asarray(to_numpy(signal.field))
    components = list(field) if signal.polarized else [field]
    fig, axes = plt.subplots(len(components), 2, figsize=(9, 3.6 * len(components)), squeeze=False)
    names = ["Ex", "Ey"] if signal.polarized else [""]
    for row, (component, name) in enumerate(zip(components, names, strict=True)):
        intensity = np.abs(component) ** 2
        _image(
            axes[row, 0],
            intensity,
            signal.grid,
            colorbar_label=_unit_label(signal.intensity_unit),
            cmap="inferno",
        )
        axes[row, 0].set_title(f"intensity {name}".strip())
        _phase_image(axes[row, 1], component, signal.grid, threshold=threshold, colorbar=True)
        axes[row, 1].set_title(f"phase {name}".strip())
    fig.tight_layout()
    return fig


def plot_radiance(
    field: RadianceField,
    *,
    ax: Any = None,
    channel: int = 0,
    stokes: int = 0,
    colorbar: bool = True,
    **kwargs: Any,
) -> Any:
    """Draw one spectral channel (and Stokes parameter) of a radiance field."""
    ax = _axes(ax)
    data = np.asarray(to_numpy(field.data))[channel]
    if field.has_stokes:
        data = data[stokes]
    kwargs.setdefault("cmap", "magma")
    _image(ax, data, field.grid, colorbar_label=_unit_label(field.unit) if colorbar else None, **kwargs)
    ax.set_title(f"λ = {field.wavelengths[channel]:.4g}")
    return ax


def plot_data(product: DataProduct, *, ax: Any = None, colorbar: bool = True, **kwargs: Any) -> Any:
    """Draw a 1D data product as a line or a 2D one as an image with its physical axes."""
    ax = _axes(ax)
    names = product.axis_names
    values = np.asarray(product.data.value)
    if values.ndim == 1:
        axis = product.axes[names[0]]
        ax.plot(axis.value, values, **kwargs)
        ax.set_xlabel(f"{names[0]} ({_unit_label(axis.unit)})")
        ax.set_ylabel(_unit_label(product.unit))
        return ax
    if values.ndim != 2:
        raise ValueError("plot_data draws 1D and 2D products; select a slice of higher-dimensional data")
    y, x = (product.axes[name] for name in names)

    def edges(c: np.ndarray) -> tuple[float, float]:
        step = c[1] - c[0] if c.size > 1 else 1.0
        return float(c[0] - step / 2), float(c[-1] + step / 2)

    extent = [*edges(np.asarray(x.value)), *edges(np.asarray(y.value))]
    kwargs.setdefault("cmap", "gray")
    im = ax.imshow(values, origin="lower", extent=extent, **kwargs)
    ax.set_xlabel(f"{names[1]} ({_unit_label(x.unit)})")
    ax.set_ylabel(f"{names[0]} ({_unit_label(y.unit)})")
    if colorbar:
        ax.figure.colorbar(im, ax=ax).set_label(_unit_label(product.unit))
    return ax
