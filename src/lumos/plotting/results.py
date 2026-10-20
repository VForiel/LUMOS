"""Plots of simulation results (step P7.2).

PSF, radial profile, spectra, transmission maps, chip outputs and the graph of a context. Every function
accepts an existing matplotlib ``Axes`` (``ax=``) and returns the axes it drew on.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from astropy import units as u

from lumos.backend import to_numpy
from lumos.fields import DataProduct, Grid, IncoherentBundle, SourceList, Wavecube, Wavefront
from lumos.plotting.style import CMAPS, axis_label, unit_label
from lumos.processing.profiles import radial_profile

__all__ = [
    "plot_chip_outputs",
    "plot_context",
    "plot_psf",
    "plot_radial_profile",
    "plot_spectrum",
    "plot_transmission_map",
    "save_context_diagram",
]


def _axes(ax: Any) -> Any:
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    return ax


def _intensity(signal: Any) -> tuple[np.ndarray, Any]:
    if isinstance(signal, Wavecube):
        return np.asarray(to_numpy(signal.integrated_intensity)), signal.intensity_unit
    if isinstance(signal, Wavefront | IncoherentBundle):
        intensity = np.asarray(to_numpy(signal.intensity))
        if intensity.ndim == 3:  # polychromatic bundle: integrate the channels
            intensity = intensity.sum(axis=0)
        return intensity, signal.intensity_unit
    raise TypeError(f"cannot plot the intensity of a {type(signal).__name__}")


def _value(quantity: Any, unit: Any) -> float:
    return float(quantity.to_value(unit))


def _mean_wavelength(signal: Any) -> float:
    if isinstance(signal, Wavefront):
        return signal.wavelength_si
    if isinstance(signal, Wavecube):
        return float(np.mean(signal.wavelengths_si))
    member = signal[0]
    return _mean_wavelength(member)


def plot_psf(
    signal: Any,
    *,
    ax: Any = None,
    diameter: u.Quantity | None = None,
    focal_length: u.Quantity | None = None,
    log: bool = True,
    dynamic_range: float = 1e6,
    colorbar: bool = True,
    **kwargs: Any,
) -> Any:
    r"""Draw a point-spread function normalised to its peak.

    Parameters
    ----------
    signal
        Field in the focal plane (:class:`~lumos.fields.Wavefront`, :class:`~lumos.fields.Wavecube` or
        :class:`~lumos.fields.IncoherentBundle`).
    diameter
        Pupil diameter :math:`D`: the axes are then in units of :math:`\lambda / D` (mean wavelength of the
        signal). On a spatial grid, ``focal_length`` :math:`f` is needed too, the unit being
        :math:`\lambda f / D`.
    log
        Logarithmic colour scale over ``dynamic_range`` decades below the peak.
    """
    from matplotlib.colors import LogNorm

    ax = _axes(ax)
    data, _ = _intensity(signal)
    peak = float(data.max())
    if peak <= 0:
        raise ValueError("the PSF is zero everywhere")
    data = data / peak
    grid: Grid = signal.grid
    unit_name = unit_label(grid.unit)
    extent = [b.to_value(grid.unit) for b in grid.bounds]
    if diameter is not None:
        scale = _mean_wavelength(signal) / _value(diameter, u.m)
        if grid.plane == "spatial":
            if focal_length is None:
                raise ValueError("a spatial focal plane needs focal_length to be drawn in λ/D units")
            scale *= _value(focal_length, u.m)
            unit_name = "λf/D"
        else:
            unit_name = "λ/D"
        extent = [_value(b, u.m if grid.plane == "spatial" else u.rad) / scale for b in grid.bounds]
    if log:
        kwargs.setdefault("norm", LogNorm(vmin=1 / dynamic_range, vmax=1.0))
    kwargs.setdefault("cmap", CMAPS["intensity"])
    image = ax.imshow(np.maximum(data, 0.0), origin="lower", extent=extent, **kwargs)
    ax.set_xlabel(f"x ({unit_name})")
    ax.set_ylabel(f"y ({unit_name})")
    if colorbar:
        ax.figure.colorbar(image, ax=ax).set_label("normalised intensity")
    return ax


def plot_radial_profile(
    signal: Any,
    *,
    ax: Any = None,
    center: tuple[u.Quantity, u.Quantity] | None = None,
    bin_width: u.Quantity | None = None,
    log: bool = True,
    **kwargs: Any,
) -> Any:
    """Draw the azimuthal average of the intensity of a field (:func:`lumos.processing.radial_profile`)."""
    ax = _axes(ax)
    data, unit = _intensity(signal)
    grid: Grid = signal.grid
    radii, profile = radial_profile(data, grid, center=center, bin_width=bin_width)
    ax.plot(radii.to_value(grid.unit), profile, **kwargs)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel(axis_label("r", grid.unit))
    ax.set_ylabel(axis_label("intensity", unit))
    return ax


def plot_spectrum(signal: Any, *, ax: Any = None, wavelength_unit: Any = u.um, **kwargs: Any) -> Any:
    """Draw spectra against wavelength.

    - :class:`~lumos.fields.SourceList`: the spectral flux density of each source (one line per source);
    - :class:`~lumos.fields.Wavecube`: the power of each channel;
    - :class:`~lumos.fields.DataProduct`: a 1D product whose axis is a wavelength.
    """
    ax = _axes(ax)
    if isinstance(signal, SourceList):
        lam = signal.wavelengths.to_value(wavelength_unit)
        flux = signal.flux
        names = signal.names
        for row, name in zip(np.atleast_2d(flux.value), names, strict=True):
            ax.plot(lam, row, marker=".", label=name, **kwargs)
        ax.set_ylabel(axis_label("flux density", flux.unit))
        if len(names) > 1:
            ax.legend()
    elif isinstance(signal, Wavecube):
        power = signal.power
        ax.plot(signal.wavelengths.to_value(wavelength_unit), power.value, marker=".", **kwargs)
        ax.set_ylabel(axis_label("power", power.unit))
    elif isinstance(signal, DataProduct):
        if len(signal.axis_names) != 1:
            raise ValueError("plot_spectrum draws 1D data products")
        axis = signal.axes[signal.axis_names[0]]
        if not u.Unit(cast(Any, axis.unit)).is_equivalent(u.m, equivalencies=u.spectral()):
            raise ValueError(f"the axis {signal.axis_names[0]!r} is not a wavelength")
        lam = axis.to_value(wavelength_unit, equivalencies=u.spectral())
        ax.plot(lam, np.asarray(signal.data.value), marker=".", **kwargs)
        ax.set_ylabel(axis_label("value", signal.unit))
    else:
        raise TypeError(f"cannot plot the spectrum of a {type(signal).__name__}")
    ax.set_xlabel(axis_label("wavelength", wavelength_unit))
    return ax


def plot_transmission_map(
    product: DataProduct, *, ax: Any = None, colorbar: bool = True, **kwargs: Any
) -> Any:
    """Draw a 2D transmission map (a :class:`~lumos.fields.DataProduct` with two physical axes)."""
    from lumos.plotting import plot_data

    kwargs.setdefault("cmap", CMAPS["data"])
    ax = plot_data(product, ax=ax, colorbar=colorbar, **kwargs)
    if colorbar:
        ax.images[0].colorbar.set_label("transmission")
    return ax


def plot_chip_outputs(
    product: DataProduct, *, ax: Any = None, labels: Sequence[str] | None = None, **kwargs: Any
) -> Any:
    """Draw the power (or current) of each output of a photonic chip as bars."""
    ax = _axes(ax)
    values = np.ravel(np.asarray(product.data.value))
    positions = np.arange(values.size)
    ax.bar(positions, values, **kwargs)
    ax.set_xticks(positions)
    names = list(labels) if labels is not None else [f"out {k}" for k in positions]
    if len(names) != values.size:
        raise ValueError(f"{len(names)} labels for {values.size} outputs")
    ax.set_xticklabels(names)
    ax.set_xlabel("output")
    ax.set_ylabel(axis_label("signal", product.unit))
    return ax


_SIGNAL_COLOURS = {
    "SourceList": "#e6a23c",
    "Wavefront": "#3c78d8",
    "Wavecube": "#6a4cc2",
    "IncoherentBundle": "#2a9d8f",
    "RadianceField": "#d1495b",
    "ModalField": "#00a6c8",
    "Bus": "#00a6c8",
    "DataProduct": "#555555",
}


def _depths(graph: Any) -> dict[int, int]:
    depth: dict[int, int] = {}
    for element in graph.topological_order():
        parents = graph.upstream(element)
        depth[id(element)] = 1 + max((depth[id(p)] for p in parents), default=-1)
    return depth


def plot_context(ctx: Any, *, ax: Any = None) -> Any:
    """Draw the graph of a context: one box per element, arrows from outputs to inputs.

    Elements are placed in columns by their depth in the graph (sources on the left) and the arrows are
    coloured by the signal type of the output port.
    """
    from matplotlib.patches import FancyArrowPatch

    graph = ctx.graph
    depth = _depths(graph)
    columns: dict[int, list[Any]] = {}
    for element in graph.topological_order():
        columns.setdefault(depth[id(element)], []).append(element)
    if ax is None:
        import matplotlib.pyplot as plt

        width = max(3.0, 2.2 * len(columns))
        height = max(2.0, 1.2 * max((len(c) for c in columns.values()), default=1))
        _, ax = plt.subplots(figsize=(width, height))
    position: dict[int, tuple[float, float]] = {}
    for column, elements in columns.items():
        for row, element in enumerate(elements):
            position[id(element)] = (float(column), (len(elements) - 1) / 2 - row)
    for element in graph:
        x, y = position[id(element)]
        ax.text(
            x,
            y,
            f"{graph.id_of(element)}\n{type(element).__name__}",
            ha="center",
            va="center",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f4f4f4", "edgecolor": "#444444"},
        )
    for connection in graph.connections:
        start = position[id(connection.source.owner)]
        end = position[id(connection.target.owner)]
        colour = _SIGNAL_COLOURS.get(connection.source.signal_type.__name__, "#333333")
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            shrinkA=28,
            shrinkB=28,
            color=colour,
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(arrow)
    xs = [p[0] for p in position.values()] or [0.0]
    ys = [p[1] for p in position.values()] or [0.0]
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    ax.set_axis_off()
    title = getattr(ctx, "name", None)
    if title:
        ax.set_title(title)
    return ax


def save_context_diagram(ctx: Any, path: str | Path) -> Path:
    """Save the diagram of the graph of ``ctx`` (format from the suffix: ``.svg``, ``.png``, ``.pdf``...)."""
    import matplotlib.pyplot as plt

    path = Path(path)
    ax = plot_context(ctx)
    ax.figure.savefig(path)
    plt.close(ax.figure)
    return path
