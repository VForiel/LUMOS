"""Compact summaries and figures of signals, for the web interface and AI assistants (step P13.5)."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from astropy import units as u

from lumos.backend import to_numpy
from lumos.service.models import PortSummary
from lumos.service.values import format_value

__all__ = ["figure_png", "summarize"]

_MAX_VALUES = 16


def _statistics(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)] if values.size else values
    if finite.size == 0:
        return {}
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "sum": float(finite.sum()),
    }


def _grid_text(grid: Any) -> str:
    unit = "rad" if grid.plane == "angular" else "m"
    return f"{grid.nx} x {grid.ny} pixels of {format_value(grid.dx_si * u.Unit(unit))}"


def summarize(port: str, signal: Any, *, figure: bool = False) -> PortSummary:
    """Return the :class:`PortSummary` of ``signal``, with a PNG figure if requested and possible."""
    from lumos.fields import DataProduct, ModalField, Wavecube, Wavefront

    kind = type(signal).__name__
    summary = PortSummary(port=port, kind=kind)
    if isinstance(signal, DataProduct):
        values = np.asarray(signal.data.value, dtype=float)
        summary.unit = signal.unit.to_string()
        summary.shape = list(values.shape)
        summary.statistics = _statistics(values)
        if values.size <= _MAX_VALUES:
            summary.values = [float(v) for v in values.ravel()]
        summary.details = {"axes": ", ".join(signal.axis_names)}
        if signal.uncertainty is not None:
            summary.details["uncertainty"] = "per sample"
    elif isinstance(signal, Wavefront | Wavecube):
        field = np.asarray(to_numpy(signal.field))
        intensity = np.abs(field) ** 2
        summary.unit = f"{signal.power_unit.to_string()} / m2"
        summary.shape = list(field.shape)
        summary.statistics = _statistics(intensity)
        summary.details = {
            "grid": _grid_text(signal.grid),
            "power": format_value(signal.power if isinstance(signal, Wavefront) else signal.power.sum()),
        }
        if isinstance(signal, Wavefront):
            summary.details["wavelength"] = format_value(signal.wavelength.to(u.nm))
        else:
            summary.details["wavelengths"] = format_value(signal.wavelengths.to(u.nm))
    elif isinstance(signal, ModalField):
        power = np.asarray(to_numpy(np.abs(signal.amplitudes) ** 2), dtype=float)
        per_channel = power.reshape(-1, power.shape[-1]).sum(axis=0)
        summary.unit = signal.power_unit.to_string()
        summary.shape = list(power.shape)
        summary.values = [float(v) for v in per_channel[:_MAX_VALUES]]
        summary.details = {"channels": str(per_channel.size)}
    elif kind == "SymbolicField":
        summary.details = {"latex": signal.latex(), "plane": signal.plane}
        if signal.wavelength is not None:
            summary.details["wavelength"] = format_value(signal.wavelength.to(u.nm))
    else:
        summary.details = {"repr": repr(signal)[:200]}
    if figure:
        summary.figure_png = figure_png(signal)
    return summary


def figure_png(signal: Any) -> bytes | None:
    """Draw ``signal`` (intensity of fields, images and curves of data products) as PNG bytes."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from lumos.fields import DataProduct, IncoherentBundle, Wavecube, Wavefront
    from lumos.plotting import plot_data, plot_intensity

    fig = Figure(figsize=(5, 4), dpi=90)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    if isinstance(signal, DataProduct) and np.asarray(signal.data.value).ndim in (1, 2):
        plot_data(signal, ax=ax)
    elif isinstance(signal, Wavefront | Wavecube | IncoherentBundle):
        plot_intensity(signal, ax=ax)
    else:
        return None
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    return buffer.getvalue()
