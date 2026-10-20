"""Compact summaries and figures of signals, for the web interface and AI assistants (step P13.5)."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from astropy import units as u

from lumos.backend import to_numpy
from lumos.service.models import PortSummary
from lumos.service.values import format_value

__all__ = ["figure_of", "figure_png", "job_text", "study_text", "summarize", "summary_text"]

_MAX_VALUES = 16
_MAX_LANE_NAMES = 8


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


def _beam_radius(intensity: np.ndarray, grid: Any) -> str:
    """1/e² radii along x and y from the second moments, ``w = 2 σ`` (exact for a Gaussian beam)."""
    intensity = intensity.reshape(-1, *intensity.shape[-2:]).sum(axis=0)  # wavelengths or modes summed
    total = float(intensity.sum())
    if not total > 0:
        return "undefined (no light)"
    x = grid.x.si.value
    y = grid.y.si.value
    px, py = intensity.sum(axis=0) / total, intensity.sum(axis=1) / total
    sx = np.sqrt(max(float(np.sum(px * (x - np.sum(px * x)) ** 2)), 0.0))
    sy = np.sqrt(max(float(np.sum(py * (y - np.sum(py * y)) ** 2)), 0.0))
    unit = u.mas if grid.plane == "angular" else u.mm
    wx, wy = ((2 * s * (u.rad if grid.plane == "angular" else u.m)).to_value(unit) for s in (sx, sy))
    return f"{wx:.4g}, {wy:.4g} {unit.to_string()}"


def _wavelengths(signal: Any) -> dict[str, str]:
    if hasattr(signal, "wavelengths"):
        waves = signal.wavelengths.to(u.nm)
        if waves.size == 1:
            return {"wavelength": format_value(waves[0])}
        return {
            "wavelengths": f"{waves.size} from {format_value(waves.min())} to {format_value(waves.max())}"
        }
    return {"wavelength": format_value(signal.wavelength.to(u.nm))}


def summarize(port: str, signal: Any, *, figure: bool = False) -> PortSummary:
    """Return the :class:`PortSummary` of ``signal``, with a PNG figure if requested and possible."""
    from lumos.fields import Bus, DataProduct, IncoherentBundle, ModalField, Wavecube, Wavefront

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
        intensity = np.asarray(to_numpy(signal.intensity), dtype=float)
        summary.unit = signal.intensity_unit.to_string()
        summary.shape = list(np.shape(to_numpy(signal.field)))
        summary.statistics = _statistics(intensity)
        summary.details = {
            "grid": _grid_text(signal.grid),
            "power": format_value(signal.power if isinstance(signal, Wavefront) else signal.power.sum()),
            "beam radius": _beam_radius(intensity, signal.grid),
        }
        summary.details.update(_wavelengths(signal))
    elif isinstance(signal, IncoherentBundle):
        intensity = np.asarray(to_numpy(signal.intensity), dtype=float)
        summary.unit = signal.intensity_unit.to_string()
        summary.shape = list(signal.shape)
        summary.statistics = _statistics(intensity)
        summary.details = {
            "grid": _grid_text(signal.grid),
            "modes": str(signal.n_modes),
            "power": format_value(signal.total_power),
            "power per mode": format_value(signal.power),
            "beam radius": _beam_radius(intensity, signal.grid),
        }
        summary.details.update(_wavelengths(signal[0]))
    elif isinstance(signal, ModalField):
        power = np.asarray(to_numpy(np.abs(signal.amplitudes) ** 2), dtype=float)
        per_mode = power.reshape(-1, power.shape[-1]).sum(axis=0)
        summary.unit = signal.power_unit.to_string()
        summary.shape = list(power.shape)
        summary.values = [float(v) for v in per_mode[:_MAX_VALUES]]
        summary.details = {"modes": str(per_mode.size)}
    elif isinstance(signal, Bus):
        summary.shape = [len(signal)]
        summary.details = {
            "lane signal": signal.lane_type.__name__,
            "lanes": f"{len(signal)}: {', '.join(signal.names[:_MAX_LANE_NAMES])}"
            + (" ..." if len(signal) > _MAX_LANE_NAMES else ""),
            "packed": "yes" if signal.packed else f"no ({Bus.packing_problem(signal.lanes)})",
        }
        if signal.lane_type is ModalField:
            # power of each lane, summed over its sources, wavelengths and modes
            powers = [float(np.sum(to_numpy(np.abs(lane.amplitudes) ** 2))) for lane in signal]
            summary.unit = signal[0].power_unit.to_string()
            summary.values = powers[:_MAX_VALUES]
            counts = signal.mode_counts[:_MAX_LANE_NAMES]
            summary.details["modes per lane"] = ", ".join(str(n) for n in counts)
    elif kind == "SymbolicField":
        summary.details = {"latex": signal.latex(), "plane": signal.plane}
        if signal.wavelength is not None:
            summary.details["wavelength"] = format_value(signal.wavelength.to(u.nm))
    else:
        summary.details = {"repr": repr(signal)[:200]}
    if figure:
        summary.figure_png = figure_png(signal)
    return summary


def figure_of(signal: Any, *, dpi: int = 90) -> Any:
    """Return a matplotlib figure of ``signal`` (field intensity, 1-D or 2-D data product), or None."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from lumos.fields import DataProduct, IncoherentBundle, Wavecube, Wavefront
    from lumos.plotting import plot_data, plot_intensity

    fig = Figure(figsize=(5, 4), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    if isinstance(signal, DataProduct) and np.asarray(signal.data.value).ndim in (1, 2):
        plot_data(signal, ax=ax)
    elif isinstance(signal, Wavefront | Wavecube | IncoherentBundle):
        plot_intensity(signal, ax=ax)
    else:
        return None
    fig.tight_layout()
    return fig


def figure_png(signal: Any) -> bytes | None:
    """Draw ``signal`` (intensity of fields, images and curves of data products) as PNG bytes."""
    fig = figure_of(signal)
    if fig is None:
        return None
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    return buffer.getvalue()


def _g(value: float) -> str:
    return f"{value:.4g}"


def summary_text(summary: PortSummary) -> str:
    """One short paragraph describing a :class:`PortSummary`, with units, for a language model."""
    unit = f" {summary.unit}" if summary.unit else ""
    shape = " x ".join(str(n) for n in summary.shape)
    parts = [f"{summary.port}: {summary.kind}" + (f" of shape {shape}" if shape else "")]
    if summary.statistics:
        stats = summary.statistics
        parts.append(
            f"values from {_g(stats['min'])} to {_g(stats['max'])}{unit} (mean {_g(stats['mean'])}{unit})"
        )
    if summary.values is not None and len(summary.values) <= 8:
        parts.append(f"values {[float(_g(v)) for v in summary.values]}{unit}")
    parts.extend(f"{key} {value}" for key, value in summary.details.items() if key != "latex")
    if "latex" in summary.details:
        parts.append(f"expression {summary.details['latex']}")
    text = "; ".join(parts) + "."
    if summary.figure_png is not None:
        text += " The figure follows."
    return text


def job_text(job: Any) -> str:
    """One line describing a run (:class:`~lumos.service.models.JobInfo`) for a language model."""
    text = f"job {job.id} {job.status}: {', '.join(job.outputs)}"
    if job.status in ("queued", "running"):
        text += f" ({job.progress:.0%}, {job.message or 'waiting'}); follow it with get_job"
    if job.seconds is not None:
        text += f" in {job.seconds:.3g} s"
    if job.error:
        text += f". Error: {job.error}. Hint: {job.hint}"
    elif job.status == "done" and getattr(job, "kind", "run") in ("sweep", "sample", "loop"):
        text += f". Read its points with get_study(job='{job.id}')"
    elif job.status == "done":
        text += f". Read the outputs with get_result(job='{job.id}', port=...)"
    if job.archive:
        text += f". Archived in {job.archive}"
    return text + "."


def study_text(study: Any, *, max_rows: int = 50) -> str:
    """Describe a study (:class:`~lumos.service.models.StudyResult`) as a header and a CSV table."""
    from lumos.service.studies import study_table

    kind = {"sweep": "sweep", "loop": "loop"}.get(study.kind, "Monte-Carlo run")
    lines = study_table(study).splitlines()
    text = (
        f"{kind} {study.job}: {len(study.points)} points (seed {study.seed}, noise {study.noise}); "
        "non-scalar outputs are summarised by their min, max, mean and sum.\n"
    )
    shown = lines[: max_rows + 1]
    text += "\n".join(shown)
    if len(lines) > len(shown):
        text += f"\n... {len(lines) - len(shown)} more rows (get_study with a larger max_rows)"
    return text
