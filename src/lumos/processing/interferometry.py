r"""Interferometric processing: null depth, kernel observables, transmission maps (step P11.2, REQ-PIPE-3).

- :func:`null_depth`: :math:`N = P_\mathrm{dark}/P_\mathrm{bright}` of two outputs of a nuller, with
  :math:`\sigma_N^2 = (\sigma_d/P_b)^2 + (P_d\sigma_b/P_b^2)^2`. A two-beam nuller of equal amplitudes and
  phase error :math:`\varphi` has :math:`N = \tan^2(\varphi/2) \approx \varphi^2/4` (Serabyn 2000).
- :func:`kernel_observables`: differences of the paired dark outputs of the kernel nuller
  (:data:`lumos.elements.photonics.KERNEL_PAIRS`), robust to second order in the piston errors (Martinache &
  Ireland 2018, A&A 619, A87); optionally divided by the bright output. Uncertainties add in quadrature.
- :func:`transmission_map`: response of a context to a point source moved over a grid of sky positions, the
  power of one output divided by the total power of all outputs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from astropy import units as u

from lumos.core.element import Element, Values
from lumos.core.param import Setting
from lumos.core.port import Input, Output
from lumos.core.registry import register, registry
from lumos.fields.dataproduct import DataProduct

__all__ = ["KernelObservables", "NullDepth", "kernel_observables", "null_depth", "transmission_map"]


def _channels(outputs: DataProduct) -> tuple[np.ndarray, np.ndarray | None]:
    values = np.asarray(outputs.data.value, dtype=float)
    sigma = (
        None
        if outputs.uncertainty is None
        else np.asarray(outputs.uncertainty.to_value(outputs.unit), dtype=float)
    )
    return values, sigma


def null_depth(outputs: DataProduct, *, dark: int = 0, bright: int = 1) -> DataProduct:
    """Return the null depth ``outputs[dark] / outputs[bright]`` (last axis: the output channels)."""
    values, sigma = _channels(outputs)
    d, b = values[..., dark], values[..., bright]
    depth = np.atleast_1d(d / b)
    uncertainty = None
    if sigma is not None:
        sd, sb = sigma[..., dark], sigma[..., bright]
        uncertainty = np.atleast_1d(np.sqrt((sd / b) ** 2 + (d * sb / b**2) ** 2)) * u.one
    return DataProduct(
        depth * u.one, uncertainty=uncertainty, meta={"dark": int(dark), "bright": int(bright)}
    )


def kernel_observables(
    outputs: DataProduct,
    *,
    pairs: Sequence[tuple[int, int]] | None = None,
    normalize: bool = False,
    bright: int = 0,
) -> DataProduct:
    """Return the kernels, differences of paired outputs (default: those of the LUMOS kernel nuller).

    Parameters
    ----------
    outputs
        Powers of the outputs of the nuller (last axis: channels).
    pairs
        Pairs ``(a, b)`` of outputs whose differences ``a - b`` are the kernels.
    normalize
        Divide the kernels by the bright output ``bright`` (contrast units).
    """
    from lumos.elements.photonics import KERNEL_PAIRS

    pairs = list(pairs or KERNEL_PAIRS)
    values, sigma = _channels(outputs)
    kernels = np.stack([values[..., a] - values[..., b] for a, b in pairs], axis=-1)
    variance = None
    if sigma is not None:
        variance = np.stack([sigma[..., a] ** 2 + sigma[..., b] ** 2 for a, b in pairs], axis=-1)
    unit = outputs.unit
    if normalize:
        reference = values[..., bright][..., None]
        if variance is not None:
            sr = sigma[..., bright][..., None] if sigma is not None else 0.0
            variance = variance / reference**2 + (kernels * sr / reference**2) ** 2
        kernels = kernels / reference
        unit = u.one
    return DataProduct(
        kernels * unit,
        uncertainty=None if variance is None else np.sqrt(variance) * unit,
        meta={"pairs": [list(p) for p in pairs], "normalized": bool(normalize)},
    )


def transmission_map(
    ctx: Any,
    source: Any,
    output: str,
    *,
    theta_x: u.Quantity,
    theta_y: u.Quantity,
    channel: int = 0,
) -> DataProduct:
    """Return the transmission of output ``channel`` for a point source moved over a grid of positions.

    Parameters
    ----------
    ctx
        Context holding the source and the detector.
    source
        Point-like source with ``theta_x`` and ``theta_y`` parameters (moved, then restored).
    output
        Output port (``"id.port"``) of a detector giving the power of every output channel.
    theta_x, theta_y
        Positions of the grid.

    Returns
    -------
    DataProduct
        ``P[channel] / ΣP`` with axes ``theta_y`` and ``theta_x``.
    """
    xs = np.atleast_1d(theta_x)
    ys = np.atleast_1d(theta_y)
    original = (source.theta_x, source.theta_y)
    result = np.zeros((ys.size, xs.size))
    try:
        for i, ty in enumerate(ys):
            for j, tx in enumerate(xs):
                source.theta_x, source.theta_y = tx, ty
                power = np.asarray(ctx.run([output])[output].data.value, dtype=float)
                total = float(power.sum())
                result[i, j] = float(power[..., channel]) / total if total > 0 else math.nan
    finally:
        source.theta_x, source.theta_y = original
    return DataProduct(
        result * u.one, axes={"theta_y": ys, "theta_x": xs}, meta={"output": output, "channel": int(channel)}
    )


@register("lumos.processing.NullDepth")
class NullDepth(Element):
    """Null depth: power of the dark output divided by the power of the bright output."""

    dark = Setting(int, default=0, doc="index of the dark output")
    bright = Setting(int, default=1, doc="index of the bright output")
    in_ = Input(DataProduct)
    out = Output(DataProduct)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the null depth."""
        return {"out": null_depth(inputs["in"], dark=int(values.dark), bright=int(values.bright))}


@register("lumos.processing.KernelObservables", version=2)
class KernelObservables(Element):
    """Kernels of the kernel nuller (differences of paired dark outputs)."""

    normalize = Setting(bool, default=False, doc="divide by the bright output")
    bright = Setting(int, default=0, doc="index of the bright output")
    in_ = Input(DataProduct)
    out = Output(DataProduct)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the kernels."""
        return {
            "out": kernel_observables(
                inputs["in"], normalize=bool(values.normalize), bright=int(values.bright)
            )
        }


@registry.add_migration("lumos.processing.KernelObservables", from_version=1)
def _rename_normalise(body: dict[str, Any]) -> dict[str, Any]:
    """Version 2 spells the setting ``normalize`` (ADR-0021, American spelling of the identifiers)."""
    settings = dict(body.get("settings", {}))
    if "normalise" in settings:
        settings["normalize"] = settings.pop("normalise")
    return {**body, "settings": settings}
