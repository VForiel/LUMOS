r"""Reduction of detected data: dark, flat field, stacking, cropping and resampling (step P11.1, REQ-PIPE-2).

Every function takes and returns :class:`~lumos.fields.DataProduct` objects and propagates their
uncertainties for independent errors (JCGM 100:2008, *GUM*, §5.1):

- :func:`subtract_dark`: :math:`S - D`, :math:`\sigma^2 = \sigma_S^2 + \sigma_D^2`;
- :func:`flat_field`: :math:`S / F`, :math:`\sigma^2 = (\sigma_S/F)^2 + (S\sigma_F/F^2)^2`, the flat being
  first divided by its mean (``normalize=True``) so that the mean level is kept;
- :func:`stack`: mean (:math:`\sigma/\sqrt N`), median (:math:`\sqrt{\pi/2}\,\sigma/\sqrt N`, the
  efficiency of the median of normal samples, Kenney & Keeping 1962) or sigma-clipped mean, the clipping
  using the median and the normalised median absolute deviation :math:`1.4826\,\mathrm{MAD}`. Without
  per-frame uncertainties, :math:`\sigma` is the scatter of the frames;
- :func:`crop` and :func:`rebin` (sums of blocks, uncertainties added in quadrature);
- :func:`resample`: flux-conserving resampling on new pixel centres, each old pixel shared among the new ones
  in proportion to their overlap (separable in x and y); the variances follow the squared weights.

The last two axes of images are ``y`` and ``x``. The elements at the end of the module wrap these functions
with :class:`~lumos.fields.DataProduct` ports (REQ-PIPE-1).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from astropy import units as u

from lumos.core.element import Element, Values
from lumos.core.param import Setting
from lumos.core.port import Input, Output
from lumos.core.registry import register, registry
from lumos.fields.dataproduct import DataProduct
from lumos.processing.settings import QuantitySetting

__all__ = [
    "Crop",
    "DarkSubtraction",
    "FlatField",
    "Rebin",
    "Stack",
    "crop",
    "flat_field",
    "rebin",
    "resample",
    "stack",
    "subtract_dark",
]

_MEDIAN_EFFICIENCY = math.sqrt(math.pi / 2)
_MAD_TO_SIGMA = 1.4826


def _values(product: DataProduct) -> np.ndarray:
    return np.asarray(product.data.value, dtype=float)


def _sigma(product: DataProduct, unit: Any = None) -> np.ndarray | None:
    if product.uncertainty is None:
        return None
    return np.asarray(product.uncertainty.to_value(unit or product.unit), dtype=float)


def subtract_dark(frame: DataProduct, dark: DataProduct) -> DataProduct:
    """Return ``frame - dark`` with uncertainties added in quadrature and masks combined."""
    if dark.unit != frame.unit:
        dark = DataProduct(
            dark.data.to(frame.unit),
            axes=dark.axes,
            mask=dark.mask,
            uncertainty=None if dark.uncertainty is None else dark.uncertainty.to(frame.unit),
        )
    return frame - DataProduct(dark.data, axes=frame.axes, mask=dark.mask, uncertainty=dark.uncertainty)


def flat_field(frame: DataProduct, flat: DataProduct, *, normalize: bool = True) -> DataProduct:
    """Divide ``frame`` by ``flat`` (normalised to a mean of 1 by default), propagating the uncertainties."""
    f = _values(flat)
    sigma_f = _sigma(flat)
    unit = u.dimensionless_unscaled
    if normalize:
        valid = np.ones(f.shape, bool) if flat.mask is None else ~flat.mask
        level = float(np.mean(f[valid]))
        f = f / level
        sigma_f = None if sigma_f is None else sigma_f / level
    else:
        unit = flat.unit
    s = _values(frame)
    sigma_s = _sigma(frame)
    with np.errstate(divide="ignore", invalid="ignore"):
        data = s / f
        variance = np.zeros_like(data)
        if sigma_s is not None:
            variance = variance + (sigma_s / f) ** 2
        if sigma_f is not None:
            variance = variance + (s * sigma_f / f**2) ** 2
    bad = ~np.isfinite(data) | (f == 0)
    masks = [m for m in (frame.mask, flat.mask, bad if bad.any() else None) if m is not None]
    out_unit = frame.unit / unit
    uncertainty = None if sigma_s is None and sigma_f is None else u.Quantity(np.sqrt(variance), out_unit)
    return DataProduct(
        u.Quantity(np.where(bad, 0.0, data), out_unit),
        axes=frame.axes,
        mask=np.logical_or.reduce(masks) if masks else None,
        uncertainty=uncertainty,
        meta=frame.meta,
    )


def _as_cube(frames: DataProduct | Sequence[DataProduct]) -> tuple[np.ndarray, np.ndarray | None, Any, Any]:
    """Return values (N, ...), uncertainties (N, ...) or None, the unit and the axes of one frame."""
    if isinstance(frames, DataProduct):
        names = frames.axis_names
        axes = {name: frames.axes[name] for name in names[1:]}
        return _values(frames), _sigma(frames), frames.unit, axes
    frames = list(frames)
    if not frames:
        raise ValueError("no frame to stack")
    unit = frames[0].unit
    values = np.stack([np.asarray(f.data.to_value(unit), dtype=float) for f in frames])
    sigmas = [_sigma(f, unit) for f in frames]
    sigma = None if any(s is None for s in sigmas) else np.stack(cast(list[np.ndarray], sigmas))
    return values, sigma, unit, frames[0].axes


def _clip(values: np.ndarray, sigma: float, iterations: int = 5) -> np.ndarray:
    keep = np.ones(values.shape, bool)
    for _ in range(iterations):
        masked = np.where(keep, values, np.nan)
        median = np.nanmedian(masked, axis=0)
        spread = _MAD_TO_SIGMA * np.nanmedian(np.abs(masked - median), axis=0)
        spread = np.where(spread > 0, spread, np.inf)
        new = np.abs(values - median) <= sigma * spread
        if np.array_equal(new, keep):
            break
        keep = new
    return keep


def stack(
    frames: DataProduct | Sequence[DataProduct], *, method: str = "mean", sigma: float = 3.0
) -> DataProduct:
    """Combine frames (a list, or a product whose first axis lists the frames).

    Parameters
    ----------
    frames
        Frames with the same axes and units.
    method
        ``"mean"``, ``"median"`` or ``"sigma_clip"`` (mean of the samples within ``sigma`` robust standard
        deviations of the median).
    sigma
        Clipping threshold of ``"sigma_clip"``.
    """
    values, sigmas, unit, axes = _as_cube(frames)
    n = values.shape[0]
    if method == "mean":
        keep = np.ones(values.shape, bool)
    elif method == "median":
        keep = None
    elif method == "sigma_clip":
        keep = _clip(values, float(sigma))
    else:
        raise ValueError(f"unknown stacking method {method!r}: use 'mean', 'median' or 'sigma_clip'")
    if keep is None:
        data = np.median(values, axis=0)
        per_frame = (
            np.sqrt(np.mean(sigmas**2, axis=0)) if sigmas is not None else np.std(values, axis=0, ddof=1)
        )
        uncertainty = _MEDIAN_EFFICIENCY * per_frame / math.sqrt(n)
    else:
        count = keep.sum(axis=0)
        safe = np.maximum(count, 1)
        data = np.where(keep, values, 0.0).sum(axis=0) / safe
        if sigmas is not None:
            uncertainty = np.sqrt(np.where(keep, sigmas**2, 0.0).sum(axis=0)) / safe
        else:
            deviation = np.where(keep, (values - data) ** 2, 0.0).sum(axis=0) / np.maximum(count - 1, 1)
            uncertainty = np.sqrt(deviation) / np.sqrt(safe)
    meta = {"stacked_frames": int(n), "stack_method": method}
    return DataProduct(data * unit, axes=axes, uncertainty=np.asarray(uncertainty) * unit, meta=meta)


# --- Geometry -----------------------------------------------------------------------------------------------


def _image_axes(product: DataProduct) -> tuple[str, str]:
    names = product.axis_names
    if len(names) < 2:
        raise ValueError("images need two axes (y, x)")
    return names[-2], names[-1]


def crop(
    product: DataProduct, *, center: tuple[u.Quantity, u.Quantity], size: tuple[int, int]
) -> DataProduct:
    """Return the ``size = (nx, ny)`` pixels around the pixel nearest to ``center = (x, y)``.

    The centre pixel is at index ``n // 2`` of the cropped axes, as in every LUMOS grid.
    """
    y_name, x_name = _image_axes(product)
    axes = product.axes
    slices: list[slice] = [slice(None)] * len(product.shape)
    for position, name, n, dim in ((center[0], x_name, size[0], -1), (center[1], y_name, size[1], -2)):
        coordinates: Any = axes[name]
        index = int(np.argmin(np.abs(np.asarray((coordinates - position).to_value(coordinates.unit)))))
        start = index - int(n) // 2
        if start < 0 or start + int(n) > coordinates.size:
            raise ValueError(f"the cropped region leaves the image along {name}")
        slices[dim] = slice(start, start + int(n))
        axes[name] = coordinates[slices[dim]]
    region = tuple(slices)
    return DataProduct(
        cast(u.Quantity, cast(Any, product.data)[region]),
        axes=axes,
        mask=None if product.mask is None else product.mask[region],
        uncertainty=None
        if product.uncertainty is None
        else cast(u.Quantity, cast(Any, product.uncertainty)[region]),
        meta=product.meta,
    )


def _block_sum(array: np.ndarray, factor: int) -> np.ndarray:
    *lead, ny, nx = array.shape
    return array.reshape(*lead, ny // factor, factor, nx // factor, factor).sum(axis=(-3, -1))


def rebin(product: DataProduct, factor: int) -> DataProduct:
    """Sum blocks of ``factor`` × ``factor`` pixels (flux conserved, uncertainties added in quadrature)."""
    factor = int(factor)
    y_name, x_name = _image_axes(product)
    ny, nx = product.shape[-2:]
    if factor < 1 or ny % factor or nx % factor:
        raise ValueError(f"the image shape {(ny, nx)} is not a multiple of the factor {factor}")
    axes = product.axes
    for name in (y_name, x_name):
        coordinates: Any = axes[name]
        axes[name] = coordinates.reshape(-1, factor).mean(axis=1)
    sigma = _sigma(product)
    mask = None if product.mask is None else _block_sum(product.mask.astype(int), factor) > 0
    return DataProduct(
        _block_sum(_values(product), factor) * product.unit,
        axes=axes,
        mask=mask,
        uncertainty=None if sigma is None else np.sqrt(_block_sum(sigma**2, factor)) * product.unit,
        meta=product.meta,
    )


def _edges(centres: np.ndarray) -> np.ndarray:
    if centres.size == 1:
        raise ValueError("an axis of one pixel has no pixel size")
    middle = (centres[1:] + centres[:-1]) / 2
    return np.concatenate([[2 * centres[0] - middle[0]], middle, [2 * centres[-1] - middle[-1]]])


def _overlap(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Fraction of each old pixel (columns) falling in each new pixel (rows); edges may decrease."""
    old_lo, old_hi = np.minimum(old[:-1], old[1:]), np.maximum(old[:-1], old[1:])
    new_lo, new_hi = np.minimum(new[:-1], new[1:]), np.maximum(new[:-1], new[1:])
    width = np.clip(
        np.minimum(new_hi[:, None], old_hi[None, :]) - np.maximum(new_lo[:, None], old_lo[None, :]), 0, None
    )
    return width / (old_hi - old_lo)[None, :]


def resample(product: DataProduct, *, x: u.Quantity, y: u.Quantity) -> DataProduct:
    """Resample an image of counts on new pixel centres ``x`` and ``y``, conserving the flux."""
    y_name, x_name = _image_axes(product)
    axes = product.axes
    unit_x, unit_y = axes[x_name].unit, axes[y_name].unit
    wx = _overlap(_edges(axes[x_name].value), _edges(np.asarray(x.to_value(unit_x))))
    wy = _overlap(_edges(axes[y_name].value), _edges(np.asarray(y.to_value(unit_y))))
    values = _values(product)
    if product.mask is not None:
        values = np.where(product.mask, 0.0, values)
    data = np.einsum("ij,...jk,lk->...il", wy, values, wx)
    sigma = _sigma(product)
    uncertainty = None
    if sigma is not None:
        uncertainty = np.sqrt(np.einsum("ij,...jk,lk->...il", wy**2, sigma**2, wx**2)) * product.unit
    axes[x_name], axes[y_name] = x.to(unit_x), y.to(unit_y)
    return DataProduct(data * product.unit, axes=axes, uncertainty=uncertainty, meta=product.meta)


# --- Elements -----------------------------------------------------------------------------------------------


@register("lumos.processing.DarkSubtraction")
class DarkSubtraction(Element):
    """Subtract a dark frame (input ``dark``) from the science frame (input ``in``)."""

    in_ = Input(DataProduct)
    dark = Input(DataProduct)
    out = Output(DataProduct)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return ``in - dark``."""
        return {"out": subtract_dark(inputs["in"], inputs["dark"])}


@register("lumos.processing.FlatField", version=2)
class FlatField(Element):
    """Divide the science frame by a flat field (normalised to a mean of 1 by default)."""

    normalize = Setting(bool, default=True, doc="divide the flat by its mean first")
    in_ = Input(DataProduct)
    flat = Input(DataProduct)
    out = Output(DataProduct)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return ``in / flat``."""
        return {"out": flat_field(inputs["in"], inputs["flat"], normalize=bool(values.normalize))}


@register("lumos.processing.Stack")
class Stack(Element):
    """Combine the frames connected to ``in`` (or the first axis of a single cube)."""

    method = Setting(
        str, default="mean", choices=("mean", "median", "sigma_clip"), doc="'mean', 'median' or 'sigma_clip'"
    )
    sigma = Setting(float, default=3.0, doc="clipping threshold of sigma_clip")
    in_ = Input(DataProduct, multiple=True)
    out = Output(DataProduct)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the stacked frame."""
        frames = list(inputs["in"])
        source: Any = frames[0] if len(frames) == 1 else frames
        return {"out": stack(source, method=str(values.method), sigma=float(values.sigma))}


@register("lumos.processing.Rebin")
class Rebin(Element):
    """Sum blocks of ``factor`` × ``factor`` pixels."""

    factor = Setting(int, default=2, doc="pixels summed per output pixel along each axis")
    in_ = Input(DataProduct)
    out = Output(DataProduct)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the binned image."""
        return {"out": rebin(inputs["in"], int(values.factor))}


def _on_axis(value: u.Quantity, axis: u.Quantity) -> u.Quantity:
    """Express ``value`` in the unit of ``axis``; zero (the default centre, ``0 pix``) fits every unit."""
    if value.value == 0:
        return u.Quantity(0.0, axis.unit)
    return value.to(axis.unit)


@register("lumos.processing.Crop")
class Crop(Element):
    """Keep ``nx`` × ``ny`` pixels around the pixel nearest to (``x``, ``y``)."""

    nx = Setting(int, default=64, doc="pixels kept along x")
    ny = Setting(int, default=64, doc="pixels kept along y")
    x = QuantitySetting(default=0 * u.pix, doc="centre along x (in a unit of the x axis)")
    y = QuantitySetting(default=0 * u.pix, doc="centre along y (in a unit of the y axis)")
    in_ = Input(DataProduct)
    out = Output(DataProduct)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the cropped image."""
        product = inputs["in"]
        y_name, x_name = _image_axes(product)
        cx, cy = _on_axis(values.x, product.axes[x_name]), _on_axis(values.y, product.axes[y_name])
        return {"out": crop(product, center=(cx, cy), size=(int(values.nx), int(values.ny)))}

@registry.add_migration("lumos.processing.FlatField", from_version=1)
def _rename_normalise(body: dict[str, Any]) -> dict[str, Any]:
    """Version 2 spells the setting ``normalize`` (ADR-0021, American spelling of the identifiers)."""
    settings = dict(body.get("settings", {}))
    if "normalise" in settings:
        settings["normalize"] = settings.pop("normalise")
    return {**body, "settings": settings}
