r"""Photometry, signal-to-noise ratio and contrast curves (step P11.1, REQ-PIPE-2).

- :func:`aperture_photometry`: sum of an image over a circular aperture, each pixel weighted by its **exact**
  overlap with the disk (analytic area of the intersection of a disk and a rectangle), minus the median
  background of an optional annulus. The variance of the flux adds the pixel variances (weighted) and the
  error of the background level, :math:`A^2\,\sigma_b^2/n_b` (:math:`A` aperture area, :math:`n_b` annulus
  area in pixels).
- :func:`ccd_snr`: the CCD equation :math:`S/\sqrt{S + n(B + D + R^2)}` (Howell 2006, *Handbook of CCD
  Astronomy*, eq. 4.4), all terms in electrons over the exposure.
- :func:`contrast_curve`: detection limit in contrast versus separation, for resolution elements of size
  ``resolution`` (λ/D in pixels or in the unit of the image axes). At separation :math:`r` the noise
  :math:`\sigma_r` is the standard deviation of the pixels of the annulus :math:`[r - \lambda/D/2,
  r + \lambda/D/2)`; the threshold includes the small-sample penalty of Mawet et al. (2014, ApJ 792, 97,
  eq. 9): with :math:`n = \lfloor 2\pi r/(\lambda/D) \rfloor` resolution elements, the threshold is
  :math:`\sigma_r\,t_{1-f,\,n-1}\sqrt{1 + 1/n}`, where the false positive fraction :math:`f` is that of a
  Gaussian threshold of ``sigma`` standard deviations. The contrast is the threshold divided by the peak of
  the off-axis PSF ``peak``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy import units as u

from lumos.core.element import Element, Values
from lumos.core.param import Setting
from lumos.core.port import Input, Output
from lumos.core.registry import register
from lumos.fields.dataproduct import DataProduct
from lumos.processing.settings import QuantitySetting

__all__ = [
    "AperturePhotometry",
    "ApertureResult",
    "ContrastCurve",
    "RadialProfile",
    "SignalToNoise",
    "aperture_photometry",
    "ccd_snr",
    "circle_overlap",
    "contrast_curve",
]


def _f(quantity: Any, unit: Any) -> float:
    return float(np.asarray(quantity.to_value(unit)))


def _a(quantity: Any, unit: Any) -> np.ndarray:
    return np.asarray(quantity.to_value(unit), dtype=float)


def _primitive(x: np.ndarray, radius: float) -> np.ndarray:
    """∫ √(R² - x²) dx."""
    x = np.clip(x, -radius, radius)
    return 0.5 * (x * np.sqrt(np.maximum(radius**2 - x**2, 0.0)) + radius**2 * np.arcsin(x / radius))


def _quadrant(a: np.ndarray, b: np.ndarray, radius: float) -> np.ndarray:
    """Area of the disk of ``radius`` (centred on 0) with ``x >= a`` and ``y >= b``."""
    a = np.clip(a, -radius, radius)
    c = np.sqrt(np.maximum(radius**2 - b**2, 0.0))
    c = np.where(b >= radius, 0.0, c)
    lo, hi = np.maximum(a, -c), c
    # where |x| < c the vertical extent is h(x) - b; for b < 0 and |x| > c it is 2 h(x)
    inner = np.where(hi > lo, (_primitive(hi, radius) - b * hi) - (_primitive(lo, radius) - b * lo), 0.0)
    negative = b < 0
    left = np.where(negative & (a < -c), 2 * (_primitive(-c, radius) - _primitive(a, radius)), 0.0)
    start = np.maximum(a, c)
    right = np.where(
        negative, 2 * (_primitive(np.full_like(start, radius), radius) - _primitive(start, radius)), 0.0
    )
    return inner + left + right


def circle_overlap(
    x_edges: np.ndarray, y_edges: np.ndarray, center: tuple[float, float], radius: float
) -> np.ndarray:
    """Exact area of the intersection of each pixel (edges in x and y) with a disk, shape ``(ny, nx)``."""
    if radius <= 0:
        return np.zeros((len(y_edges) - 1, len(x_edges) - 1))
    x0, x1 = x_edges[:-1] - center[0], x_edges[1:] - center[0]
    y0, y1 = y_edges[:-1, None] - center[1], y_edges[1:, None] - center[1]
    x0, x1 = np.minimum(x0, x1)[None, :], np.maximum(x0, x1)[None, :]
    y0, y1 = np.minimum(y0, y1), np.maximum(y0, y1)
    q = _quadrant
    return q(x0, y0, radius) - q(x1, y0, radius) - q(x0, y1, radius) + q(x1, y1, radius)


def _pixel_edges(centres: np.ndarray) -> np.ndarray:
    middle = (centres[1:] + centres[:-1]) / 2
    return np.concatenate([[2 * centres[0] - middle[0]], middle, [2 * centres[-1] - middle[-1]]])


@dataclass(frozen=True)
class ApertureResult:
    """Result of :func:`aperture_photometry`.

    Attributes
    ----------
    flux
        Background-subtracted sum over the aperture.
    uncertainty
        Standard deviation of ``flux`` (``None`` without pixel uncertainties nor annulus).
    background
        Background level per pixel (0 without annulus).
    area
        Area of the aperture (pixels).
    background_area
        Area of the annulus (pixels, 0 without annulus).
    """

    flux: u.Quantity
    uncertainty: u.Quantity | None
    background: u.Quantity
    area: u.Quantity
    background_area: u.Quantity


def _weights(
    image: DataProduct, center: tuple[u.Quantity, u.Quantity], radius: u.Quantity
) -> tuple[np.ndarray, float]:
    names = image.axis_names[-2:]
    y_axis, x_axis = image.axes[names[0]], image.axes[names[1]]
    unit = x_axis.unit
    x_edges = _pixel_edges(np.asarray(x_axis.value, dtype=float))
    y_edges = _pixel_edges(np.asarray(y_axis.to_value(unit), dtype=float))
    c = (_f(center[0], unit), _f(center[1], unit))
    areas = circle_overlap(x_edges, y_edges, c, _f(radius, unit))
    pixel_area = np.abs(np.diff(x_edges))[None, :] * np.abs(np.diff(y_edges))[:, None]
    return areas / pixel_area, float(np.mean(pixel_area))


def aperture_photometry(
    image: DataProduct,
    *,
    center: tuple[u.Quantity, u.Quantity],
    radius: u.Quantity,
    annulus: tuple[u.Quantity, u.Quantity] | None = None,
) -> ApertureResult:
    """Measure the flux of ``image`` in a circular aperture, with an optional background annulus.

    Parameters
    ----------
    image
        Image whose last two axes are ``y`` and ``x`` (in pixels or physical units); its values are counts or
        fluxes per pixel.
    center
        ``(x, y)`` of the aperture, in the unit of the image axes.
    radius
        Radius of the aperture.
    annulus
        ``(inner, outer)`` radii of the background annulus.
    """
    values = np.asarray(image.data.value, dtype=float)
    valid = np.ones(values.shape, bool) if image.mask is None else ~image.mask
    weights, _ = _weights(image, center, radius)
    weights = weights * valid
    area = float(weights.sum())
    background, background_area, background_variance = 0.0, 0.0, 0.0
    if annulus is not None:
        outer, _ = _weights(image, center, annulus[1])
        inner, _ = _weights(image, center, annulus[0])
        ring = (outer - inner) * valid
        background_area = float(ring.sum())
        selected = values[ring > 0.5]  # pixels mostly inside the annulus
        if selected.size < 2:
            raise ValueError("the background annulus holds fewer than 2 pixels")
        background = float(np.median(selected))
        background_variance = float(np.var(selected, ddof=1)) * math.pi / 2 / selected.size  # median error
    flux = float(np.sum(weights * values)) - background * area
    variance = area**2 * background_variance
    has_uncertainty = image.uncertainty is not None or annulus is not None
    if image.uncertainty is not None:
        sigma = np.asarray(image.uncertainty.to_value(image.unit), dtype=float)
        variance += float(np.sum(weights**2 * sigma**2))
    return ApertureResult(
        flux=flux * image.unit,
        uncertainty=math.sqrt(variance) * image.unit if has_uncertainty else None,
        background=background * image.unit,
        area=area * u.pix,
        background_area=background_area * u.pix,
    )


def ccd_snr(
    signal: u.Quantity,
    *,
    n_pixels: float,
    background: u.Quantity = 0 * u.electron,
    dark: u.Quantity = 0 * u.electron,
    read_noise: u.Quantity = 0 * u.electron,
) -> float:
    """Signal-to-noise ratio of the CCD equation (all inputs in electrons, per pixel for the noises)."""
    e = u.electron
    s = _f(signal, e)
    noise = s + float(n_pixels) * (_f(background, e) + _f(dark, e) + _f(read_noise, e) ** 2)
    return s / math.sqrt(noise)


def contrast_curve(
    image: DataProduct,
    *,
    peak: u.Quantity,
    resolution: u.Quantity,
    sigma: float = 5.0,
    center: tuple[u.Quantity, u.Quantity] | None = None,
    small_sample: bool = True,
) -> DataProduct:
    """Return the detection limit in contrast versus separation (multiples of ``resolution``).

    Parameters
    ----------
    image
        Residual image (after starlight subtraction), last two axes ``y`` and ``x``.
    peak
        Peak of the off-axis PSF of the star, in the unit of the image.
    resolution
        Size of a resolution element (λ/D), in the unit of the image axes.
    sigma
        Gaussian-equivalent detection threshold.
    center
        Position of the star (default: the origin of the axes).
    small_sample
        Apply the penalty of Mawet et al. (2014).
    """
    from scipy import stats

    names = image.axis_names[-2:]
    y_axis, x_axis = image.axes[names[0]], image.axes[names[1]]
    unit = x_axis.unit
    cx, cy = (0.0, 0.0) if center is None else (_f(center[0], unit), _f(center[1], unit))
    xx, yy = np.meshgrid(_a(x_axis, unit) - cx, _a(y_axis, unit) - cy)
    rr = np.hypot(xx, yy)
    values = np.asarray(image.data.value, dtype=float)
    valid = np.ones(values.shape, bool) if image.mask is None else ~image.mask
    step = _f(resolution, unit)
    limit = min(np.abs(_a(x_axis, unit) - cx).max(), np.abs(_a(y_axis, unit) - cy).max())
    fpf = float(stats.norm.sf(sigma))
    separations, contrasts = [], []
    k = 1
    while k * step + step / 2 <= limit:
        r = k * step
        ring = values[valid & (rr >= r - step / 2) & (rr < r + step / 2)]
        n = math.floor(2 * math.pi * r / step)
        if ring.size >= 2 and n >= 2:
            noise = float(np.std(ring, ddof=1))
            factor = float(stats.t.ppf(1 - fpf, n - 1)) * math.sqrt(1 + 1 / n) if small_sample else sigma
            separations.append(r)
            contrasts.append(noise * factor / _f(peak, image.unit))
        k += 1
    return DataProduct(
        np.asarray(contrasts) * u.one,
        axes={"separation": u.Quantity(np.asarray(separations), unit)},
        meta={"sigma": float(sigma), "small_sample": bool(small_sample)},
    )


# --- Elements -----------------------------------------------------------------------------------------------


@register("lumos.processing.AperturePhotometry")
class AperturePhotometry(Element):
    """Flux in a circular aperture, with an optional background annulus (a product holding the flux)."""

    x = QuantitySetting(default=0 * u.pix, doc="centre along x")
    y = QuantitySetting(default=0 * u.pix, doc="centre along y")
    radius = QuantitySetting(default=3 * u.pix)
    annulus_inner = QuantitySetting(default=0 * u.pix, doc="inner radius of the annulus (0: no annulus)")
    annulus_outer = QuantitySetting(default=0 * u.pix)
    in_ = Input(DataProduct)
    out = Output(DataProduct)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the flux, its uncertainty and the background in the metadata."""
        annulus = None
        if values.annulus_outer.value > 0:
            annulus = (values.annulus_inner, values.annulus_outer)
        result = aperture_photometry(
            inputs["in"], center=(values.x, values.y), radius=values.radius, annulus=annulus
        )
        uncertainty = None if result.uncertainty is None else np.atleast_1d(result.uncertainty)
        meta = {"background": result.background, "area": result.area}
        return {"out": DataProduct(np.atleast_1d(result.flux), uncertainty=uncertainty, meta=meta)}


@register("lumos.processing.ContrastCurve")
class ContrastCurve(Element):
    """Detection limit in contrast versus separation of a residual image (Mawet et al. 2014)."""

    peak = QuantitySetting(default=1 * u.electron, doc="peak of the off-axis PSF, in the unit of the image")
    resolution = QuantitySetting(default=4 * u.pix, doc="λ/D in the unit of the image axes")
    sigma = Setting(float, default=5.0)
    in_ = Input(DataProduct)
    out = Output(DataProduct)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the contrast curve."""
        curve = contrast_curve(
            inputs["in"], peak=values.peak, resolution=values.resolution, sigma=float(values.sigma)
        )
        return {"out": curve}


@register("lumos.processing.RadialProfile")
class RadialProfile(Element):
    """Azimuthal average of an image around (``x``, ``y``), in rings of ``width`` (default: one pixel)."""

    x = QuantitySetting(default=0 * u.pix, doc="centre along x")
    y = QuantitySetting(default=0 * u.pix, doc="centre along y")
    width = QuantitySetting(default=0 * u.pix, doc="ring width (0: the pixel step)")
    in_ = Input(DataProduct)
    out = Output(DataProduct)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the mean of each ring, with the standard error of the mean when the image has none."""
        image = inputs["in"]
        names = image.axis_names[-2:]
        y_axis, x_axis = image.axes[names[0]], image.axes[names[1]]
        unit = x_axis.unit
        xx, yy = np.meshgrid(_a(x_axis, unit) - _f(values.x, unit), _a(y_axis, unit) - _f(values.y, unit))
        width = (
            _f(values.width, unit)
            if values.width.value > 0
            else float(np.abs(np.diff(_a(x_axis, unit))).min())
        )
        data = np.asarray(image.data.value, dtype=float)
        valid = np.ones(data.shape, bool) if image.mask is None else ~image.mask
        rings = np.floor(np.hypot(xx, yy) / width).astype(int)
        n = int(rings[valid].max()) + 1
        counts = np.bincount(rings[valid], minlength=n)
        sums = np.bincount(rings[valid], weights=data[valid], minlength=n)
        squares = np.bincount(rings[valid], weights=data[valid] ** 2, minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = sums / counts
            spread = np.sqrt(np.maximum(squares / counts - mean**2, 0.0) * counts / np.maximum(counts - 1, 1))
            error = spread / np.sqrt(counts)
        keep = counts > 0
        radii = u.Quantity((np.arange(n)[keep] + 0.5) * width, unit)
        return {
            "out": DataProduct(
                u.Quantity(mean[keep], image.unit),
                axes={"radius": radii},
                uncertainty=u.Quantity(error[keep], image.unit),
            )
        }


@register("lumos.processing.SignalToNoise")
class SignalToNoise(Element):
    """Map of the signal-to-noise ratio, the data divided by their uncertainty."""

    in_ = Input(DataProduct)
    out = Output(DataProduct)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return ``data / uncertainty`` (masked where the uncertainty is 0)."""
        image = inputs["in"]
        if image.uncertainty is None:
            raise ValueError("the signal-to-noise ratio needs the uncertainties of the data")
        sigma = np.asarray(image.uncertainty.to_value(image.unit), dtype=float)
        data = np.asarray(image.data.value, dtype=float)
        bad = sigma <= 0
        snr = np.where(bad, 0.0, data / np.where(bad, 1.0, sigma))
        mask = bad if image.mask is None else bad | image.mask
        return {"out": DataProduct(u.Quantity(snr, u.one), axes=image.axes, mask=mask)}
