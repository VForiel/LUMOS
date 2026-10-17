r"""Classical calibration: phase fitting, MTF and PSF measurement (step P11.4, REQ-PIPE-5).

Phase fitting (:func:`fit_phases`)
    Minimises a measured quantity (a null depth, a dark output, a wavefront error) over phase settings with
    one of three methods: ``"gradient"`` (L-BFGS-B with finite-difference gradients), ``"scan"`` (trial and
    error: each phase in turn is scanned over its range and set to its best value, then the ranges shrink)
    and ``"genetic"`` (differential evolution, Storn & Price 1997, then a local polish). The measurement is
    any callable, typically setting phase shifters of a context and running it.

Slanted-edge MTF (:func:`mtf_slanted_edge`, ISO 12233; Burns 2000, Proc. IS&T PICS, 135)
    The edge position in each row is the centroid of the derivative of the row; a straight line is fitted to
    these positions. Every pixel is then placed at its distance to the edge, measured across it, in bins of
    ``1/oversampling`` pixel: this gives a supersampled edge spread function. Its derivative (the line spread
    function), apodised by a Hamming window, is Fourier transformed; the modulus normalised at zero frequency
    is the MTF. The transfers of the finite difference and of the binning, both :math:`\mathrm{sinc}(f\Delta)`
    for bins of :math:`\Delta`, are divided out.

PSF measurement (:func:`measure_psf`)
    Centroid (first moment over a disk of two resolution elements, re-centred until it converges), peak,
    FWHM (radius where an even polynomial fitted to the core falls to half of its central value), encircled
    energy (exact disk overlap, :func:`~lumos.processing.photometry.circle_overlap`) and Strehl ratio against
    a reference PSF of the same incoming flux (ratio of the peaks).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy import units as u

from lumos.core.element import Element, Values
from lumos.core.param import Setting
from lumos.core.port import Input, Output
from lumos.core.registry import register
from lumos.fields.dataproduct import DataProduct
from lumos.processing.photometry import _pixel_edges, circle_overlap

__all__ = [
    "MTFMeasurement",
    "PSFMeasurement",
    "PSFMeasurementElement",
    "PhaseFit",
    "fit_phases",
    "measure_psf",
    "mtf_slanted_edge",
]

# --- Phase fitting -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseFit:
    """Result of :func:`fit_phases`: best phases, measured value there, number of measurements, method."""

    phases: u.Quantity
    value: float
    evaluations: int
    method: str


def _scan(
    measure: Callable[[np.ndarray], float],
    start: np.ndarray,
    half_ranges: np.ndarray,
    *,
    points: int,
    iterations: int,
    shrink: float,
) -> np.ndarray:
    best = start.copy()
    half = half_ranges.copy()
    for _ in range(iterations):
        for k in range(best.size):
            trials = best[k] + np.linspace(-half[k], half[k], points)
            values = []
            for value in trials:
                candidate = best.copy()
                candidate[k] = value
                values.append(measure(candidate))
            best[k] = trials[int(np.argmin(values))]
        half *= shrink
    return best


def fit_phases(
    measure: Callable[[np.ndarray], float],
    *,
    initial: u.Quantity,
    method: str = "gradient",
    half_range: u.Quantity = math.pi * u.rad,
    seed: int | None = None,
    points: int = 21,
    iterations: int = 12,
) -> PhaseFit:
    """Find the phases minimising ``measure``.

    Parameters
    ----------
    measure
        Function of the phases in radians (NumPy array) returning the quantity to minimise.
    initial
        Starting phases (searched within ``initial ± half_range``).
    method
        ``"gradient"``, ``"scan"`` (trial and error) or ``"genetic"``.
    seed
        Seed of the genetic algorithm.
    points, iterations
        Trials per phase and number of sweeps of ``"scan"`` (the ranges shrink by 0.3 at each sweep).
    """
    from scipy import optimize

    x0 = np.atleast_1d(np.asarray(initial.to_value(u.rad), dtype=float))
    half = float(np.asarray(half_range.to_value(u.rad)))
    bounds = [(v - half, v + half) for v in x0]
    count = 0

    def counted(phases: np.ndarray) -> float:
        nonlocal count
        count += 1
        return float(measure(np.asarray(phases, dtype=float)))

    if method == "gradient":
        result = optimize.minimize(
            counted, x0, method="L-BFGS-B", bounds=bounds, options={"ftol": 1e-15, "gtol": 1e-12}
        )
        best = np.asarray(result.x, dtype=float)
    elif method == "scan":
        best = _scan(counted, x0, np.full(x0.size, half), points=points, iterations=iterations, shrink=0.3)
    elif method == "genetic":
        import inspect

        # SciPy >= 1.15 names the random state ``rng`` and deprecates ``seed``
        keyword = "rng" if "rng" in inspect.signature(optimize.differential_evolution).parameters else "seed"
        options: dict[str, Any] = {keyword: seed, "tol": 1e-12, "polish": True}
        result = optimize.differential_evolution(counted, bounds, **options)
        best = np.asarray(result.x, dtype=float)
    else:
        raise ValueError(f"unknown method {method!r}: use 'gradient', 'scan' or 'genetic'")
    value = counted(best)
    return PhaseFit(best * u.rad, value, count, method)


# --- MTF --------------------------------------------------------------------------------------------------


def _image(product: DataProduct) -> tuple[np.ndarray, Any]:
    data = np.asarray(product.data.value, dtype=float)
    if data.ndim != 2:
        raise ValueError("the measurement needs a 2-D image")
    x_axis = product.axes[product.axis_names[-1]]
    step = float(np.mean(np.diff(np.asarray(x_axis.value, dtype=float))))
    return data, u.Quantity(step, x_axis.unit)


def mtf_slanted_edge(image: DataProduct, *, oversampling: int = 4, window: bool = True) -> DataProduct:
    """Measure the MTF across a slanted edge (ISO 12233).

    Parameters
    ----------
    image
        Image of a straight edge tilted by a few degrees from the columns (or the rows).
    oversampling
        Bins per pixel of the edge spread function.
    window
        Apodise the line spread function by a Hamming window.

    Returns
    -------
    DataProduct
        MTF on the axis ``frequency`` (cycles per unit of the image axes), up to the pixel frequency.
    """
    data, step = _image(image)
    gx = np.abs(np.diff(data, axis=1)).sum()
    gy = np.abs(np.diff(data, axis=0)).sum()
    transposed = gy > gx  # horizontal edge: work on the transpose
    if transposed:
        data = data.T
    ny, nx = data.shape
    derivative = np.diff(data, axis=1)
    centres = np.arange(nx - 1) + 0.5
    weights = np.abs(derivative)
    positions = (weights * centres).sum(axis=1) / weights.sum(axis=1)
    rows = np.arange(ny)
    slope, intercept = np.polyfit(rows, positions, 1)
    angle = math.atan(slope)
    yy, xx = np.mgrid[0:ny, 0:nx]
    distance = (xx - (intercept + slope * yy)) * math.cos(angle)
    delta = 1.0 / int(oversampling)
    bins = np.floor(distance / delta).astype(int)
    bins -= bins.min()
    counts = np.bincount(bins.ravel())
    sums = np.bincount(bins.ravel(), weights=data.ravel())
    filled = counts > 0
    index = np.arange(counts.size)
    esf = np.interp(index, index[filled], sums[filled] / counts[filled])
    if esf[-1] < esf[0]:
        esf = esf[::-1]
    lsf = np.diff(esf)
    if window:
        peak = int(np.argmax(lsf))
        n = lsf.size
        half = max(peak, n - 1 - peak)
        offsets = np.arange(n) - peak
        lsf = lsf * (0.54 + 0.46 * np.cos(np.pi * offsets / max(half, 1)))
    spectrum = np.abs(np.fft.rfft(lsf))
    frequency = np.fft.rfftfreq(lsf.size, d=delta)  # cycles per pixel
    keep = frequency <= 1.0
    correction = np.sinc(frequency[keep] * delta) ** 2  # finite difference and binning
    mtf = spectrum[keep] / spectrum[0] / correction
    return DataProduct(
        mtf * u.one,
        axes={"frequency": frequency[keep] / step},
        meta={
            "edge_angle_deg": math.degrees(angle),
            "oversampling": int(oversampling),
            "transposed": bool(transposed),
        },
    )


# --- PSF --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PSFMeasurement:
    """Measured properties of a PSF (positions and widths in the unit of the image axes)."""

    centroid: tuple[u.Quantity, u.Quantity]
    peak: u.Quantity
    fwhm: u.Quantity
    total: u.Quantity
    strehl: float | None
    _image: DataProduct

    def encircled_energy(self, radius: u.Quantity, *, total: u.Quantity | None = None) -> float:
        """Fraction of ``total`` (default: the sum of the image) within ``radius`` of the centroid."""
        weights = _disk(self._image, self.centroid, radius)
        values = np.asarray(self._image.data.value, dtype=float)
        reference = self.total if total is None else total
        return float(np.sum(weights * values)) / float(np.asarray(reference.to_value(self._image.unit)))


def _axes(image: DataProduct) -> tuple[np.ndarray, np.ndarray, Any]:
    names = image.axis_names[-2:]
    y_axis, x_axis = image.axes[names[0]], image.axes[names[1]]
    unit = x_axis.unit
    return np.asarray(x_axis.value, dtype=float), np.asarray(y_axis.to_value(unit), dtype=float), unit


def _disk(image: DataProduct, center: tuple[u.Quantity, u.Quantity], radius: u.Quantity) -> np.ndarray:
    x, y, unit = _axes(image)
    xe, ye = _pixel_edges(x), _pixel_edges(y)
    c = (float(np.asarray(center[0].to_value(unit))), float(np.asarray(center[1].to_value(unit))))
    areas = circle_overlap(xe, ye, c, float(np.asarray(radius.to_value(unit))))
    return areas / (np.abs(np.diff(xe))[None, :] * np.abs(np.diff(ye))[:, None])


def _half_width(values: np.ndarray, radii: np.ndarray) -> tuple[float, float]:
    """Fit ``a0 + a1 r² + a2 r⁴ + a3 r⁶`` to the core and return (a0, radius at a0 / 2)."""
    core = values > 0.2 * values.max()
    r2 = radii[core] ** 2
    design = np.stack([np.ones_like(r2), r2, r2**2, r2**3], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, values[core], rcond=None)
    a0 = float(coefficients[0])
    roots = np.roots([coefficients[3], coefficients[2], coefficients[1], coefficients[0] - a0 / 2])
    real = sorted(float(z.real) for z in roots if abs(z.imag) < 1e-9 * max(abs(z), 1) and z.real > 0)
    if not real:
        raise ValueError("the PSF core has no half-maximum radius")
    return a0, math.sqrt(real[0])


def measure_psf(
    image: DataProduct, *, resolution: u.Quantity | None = None, reference: DataProduct | None = None
) -> PSFMeasurement:
    """Measure the centroid, peak, FWHM, total and (with ``reference``) the Strehl ratio of a PSF image.

    Parameters
    ----------
    image
        Image of the PSF (last two axes ``y`` and ``x``).
    resolution
        Expected width (λ/D) used for the centroid window (default: from the pixels above half maximum).
    reference
        Ideal PSF for the same incoming flux, for the Strehl ratio (ratio of the peaks).
    """
    values = np.asarray(image.data.value, dtype=float)
    x, y, unit = _axes(image)
    xx, yy = np.meshgrid(x, y)
    iy, ix = np.unravel_index(int(np.argmax(values)), values.shape)
    peak = float(values[iy, ix])
    if resolution is None:
        width = 2 * math.sqrt(
            np.count_nonzero(values > peak / 2) * abs((x[1] - x[0]) * (y[1] - y[0])) / math.pi
        )
    else:
        width = float(np.asarray(resolution.to_value(unit)))
    cx, cy = float(x[ix]), float(y[iy])
    for _ in range(20):
        weights = _disk(image, (cx * unit, cy * unit), 2 * width * unit) * values
        nx_, ny_ = float((weights * xx).sum() / weights.sum()), float((weights * yy).sum() / weights.sum())
        moved = math.hypot(nx_ - cx, ny_ - cy)
        cx, cy = nx_, ny_
        if moved < 1e-9 * width:
            break
    radii = np.hypot(xx - cx, yy - cy)
    _, half = _half_width(values.ravel(), radii.ravel())
    strehl = None
    if reference is not None:
        strehl = peak / float(np.max(np.asarray(reference.data.to_value(image.unit), dtype=float)))
    return PSFMeasurement(
        centroid=(cx * unit, cy * unit),
        peak=peak * image.unit,
        fwhm=2 * half * unit,
        total=float(values.sum()) * image.unit,
        strehl=strehl,
        _image=image,
    )


# --- Elements ---------------------------------------------------------------------------------------------


@register("lumos.processing.MTFMeasurement")
class MTFMeasurement(Element):
    """MTF measured on the image of a slanted edge (ISO 12233)."""

    oversampling = Setting(int, default=4)
    in_ = Input(DataProduct)
    out = Output(DataProduct)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the MTF."""
        return {"out": mtf_slanted_edge(inputs["in"], oversampling=int(values.oversampling))}


@register("lumos.processing.PSFMeasurement")
class PSFMeasurementElement(Element):
    """Centroid, peak, FWHM and encircled energy of a PSF image (a product of scalars with named meta)."""

    in_ = Input(DataProduct)
    out = Output(DataProduct)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return ``[x, y, fwhm]`` in the unit of the image axes, with the peak and total in the metadata."""
        psf = measure_psf(inputs["in"])
        unit = psf.fwhm.unit
        data = u.Quantity([float(np.asarray(q.to_value(unit))) for q in (*psf.centroid, psf.fwhm)], unit)
        meta = {"quantities": ["x", "y", "fwhm"], "peak": psf.peak, "total": psf.total}
        return {"out": DataProduct(data, meta=meta)}
