r"""Imaging of extended incoherent scenes (step P8.5, REQ-SCN-6, ADR-0011).

An extended scene is a radiance map :math:`L(\theta, \lambda)` (:class:`~lumos.fields.RadianceField`). Its
points are mutually incoherent, so the image is the sum of the images of its points, weighted by their flux
(Goodman, *Introduction to Fourier Optics*, §6.1). :func:`image_scene` computes it through an
:class:`ImagingSystem`, a part of a context going from an input that receives point sources
(:class:`~lumos.fields.SourceList`, e.g. ``"telescope.in"``) to an output field (e.g. ``"camera_plane.out"``),
with one of three strategies:

``"isoplanatic"``
    The point-spread function (PSF) does not depend on the position in the field: the image is the
    convolution of the scene by the on-axis PSF :math:`h`, computed with FFTs channel by channel,

    .. math:: I(\mathbf{x}) = \sum_j F_j\, h(\mathbf{x} - \mathbf{p}\,\theta_j),

    with :math:`F_j = L(\theta_j)\,\Delta\Omega` the flux of scene pixel :math:`j` and :math:`\mathbf{p}` the
    plate scale (focal-plane position per unit angle), measured on the system from the images of two
    off-axis points. The scene is first resampled, conserving its flux, onto an angular grid whose pixels
    map exactly onto the detector pixels.
``"anisoplanatic"``
    The PSF varies slowly across the field: PSFs :math:`h_k` are computed on a grid of field points
    :math:`\theta_k` and interpolated with bilinear weights :math:`w_k(\theta)`,
    :math:`I = \sum_k (w_k F) * h_k` (weighted convolutions; Lauer 2002, Proc. SPIE 4847, 167).
``"decomposition"``
    Every scene pixel becomes a point source and the whole system is computed for all of them at once (an
    :class:`~lumos.fields.IncoherentBundle` mode per pixel). It is exact for any system and compulsory for
    systems that are not shift-invariant: coronagraphs, interferometers, fibre injection.

``strategy="auto"`` probes the system: it compares the image of an off-axis point, shifted back by the plate
scale, with the on-axis PSF. When they agree within 1 % (L1 norm) the system is shift-invariant and the
convolution is used; otherwise the scene is decomposed. Forcing ``"isoplanatic"`` on a system that fails the
probe raises an error. The choice and its reason are logged on ``lumos.scenes``.

The same decomposition is available inside a graph with the element :class:`SceneToSources`
(``scene.radiance >> SceneToSources() >> telescope``).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from astropy import units as u
from scipy.signal import fftconvolve

from lumos.backend._core import use
from lumos.core.element import Element, Values
from lumos.core.param import Setting
from lumos.core.port import Input, Output, PortRef
from lumos.core.registry import register
from lumos.fields.grid import Grid
from lumos.fields.radiance import RadianceField
from lumos.fields.sources import PHOTON_FLUX_DENSITY, SourceList
from lumos.fields.wavecube import Wavecube
from lumos.scenes.scene import RADIANCE_UNIT

__all__ = [
    "STRATEGIES",
    "ImageResult",
    "ImagingSystem",
    "SceneToSources",
    "image_scene",
    "scene_to_sources",
]

logger = logging.getLogger("lumos.scenes")

#: Imaging strategies of extended scenes.
STRATEGIES: tuple[str, ...] = ("auto", "isoplanatic", "anisoplanatic", "decomposition")

#: Relative L1 difference below which the probe declares a system shift-invariant.
INVARIANCE_TOLERANCE = 1e-2

_ENERGY_FLUX = u.W / u.m**2 / u.m


# --- Scene → point sources ---------------------------------------------------------------------------------


def scene_to_sources(
    radiance: RadianceField, bandwidths: u.Quantity, *, threshold: float = 0.0
) -> SourceList:
    r"""Turn every pixel of a scene into a point source of flux :math:`L(\theta_j)\,\Delta\Omega`.

    Parameters
    ----------
    radiance
        Scene on an angular grid.
    bandwidths
        Widths of the spectral channels of the scene.
    threshold
        Pixels whose total radiance is at most ``threshold`` times the brightest pixel are dropped
        (``0``: only the dark pixels).
    """
    if radiance.grid.plane != "angular":
        raise ValueError("scene_to_sources needs a radiance field on an angular grid")
    if radiance.has_stokes:
        raise ValueError("scene_to_sources needs an unpolarised radiance field")
    grid = radiance.grid
    values = np.asarray(radiance.radiance.to_value(RADIANCE_UNIT))  # (nλ, ny, nx)
    total = values.sum(axis=0)
    keep = total > threshold * float(total.max()) if total.max() > 0 else np.zeros(total.shape, bool)
    x, y = (np.asarray(c) for c in grid.coords(backend="numpy"))
    omega = grid.dx_si * grid.dy_si
    flux = values[:, keep].T * omega  # (M, nλ) in W m-2 m-1
    return SourceList(
        x[keep] * u.rad,
        y[keep] * u.rad,
        radiance.wavelengths,
        bandwidths,
        flux * _ENERGY_FLUX,
    )


@register("lumos.scenes.SceneToSources")
class SceneToSources(Element):
    """Decompose a radiance map into incoherent point sources, one per pixel (``"decomposition"``)."""

    observation_dependent = True
    threshold = Setting(float, default=0.0, doc="drop pixels fainter than this fraction of the brightest")
    in_ = Input(RadianceField)
    out = Output(SourceList)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return one point source per pixel of the scene."""
        radiance: RadianceField = inputs["in"]
        bandwidths = np.asarray(values.bandwidths, dtype=float) * u.m
        if len(bandwidths) != radiance.n_wavelengths:
            raise ValueError("the scene is not sampled at the channels of the observation")
        return {"out": scene_to_sources(radiance, bandwidths, threshold=float(values.threshold))}


# --- Imaging system -----------------------------------------------------------------------------------------


class ImagingSystem:
    """The part of a context between an input receiving point sources and an output field.

    Parameters
    ----------
    ctx
        The :class:`~lumos.Context` holding the system.
    entrance
        Input port receiving a :class:`~lumos.fields.SourceList` (``"id.port"`` or a port), left unconnected.
    exit
        Output port giving the field in the image plane.
    """

    def __init__(self, ctx: Any, *, entrance: str | PortRef, exit: str | PortRef) -> None:
        self.ctx = ctx
        self.entrance = ctx.port(entrance) if isinstance(entrance, str) else entrance
        self.exit = ctx.port(exit) if isinstance(exit, str) else exit
        if self.entrance.direction != "in" or not issubclass(SourceList, self.entrance.signal_type):
            raise ValueError(f"{self.entrance!r} must be an input accepting point sources (SourceList)")
        if self.exit.direction != "out":
            raise ValueError(f"{self.exit!r} must be an output port")
        if ctx.graph.incoming(self.entrance):
            raise ValueError(
                f"{self.entrance!r} is connected; the entrance of an imaging system must be free"
            )
        self.grid: Grid | None = None
        self.weights: np.ndarray | None = None
        self.power_unit: u.UnitBase | None = None

    def bandwidths(self) -> u.Quantity:
        """Widths of the spectral channels of the observation of the context."""
        observation = self.ctx.observation
        if observation.band is None:
            raise ValueError("the context needs an observation band (ctx.observation.band)")
        _, weights = observation.wavelengths()
        low, high = (float(q.to_value(u.m, equivalencies=u.spectral())) for q in observation.band)
        return np.asarray(weights, dtype=float) * (high - low) * u.m

    def response(self, sources: SourceList) -> np.ndarray:
        """Return the power per pixel and channel, ``(nλ, ny, nx)``, of the image of ``sources``."""
        from lumos.elements.detectors import _spectral_power

        graph = self.ctx.graph
        graph.invalidate(self.entrance.owner)
        feed = [sources] if self.entrance.port.multiple else sources
        try:
            if self.ctx.backend == "auto":
                result = graph.run([self.exit], feeds={self.entrance: feed})
            else:
                with use(self.ctx.backend):
                    result = graph.run([self.exit], feeds={self.entrance: feed})
        finally:
            graph.invalidate(self.entrance.owner)
        signal = result[self.exit]
        _, weights, power, unit = _spectral_power(signal)
        self.grid = signal.grid
        self.weights = np.asarray(weights)
        self.power_unit = unit
        return np.asarray(power, dtype=float)

    def point(
        self, theta_x: float, theta_y: float, wavelengths: u.Quantity, bandwidths: u.Quantity
    ) -> np.ndarray:
        """Image of a point source of unit photon flux density (1 photon s⁻¹ m⁻² m⁻¹) at ``(θx, θy)`` rad."""
        unit = SourceList(
            [theta_x] * u.rad,
            [theta_y] * u.rad,
            wavelengths,
            bandwidths,
            np.ones((1, len(wavelengths))) * PHOTON_FLUX_DENSITY,
        )
        return self.response(unit)


@dataclass
class ImageResult:
    """Image of a scene.

    Attributes
    ----------
    power
        Power per pixel and spectral channel, ``(nλ, ny, nx)``, in ``power_unit``.
    grid
        Sampling of the image plane.
    wavelengths
        Spectral channels.
    weights
        Quadrature weights of the channels.
    power_unit
        Unit of ``power`` (photon s⁻¹ for scenes imaged through telescopes).
    strategy, reason
        Strategy used and why it was chosen.
    """

    power: np.ndarray
    grid: Grid
    wavelengths: u.Quantity
    weights: np.ndarray
    power_unit: u.UnitBase
    strategy: str
    reason: str

    def to_wavecube(self) -> Wavecube:
        """Return the image as a :class:`~lumos.fields.Wavecube` with these intensities, for detectors.

        The phase of an incoherent image is meaningless: detect the cube, do not propagate it.
        """
        area = self.grid.dx_si * self.grid.dy_si
        field = np.sqrt(np.maximum(self.power, 0.0) / area)
        return Wavecube(field, self.grid, self.wavelengths, weights=self.weights, power_unit=self.power_unit)


# --- Probing the system -------------------------------------------------------------------------------------


def _centroid(power: np.ndarray, grid: Grid, radius: int = 4) -> tuple[float, float]:
    """Centroid of the image within ``radius`` pixels of its brightest pixel (insensitive to the wings)."""
    image = power.sum(axis=0)
    if float(image.max()) <= 0:
        return math.nan, math.nan
    row, column = np.unravel_index(int(np.argmax(image)), image.shape)
    rows = slice(max(row - radius, 0), row + radius + 1)
    columns = slice(max(column - radius, 0), column + radius + 1)
    window = image[rows, columns]
    x, y = (np.asarray(c)[rows, columns] for c in grid.coords(backend="numpy"))
    total = float(window.sum())
    return float((window * x).sum() / total), float((window * y).sum() / total)


def _shifted(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift the last two axes by whole pixels, filling with zeros (no wrap-around)."""
    out = np.zeros_like(image)
    ny, nx = image.shape[-2:]
    ys, yd = (slice(0, ny - dy), slice(dy, ny)) if dy >= 0 else (slice(-dy, ny), slice(0, ny + dy))
    xs, xd = (slice(0, nx - dx), slice(dx, nx)) if dx >= 0 else (slice(-dx, nx), slice(0, nx + dx))
    out[..., yd, xd] = image[..., ys, xs]
    return out


@dataclass
class _Calibration:
    psf: np.ndarray  # on-axis PSF per unit flux density, (nλ, ny, nx)
    scale: tuple[float, float]  # detector pixels per radian along x and y
    invariant: bool
    difference: float


def _calibrate(system: ImagingSystem, scene: RadianceField, bandwidths: u.Quantity) -> _Calibration:
    lam = scene.wavelengths
    psf = system.point(0.0, 0.0, lam, bandwidths)
    grid = cast(Grid, system.grid)
    half = 0.25 * min(scene.grid.nx * scene.grid.dx_si, scene.grid.ny * scene.grid.dy_si)
    probe_x = system.point(half, 0.0, lam, bandwidths)
    probe_y = system.point(0.0, half, lam, bandwidths)
    total0, total1 = float(psf.sum()), float(probe_x.sum())
    c0, c1, c2 = _centroid(psf, grid), _centroid(probe_x, grid), _centroid(probe_y, grid)
    if total0 <= 0 or not all(math.isfinite(v) for v in (*c0, *c1, *c2)):
        return _Calibration(psf, (math.nan, math.nan), False, math.inf)
    scale = ((c1[0] - c0[0]) / half / grid.dx_si, (c2[1] - c0[1]) / half / grid.dy_si)
    shift = round(scale[0] * half)
    if shift == 0 or abs(total1 - total0) > INVARIANCE_TOLERANCE * total0:
        difference = abs(total1 - total0) / total0
        return _Calibration(psf, scale, False, difference if shift else math.inf)
    exact = system.point(shift / scale[0], 0.0, lam, bandwidths)  # whole-pixel displacement
    back = _shifted(exact, 0, -shift)
    nx = psf.shape[-1]
    valid = slice(0, nx - shift) if shift > 0 else slice(-shift, nx)  # columns present in both images
    reference = psf[..., valid]
    difference = float(np.abs(back[..., valid] - reference).sum() / max(np.abs(reference).sum(), 1e-300))
    return _Calibration(psf, scale, difference < INVARIANCE_TOLERANCE, difference)


# --- Convolution strategies ---------------------------------------------------------------------------------


def _angular_grid(scene: RadianceField, scale: tuple[float, float]) -> Grid:
    """Odd-sized angular grid centred on the axis, one pixel per detector pixel, covering the scene."""
    sx, sy = abs(scale[0]), abs(scale[1])
    width = scene.grid.nx * scene.grid.dx_si + 2 * abs(scene.grid.x0_si)
    height = scene.grid.ny * scene.grid.dy_si + 2 * abs(scene.grid.y0_si)
    nx = 2 * math.ceil(width * sx / 2) + 1
    ny = 2 * math.ceil(height * sy / 2) + 1
    return Grid(nx, ny, dx=1 / sx * u.rad, dy=1 / sy * u.rad)


def _flux_map(scene: RadianceField, grid: Grid, scale: tuple[float, float]) -> np.ndarray:
    """Photon flux density of each pixel of ``grid`` (photon s⁻¹ m⁻² m⁻¹), oriented like the image."""
    resampled = scene.resample(grid)
    energy = np.asarray(resampled.radiance.to_value(RADIANCE_UNIT)) * grid.dx_si * grid.dy_si
    lam = np.asarray(scene.wavelengths_si)
    photons = energy * (lam / (6.62607015e-34 * 299_792_458.0))[:, None, None]
    if scale[0] < 0:
        photons = photons[..., ::-1]
    if scale[1] < 0:
        photons = photons[..., ::-1, :]
    return photons


def _convolve(flux: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """``out[i] = Σ_j flux[j] psf[i - (j - c)]`` with ``c`` the centre of the (odd-sized) flux map."""
    full = fftconvolve(psf, flux, mode="full", axes=(-2, -1))
    cy, cx = flux.shape[-2] // 2, flux.shape[-1] // 2
    ny, nx = psf.shape[-2:]
    return np.asarray(full[..., cy : cy + ny, cx : cx + nx])


def _isoplanatic(scene: RadianceField, calibration: _Calibration) -> np.ndarray:
    grid = _angular_grid(scene, calibration.scale)
    return _convolve(_flux_map(scene, grid, calibration.scale), calibration.psf)


def _tent(coordinates: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Bilinear (tent) weights of ``coordinates`` on 1D ``nodes``, ``(len(nodes), len(coordinates))``."""
    clipped = np.clip(coordinates, nodes[0], nodes[-1])
    weights = np.zeros((nodes.size, coordinates.size))
    if nodes.size == 1:
        weights[0] = 1.0
        return weights
    index = np.clip(np.searchsorted(nodes, clipped) - 1, 0, nodes.size - 2)
    t = (clipped - nodes[index]) / (nodes[index + 1] - nodes[index])
    columns = np.arange(coordinates.size)
    weights[index, columns] = 1 - t
    weights[index + 1, columns] = t
    return weights


def _anisoplanatic(
    scene: RadianceField,
    system: ImagingSystem,
    calibration: _Calibration,
    bandwidths: u.Quantity,
    n_field: int,
) -> np.ndarray:
    grid = _angular_grid(scene, calibration.scale)
    flux = _flux_map(scene, grid, calibration.scale)
    sx, sy = calibration.scale
    # field points at whole detector pixels, across the scene
    extent_x = (scene.grid.nx * scene.grid.dx_si) / 2
    extent_y = (scene.grid.ny * scene.grid.dy_si) / 2
    pixels_x = np.unique(np.round(np.linspace(-extent_x, extent_x, n_field) * abs(sx)).astype(int))
    pixels_y = np.unique(np.round(np.linspace(-extent_y, extent_y, n_field) * abs(sy)).astype(int))
    x, y = (np.asarray(a) for a in grid.axes(backend="numpy"))
    wx = _tent(x * abs(sx), pixels_x.astype(float))
    wy = _tent(y * abs(sy), pixels_y.astype(float))
    if sx < 0:
        wx = wx[:, ::-1]
    if sy < 0:
        wy = wy[:, ::-1]
    image = np.zeros_like(calibration.psf)
    for j, py in enumerate(pixels_y):
        for i, px in enumerate(pixels_x):
            theta_x = px / abs(sx)  # field point on the scene; its PSF lands at sx · θx
            theta_y = py / abs(sy)
            psf = system.point(float(theta_x), float(theta_y), scene.wavelengths, bandwidths)
            shift_x = round(sx * theta_x)
            shift_y = round(sy * theta_y)
            centred = _shifted(psf, -shift_y, -shift_x)
            weight = wy[j][:, None] * wx[i][None, :]
            image += _convolve(flux * weight[None], centred)
    return image


# --- Decomposition ------------------------------------------------------------------------------------------


def _subset(sources: SourceList, start: int, stop: int) -> SourceList:
    every: Any = sources
    return SourceList(
        every.theta_x[start:stop],
        every.theta_y[start:stop],
        sources.wavelengths,
        sources.bandwidths,
        every.flux[start:stop],
    )


def _decomposition(
    scene: RadianceField, system: ImagingSystem, bandwidths: u.Quantity, batch: int
) -> np.ndarray:
    sources = scene_to_sources(scene, bandwidths)
    image: np.ndarray | None = None
    for start in range(0, max(sources.n_sources, 1), batch):
        part = _subset(sources, start, min(start + batch, sources.n_sources))
        if part.n_sources == 0:
            continue
        power = system.response(part)
        image = power if image is None else image + power
    if image is None:  # a dark scene: the image of a dark source
        image = 0.0 * system.point(0.0, 0.0, scene.wavelengths, bandwidths)
    return image


# --- Entry point --------------------------------------------------------------------------------------------


def image_scene(
    scene: RadianceField,
    system: ImagingSystem,
    *,
    strategy: str = "auto",
    points: SourceList | Sequence[SourceList] | None = None,
    field_points: int = 3,
    batch: int = 1024,
) -> ImageResult:
    """Image an extended scene (and optional point sources) through ``system``.

    Parameters
    ----------
    scene
        Radiance of the scene on an angular grid, sampled at the channels of the context of ``system``.
    system
        The :class:`ImagingSystem`.
    strategy
        ``"auto"`` (default), ``"isoplanatic"``, ``"anisoplanatic"`` or ``"decomposition"`` (see the module
        documentation).
    points
        Point sources of the scene, added exactly (their image is computed by the system).
    field_points
        Number of field points along each axis for the anisoplanatic strategy.
    batch
        Number of point sources computed together by the decomposition.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; use one of {', '.join(STRATEGIES)}")
    bandwidths = system.bandwidths()
    if len(bandwidths) != scene.n_wavelengths:
        raise ValueError("the scene is not sampled at the channels of the observation of the system")
    calibration: _Calibration | None = None
    if strategy != "decomposition":
        calibration = _calibrate(system, scene, bandwidths)
    if strategy == "auto":
        assert calibration is not None
        if calibration.invariant:
            strategy = "isoplanatic"
            reason = (
                "the PSF is shift-invariant over the field (relative difference "
                f"{calibration.difference:.1e} < {INVARIANCE_TOLERANCE:g}): convolution by the on-axis PSF"
            )
        else:
            strategy = "decomposition"
            reason = (
                f"the system is not shift-invariant (an off-axis PSF differs from the on-axis one by "
                f"{calibration.difference:.1e}): explicit decomposition into point sources"
            )
    elif strategy == "isoplanatic":
        assert calibration is not None
        if not calibration.invariant:
            raise ValueError(
                "the system is not shift-invariant (an off-axis PSF differs from the on-axis one by "
                f"{calibration.difference:.1e}); use strategy='decomposition'"
            )
        reason = "requested by the user"
    else:
        reason = "requested by the user"
    if strategy == "anisoplanatic" and calibration is not None and not math.isfinite(calibration.scale[0]):
        raise ValueError("the plate scale of the system cannot be measured; use strategy='decomposition'")
    logger.info("extended imaging strategy: %s (%s)", strategy, reason)

    if strategy == "isoplanatic":
        assert calibration is not None
        power = _isoplanatic(scene, calibration)
    elif strategy == "anisoplanatic":
        assert calibration is not None
        power = _anisoplanatic(scene, system, calibration, bandwidths, int(field_points))
    else:
        power = _decomposition(scene, system, bandwidths, int(batch))

    for sources in [points] if isinstance(points, SourceList) else list(points or []):
        power = power + system.response(sources)

    return ImageResult(
        power=power,
        grid=cast(Grid, system.grid),
        wavelengths=scene.wavelengths,
        weights=cast(np.ndarray, system.weights),
        power_unit=cast(u.UnitBase, system.power_unit),
        strategy=strategy,
        reason=reason,
    )
