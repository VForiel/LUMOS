r"""Parametric diffuse components of scenes (step P8.3, REQ-SCN-3).

Each component produces a :class:`~lumos.fields.RadianceField` (W m⁻² sr⁻¹ m⁻¹) on its own angular grid of
``n_pixels`` × ``n_pixels`` pixels of ``pixel_scale``, centred on its position ``(x0, y0)``, at the spectral
channels of the context. Pixels are averaged over ``supersample`` × ``supersample`` sub-pixels, which
anti-aliases sharp edges. Add components to a :class:`~lumos.scenes.Scene` to sum them on the scene grid.

Compact components
------------------
:class:`UniformDisk`, :class:`GaussianComponent`, :class:`SersicComponent` and :class:`InclinedRing` hold the
total spectral flux density ``flux_density`` :math:`F_{\nu,0}` at ``reference_wavelength`` :math:`\lambda_0`.
With a shape :math:`S(\theta)` normalised by its integral over the sky, the radiance is

.. math:: L_\lambda(\theta) = F_\nu(\lambda)\,\frac{c}{\lambda^2}\,\frac{S(\theta)}{\int S\,d\Omega},
          \qquad F_\nu(\lambda) = F_{\nu,0}\,\frac{s(\lambda)}{s(\lambda_0)},

with the spectral shape :math:`s = 1` (``spectrum="flat"``, flat in :math:`F_\nu`) or the Planck function
:math:`s = B_\nu(\lambda, T)` (``spectrum="blackbody"``). Flux falling outside the grid of the component is
lost (the normalisation uses the analytic integral of the shape when it exists).

- **Uniform disk** of radius :math:`R`: :math:`S = 1` for :math:`r \le R`, integral :math:`\pi R^2`.
- **Gaussian** of full width at half maximum :math:`w`: :math:`S = e^{-r^2 / 2\sigma^2}` with
  :math:`\sigma = w / (2\sqrt{2\ln 2})`, integral :math:`2\pi\sigma^2`.
- **Sérsic** profile of index :math:`n` and effective radius :math:`R_e` (Sérsic 1963):
  :math:`S = \exp\{-b_n[(r/R_e)^{1/n} - 1]\}`, where :math:`b_n` solves
  :math:`\gamma(2n, b_n) = \Gamma(2n)/2` so that :math:`R_e` encloses half of the light; integral
  :math:`2\pi n R_e^2 e^{b_n}\Gamma(2n) / b_n^{2n}` (Graham & Driver 2005, PASA 22, 118, eqs. 1-4).
- **Inclined ring** (debris disks, exo-zodiacal disks): an optically thin flat ring between
  ``inner_radius`` and ``outer_radius`` with a surface brightness :math:`\propto r^{-p}` in its plane, seen
  with the ``inclination`` :math:`i` (0: face-on) and the major axis at ``position_angle``, measured from
  North (+y) towards East (+x). A sky point :math:`(x, y)` maps to the disk-plane radius
  :math:`r = \sqrt{u^2 + (v / \cos i)^2}` with :math:`u` along the major axis and :math:`v` along the minor
  axis; the brightness on the sky is the disk brightness divided by :math:`\cos i` (longer path through the
  disk), and the total flux is normalised on the grid.

Backgrounds
-----------
:class:`UniformBackground` and :class:`ZodiacalLight` cover their whole grid with a surface brightness
``surface_brightness`` :math:`\Sigma_{\nu,0}` (e.g. Jy arcsec⁻²) at ``reference_wavelength``:
:math:`L_\lambda = \Sigma_\nu(\lambda)\,c/\lambda^2`. The zodiacal light has the two-component spectral shape
of sunlight scattered by interplanetary dust plus the thermal emission of the dust,
:math:`s(\lambda) = A\,(R_\odot / 1\,\mathrm{au})^2\,B_\nu(\lambda, T_\odot) + (1 - A)\,B_\nu(\lambda,
T_\mathrm{dust})`, with the albedo :math:`A` and the dust temperature near 1 au (about 265 K; Leinert et al.
1998, A&AS 127, 1, §8): a simple model whose absolute level comes from the reference brightness.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from astropy import units as u
from scipy.special import gamma, gammaincinv

from lumos.core.element import Element, Values
from lumos.core.param import Param, Setting
from lumos.core.port import Output
from lumos.core.registry import register
from lumos.fields.grid import Grid
from lumos.fields.radiance import RadianceField
from lumos.scenes.scene import RADIANCE_UNIT

__all__ = [
    "DiffuseComponent",
    "GaussianComponent",
    "InclinedRing",
    "SersicComponent",
    "SurfaceBrightnessComponent",
    "UniformBackground",
    "UniformDisk",
    "ZodiacalLight",
    "sersic_b",
]

_H = 6.62607015e-34
_C = 299_792_458.0
_K = 1.380649e-23
_SUN_TEMPERATURE = 5772.0  # K, IAU 2015 Resolution B3 nominal effective temperature
_SUN_DILUTION = (6.957e8 / 1.495978707e11) ** 2  # (R_sun / 1 au)², IAU 2015 nominal values

Shape = Callable[[Values, np.ndarray, np.ndarray], np.ndarray]
Integral = Callable[[Values], float | None]


def _planck_nu(lam: np.ndarray, temperature: float) -> np.ndarray:
    """Planck function per unit frequency, up to a constant factor (shape only)."""
    nu = _C / lam
    with np.errstate(over="ignore"):
        return nu**3 / np.expm1(_H * nu / (_K * temperature))


def _spectral_shape(values: Values, lam: np.ndarray) -> np.ndarray:
    """Spectral shape s(λ) / s(λ0) of the flux density per unit frequency."""
    if values.spectrum == "flat":
        return np.ones_like(lam)
    reference = np.array([float(values.reference_wavelength)])
    return _planck_nu(lam, values.temperature) / _planck_nu(reference, values.temperature)


def _component_grid(values: Values) -> Grid:
    step = float(values.pixel_scale)
    return Grid(
        int(values.n_pixels), dx=step * u.rad, x0=float(values.x0) * u.rad, y0=float(values.y0) * u.rad
    )


def _supersampled(values: Values, grid: Grid, shape: Shape) -> np.ndarray:
    """Average ``shape`` over sub-pixels; coordinates relative to the centre of the component."""
    x, y = (np.asarray(c) for c in grid.axes(backend="numpy"))
    x, y = x - float(values.x0), y - float(values.y0)
    s = max(int(values.supersample), 1)
    offsets = ((np.arange(s) + 0.5) / s - 0.5) * grid.dx_si
    total = np.zeros(grid.shape)
    for oy in offsets:
        for ox in offsets:
            total += shape(values, (x + ox)[None, :], (y + oy)[:, None])
    return total / (s * s)


def _compact_transfer(shape: Shape, integral: Integral) -> staticmethod:
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the radiance of the component on its grid."""
        lam = np.asarray(values.wavelengths, dtype=float)
        grid = _component_grid(values)
        map_ = _supersampled(values, grid, shape)
        norm = integral(values)
        if norm is None:
            norm = float(map_.sum()) * grid.dx_si * grid.dy_si
        f_lambda = float(values.flux_density) * _spectral_shape(values, lam) * _C / lam**2
        data = f_lambda[:, None, None] * map_[None] / (norm if norm > 0 else 1.0)
        return {"out": RadianceField(data * RADIANCE_UNIT, grid, lam * u.m)}

    return staticmethod(transfer)


class DiffuseComponent(Element):
    """Base class of compact diffuse components: total flux, spectrum, position and sampling."""

    observation_dependent = True
    flux_density = Param(
        unit=u.Jy, default=1 * u.Jy, bounds=(0 * u.Jy, None), doc="total F_nu at the reference wavelength"
    )
    reference_wavelength = Param(
        unit=u.um, default=1 * u.um, bounds=(0 * u.m, None), inclusive=(False, True), doc="wavelength of F_nu"
    )
    spectrum = Setting(str, default="flat", choices=("flat", "blackbody"), doc="spectral shape of F_nu")
    temperature = Param(
        unit=u.K,
        default=5772 * u.K,
        bounds=(0 * u.K, None),
        inclusive=(False, True),
        doc="temperature of the blackbody spectrum",
    )
    x0 = Param(unit=u.mas, default=0 * u.mas, doc="position along x (East)")
    y0 = Param(unit=u.mas, default=0 * u.mas, doc="position along y (North)")
    pixel_scale = Param(
        unit=u.mas,
        default=2 * u.mas,
        bounds=(0 * u.rad, None),
        inclusive=(False, True),
        doc="angular pixel size of the map",
    )
    n_pixels = Setting(int, default=64, doc="pixels per side of the component grid")
    supersample = Setting(int, default=4, doc="sub-pixels per pixel side")
    out = Output(RadianceField)


# --- Uniform disk -------------------------------------------------------------------------------------------


def _disk(values: Values, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x**2 + y**2 <= float(values.radius) ** 2).astype(float)


@register("lumos.scenes.UniformDisk")
class UniformDisk(DiffuseComponent):
    """Disk of uniform surface brightness (resolved star, planet, uniform nebula)."""

    radius = Param(
        unit=u.mas,
        default=10 * u.mas,
        bounds=(0 * u.rad, None),
        inclusive=(False, True),
        doc="angular radius of the disk",
    )
    transfer = _compact_transfer(_disk, lambda v: math.pi * float(v.radius) ** 2)


# --- Gaussian -----------------------------------------------------------------------------------------------


def _sigma(values: Values) -> float:
    return float(values.fwhm) / (2 * math.sqrt(2 * math.log(2)))


def _gaussian(values: Values, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.exp(-(x**2 + y**2) / (2 * _sigma(values) ** 2))


@register("lumos.scenes.GaussianComponent")
class GaussianComponent(DiffuseComponent):
    """Circular Gaussian surface brightness."""

    fwhm = Param(
        unit=u.mas,
        default=10 * u.mas,
        bounds=(0 * u.rad, None),
        inclusive=(False, True),
        doc="full width at half maximum",
    )
    transfer = _compact_transfer(_gaussian, lambda v: 2 * math.pi * _sigma(v) ** 2)


# --- Sérsic -------------------------------------------------------------------------------------------------


def sersic_b(index: float) -> float:
    r"""Return :math:`b_n`, solution of :math:`P(2n, b_n) = 1/2` (half of the light within :math:`R_e`)."""
    return float(gammaincinv(2 * index, 0.5))


def _sersic(values: Values, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n, re = float(values.index), float(values.effective_radius)
    return np.exp(-sersic_b(n) * ((np.hypot(x, y) / re) ** (1 / n) - 1))


def _sersic_integral(values: Values) -> float:
    n, re = float(values.index), float(values.effective_radius)
    b = sersic_b(n)
    return float(2 * math.pi * n * re**2 * math.exp(b) * gamma(2 * n) / b ** (2 * n))


@register("lumos.scenes.SersicComponent")
class SersicComponent(DiffuseComponent):
    """Sérsic profile (n = 1: exponential disk, n = 4: de Vaucouleurs elliptical galaxy)."""

    effective_radius = Param(
        unit=u.mas,
        default=10 * u.mas,
        bounds=(0 * u.rad, None),
        inclusive=(False, True),
        doc="radius enclosing half of the flux",
    )
    index = Param(default=1.0, bounds=(0.2, 10), doc="Sérsic index n")
    transfer = _compact_transfer(_sersic, _sersic_integral)


# --- Inclined ring ------------------------------------------------------------------------------------------


def _ring(values: Values, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    pa = float(values.position_angle)
    cos_i = max(math.cos(float(values.inclination)), 1e-6)
    major = x * math.sin(pa) + y * math.cos(pa)  # along the major axis (PA from North towards East)
    minor = x * math.cos(pa) - y * math.sin(pa)
    r = np.hypot(major, minor / cos_i)
    inside = (r >= float(values.inner_radius)) & (r <= float(values.outer_radius))
    with np.errstate(divide="ignore"):
        brightness = np.where(inside, np.maximum(r, 1e-300) ** (-float(values.power_law)), 0.0)
    return brightness / cos_i


@register("lumos.scenes.InclinedRing")
class InclinedRing(DiffuseComponent):
    """Optically thin flat ring or disk seen inclined (debris disks, exo-zodiacal dust)."""

    inner_radius = Param(
        unit=u.mas, default=20 * u.mas, bounds=(0 * u.rad, None), doc="inner radius of the ring"
    )
    outer_radius = Param(
        unit=u.mas,
        default=30 * u.mas,
        bounds=(0 * u.rad, None),
        inclusive=(False, True),
        doc="outer radius of the ring",
    )
    inclination = Param(unit=u.deg, default=0 * u.deg, bounds=(0 * u.deg, 90 * u.deg), doc="0: face-on")
    position_angle = Param(unit=u.deg, default=0 * u.deg, doc="major axis, from North towards East")
    power_law = Param(default=0.0, doc="index p of the surface brightness r^-p in the plane of the ring")
    transfer = _compact_transfer(_ring, lambda v: None)


# --- Backgrounds --------------------------------------------------------------------------------------------


def _background_transfer(shape: Callable[[Values, np.ndarray], np.ndarray]) -> staticmethod:
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the uniform radiance of the background on its grid."""
        lam = np.asarray(values.wavelengths, dtype=float)
        grid = _component_grid(values)
        reference = np.array([float(values.reference_wavelength)])
        relative = shape(values, lam) / shape(values, reference)
        sigma_nu = float(values.surface_brightness)  # W m-2 Hz-1 sr-1 (SI of Jy / sr)
        radiance = sigma_nu * relative * _C / lam**2
        data = np.broadcast_to(radiance[:, None, None], (lam.size, *grid.shape)).copy()
        return {"out": RadianceField(data * RADIANCE_UNIT, grid, lam * u.m)}

    return staticmethod(transfer)


class SurfaceBrightnessComponent(Element):
    """Base class of uniform backgrounds: surface brightness, spectrum and sampling."""

    observation_dependent = True
    surface_brightness = Param(
        unit=u.Jy / u.arcsec**2,
        default=1e-6 * u.Jy / u.arcsec**2,
        bounds=(0 * u.Jy / u.sr, None),
        doc="surface brightness (F_nu per solid angle) at the reference wavelength",
    )
    reference_wavelength = Param(
        unit=u.um,
        default=1 * u.um,
        bounds=(0 * u.m, None),
        inclusive=(False, True),
        doc="wavelength of the surface brightness",
    )
    x0 = Param(unit=u.mas, default=0 * u.mas, doc="centre of the grid along x")
    y0 = Param(unit=u.mas, default=0 * u.mas, doc="centre of the grid along y")
    pixel_scale = Param(
        unit=u.mas,
        default=10 * u.mas,
        bounds=(0 * u.rad, None),
        inclusive=(False, True),
        doc="angular pixel size of the map",
    )
    n_pixels = Setting(int, default=64, doc="pixels per side of the background grid")
    out = Output(RadianceField)


def _background_shape(values: Values, lam: np.ndarray) -> np.ndarray:
    if values.spectrum == "flat":
        return np.ones_like(lam)
    return _planck_nu(lam, values.temperature)


@register("lumos.scenes.UniformBackground")
class UniformBackground(SurfaceBrightnessComponent):
    """Uniform background (sky, thermal background of the instrument) with a flat or blackbody spectrum."""

    spectrum = Setting(
        str, default="flat", choices=("flat", "blackbody"), doc="'flat' or 'blackbody' spectrum"
    )
    temperature = Param(
        unit=u.K,
        default=280 * u.K,
        bounds=(0 * u.K, None),
        inclusive=(False, True),
        doc="temperature of the blackbody spectrum",
    )
    transfer = _background_transfer(_background_shape)


def _zodi_shape(values: Values, lam: np.ndarray) -> np.ndarray:
    albedo = float(values.albedo)
    scattered = albedo * _SUN_DILUTION * _planck_nu(lam, _SUN_TEMPERATURE)
    thermal = (1 - albedo) * _planck_nu(lam, float(values.dust_temperature))
    return scattered + thermal


@register("lumos.scenes.ZodiacalLight")
class ZodiacalLight(SurfaceBrightnessComponent):
    """Local zodiacal light: uniform, with scattered sunlight and thermal dust emission."""

    albedo = Param(default=0.2, bounds=(0, 1), doc="albedo of the interplanetary dust")
    dust_temperature = Param(
        unit=u.K,
        default=265 * u.K,
        bounds=(0 * u.K, None),
        inclusive=(False, True),
        doc="temperature of the thermal emission of the dust",
    )
    transfer = _background_transfer(_zodi_shape)
