r"""Procedural scene components: random fields and test targets (step P8.6, REQ-SCN-8, §5.9).

Each component draws a pattern :math:`P \in [0, 1]` (relative level) on its own grid and produces the
spectral radiance :math:`L_\lambda = \Sigma_\nu(\lambda)\,P\,c/\lambda^2`, with the surface brightness
``surface_brightness`` :math:`\Sigma_{\nu,0}` at ``reference_wavelength`` and a flat or blackbody spectral
shape, as the backgrounds of :mod:`lumos.scenes.diffuse`.

Grids
-----
The grid has ``n_pixels`` × ``n_pixels`` pixels. By default it is angular, with pixels of ``pixel_scale``
(sky scenes); with ``pixel_size`` > 0 it is spatial, with pixels of that length (laboratory scenes, at the
distance of the :class:`~lumos.scenes.Scene`). The **sizes of the patterns** (radius, period, length) are
angles on angular grids, lengths on spatial grids, or numbers of pixels (``u.pix`` or a plain number) on both,
so that the same target serves both kinds of scene; a size :math:`s` covers :math:`s/p` pixels of size
:math:`p`. Positions are measured from the central pixel ``N // 2`` (§9 of the specifications). Edges are
anti-aliased by averaging ``supersample`` × ``supersample`` sub-pixels.

Patterns
--------
- :class:`GaussianRandomField`: a log-normal random texture of mean ``surface_brightness``, relative standard
  deviation ``contrast`` and power spectrum :math:`\propto k^{-\beta}` (``spectral_index`` β). White Gaussian
  noise is filtered by :math:`k^{-\beta/2}` in Fourier space, standardised to :math:`z`, then
  :math:`P \propto \exp(s z)` with :math:`s^2 = \ln(1 + c^2)` (log-normal of coefficient of variation
  :math:`c`), and finally rescaled to the exact mean and contrast. For small contrasts the power spectrum
  keeps the slope :math:`-\beta` (Wiener-Khinchin theorem). The noise comes from the reproducible random
  stream of the element (ADR-0007).
- :class:`SiemensStar`: ``n_spokes`` bright sectors alternating with dark ones within ``radius``; the local
  period grows with the radius, which probes all spatial frequencies (ISO 12233).
- :class:`Checkerboard`: squares of side ``period / 2`` with a corner on the axis.
- :class:`BarTarget`: ``n_bars`` vertical bars of width ``period / 2`` and length ``length``, centred, as
  in one element of the USAF 1951 resolution chart (MIL-STD-150A).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from astropy import units as u

from lumos.core.element import Element, Values, generator
from lumos.core.param import Param, Setting
from lumos.core.port import Output
from lumos.core.registry import register
from lumos.fields.grid import Grid
from lumos.fields.radiance import RadianceField
from lumos.scenes.diffuse import _planck_nu
from lumos.scenes.scene import RADIANCE_UNIT
from lumos.units import quantity_from_json, quantity_to_json

__all__ = [
    "BarTarget",
    "Checkerboard",
    "GaussianRandomField",
    "ProceduralComponent",
    "SiemensStar",
    "TargetSize",
]

_C = 299_792_458.0

Pattern = Callable[[Values, np.ndarray, np.ndarray], np.ndarray]


def _grid(values: Values) -> Grid:
    n = int(values.n_pixels)
    if float(values.pixel_size) > 0:
        return Grid(n, dx=float(values.pixel_size) * u.m)
    return Grid(n, dx=float(values.pixel_scale) * u.rad)


class TargetSize(Setting):
    """Size of a test target: an angle, a length, pixels (``u.pix``) or a plain number of pixels."""

    def __init__(self, *, default: float, doc: str = "") -> None:
        super().__init__(object, default=default, doc=doc)

    def validate(self, value: Any) -> tuple[Any, Any]:
        """Accept a positive number (pixels) or a scalar quantity of pixels, angle or length."""
        if isinstance(value, u.Quantity):
            if not value.isscalar:
                raise ValueError(f"{self.name}: expected a scalar size")
            unit = cast(u.UnitBase, value.unit)
            if not (unit == u.pix or unit.is_equivalent(u.rad) or unit.is_equivalent(u.m)):
                raise u.UnitsError(f"{self.name}: expected an angle, a length or pixels, got {value.unit}")
            size = value
        elif isinstance(value, int | float) and not isinstance(value, bool):
            size = float(value)
        else:
            raise TypeError(
                f"{self.name}: expected a size (number of pixels or quantity), got {type(value).__name__}"
            )
        if float(np.asarray(getattr(size, "value", size))) <= 0:
            raise ValueError(f"{self.name}: the size must be positive")
        return size, size

    def encode(self, value: Any) -> Any:
        """Return the JSON form: a number of pixels or a quantity."""
        return quantity_to_json(value) if isinstance(value, u.Quantity) else value

    def decode(self, data: Any) -> Any:
        """Return the size of a JSON form."""
        return quantity_from_json(data) if isinstance(data, dict) else float(data)


def _pixels(values: Values, name: str) -> float:
    """Return the size ``name`` of the pattern in pixels of its grid."""
    size = getattr(values, name)
    if not isinstance(size, u.Quantity) or size.unit == u.pix:
        return float(getattr(size, "value", size))
    spatial = float(values.pixel_size) > 0
    if cast(u.UnitBase, size.unit).is_equivalent(u.rad):
        if spatial:
            raise ValueError(f"{name} = {size} is an angle: it needs an angular grid (pixel_size = 0)")
        return float(np.asarray(size.to_value(u.rad))) / float(values.pixel_scale)
    if not spatial:
        raise ValueError(f"{name} = {size} is a length: it needs a spatial grid (pixel_size > 0)")
    return float(np.asarray(size.to_value(u.m))) / float(values.pixel_size)


def _pixel_axes(n: int) -> np.ndarray:
    return np.arange(n) - n // 2


def _spectral(values: Values, lam: np.ndarray) -> np.ndarray:
    """Surface brightness per unit frequency (W m-2 Hz-1 sr-1) at each wavelength."""
    if values.spectrum == "flat":
        relative = np.ones_like(lam)
    else:
        reference = np.array([float(values.reference_wavelength)])
        relative = _planck_nu(lam, values.temperature) / _planck_nu(reference, values.temperature)
    return float(values.surface_brightness) * relative


def _radiance(values: Values, pattern: np.ndarray) -> dict[str, Any]:
    lam = np.asarray(values.wavelengths, dtype=float)
    sigma = _spectral(values, lam) * _C / lam**2
    data = sigma[:, None, None] * pattern[None]
    return {"out": RadianceField(data * RADIANCE_UNIT, _grid(values), lam * u.m)}


def _anti_aliased(pattern: Pattern) -> staticmethod:
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the radiance of the pattern on its grid."""
        n = int(values.n_pixels)
        s = max(int(values.supersample), 1)
        offsets = (np.arange(s) + 0.5) / s - 0.5
        x, y = _pixel_axes(n).astype(float), _pixel_axes(n).astype(float)
        total = np.zeros((n, n))
        for oy in offsets:
            for ox in offsets:
                total += pattern(values, (x + ox)[None, :], (y + oy)[:, None])
        return _radiance(values, total / (s * s))

    return staticmethod(transfer)


class ProceduralComponent(Element):
    """Base class of procedural components: brightness, spectrum and grid (angular or spatial)."""

    observation_dependent = True
    surface_brightness = Param(
        unit=u.Jy / u.arcsec**2,
        default=1e-3 * u.Jy / u.arcsec**2,
        bounds=(0 * u.Jy / u.sr, None),
        doc="brightest level (random fields: mean) at the reference wavelength",
    )
    reference_wavelength = Param(
        unit=u.um,
        default=1 * u.um,
        bounds=(0 * u.m, None),
        inclusive=(False, True),
        doc="wavelength of the brightness",
    )
    spectrum = Setting(
        str, default="flat", choices=("flat", "blackbody"), doc="'flat' or 'blackbody' spectrum"
    )
    temperature = Param(
        unit=u.K,
        default=3000 * u.K,
        bounds=(0 * u.K, None),
        inclusive=(False, True),
        doc="temperature of the blackbody spectrum",
    )
    pixel_scale = Param(
        unit=u.mas,
        default=1 * u.mas,
        bounds=(0 * u.rad, None),
        inclusive=(False, True),
        doc="pixel size of the map",
    )
    pixel_size = Param(
        unit=u.um, default=0 * u.um, bounds=(0 * u.m, None), doc="> 0: spatial grid (laboratory)"
    )
    n_pixels = Setting(int, default=128, doc="pixels per side")
    supersample = Setting(int, default=4, doc="sub-pixels per pixel side for anti-aliasing")
    out = Output(RadianceField)


# --- Random field ------------------------------------------------------------------------------------------


@register("lumos.scenes.GaussianRandomField")
class GaussianRandomField(ProceduralComponent):
    """Log-normal random texture with a power-law power spectrum."""

    stochastic = True
    contrast = Param(default=0.1, bounds=(0, None), doc="standard deviation divided by the mean")
    spectral_index = Param(default=3.0, bounds=(0, None), doc="β of the power spectrum k^-β")

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return a reproducible random texture."""
        n = int(values.n_pixels)
        rng = generator(values)
        noise = rng.standard_normal((n, n))
        k = np.hypot(*np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="xy"))
        k[0, 0] = np.inf  # no constant term
        filtered = np.real(np.fft.ifft2(np.fft.fft2(noise) * k ** (-float(values.spectral_index) / 2)))
        z = (filtered - filtered.mean()) / filtered.std()
        c = float(values.contrast)
        texture = np.exp(math.sqrt(math.log1p(c * c)) * z)
        if c == 0:
            return _radiance(values, np.ones_like(texture))
        texture = 1 + (texture - texture.mean()) * (c / texture.std())
        return _radiance(values, np.maximum(texture, 0.0))


# --- Test targets -------------------------------------------------------------------------------------------


def _siemens(values: Values, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    phi = np.arctan2(y, x)
    inside = np.hypot(x, y) <= _pixels(values, "radius")
    return (inside & (np.sin(int(values.n_spokes) * phi) > 0)).astype(float)


@register("lumos.scenes.SiemensStar")
class SiemensStar(ProceduralComponent):
    """Siemens star: alternating bright and dark sectors."""

    n_spokes = Setting(int, default=16, doc="number of bright sectors")
    radius = TargetSize(default=60.0, doc="radius (angle, length or pixels)")
    transfer = _anti_aliased(_siemens)


def _checkerboard(values: Values, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    side = _pixels(values, "period") / 2
    return ((np.floor(x / side) + np.floor(y / side)) % 2 == 0).astype(float)


@register("lumos.scenes.Checkerboard")
class Checkerboard(ProceduralComponent):
    """Checkerboard of squares of side ``period / 2``, a corner on the axis."""

    period = TargetSize(default=8.0, doc="period: two squares (angle, length or pixels)")
    transfer = _anti_aliased(_checkerboard)


def _bars(values: Values, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    period = _pixels(values, "period")
    n = int(values.n_bars)
    start = -(n - 0.5) * period / 2  # left edge of the first bar; the group of bars is centred
    position = (x - start) / period
    in_bar = (position >= 0) & (position < n) & ((position % 1) < 0.5)
    in_length = np.abs(y) <= _pixels(values, "length") / 2
    return (in_bar & in_length).astype(float)


@register("lumos.scenes.BarTarget")
class BarTarget(ProceduralComponent):
    """Group of bright vertical bars (one element of a USAF 1951 chart)."""

    n_bars = Setting(int, default=3, doc="bars of each group")
    period = TargetSize(default=10.0, doc="period: a bar and a gap (angle, length or pixels)")
    length = TargetSize(default=25.0, doc="length of the bars (angle, length or pixels)")
    transfer = _anti_aliased(_bars)
