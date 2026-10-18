r"""Bulk optics (§5.9, step P6.3).

Free space, thin lenses and mirrors, beam splitters, filters, delay lines, polarisers, wave plates and
Zernike phase plates. Every
element acts on :class:`~lumos.fields.Wavefront`, :class:`~lumos.fields.Wavecube` and
:class:`~lumos.fields.IncoherentBundle` signals (REQ-EL-8), channel by channel for polychromatic light and
mode by mode for bundles; the spectral filter also acts on :class:`~lumos.fields.RadianceField` signals.

Conventions and physics
-----------------------
- Fields evolve as :math:`\exp[i(kz - \omega t)]`: a vacuum path difference :math:`d` multiplies the field by
  :math:`\exp(2i\pi d / \lambda)` (:class:`DelayLine`).
- A thin lens of focal length :math:`f` multiplies the field by :math:`\exp(-ik r^2 / 2f)` (Goodman,
  *Introduction to Fourier Optics*, §5.1), with :math:`r` measured from the origin of the grid coordinates.
  A thin mirror is modelled *unfolded*: a spherical mirror of focal length :math:`f = R/2` acts like the lens
  of the same focal length on the field travelling along the folded axis.
- A lossless beam splitter of power reflectance :math:`R` has the amplitude coefficients
  :math:`t = \sqrt{1 - R}` and :math:`r = i\sqrt{R}`, which make its scattering matrix unitary
  (Saleh & Teich, *Fundamentals of Photonics*, §13.1).
- A spectral filter has a top-hat power transmission :math:`T` for :math:`|\lambda - \lambda_0| \le
  \Delta\lambda / 2` (amplitude :math:`\sqrt{T}`) and blocks the other channels.
- Polarisers and wave plates apply the Jones matrices of :mod:`lumos.fields.jones` (Hecht, *Optics*, §8).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from astropy import units as u

from lumos.backend._core import get
from lumos.core.element import Element, Values
from lumos.core.param import Param, Setting
from lumos.core.port import Input, Output
from lumos.core.registry import register
from lumos.core.signal import OpticalSignal
from lumos.fields import jones
from lumos.fields.grid import GridSetting
from lumos.fields.incoherent import IncoherentBundle
from lumos.fields.radiance import RadianceField
from lumos.fields.wavecube import Wavecube
from lumos.fields.wavefront import Wavefront
from lumos.propagation import propagate

__all__ = [
    "BeamSplitter",
    "DelayLine",
    "FreeSpace",
    "PhasePlate",
    "Polarizer",
    "SpectralFilter",
    "ThinLens",
    "ThinMirror",
    "WavePlate",
    "chromatic_factor",
]

Coherent = Wavefront | Wavecube | IncoherentBundle


def _coherent(signal: Any, element: str) -> Coherent:
    if not isinstance(signal, Wavefront | Wavecube | IncoherentBundle):
        raise TypeError(
            f"{element} acts on wavefronts, wavecubes and bundles, not on a {type(signal).__name__}"
        )
    return signal


def _wavelengths_si(signal: Any) -> np.ndarray:
    member = signal[0] if isinstance(signal, IncoherentBundle) else signal
    if isinstance(member, Wavefront):
        return np.array([member.wavelength_si])
    return np.asarray(member.wavelengths_si, dtype=float)


def _monochromatic(signal: Any) -> bool:
    member = signal[0] if isinstance(signal, IncoherentBundle) else signal
    return isinstance(member, Wavefront)


def chromatic_factor(signal: Coherent, factor: Callable[[Any, Any], Any]) -> Coherent:
    """Multiply every spectral channel of ``signal`` by ``factor(wavelengths, xp)``.

    Parameters
    ----------
    signal
        Wavefront, wavecube or bundle.
    factor
        Function of the vacuum wavelengths (a backend array of shape ``(nλ, 1, 1)``, in metres) and of the
        array namespace, returning a transmission broadcastable to ``(nλ, ny, nx)``.
    """
    xp = get(signal.backend).xp
    lam = get(signal.backend).asarray(_wavelengths_si(signal)[:, None, None])
    values = factor(lam, xp)
    if _monochromatic(signal):
        values = values[0]
    return signal * values


def _phase(signal: Coherent, opd: Callable[[Any, Any], Any]) -> Coherent:
    r"""Multiply ``signal`` by :math:`\exp(2i\pi\,\mathrm{OPD}/\lambda)` for the OPD map ``opd(grid, xp)``."""

    def factor(lam: Any, xp: Any) -> Any:
        return xp.exp(1j * (2 * math.pi / lam) * opd(signal.grid, xp))

    return chromatic_factor(signal, factor)


# --- Free space, lenses and mirrors -----------------------------------------------------------------


@register("lumos.optics.FreeSpace")
class FreeSpace(Element):
    """Propagation in vacuum over a distance, with the methods of :func:`lumos.propagation.propagate`."""

    distance = Param(unit=u.mm, default=0 * u.mm, doc="propagation distance (negative to back-propagate)")
    method = Setting(str, default="auto", doc="propagation method ('auto' lets the planner choose)")
    output_grid = GridSetting(doc="sampling of the output plane (None: the input grid)")
    in_ = Input(OpticalSignal)
    out = Output(OpticalSignal)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Propagate the incoming field to the output plane."""
        signal = _coherent(inputs["in"], "FreeSpace")
        out = propagate(signal, values.distance * u.m, output_grid=values.output_grid, method=values.method)
        return {"out": out}

    @staticmethod
    def transfer_symbolic(values: Values, inputs: Any) -> dict[str, Any]:
        """Propagate a symbolic field: closed form when known, logged numerical fallback otherwise."""
        from lumos.symbolic.closed_forms import propagate_symbolic

        out = propagate_symbolic(
            inputs["in"], float(values.distance), output_grid=values.output_grid, method=values.method
        )
        return {"out": out}


def _lens(values: Values, inputs: Any, element: str) -> dict[str, Any]:
    signal = _coherent(inputs["in"], element)
    f = values.focal_length

    def opd(grid: Any, xp: Any) -> Any:
        xx, yy = grid.coords(backend=signal.backend)
        return -(xx**2 + yy**2) / (2 * f)

    return {"out": _phase(signal, opd)}


def _symbolic_lens(values: Values, inputs: Any) -> dict[str, Any]:
    from lumos.symbolic.closed_forms import through_lens

    return {"out": through_lens(inputs["in"], float(values.focal_length))}


@register("lumos.optics.ThinLens")
class ThinLens(Element):
    r"""Thin lens: multiplies the field by :math:`\exp(-ik r^2 / 2f)` (converging for :math:`f > 0`)."""

    focal_length = Param(unit=u.mm, default=100 * u.mm, doc="focal length (negative: diverging lens)")
    in_ = Input(OpticalSignal)
    out = Output(OpticalSignal)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Apply the quadratic phase of the lens."""
        return _lens(values, inputs, "ThinLens")

    transfer_symbolic = staticmethod(_symbolic_lens)


@register("lumos.optics.ThinMirror")
class ThinMirror(Element):
    """Thin spherical mirror of focal length ``R / 2``, modelled unfolded as the equivalent thin lens."""

    focal_length = Param(unit=u.mm, default=100 * u.mm, doc="focal length R/2 (negative: convex mirror)")
    in_ = Input(OpticalSignal)
    out = Output(OpticalSignal)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Apply the quadratic phase of the mirror."""
        return _lens(values, inputs, "ThinMirror")

    transfer_symbolic = staticmethod(_symbolic_lens)


# --- Beam splitter, delay line, filter ----------------------------------------------------------------


@register("lumos.optics.BeamSplitter")
class BeamSplitter(Element):
    r"""Lossless beam splitter: :math:`t = \sqrt{1 - R}` (transmitted), :math:`r = i\sqrt{R}` (reflected)."""

    reflectance = Param(default=0.5, bounds=(0, 1), doc="power reflectance R")
    in_ = Input(OpticalSignal)
    reflected = Output(OpticalSignal)
    transmitted = Output(OpticalSignal)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Split the incoming light in two beams."""
        signal = inputs["in"]
        r = values.reflectance
        if isinstance(signal, RadianceField):  # incoherent radiance: split the power
            return {"reflected": signal * r, "transmitted": signal * (1 - r)}
        return {"reflected": signal * (1j * r**0.5), "transmitted": signal * (1 - r) ** 0.5}

    @staticmethod
    def transfer_symbolic(values: Values, inputs: Any) -> dict[str, Any]:
        r"""Split a symbolic field: :math:`i\sqrt{R}\,U` and :math:`\sqrt{1 - R}\,U`."""
        import sympy as sp

        field = inputs["in"]
        reflectance = sp.Float(values.reflectance)
        return {
            "reflected": field.multiply(sp.I * sp.sqrt(reflectance)),
            "transmitted": field.multiply(sp.sqrt(1 - reflectance)),
        }


@register("lumos.optics.DelayLine")
class DelayLine(Element):
    r"""Vacuum optical path delay :math:`d`: multiplies each channel by :math:`\exp(2i\pi d / \lambda)`."""

    delay = Param(unit=u.um, default=0 * u.um, doc="optical path difference added")
    in_ = Input(OpticalSignal)
    out = Output(OpticalSignal)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Apply the chromatic phase of the delay."""
        signal = _coherent(inputs["in"], "DelayLine")
        d = values.delay
        return {"out": chromatic_factor(signal, lambda lam, xp: xp.exp(1j * (2 * math.pi * d) / lam))}

    @staticmethod
    def transfer_symbolic(values: Values, inputs: Any) -> dict[str, Any]:
        r"""Multiply a symbolic field by :math:`\exp(2i\pi d / \lambda)`."""
        import sympy as sp

        from lumos.symbolic.symbols import lam

        return {"out": inputs["in"].multiply(sp.exp(sp.I * 2 * sp.pi * sp.Float(values.delay) / lam))}


@register("lumos.optics.SpectralFilter")
class SpectralFilter(Element):
    """Top-hat band-pass filter: power transmission inside the band, opaque outside."""

    center = Param(
        unit=u.nm,
        default=550 * u.nm,
        bounds=(0 * u.m, None),
        inclusive=(False, True),
        doc="central wavelength of the band",
    )
    width = Param(
        unit=u.nm,
        default=100 * u.nm,
        bounds=(0 * u.m, None),
        inclusive=(False, True),
        doc="full width of the band",
    )
    transmission = Param(default=1.0, bounds=(0, 1), doc="power transmission in the band")
    in_ = Input(OpticalSignal)
    out = Output(OpticalSignal)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Keep the channels inside the band."""
        signal = inputs["in"]
        lam = signal.wavelengths_si if isinstance(signal, RadianceField) else _wavelengths_si(signal)
        inside = np.abs(lam - values.center) <= values.width / 2 * (1 + 1e-12)
        power = np.where(inside, values.transmission, 0.0)
        if isinstance(signal, RadianceField):
            shape = (-1, 1, 1, 1) if signal.has_stokes else (-1, 1, 1)
            return {"out": signal * get(signal.backend).asarray(power.reshape(shape))}
        amplitude = np.sqrt(power)
        signal = _coherent(signal, "SpectralFilter")
        return {
            "out": chromatic_factor(
                signal, lambda _lam, xp: get(signal.backend).asarray(amplitude)[:, None, None]
            )
        }

    @staticmethod
    def transfer_symbolic(values: Values, inputs: Any) -> dict[str, Any]:
        """Multiply a symbolic field by the amplitude transmission of the band (``Piecewise`` in λ)."""
        import sympy as sp

        from lumos.symbolic.symbols import lam

        field = inputs["in"]
        amplitude = sp.sqrt(sp.Float(values.transmission))
        half = values.width / 2 * (1 + 1e-12)
        if not field.depends_on_wavelength and field.wavelength is not None:
            wavelength = float(field.wavelength.to_value(u.m, equivalencies=u.spectral()))
            factor = amplitude if abs(wavelength - values.center) <= half else sp.Integer(0)
        else:
            factor = sp.Piecewise(
                (amplitude, sp.Abs(lam - sp.Float(values.center)) <= sp.Float(half)), (0, True)
            )
        return {"out": field.multiply(factor)}


# --- Polarisation -----------------------------------------------------------------------------------


@register("lumos.optics.Polarizer")
class Polarizer(Element):
    """Ideal linear polariser transmitting the polarisation at ``angle`` from +x."""

    angle = Param(unit=u.deg, default=0 * u.deg, doc="transmission axis, from +x")
    in_ = Input(OpticalSignal)
    out = Output(OpticalSignal)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Apply the Jones matrix of the polariser."""
        signal = _coherent(inputs["in"], "Polarizer")
        return {"out": jones.apply(jones.polarizer(values.angle * u.rad), signal)}


@register("lumos.optics.WavePlate")
class WavePlate(Element):
    """Linear retarder with its fast axis at ``angle`` (a ``retardance`` of 180° makes a half-wave plate)."""

    retardance = Param(unit=u.deg, default=180 * u.deg, doc="phase delay of the slow axis")
    angle = Param(unit=u.deg, default=0 * u.deg, doc="fast axis, from +x")
    in_ = Input(OpticalSignal)
    out = Output(OpticalSignal)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Apply the Jones matrix of the retarder."""
        signal = _coherent(inputs["in"], "WavePlate")
        matrix = jones.retarder(values.retardance * u.rad, values.angle * u.rad)
        return {"out": jones.apply(matrix, signal)}


# --- Phase plates (step P6.5) -------------------------------------------------------------------------


def _plate_opd(values: Values, grid: Any) -> np.ndarray:
    from lumos.physics.zernike import zernike_basis

    coefficients = np.atleast_1d(np.asarray(values.coefficients, dtype=float))
    basis = zernike_basis(grid, values.diameter * u.m, len(coefficients))
    return np.tensordot(coefficients, basis, axes=1)


@register("lumos.optics.PhasePlate")
class PhasePlate(Element):
    """Phase plate adding the OPD ``Σ a_j Z_j`` (Noll's Zernike modes, j from 1) over a disk."""

    diameter = Param(
        unit=u.mm,
        default=10 * u.mm,
        bounds=(0 * u.m, None),
        inclusive=(False, True),
        doc="diameter over which the Zernike modes are normalised",
    )
    coefficients = Param(unit=u.nm, shape=None, default=[0.0] * u.nm, doc="RMS OPD of modes 1, 2, ...")
    in_ = Input(OpticalSignal)
    out = Output(OpticalSignal)

    def opd(self, grid: Any) -> u.Quantity:
        """Return the optical path difference of the plate on ``grid`` (zero outside the disk)."""
        return _plate_opd(self.values(), grid) * u.m

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Add the optical path of the plate to every channel and mode."""
        signal = _coherent(inputs["in"], "PhasePlate")
        opd = _plate_opd(values, signal.grid)
        return {"out": _phase(signal, lambda grid, xp: xp.asarray(opd))}
