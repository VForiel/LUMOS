r"""Optical fibres: injection, guided propagation and extraction (§5.9, step P6.7).

A :class:`Fiber` receives the field on its input face (a :class:`~lumos.fields.Wavefront`,
:class:`~lumos.fields.Wavecube` or :class:`~lumos.fields.IncoherentBundle`, the fibre axis at the coordinate
origin) and projects it on its guided modes :math:`\psi_n`, normalised on the grid
(:math:`\sum |\psi_n|^2 \Delta x \Delta y = 1`):

.. math:: a_n = \sum E\,\psi_n^*\, \Delta x\, \Delta y,

so :math:`|a_n|^2` is the power coupled into mode :math:`n` (the overlap-integral efficiency of Shaklan &
Roddier 1988 for a single mode). Over the length :math:`L` every mode gets the phase :math:`\beta_n L`,
:math:`\beta_n = 2\pi n_{\mathrm{eff},n}/\lambda`, and the power transmission :math:`10^{-\alpha L/10}` of the
attenuation :math:`\alpha`. Outputs: ``modes``, the amplitudes at the output end (a
:class:`~lumos.fields.ModalField`, one channel per mode and polarisation), and ``out``, the near field
:math:`\sum_n a_n \psi_n` on the output face, sampled on the input grid.

Modes come from :mod:`lumos.physics.fibers`: exact LP modes (``mode_model="lp"``; modes with
:math:`\ell \ge 1` have a cosine ``a`` and a sine ``b`` orientation), or the Gaussian approximation of LP01
(``"gaussian"``) with the given mode field diameter or Marcuse's waist. The channels are the modes guided at
the shortest wavelength; at longer wavelengths the modes beyond cutoff carry no light. Material dispersion is
not modelled (constant ``cladding_index`` and numerical aperture).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from astropy import units as u

from lumos.backend._core import get
from lumos.core.element import Element, Values
from lumos.core.param import Param, Setting
from lumos.core.port import Input, Output
from lumos.core.registry import register
from lumos.core.signal import OpticalSignal
from lumos.elements.optics import _coherent, _wavelengths_si
from lumos.fields.grid import Grid
from lumos.fields.incoherent import IncoherentBundle
from lumos.fields.modal import ModalField
from lumos.fields.wavecube import Wavecube
from lumos.physics.fibers import LPMode, lp_modes, lp_radial, marcuse_waist
from lumos.units import require_unit, to_si

__all__ = ["Fiber"]

#: SI value of 1 dB/km (astropy stores logarithmic units in dex).
_DB_PER_KM = float(to_si(1 * u.dB / u.km))

# (label, ell, m, orientation): orientation 0 is cos(ell φ), 1 is sin(ell φ)
Key = tuple[str, int, int, int]


def _v(values: Values, wavelength: float) -> float:
    return 2 * math.pi * values.core_radius * values.numerical_aperture / wavelength


def _keys(values: Values, wavelength: float) -> list[Key]:
    if values.mode_model == "gaussian":
        return [("LP01", 0, 1, 0)]
    keys: list[Key] = []
    for mode in lp_modes(_v(values, wavelength)):
        if mode.ell == 0:
            keys.append((mode.label, 0, mode.m, 0))
        else:
            keys += [(mode.label + "a", mode.ell, mode.m, 0), (mode.label + "b", mode.ell, mode.m, 1)]
    return keys


def _guided(values: Values, wavelength: float) -> dict[tuple[int, int], LPMode]:
    return {(mode.ell, mode.m): mode for mode in lp_modes(_v(values, wavelength))}


def _fields(values: Values, grid: Grid, wavelength: float, keys: list[Key]) -> np.ndarray:
    """Mode fields normalised on ``grid``, shape ``(n, ny, nx)``; zero for modes beyond cutoff."""
    xx, yy = (np.asarray(c, dtype=float) for c in grid.coords(backend="numpy"))
    r = np.hypot(xx, yy)
    phi = np.arctan2(yy, xx)
    area = grid.dx_si * grid.dy_si
    fields = np.zeros((len(keys), *grid.shape))
    if values.mode_model == "gaussian":
        waist = values.mode_field_diameter / 2
        if not waist > 0:
            waist = marcuse_waist(_v(values, wavelength)) * values.core_radius
        fields[0] = np.exp(-(r**2) / waist**2)
    else:
        guided = _guided(values, wavelength)
        for index, (_label, ell, m, orientation) in enumerate(keys):
            mode = guided.get((ell, m))
            if mode is None:
                continue
            angular = np.cos(ell * phi) if orientation == 0 else np.sin(ell * phi)
            fields[index] = lp_radial(mode, r / values.core_radius) * angular
    norms = np.sqrt(np.sum(fields**2, axis=(1, 2)) * area)
    return np.divide(fields, norms[:, None, None], out=np.zeros_like(fields), where=norms[:, None, None] > 0)


def _effective_indices(values: Values, wavelength: float, keys: list[Key]) -> np.ndarray:
    guided = _guided(values, wavelength)
    n_cl, na = values.cladding_index, values.numerical_aperture
    result = np.full(len(keys), np.nan)
    for index, (_label, ell, m, _orientation) in enumerate(keys):
        mode = guided.get((ell, m))
        if mode is not None:
            result[index] = math.sqrt(n_cl**2 + mode.b * na**2)
    return result


def _transfer(values: Values, signal: Any) -> tuple[ModalField, Any]:
    backend = get(signal.backend)
    lams = _wavelengths_si(signal)
    keys = _keys(values, float(lams.min()))
    grid = signal.grid
    area = grid.dx_si * grid.dy_si
    bundle = isinstance(signal, IncoherentBundle)
    member = signal[0] if bundle else signal
    polarized = member.polarized
    field = np.asarray(backend.to_numpy(signal.field))
    shape = field.shape
    if not bundle:
        field = field[None]
    if not isinstance(member, Wavecube):
        field = field[:, None]
    if not polarized:
        field = field[:, :, None]
    # field: (M, nλ, P, ny, nx)
    decibels = values.attenuation / _DB_PER_KM * values.length / 1000  # attenuation over the length, in dB
    loss = 10 ** (-decibels / 20)  # amplitude
    amplitudes = np.zeros((*field.shape[:3], len(keys)), dtype=complex)
    exit_field = np.zeros_like(field, dtype=complex)
    for k, lam in enumerate(lams):
        modes = _fields(values, grid, float(lam), keys)
        beta = 2 * math.pi / lam * np.nan_to_num(_effective_indices(values, float(lam), keys))
        propagation = loss * np.exp(1j * beta * values.length) * (np.sum(modes**2, axis=(1, 2)) > 0)
        coupled = np.einsum("mpyx,nyx->mpn", field[:, k], modes) * area * propagation
        amplitudes[:, k] = coupled
        exit_field[:, k] = np.einsum("mpn,nyx->mpyx", coupled, modes)

    n_pol = field.shape[2]
    labels = [label for label, *_ in keys]
    if n_pol == 2:
        labels = [f"{label}x" for label in labels] + [f"{label}y" for label in labels]
    channels = amplitudes.reshape(field.shape[0], len(lams), n_pol * len(keys))
    weights = member.weights if isinstance(member, Wavecube) else None
    modal = ModalField(
        channels if bundle else channels[0],
        lams * u.m,
        weights=weights,
        labels=labels,
        power_unit=signal.power_unit,
    )
    out = signal._with_field(backend.asarray(exit_field.reshape(shape)))
    return modal, out


@register("lumos.fibers.Fiber")
class Fiber(Element):
    """Step-index fibre: injection into LP modes, propagation over its length, near field at its output."""

    core_radius = Param(
        unit=u.um,
        default=4.1 * u.um,
        bounds=(0 * u.m, None),
        inclusive=(False, True),
        doc="radius of the core",
    )
    numerical_aperture = Param(
        default=0.12, bounds=(0, 1), inclusive=(False, True), doc="numerical aperture of the step-index fibre"
    )
    cladding_index = Param(default=1.444, bounds=(1, None), doc="refractive index of the cladding")
    length = Param(unit=u.m, default=1 * u.m, bounds=(0 * u.m, None), doc="length of the fibre")
    attenuation = Param(
        unit=u.dB / u.km, default=0 * u.dB / u.km, bounds=(0 * u.dB / u.km, None), doc="propagation loss"
    )
    mode_field_diameter = Param(
        unit=u.um, default=0 * u.um, bounds=(0 * u.m, None), doc="Gaussian model; 0: Marcuse"
    )
    mode_model = Setting(str, default="lp", choices=("lp", "gaussian"), doc="exact LP modes or Gaussian LP01")
    in_ = Input(OpticalSignal)
    modes = Output(ModalField, doc="modal amplitudes at the output end")
    out = Output(OpticalSignal, doc="near field on the output face")

    def v_number(self, wavelength: u.Quantity) -> float:
        """Normalised frequency ``V = 2π a NA / λ``."""
        return _v(self.values(), _wavelength(wavelength))

    def mode_labels(self, wavelength: u.Quantity) -> list[str]:
        """Names of the scalar modes guided at ``wavelength``."""
        return [label for label, *_ in _keys(self.values(), _wavelength(wavelength))]

    def mode_fields(self, grid: Grid, wavelength: u.Quantity) -> np.ndarray:
        """Normalised mode fields at ``wavelength`` sampled on ``grid``, shape ``(n, ny, nx)``."""
        lam = _wavelength(wavelength)
        values = self.values()
        return _fields(values, grid, lam, _keys(values, lam))

    def effective_indices(self, wavelength: u.Quantity) -> np.ndarray:
        """Effective indices of the modes guided at ``wavelength``."""
        lam = _wavelength(wavelength)
        values = self.values()
        return _effective_indices(values, lam, _keys(values, lam))

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Couple the light into the modes, propagate it and rebuild the output near field."""
        modal, out = _transfer(values, _coherent(inputs["in"], "Fiber"))
        return {"modes": modal, "out": out}


def _wavelength(value: u.Quantity) -> float:
    return float(to_si(require_unit(value, u.m, name="wavelength", equivalencies=u.spectral())))
