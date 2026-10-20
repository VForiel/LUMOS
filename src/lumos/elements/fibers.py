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
:class:`~lumos.fields.ModalField`, one guided lane whose modes are the LP modes and polarisations), and
``out``, the near field :math:`\sum_n a_n \psi_n` on the output face, sampled on the input grid. On a
:class:`~lumos.fields.Bus` of beams, the fibre acts on every lane (one identical fibre per lane, ADR-0023)
and both outputs are buses.

Injection from a pupil plane
----------------------------
With ``injection_focal_length`` :math:`f > 0`, the input is a field in a pupil plane (the output of a
:class:`~lumos.elements.telescopes.TelescopeArray` or a :class:`~lumos.elements.telescopes.Telescope`) focused
on the fibre face by an ideal lens. The lens maps the face to the pupil by a Fourier transform, which is
unitary, so the overlap is computed in the pupil with the modes imaged there,

.. math:: \tilde\psi_n(x, y) = \frac{1}{\lambda f} \iint \psi_n(\xi, \eta)\,
          e^{-2i\pi (x\xi + y\eta)/(\lambda f)}\, d\xi\, d\eta ,

a Gaussian of waist :math:`w_p = \lambda f / (\pi w_0)` for the Gaussian model (computed by a matrix Fourier
transform of the sampled mode otherwise). This gives the coupling efficiency of a telescope pupil (at most
about 81 % for a circular pupil) and the antenna lobe of the fibre for off-axis sources (Shaklan & Roddier
1988; Ruilier 1998, Proc. SPIE 3350, 319). The ``out`` port is then the guided light collimated back by the
same lens, :math:`\sum_n a_n \tilde\psi_n`, on the input grid.

Modes come from :mod:`lumos.physics.fibers`: exact LP modes (``mode_model="lp"``; modes with
:math:`\ell \ge 1` have a cosine ``a`` and a sine ``b`` orientation), or the Gaussian approximation of LP01
(``"gaussian"``) with the given mode field diameter or Marcuse's waist. The modes are those guided at
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
        fields[0] = np.exp(-(r**2) / _waist(values, wavelength) ** 2)
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


def _waist(values: Values, wavelength: float) -> float:
    """Waist of the Gaussian model (given mode field diameter, or Marcuse's waist)."""
    waist = values.mode_field_diameter / 2
    return waist if waist > 0 else marcuse_waist(_v(values, wavelength)) * values.core_radius


def _pupil_fields(values: Values, grid: Grid, wavelength: float, keys: list[Key]) -> np.ndarray:
    """Modes imaged in the pupil of a lens of focal ``injection_focal_length``, ``(n, ny, nx)`` complex.

    They are normalised on the whole plane (the transform is unitary), not on ``grid``: the part of a mode
    falling outside the grid is not coupled, as with a pupil field that is zero there.
    """
    f = values.injection_focal_length
    xx, yy = (np.asarray(c, dtype=float) for c in grid.coords(backend="numpy"))
    if values.mode_model == "gaussian":
        w_p = wavelength * f / (math.pi * _waist(values, wavelength))
        gaussian = math.sqrt(2 / (math.pi * w_p**2)) * np.exp(-(xx**2 + yy**2) / w_p**2)
        return gaussian[None].astype(complex)
    # sample the modes finely on the fibre face, then a matrix Fourier transform to the pupil grid
    extent = 8 * max(values.core_radius, _waist(values, wavelength))
    n = _FACE_SAMPLES
    face = Grid(n, dx=extent / n * u.m)
    modes = _fields(values, face, wavelength, keys)  # normalised on the fine face grid
    xi = np.asarray(face.axes(backend="numpy")[0], dtype=float)
    x_axis, y_axis = (np.asarray(a, dtype=float) for a in grid.axes(backend="numpy"))
    ex = np.exp(-2j * math.pi * np.outer(x_axis, xi) / (wavelength * f))
    ey = np.exp(-2j * math.pi * np.outer(y_axis, xi) / (wavelength * f))
    scale = face.dx_si * face.dy_si / (wavelength * f)
    return np.einsum("yj,njk,xk->nyx", ey, modes, ex) * scale


#: Samples per side of the fibre face for the modes imaged in a pupil (LP model).
_FACE_SAMPLES = 128


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
    pupil = values.injection_focal_length > 0
    for k, lam in enumerate(lams):
        modes = (
            _pupil_fields(values, grid, float(lam), keys)
            if pupil
            else _fields(values, grid, float(lam), keys)
        )
        guided = np.nan_to_num(_effective_indices(values, float(lam), keys))
        beta = 2 * math.pi / lam * guided
        propagation = loss * np.exp(1j * beta * values.length)
        propagation = propagation * (np.sum(np.abs(modes) ** 2, axis=(1, 2)) > 0)
        coupled = np.einsum("mpyx,nyx->mpn", field[:, k], np.conj(modes)) * area * propagation
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
        sources=signal.sources if bundle else None,
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
    injection_focal_length = Param(
        unit=u.mm,
        default=0 * u.mm,
        bounds=(0 * u.m, None),
        doc="focal length of a lens focusing a pupil-plane input on the face (0: the input is on the face)",
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
