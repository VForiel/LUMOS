r"""Turbulent atmosphere and adaptive optics (§5.9, step P6.5).

:class:`Atmosphere` adds the optical path difference (OPD) of von Kármán turbulence layers to the light it
receives (:mod:`lumos.physics.turbulence`). Each layer :math:`i` has a fraction :math:`w_i` of the turbulence
(:math:`r_{0,i} = r_0 w_i^{-3/5}`) and is carried by the wind at :math:`\mathbf{v}_i` (Taylor's frozen flow:
:math:`\mathrm{OPD}_i(\mathbf{x}, t) = \mathrm{OPD}_i(\mathbf{x} - \mathbf{v}_i t, 0)`, ``t`` from the
context clock). The OPD is achromatic, so every channel gets the phase :math:`2\pi\,\mathrm{OPD}/\lambda` and
the Fried parameter scales as :math:`\lambda^{6/5}` (Roddier 1981). Screens are drawn from the reproducible
random stream of the element in its context (:attr:`~lumos.core.element.Element.stochastic`); the layers act
in the pupil plane (no scintillation).

A simple adaptive optics correction is built in: ``corrected_modes = J`` removes, over the pupil of diameter
``pupil_diameter``, the least-squares projection of the OPD on Noll's modes 1 … J (an ideal wavefront sensor
and corrector), leaving the residual variance :math:`\Delta_J (D/r_0)^{5/3}` of Noll (1976).

:class:`DeformableMirror` is a grid of actuators with Gaussian influence functions
:math:`w(r) = c^{(r/p)^2}` (coupling :math:`c` at the pitch :math:`p`); its surface
:math:`s = \sum_k a_k w(\mathbf{x} - \mathbf{x}_k)` changes the optical path by :math:`2s` (reflection).
:meth:`DeformableMirror.fit` finds the commands whose OPD best fits a map (least squares).

A closed adaptive optics loop (ADR-0020) adds two elements. :class:`WavefrontSensor` is an ideal
Shack-Hartmann sensor: in each square sub-aperture of side :math:`w` it measures the mean gradient of the
optical path, :math:`\langle \partial_x \mathrm{OPD} \rangle`, the tilt of the wavefront in radians (Hardy
1998, §5.3). The gradient of each pixel comes from the phase difference of its neighbours,
:math:`\arg(E_{j+1} E_{j-1}^*)\,\lambda / (4\pi\,\Delta x)`, so wavefronts of many waves are measured as
long as the phase changes by less than π between two pixels; channels and polarisations are weighted by their
intensity. It also gives the RMS optical path over the illuminated pupil, from the phase relative to the mean
complex amplitude (piston removed): exact while the residual stays below about λ/4. :class:`AOController` is
an integrator: from the slopes ``s`` it computes the correction :math:`\Delta c = -g R s` of a deformable
mirror of the same geometry, with :math:`R = (D^T D + \epsilon\,\sigma_{max}^2 I)^{-1} D^T` the regularised
pseudo-inverse of the interaction matrix :math:`D`. With separable Gaussian influence functions
:math:`f_k(t) = e^{-\alpha (t - a_k)^2}`, :math:`\alpha = -\ln c / p^2`, the mean slope of an actuator over
a sub-aperture is separable: ``D_x = 2 (M_y ⊗ K_x)`` with :math:`M` the mean of the influence over the
sub-aperture (error functions) and :math:`K` its difference across it divided by its width (the factor 2 of
reflection).
Fed back to the mirror with ``ctx.feedback("ctrl.commands", "dm.commands", mode="add")`` and run by
``ctx.loop``, the error of an aberration the mirror can reproduce decreases as :math:`(1 - g)^k`.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np
from astropy import units as u

from lumos.backend import to_numpy
from lumos.core.element import Element, Values, generator
from lumos.core.param import Param, Setting, State
from lumos.core.port import Input, Output
from lumos.core.registry import register
from lumos.core.signal import OpticalSignal
from lumos.elements.optics import _coherent, _phase
from lumos.fields.dataproduct import DataProduct
from lumos.fields.grid import Grid
from lumos.physics.turbulence import ScreenSpectrum
from lumos.physics.zernike import zernike_basis
from lumos.units import require_unit, to_si

__all__ = ["AOController", "Atmosphere", "DeformableMirror", "WavefrontSensor"]


# --- Atmosphere ------------------------------------------------------------------------------------------


def _layers(values: Values) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.atleast_1d(np.asarray(values.weights, dtype=float))
    speeds = np.atleast_1d(np.asarray(values.wind_speed, dtype=float))
    directions = np.atleast_1d(np.asarray(values.wind_direction, dtype=float))
    if not (weights.shape == speeds.shape == directions.shape) or weights.ndim != 1:
        raise ValueError("weights, wind_speed and wind_direction must have one value per layer")
    if np.any(weights < 0) or not weights.sum() > 0:
        raise ValueError("the layer weights must be non-negative and not all zero")
    return weights / weights.sum(), speeds, directions


@functools.lru_cache(maxsize=32)
def _spectra(
    random_key: tuple[int, ...],
    n: int,
    dx: float,
    r0_layers: tuple[float, ...],
    outer_scale: float,
    subharmonics: int,
) -> tuple[ScreenSpectrum, ...]:
    rng = generator({"random_key": random_key})
    return tuple(
        ScreenSpectrum.draw(
            n, dx, r0, outer_scale, rng, subharmonics=None if subharmonics < 0 else subharmonics
        )
        for r0 in r0_layers
    )


def _crop(screen: np.ndarray, grid: Grid) -> np.ndarray:
    n = screen.shape[0]
    y0 = n // 2 - grid.shape[0] // 2
    x0 = n // 2 - grid.shape[1] // 2
    return screen[y0 : y0 + grid.shape[0], x0 : x0 + grid.shape[1]]


def _turbulent_opd(values: Values, grid: Grid) -> np.ndarray:
    if grid.plane != "spatial":
        raise ValueError("the atmosphere acts on fields sampled on a spatial grid")
    if not math.isclose(grid.dx_si, grid.dy_si, rel_tol=1e-12):
        raise ValueError("the atmosphere needs square pixels (dx == dy)")
    weights, speeds, directions = _layers(values)
    n = int(values.screen_oversize) * max(grid.shape)
    r0_layers = tuple(float(values.r0 * w ** (-3 / 5)) if w > 0 else math.inf for w in weights)
    spectra = _spectra(
        tuple(values.random_key),
        n,
        grid.dx_si,
        r0_layers,
        float(values.outer_scale),
        int(values.subharmonics),
    )
    phase = np.zeros((n, n))
    for spectrum, weight, speed, direction in zip(spectra, weights, speeds, directions, strict=True):
        if weight == 0:
            continue
        sx = speed * math.cos(direction) * values.time - grid.x0_si
        sy = speed * math.sin(direction) * values.time - grid.y0_si
        phase += spectrum.evaluate(sx, sy)
    opd = _crop(phase, grid) * values.reference_wavelength / (2 * math.pi)
    if values.corrected_modes > 0:
        opd = _correct(opd, grid, values.pupil_diameter, int(values.corrected_modes))
    return opd


def _correct(opd: np.ndarray, grid: Grid, diameter: float, modes: int) -> np.ndarray:
    basis = zernike_basis(grid, diameter * u.m, modes)
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    inside = np.hypot(xx, yy) <= diameter / 2 * (1 + 1e-12)
    matrix = basis[:, inside].T
    coefficients, *_ = np.linalg.lstsq(matrix, opd[inside], rcond=None)
    corrected = opd.copy()
    corrected[inside] -= matrix @ coefficients
    return corrected


@register("lumos.atmosphere.Atmosphere")
class Atmosphere(Element):
    """Multi-layer von Kármán turbulence in frozen flow, with an optional ideal adaptive optics correction."""

    time_dependent = True
    stochastic = True
    r0 = Param(
        unit=u.cm, default=10 * u.cm, bounds=(0 * u.m, None), inclusive=(False, True), doc="Fried parameter"
    )
    reference_wavelength = Param(
        unit=u.nm, default=500 * u.nm, bounds=(0 * u.m, None), inclusive=(False, True), doc="wavelength of r0"
    )
    outer_scale = Param(unit=u.m, default=25 * u.m, bounds=(0 * u.m, None), inclusive=(False, True))
    weights = Param(shape=None, default=[1.0], doc="fraction of the turbulence in each layer")
    wind_speed = Param(unit=u.m / u.s, shape=None, default=[10.0] * u.m / u.s, doc="wind speed of each layer")
    wind_direction = Param(
        unit=u.deg, shape=None, default=[0.0] * u.deg, doc="direction of the wind, from +x"
    )
    pupil_diameter = Param(unit=u.m, default=8 * u.m, bounds=(0 * u.m, None), inclusive=(False, True))
    corrected_modes = Setting(int, default=0, doc="Noll modes removed by the adaptive optics (0: none)")
    screen_oversize = Setting(int, default=2, doc="screen size in units of the field grid")
    subharmonics = Setting(
        int, default=-1, doc="levels of low-frequency subharmonics (-1: to the outer scale)"
    )
    in_ = Input(OpticalSignal)
    out = Output(OpticalSignal)

    def opd(self, grid: Grid) -> u.Quantity:
        """Return the optical path difference added on ``grid`` at the current time."""
        return _turbulent_opd(self.values(), grid) * u.m

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Add the turbulent phase to every channel and mode."""
        signal = _coherent(inputs["in"], "Atmosphere")
        opd = _turbulent_opd(values, signal.grid)
        return {"out": _phase(signal, lambda grid, xp: xp.asarray(opd))}


# --- Deformable mirror -------------------------------------------------------------------------------------


def _actuator_axis(values: Values) -> np.ndarray:
    n = int(values.n_actuators)
    if n < 2:
        raise ValueError("a deformable mirror needs at least 2 actuators per side")
    pitch = values.diameter / (n - 1)
    return (np.arange(n) - (n - 1) / 2) * pitch


def _influence(values: Values, coordinates: np.ndarray) -> np.ndarray:
    """Separable influence factors, shape ``(n_actuators, len(coordinates))``."""
    axis = _actuator_axis(values)
    pitch = values.diameter / (int(values.n_actuators) - 1)
    return values.coupling ** (((coordinates[None, :] - axis[:, None]) / pitch) ** 2)


def _commands(values: Values) -> np.ndarray:
    n = int(values.n_actuators)
    commands = np.atleast_1d(np.asarray(values.commands, dtype=float))
    if commands.size == 1:
        return np.full((n, n), float(commands[0]))
    if commands.size != n * n:
        raise ValueError(f"the mirror has {n * n} actuators, got {commands.size} commands")
    return commands.reshape(n, n)


def _mirror_opd(values: Values, grid: Grid) -> np.ndarray:
    x, y = (np.asarray(c, dtype=float) for c in grid.axes(backend="numpy"))
    wx, wy = _influence(values, x), _influence(values, y)
    return 2 * wy.T @ _commands(values) @ wx


@register("lumos.atmosphere.DeformableMirror")
class DeformableMirror(Element):
    """Square grid of actuators with Gaussian influence functions; the OPD is twice the surface."""

    diameter = Param(
        unit=u.m,
        default=10 * u.mm,
        bounds=(0 * u.m, None),
        inclusive=(False, True),
        doc="first to last actuator",
    )
    coupling = Param(default=0.15, bounds=(0, 1), inclusive=(False, False), doc="influence at the pitch")
    commands = State(
        unit=u.nm, shape=None, default=[0.0] * u.nm, doc="surface stroke per actuator (row-major)"
    )
    n_actuators = Setting(int, default=12, doc="actuators per side")
    in_ = Input(OpticalSignal)
    out = Output(OpticalSignal)

    def actuator_positions(self) -> u.Quantity:
        """Positions ``(x, y)`` of the actuators in the order of the commands, shape ``(n², 2)``."""
        axis = _actuator_axis(self.values())
        xx, yy = np.meshgrid(axis, axis)
        return np.stack([xx.ravel(), yy.ravel()], axis=1) * u.m

    def opd(self, grid: Grid) -> u.Quantity:
        """Return the optical path difference of the mirror on ``grid``."""
        return _mirror_opd(self.values(), grid) * u.m

    def fit(self, opd: u.Quantity, grid: Grid) -> u.Quantity:
        """Return the commands whose OPD fits ``opd`` (sampled on ``grid``) with least squares."""
        target = np.asarray(to_si(require_unit(opd, u.m, name="opd")), dtype=float)
        if target.shape != grid.shape:
            raise ValueError(f"the OPD map has shape {target.shape}, the grid {grid.shape}")
        values = self.values()
        x, y = (np.asarray(c, dtype=float) for c in grid.axes(backend="numpy"))
        wx, wy = _influence(values, x), _influence(values, y)
        commands = np.linalg.pinv(wy.T) @ (target / 2) @ np.linalg.pinv(wx)
        return u.Quantity(commands.ravel(), u.m).to(u.nm)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Add the optical path of the mirror to every channel and mode."""
        signal = _coherent(inputs["in"], "DeformableMirror")
        opd = _mirror_opd(values, signal.grid)
        return {"out": _phase(signal, lambda grid, xp: xp.asarray(opd))}


# --- Wavefront sensor --------------------------------------------------------------------------------------

#: Pixels with less than this fraction of the peak intensity are dark (outside the pupil).
_DARK = 1e-2


def _channels(signal: Any) -> list[tuple[np.ndarray, float]]:
    """Complex 2-D fields of every channel, polarisation and mode of ``signal``, with their wavelength."""
    from lumos.fields import IncoherentBundle, Wavecube, Wavefront

    if isinstance(signal, IncoherentBundle):
        return [channel for m in range(signal.n_modes) for channel in _channels(signal[m])]
    shape = signal.grid.shape
    field = np.asarray(to_numpy(signal.field))
    if isinstance(signal, Wavefront):
        return [(plane, signal.wavelength_si) for plane in field.reshape(-1, *shape)]
    if isinstance(signal, Wavecube):
        found = []
        for index, wavelength in enumerate(signal.wavelengths_si):
            found += [(plane, float(wavelength)) for plane in field[index].reshape(-1, *shape)]
        return found
    raise TypeError(f"the wavefront sensor measures coherent fields, not a {type(signal).__name__}")


def _gradients(
    channels: list[tuple[np.ndarray, float]], lit: np.ndarray, dx: float, dy: float
) -> tuple[np.ndarray, ...]:
    """Intensity-weighted mean gradients of the optical path along x and y, and their weights.

    A pixel has a gradient when its two neighbours are lit.
    """
    shape = channels[0][0].shape
    gx, wx, gy, wy = (np.zeros(shape) for _ in range(4))
    both_x = lit[:, 2:] & lit[:, :-2]
    both_y = lit[2:, :] & lit[:-2, :]
    for field, wavelength in channels:
        amplitude = np.abs(field)
        cross = field[:, 2:] * np.conj(field[:, :-2])
        weight = amplitude[:, 2:] * amplitude[:, :-2] * both_x
        gx[:, 1:-1] += weight * np.angle(cross) * wavelength / (4 * math.pi * dx)
        wx[:, 1:-1] += weight
        cross = field[2:, :] * np.conj(field[:-2, :])
        weight = amplitude[2:, :] * amplitude[:-2, :] * both_y
        gy[1:-1, :] += weight * np.angle(cross) * wavelength / (4 * math.pi * dy)
        wy[1:-1, :] += weight
    return gx, wx, gy, wy


def subaperture_centres(values: Values) -> np.ndarray:
    """Centres of the sub-apertures along one axis, in metres."""
    n = int(values.n_subapertures)
    if n < 1:
        raise ValueError("the wavefront sensor needs at least one sub-aperture")
    width = values.diameter / n
    return (np.arange(n) - (n - 1) / 2) * width


@register("lumos.atmosphere.WavefrontSensor")
class WavefrontSensor(Element):
    """Ideal Shack-Hartmann sensor: mean gradients of the optical path in sub-apertures, RMS residual."""

    diameter = Param(
        unit=u.m,
        default=8 * u.m,
        bounds=(0 * u.m, None),
        inclusive=(False, True),
        doc="side of the lenslet array",
    )
    illumination = Param(
        default=0.5, bounds=(0, 1), inclusive=(False, True), doc="lit fraction of a valid sub-aperture"
    )
    n_subapertures = Setting(int, default=16, doc="sub-apertures per side")
    in_ = Input(OpticalSignal)
    slopes = Output(DataProduct, doc="mean x and y gradients of the optical path (rad), invalid ones masked")
    residual = Output(DataProduct, doc="RMS optical path over the lit pupil, piston removed")

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Measure the slopes in every sub-aperture and the RMS residual optical path."""
        signal = _coherent(inputs["in"], "WavefrontSensor")
        grid = signal.grid
        channels = _channels(signal)
        intensity = np.sum([np.abs(field) ** 2 for field, _ in channels], axis=0)
        lit = np.asarray(intensity >= _DARK * float(np.max(intensity)), dtype=bool)
        gx, wx, gy, wy = _gradients(channels, lit, grid.dx_si, grid.dy_si)

        n = int(values.n_subapertures)
        centres = subaperture_centres(values)
        width = values.diameter / n
        x, y = (np.asarray(c, dtype=float) for c in grid.axes(backend="numpy"))
        column = np.floor((x - (centres[0] - width / 2)) / width).astype(int)
        row = np.floor((y - (centres[0] - width / 2)) / width).astype(int)
        inside_x, inside_y = (column >= 0) & (column < n), (row >= 0) & (row < n)
        rr, cc = np.meshgrid(np.where(inside_y, row, 0), np.where(inside_x, column, 0), indexing="ij")
        within = inside_y[:, None] & inside_x[None, :]
        index = (rr * n + cc)[within]

        def binned(weights: np.ndarray) -> np.ndarray:
            return np.bincount(index, weights=weights[within], minlength=n * n).reshape(n, n)

        pixels = binned(np.ones(grid.shape))
        fraction = binned(lit.astype(float)) / np.maximum(pixels, 1)
        valid = (pixels > 0) & (fraction >= values.illumination)
        mask_x, mask_y = lit & (wx > 0), lit & (wy > 0)
        sx = binned(np.where(mask_x, gx / np.where(wx > 0, wx, 1), 0)) / np.maximum(binned(mask_x * 1.0), 1)
        sy = binned(np.where(mask_y, gy / np.where(wy > 0, wy, 1), 0)) / np.maximum(binned(mask_y * 1.0), 1)
        data = np.where(valid, np.stack([sx, sy]), 0.0)
        slopes = DataProduct(
            data * u.rad,
            axes={"component": np.arange(2) * u.one, "y": centres * u.m, "x": centres * u.m},
            mask=~np.stack([valid, valid]),
            meta={"subaperture": width},
        )

        # RMS optical path over the lit pixels, piston removed (phase relative to the mean amplitude)
        total, power = 0.0, 0.0
        for field, wavelength in channels:
            weight = np.abs(field[lit]) ** 2
            if not weight.sum() > 0:
                continue
            mean = np.sum(field[lit])
            opd = np.angle(field[lit] * np.conj(mean)) * wavelength / (2 * math.pi)
            total += float(np.sum(weight * opd**2))
            power += float(weight.sum())
        rms = math.sqrt(total / power) if power > 0 else 0.0
        return {"slopes": slopes, "residual": DataProduct(u.Quantity(rms, u.m).to(u.nm))}


# --- Controller ------------------------------------------------------------------------------------------


def _influence_moments(values: Values, centres: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean ``M`` and mean derivative ``K`` of each influence over each sub-aperture.

    Both have the shape ``(n_subapertures, n_actuators)``.
    """
    from scipy.special import erf

    axis = _actuator_axis(values)
    pitch = values.diameter / (int(values.n_actuators) - 1)
    alpha = -math.log(values.coupling) / pitch**2
    low = centres[:, None] - width / 2 - axis[None, :]
    high = centres[:, None] + width / 2 - axis[None, :]
    root = math.sqrt(alpha)
    mean = math.sqrt(math.pi) / (2 * root * width) * (erf(root * high) - erf(root * low))
    derivative = (np.exp(-alpha * high**2) - np.exp(-alpha * low**2)) / width
    return mean, derivative


@functools.lru_cache(maxsize=16)
def _reconstructor(
    n_actuators: int,
    diameter: float,
    coupling: float,
    regularization: float,
    centres: tuple[float, ...],
    width: float,
    valid: bytes,
) -> np.ndarray:
    values = Values({"n_actuators": n_actuators, "diameter": diameter, "coupling": coupling})
    mean, derivative = _influence_moments(values, np.asarray(centres), width)
    dx = 2 * np.kron(mean, derivative)  # x slopes: mean along y, derivative along x
    dy = 2 * np.kron(derivative, mean)
    keep = np.frombuffer(valid, dtype=bool)
    matrix = np.vstack([dx[keep], dy[keep]])
    if matrix.shape[0] == 0:
        raise ValueError("no valid sub-aperture: the pupil does not light the wavefront sensor")
    normal = matrix.T @ matrix
    scale = float(np.linalg.eigvalsh(normal)[-1])
    return np.linalg.solve(normal + regularization * scale * np.eye(normal.shape[0]), matrix.T)


@register("lumos.atmosphere.AOController")
class AOController(Element):
    """Integrator of an adaptive optics loop: command increments of a deformable mirror from the slopes."""

    gain = Param(default=0.5, bounds=(0, 2), inclusive=(False, True), doc="loop gain g of Δc = -g R s")
    diameter = Param(
        unit=u.m,
        default=10 * u.mm,
        bounds=(0 * u.m, None),
        inclusive=(False, True),
        doc="first to last actuator of the mirror",
    )
    coupling = Param(
        default=0.15, bounds=(0, 1), inclusive=(False, False), doc="influence of the mirror at the pitch"
    )
    regularization = Param(
        default=1e-3, bounds=(0, None), doc="Tikhonov weight, relative to the largest eigenvalue"
    )
    n_actuators = Setting(int, default=12, doc="actuators per side of the mirror")
    slopes = Input(DataProduct, doc="slopes of a WavefrontSensor")
    commands = Output(
        DataProduct, doc="increments of the commands of the mirror (feed them back with mode 'add')"
    )

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Compute ``Δc = -g R s`` from the valid slopes."""
        slopes = inputs["slopes"]
        data = np.asarray(slopes.data.to_value(u.rad), dtype=float)
        if data.ndim != 3 or data.shape[0] != 2 or data.shape[1] != data.shape[2]:
            raise ValueError("the controller needs the slopes of a WavefrontSensor, shape (2, n, n)")
        centres = np.asarray(slopes.axes["x"].to_value(u.m), dtype=float)
        if centres.size < 2:
            raise ValueError("the controller needs at least 2 sub-apertures per side")
        width = float(centres[1] - centres[0])
        mask = np.zeros(data.shape, dtype=bool) if slopes.mask is None else np.asarray(slopes.mask)
        valid = ~(mask[0] | mask[1])
        if int(values.n_actuators) < 2:
            raise ValueError("a deformable mirror needs at least 2 actuators per side")
        reconstructor = _reconstructor(
            int(values.n_actuators),
            float(values.diameter),
            float(values.coupling),
            float(values.regularization),
            tuple(float(c) for c in centres),
            width,
            valid.ravel().tobytes(),
        )
        measured = np.concatenate([data[0][valid], data[1][valid]])
        increments = -values.gain * (reconstructor @ measured)
        return {"commands": DataProduct(u.Quantity(increments, u.m).to(u.nm))}

    def check_feedback(self, element: Any, field: str) -> None:
        """Accept only the ``commands`` of a deformable mirror of the geometry of the controller."""
        if field != "commands" or not isinstance(element, DeformableMirror):
            raise ValueError(
                f"{self.name or 'the controller'} drives the commands of a DeformableMirror, not {field}"
            )
        mine = (self.n_actuators, self.diameter.to_value(u.m), float(self.coupling))
        theirs = (element.n_actuators, element.diameter.to_value(u.m), float(element.coupling))
        if mine[0] != theirs[0] or not np.allclose(mine[1:], theirs[1:], rtol=1e-9):
            raise ValueError(
                f"{self.name or 'the controller'} is set for a mirror of {mine[0]} actuators per side, "
                f"{mine[1]:g} m and coupling {mine[2]:g}; {element.name or 'the mirror'} has {theirs[0]}, "
                f"{theirs[1]:g} m and {theirs[2]:g}: give them the same geometry"
            )
