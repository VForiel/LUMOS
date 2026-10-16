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
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np
from astropy import units as u

from lumos.core.element import Element, Values, generator
from lumos.core.param import Param, Setting, State
from lumos.core.port import Input, Output
from lumos.core.registry import register
from lumos.core.signal import OpticalSignal
from lumos.elements.optics import _coherent, _phase
from lumos.fields.grid import Grid
from lumos.physics.turbulence import ScreenSpectrum
from lumos.physics.zernike import zernike_basis
from lumos.units import require_unit, to_si

__all__ = ["Atmosphere", "DeformableMirror"]


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
        ScreenSpectrum.draw(n, dx, r0, outer_scale, rng, subharmonics=subharmonics) for r0 in r0_layers
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
    subharmonics = Setting(int, default=3, doc="levels of low-frequency subharmonics")
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
