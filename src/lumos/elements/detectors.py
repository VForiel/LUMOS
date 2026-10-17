r"""Detectors (§5.9, step P6.9): camera, photodiode and power meter.

Detectors turn light into :class:`~lumos.fields.DataProduct` measurements. Every spectral channel :math:`k`
of weight :math:`w_k` (spectral quadrature weights of wavecubes) is converted to a photon rate
:math:`N_k = P_k \lambda_k / (hc)` when its power is in watts; incoherent modes add with their weights.

Camera
------
The pixels are the samples of the incoming field (propagate it to the detector grid first, e.g. with
:class:`~lumos.elements.optics.FreeSpace` and an ``output_grid``). During an exposure :math:`t`, a pixel
collects on average :math:`\bar n = t\,(\sum_k w_k \eta(\lambda_k) N_k + D)` electrons (quantum efficiency
:math:`\eta`, dark current :math:`D`). With ``noise=True`` the charge is a Poisson draw, clipped at the full
well, summed over ``binning`` × ``binning`` pixels (charge binning), read once with a Gaussian read noise,
converted with the gain :math:`g` (e⁻/ADU), offset by the bias, rounded and clipped to the range of the
``bits``-bit converter (Janesick, *Photon Transfer*, 2007). With ``noise=False`` the camera returns the
expected value, clipped the same way but not rounded. :meth:`Camera.expose` is a timed action lasting the
exposure time; each exposure draws a new frame from the reproducible random stream of the camera.

Photodiode and power meter
--------------------------
A photodiode delivers the current :math:`I = q \sum_k w_k \eta N_k + I_\mathrm{dark}` (responsivity
:math:`\eta q \lambda / hc`) with the shot noise :math:`\sigma_I = \sqrt{2 q I \Delta f}` over its bandwidth
(Schottky 1918); :meth:`Photodiode.acquire` takes a new sample. A power meter reads the power in watts. Both
give one value per guided channel when they receive a :class:`~lumos.fields.ModalField`.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from astropy import units as u

from lumos.backend._core import get
from lumos.core.action import action
from lumos.core.element import Element, Values, generator
from lumos.core.param import Param, Setting, State
from lumos.core.port import Input, Output
from lumos.core.registry import register
from lumos.core.signal import OpticalSignal
from lumos.fields.dataproduct import DataProduct
from lumos.fields.incoherent import IncoherentBundle
from lumos.fields.modal import ModalField
from lumos.fields.wavecube import Wavecube
from lumos.fields.wavefront import Wavefront

__all__ = ["Camera", "Photodiode", "PowerMeter"]

# exact SI values (2019 redefinition of the SI)
_H = 6.62607015e-34  # J s
_C = 299_792_458.0  # m / s
_Q = 1.602176634e-19  # C
_PHOTON_RATE = u.photon / u.s


def _spectral_power(signal: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, u.UnitBase]:
    """Return wavelengths, weights and the power per channel and sample, ``(nλ, ...)``, with its unit.

    The samples are pixels for fields (power per pixel) and guided channels for modal fields; incoherent
    modes are summed with their weights.
    """
    if isinstance(signal, ModalField):
        amplitudes = np.asarray(get(signal.backend).to_numpy(signal.amplitudes))
        power = np.abs(amplitudes) ** 2
        if power.ndim == 3:
            power = power.sum(axis=0)
        return np.asarray(signal.wavelengths_si), np.asarray(signal.weights), power, signal.power_unit
    if isinstance(signal, IncoherentBundle):
        member = signal[0]
        field = np.asarray(get(signal.backend).to_numpy(signal.field))
        density = np.abs(field) ** 2
        if member.polarized:
            density = density.sum(axis=-3)
        density = np.tensordot(signal.mode_weights, density, axes=1)
        if isinstance(member, Wavefront):
            density = density[None]
    elif isinstance(signal, Wavefront | Wavecube):
        member = signal
        density = np.asarray(get(signal.backend).to_numpy(signal.intensity))
        if isinstance(signal, Wavefront):
            density = density[None]
    else:
        raise TypeError(f"a detector cannot read a {type(signal).__name__}")
    if isinstance(member, Wavefront):
        lams, weights = np.array([member.wavelength_si]), np.ones(1)
    else:
        lams, weights = np.asarray(member.wavelengths_si), np.asarray(member.weights)
    grid = signal.grid
    return lams, weights, density * grid.dx_si * grid.dy_si, member.power_unit


def _scale(unit: u.UnitBase, target: u.UnitBase) -> float:
    return float(cast(Any, unit.to(target)))


def _photon_rates(lams: np.ndarray, power: np.ndarray, unit: u.UnitBase) -> np.ndarray:
    """Convert powers ``(nλ, ...)`` to photons per second."""
    shape = (-1,) + (1,) * (power.ndim - 1)
    if unit.is_equivalent(u.W):
        return power * _scale(unit, u.W) * (lams / (_H * _C)).reshape(shape)
    return power * _scale(unit, _PHOTON_RATE)


def _watts(lams: np.ndarray, power: np.ndarray, unit: u.UnitBase) -> np.ndarray:
    shape = (-1,) + (1,) * (power.ndim - 1)
    if unit.is_equivalent(u.W):
        return power * _scale(unit, u.W)
    return power * _scale(unit, _PHOTON_RATE) * (_H * _C / lams).reshape(shape)


def _weighted(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.tensordot(weights, values, axes=1)


# --- Camera ------------------------------------------------------------------------------------------------


def _quantum_efficiency(values: Values, lams: np.ndarray) -> np.ndarray:
    efficiency = np.atleast_1d(np.asarray(values.quantum_efficiency, dtype=float))
    table = np.atleast_1d(np.asarray(values.qe_wavelengths, dtype=float))
    if table.size == 0:
        if efficiency.size != 1:
            raise ValueError("give qe_wavelengths with a quantum-efficiency curve, or a single efficiency")
        return np.full(len(lams), float(efficiency[0]))
    if table.shape != efficiency.shape:
        raise ValueError("quantum_efficiency and qe_wavelengths must have the same length")
    return np.interp(lams, table, efficiency, left=0.0, right=0.0)


def _bin(array: np.ndarray, b: int) -> np.ndarray:
    ny, nx = (array.shape[0] // b) * b, (array.shape[1] // b) * b
    return array[:ny, :nx].reshape(ny // b, b, nx // b, b).sum(axis=(1, 3))


@register("lumos.detectors.Camera")
class Camera(Element):
    """Pixel detector: quantum efficiency, photon, dark and read noises, gain, saturation and binning."""

    stochastic = True
    lumoslabs_model = "LL-CAM"
    quantum_efficiency = Param(shape=None, default=[0.8], bounds=(0, 1), doc="one value, or a curve")
    qe_wavelengths = Param(unit=u.nm, shape=None, default=[] * u.nm, doc="wavelengths of the QE curve")
    exposure = Param(unit=u.s, default=1 * u.s, bounds=(0 * u.s, None))
    dark_current = Param(
        unit=u.electron / u.s, default=0 * u.electron / u.s, bounds=(0 * u.electron / u.s, None)
    )
    read_noise = Param(unit=u.electron, default=0 * u.electron, bounds=(0 * u.electron, None))
    gain = Param(
        unit=u.electron / u.adu,
        default=1 * u.electron / u.adu,
        bounds=(0 * u.electron / u.adu, None),
        inclusive=(False, True),
    )
    bias = Param(unit=u.adu, default=0 * u.adu)
    full_well = Param(unit=u.electron, default=1e5 * u.electron, bounds=(0 * u.electron, None))
    frame = State(default=0.0, doc="number of exposures taken")
    bits = Setting(int, default=16, doc="resolution of the analog-to-digital converter")
    binning = Setting(int, default=1, doc="pixels binned along each axis")
    noise = Setting(bool, default=True, doc="False: expected values")
    in_ = Input(OpticalSignal)
    out = Output(DataProduct)

    def twin_feed(self, inputs: Any, simulated: Any) -> Any:
        """Return the expected signal of each pixel in e⁻/s (without dark current), for an emulated camera."""
        signal = inputs["in"]
        values = self.values()
        lams, weights, power, unit = _spectral_power(signal)
        rates = _weighted(weights * _quantum_efficiency(values, lams), _photon_rates(lams, power, unit))
        return np.asarray(rates, dtype=float) * u.electron / u.s

    @action(duration=lambda self: self.exposure)
    def expose(self) -> None:
        """Take an exposure: the clock advances by the exposure time and a new frame is drawn."""
        self.frame = self.frame + 1

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the image in ADU."""
        signal = inputs["in"]
        if isinstance(signal, ModalField):
            raise TypeError("a camera needs a field sampled on its pixels, not a ModalField")
        lams, weights, power, unit = _spectral_power(signal)
        rates = _photon_rates(lams, power, unit)
        efficiency = _quantum_efficiency(values, lams)
        electrons = (_weighted(weights * efficiency, rates) + values.dark_current) * values.exposure
        rng = generator({"random_key": (*values.random_key, int(values.frame))}) if values.noise else None
        if rng is not None:
            electrons = rng.poisson(electrons).astype(float)
        electrons = np.minimum(electrons, values.full_well)
        b = int(values.binning)
        electrons = _bin(electrons, b)
        if rng is not None and values.read_noise > 0:
            electrons = electrons + rng.normal(0.0, values.read_noise, electrons.shape)
        adu = electrons / values.gain + values.bias
        if rng is not None:
            adu = np.round(adu)
        adu = np.clip(adu, 0, 2 ** int(values.bits) - 1)
        grid = signal.grid
        x, y = (np.asarray(c, dtype=float) for c in grid.axes(backend="numpy"))
        ny, nx = adu.shape
        x_binned = x[: nx * b].reshape(nx, b).mean(axis=1)
        y_binned = y[: ny * b].reshape(ny, b).mean(axis=1)
        unit_xy = u.m if grid.plane == "spatial" else u.rad
        axes = {"y": y_binned * unit_xy, "x": x_binned * unit_xy}
        meta = {"exposure_s": float(values.exposure), "frame": int(values.frame), "binning": b}
        return {"out": DataProduct(adu * u.adu, axes=axes, meta=meta)}


# --- Photodiode and power meter -----------------------------------------------------------------------


def _channel_axes(signal: Any, n: int) -> dict[str, u.Quantity] | None:
    return {"channel": np.arange(n) * u.dimensionless_unscaled} if isinstance(signal, ModalField) else None


def _totals(signal: Any, per_channel: np.ndarray) -> np.ndarray:
    """Sum the spatial samples of fields; keep the channels of modal fields."""
    if isinstance(signal, ModalField):
        return per_channel
    return np.asarray(per_channel.sum(axis=(-2, -1)))


@register("lumos.detectors.Photodiode")
class Photodiode(Element):
    """Photodiode: photocurrent with shot noise over its bandwidth."""

    stochastic = True
    quantum_efficiency = Param(default=0.8, bounds=(0, 1))
    dark_current = Param(unit=u.nA, default=0 * u.nA, bounds=(0 * u.A, None))
    bandwidth = Param(unit=u.Hz, default=1 * u.kHz, bounds=(0 * u.Hz, None), inclusive=(False, True))
    sample = State(default=0.0, doc="number of samples acquired")
    noise = Setting(bool, default=True, doc="False: expected values")
    in_ = Input(OpticalSignal)
    out = Output(DataProduct)

    @action(duration=lambda self: 1 / (2 * self.bandwidth))
    def acquire(self) -> None:
        """Acquire a new sample (lasting half the inverse bandwidth)."""
        self.sample = self.sample + 1

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the photocurrent in A (one value per guided channel for modal fields)."""
        signal = inputs["in"]
        lams, weights, power, unit = _spectral_power(signal)
        photons = _totals(signal, _weighted(weights, _photon_rates(lams, power, unit)))
        current = _Q * values.quantum_efficiency * photons + values.dark_current
        if values.noise:
            rng = generator({"random_key": (*values.random_key, int(values.sample))})
            sigma = np.sqrt(2 * _Q * np.asarray(current) * values.bandwidth)
            current = current + rng.normal(0.0, 1.0, np.shape(current)) * sigma
        current = np.asarray(current, dtype=float)
        return {"out": DataProduct(current * u.A, axes=_channel_axes(signal, current.size))}


@register("lumos.detectors.PowerMeter")
class PowerMeter(Element):
    """Ideal power meter reading the optical power in watts."""

    lumoslabs_model = "LL-PM"

    channel = Setting(int, default=-1, doc="guided channel read (-1: every channel)")
    in_ = Input(OpticalSignal)
    out = Output(DataProduct)

    def twin_feed(self, inputs: Any, simulated: Any) -> Any:
        """Return the total simulated power, for an emulated power meter."""
        return float(np.sum(simulated.data.to_value(u.W))) * u.W

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the power in W (one value per guided channel for modal fields, or the selected channel)."""
        signal = inputs["in"]
        lams, weights, power, unit = _spectral_power(signal)
        watts = np.asarray(_totals(signal, _weighted(weights, _watts(lams, power, unit))), dtype=float)
        channel = int(values.channel)
        if channel >= 0 and isinstance(signal, ModalField):
            if channel >= watts.size:
                raise ValueError(f"the power meter reads channel {channel}, but the field has {watts.size}")
            return {"out": DataProduct(np.atleast_1d(watts[channel]) * u.W)}
        return {"out": DataProduct(watts * u.W, axes=_channel_axes(signal, watts.size))}
