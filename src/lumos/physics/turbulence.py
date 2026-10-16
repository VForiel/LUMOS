r"""Atmospheric turbulence: von Kármán spectrum and random phase screens.

The phase of a wave crossing Kolmogorov turbulence of Fried parameter :math:`r_0` has the power spectral
density (Noll 1976, JOSA 66, 207; Roddier 1981, *Progress in Optics* 19, §3.5.2), with the outer scale
:math:`L_0` of the von Kármán model,

.. math:: \Phi(f) = 0.023\, r_0^{-5/3} \left(f^2 + L_0^{-2}\right)^{-11/6} \quad \mathrm{rad^2\,m^2},

and the structure function :math:`D(r) = 6.88\,(r / r_0)^{5/3}` for :math:`L_0 \to \infty`. The Fried
parameter scales with wavelength as :math:`r_0 \propto \lambda^{6/5}` (the optical path is achromatic), and
layers of fractional :math:`C_n^2` weights :math:`w_i` have :math:`r_{0,i} = r_0\, w_i^{-3/5}`.

Screens are drawn in Fourier space (McGlamery 1976): one random complex amplitude per frequency cell of a
periodic grid, plus levels of subharmonics that restore the low frequencies the grid misses (Lane, Glindemann
& Dainty 1992, *Waves in Random Media* 2, 209). Because :math:`\Phi` is steep near the origin, the variance of
each cell close to it is the integral of :math:`\Phi` over the cell rather than its value at the centre
(Johansson & Gavel 1994, Proc. SPIE 2200), and the subharmonics go down to the outer scale. Even the smallest
lags of the structure function depend on these large scales: with three levels only, :math:`D(r)` is 10 to
20 % too low. A :class:`ScreenSpectrum` keeps the amplitudes, so the screen can be evaluated at any
translation, exactly (Taylor's frozen flow).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy import units as u

from lumos.units import require_unit, to_si

__all__ = ["ScreenSpectrum", "von_karman_psd", "von_karman_psd_si"]


def von_karman_psd_si(f: Any, r0: float, outer_scale: float) -> np.ndarray:
    """Phase power spectral density in rad² m² at spatial frequencies ``f`` (m⁻¹); SI floats."""
    f = np.asarray(f, dtype=float)
    inverse_l0 = 0.0 if not math.isfinite(outer_scale) else 1.0 / outer_scale
    with np.errstate(divide="ignore"):
        return 0.023 * r0 ** (-5 / 3) * (f**2 + inverse_l0**2) ** (-11 / 6)


def von_karman_psd(f: u.Quantity, *, r0: u.Quantity, outer_scale: u.Quantity) -> u.Quantity:
    """Phase power spectral density of von Kármán turbulence (rad² m², radians omitted)."""
    f_si = to_si(require_unit(f, 1 / u.m, name="f"))
    r0_si = float(to_si(require_unit(r0, u.m, name="r0")))
    l0_si = float(to_si(require_unit(outer_scale, u.m, name="outer_scale")))
    return von_karman_psd_si(f_si, r0_si, l0_si) * u.m**2


#: FFT cells within this many steps of the origin (and the subharmonics) are explicit sinusoids.
_LOW_CELLS = 4
#: Midpoint samples per side used to integrate the spectrum over a cell.
_CELL_SAMPLES = 16
#: Largest number of subharmonic levels (3^-15 of the grid frequency step).
_MAX_LEVELS = 15


def _cell(fx: float, fy: float, step: float, r0: float, outer_scale: float) -> tuple[float, float, float]:
    """Power of the square cell of side ``step`` centred on ``(fx, fy)`` and its frequency components.

    The power is the integral of the spectrum over the cell; the components ``|fx|``, ``|fy|`` are the square
    roots of the spectrum-weighted means of ``fx²`` and ``fy²`` (with the signs of the centre), so that the
    sinusoid representing the cell has the right variance and the right small-lag structure function along
    both axes.
    """
    offsets = ((np.arange(_CELL_SAMPLES) + 0.5) / _CELL_SAMPLES - 0.5) * step
    sx, sy = np.meshgrid(fx + offsets, fy + offsets)
    density = von_karman_psd_si(np.hypot(sx, sy), r0, outer_scale)
    total = float(np.sum(density))
    mean_x2 = float(np.sum(density * sx**2)) / total
    mean_y2 = float(np.sum(density * sy**2)) / total
    sign_x = 1.0 if fx >= 0 else -1.0
    sign_y = 1.0 if fy >= 0 else -1.0
    return total / _CELL_SAMPLES**2 * step**2, sign_x * math.sqrt(mean_x2), sign_y * math.sqrt(mean_y2)


def _subharmonic_levels(df: float, outer_scale: float) -> int:
    """Levels needed for the smallest subharmonic step to fall below a tenth of ``1 / outer_scale``."""
    if not math.isfinite(outer_scale):
        return _MAX_LEVELS
    levels = math.ceil(math.log(max(10 * df * outer_scale, 1.0)) / math.log(3))
    return int(min(max(levels, 1), _MAX_LEVELS))


@dataclass(frozen=True)
class ScreenSpectrum:
    """Random Fourier amplitudes of a phase screen of ``n × n`` pixels of side ``dx`` (SI units).

    The high frequencies are FFT amplitudes on the periodic grid; the low frequencies (the cells close to the
    origin and the subharmonics) are explicit sinusoids. Build it with :meth:`draw`; :meth:`evaluate` returns
    the phase (rad) on the grid, translated by a shift.
    """

    n: int
    dx: float
    amplitudes: np.ndarray  # (n, n) complex, FFT ordering
    low_frequencies: np.ndarray  # (K, 2) in m⁻¹
    low_amplitudes: np.ndarray  # (K,) complex

    @classmethod
    def draw(
        cls,
        n: int,
        dx: float,
        r0: float,
        outer_scale: float,
        rng: np.random.Generator,
        *,
        subharmonics: int | None = None,
    ) -> ScreenSpectrum:
        """Draw the amplitudes of a screen of Fried parameter ``r0`` and outer scale ``outer_scale``.

        ``subharmonics`` is the number of subharmonic levels; ``None`` goes down to the outer scale.
        """
        df = 1.0 / (n * dx)
        f = np.fft.fftfreq(n, dx)
        fx, fy = np.meshgrid(f, f)
        power = von_karman_psd_si(np.hypot(fx, fy), r0, outer_scale) * df**2
        noise = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        k = np.round(np.fft.fftfreq(n, 1.0 / n)).astype(int)
        low = np.flatnonzero(np.abs(k) <= _LOW_CELLS)
        power[np.ix_(low, low)] = 0.0  # replaced by explicit sinusoids below
        amplitudes = noise * np.sqrt(power)

        cells: list[tuple[float, float, float]] = []
        for i in low:
            for j in low:
                if k[i] or k[j]:
                    cells.append((float(fx[i, j]), float(fy[i, j]), df))
        levels = _subharmonic_levels(df, outer_scale) if subharmonics is None else int(subharmonics)
        for level in range(1, levels + 1):
            step = df / 3**level
            cells += [(i * step, j * step, step) for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j]
        frequencies, values = [], []
        for cx, cy, step in cells:
            cell_power, ex, ey = _cell(cx, cy, step, r0, outer_scale)
            frequencies.append((ex, ey))
            values.append((rng.normal() + 1j * rng.normal()) * math.sqrt(cell_power))
        return cls(
            n=int(n),
            dx=float(dx),
            amplitudes=amplitudes,
            low_frequencies=np.asarray(frequencies, dtype=float).reshape(-1, 2),
            low_amplitudes=np.asarray(values, dtype=complex),
        )

    def evaluate(self, shift_x: float = 0.0, shift_y: float = 0.0) -> np.ndarray:
        r"""Return the phase screen (rad) at the pixel coordinates ``(i - n // 2) dx``, translated.

        The value at :math:`\mathbf{x}` is the value of the untranslated screen at
        :math:`\mathbf{x} - \mathbf{s}`, with :math:`\mathbf{s}` = ``(shift_x, shift_y)``.
        """
        n, dx = self.n, self.dx
        f = np.fft.fftfreq(n, dx)
        fx, fy = np.meshgrid(f, f)
        coordinates = (np.arange(n) - n // 2) * dx
        # centre the periodic screen on the coordinate origin, then translate it
        ramp = np.exp(-2j * math.pi * (fx * (shift_x - coordinates[0]) + fy * (shift_y - coordinates[0])))
        screen = np.real(np.fft.ifft2(self.amplitudes * ramp)) * n * n
        if len(self.low_amplitudes):
            # separable sum of the explicit sinusoids: Σ_k a_k e^{2iπ fy_k y} e^{2iπ fx_k x}
            ex = np.exp(2j * math.pi * np.outer(self.low_frequencies[:, 0], coordinates - shift_x))  # (K, n)
            ey = np.exp(2j * math.pi * np.outer(coordinates - shift_y, self.low_frequencies[:, 1]))  # (n, K)
            screen = screen + np.real((ey * self.low_amplitudes[None, :]) @ ex)
        return screen
