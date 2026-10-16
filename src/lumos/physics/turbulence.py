r"""Atmospheric turbulence: von Kármán spectrum and random phase screens.

The phase of a wave crossing Kolmogorov turbulence of Fried parameter :math:`r_0` has the power spectral
density (Noll 1976, JOSA 66, 207; Roddier 1981, *Progress in Optics* 19, §3.5.2), with the outer scale
:math:`L_0` of the von Kármán model,

.. math:: \Phi(f) = 0.023\, r_0^{-5/3} \left(f^2 + L_0^{-2}\right)^{-11/6} \quad \mathrm{rad^2\,m^2},

and the structure function :math:`D(r) = 6.88\,(r / r_0)^{5/3}` for :math:`L_0 \to \infty`. The Fried
parameter scales with wavelength as :math:`r_0 \propto \lambda^{6/5}` (the optical path is achromatic), and
layers of fractional :math:`C_n^2` weights :math:`w_i` have :math:`r_{0,i} = r_0\, w_i^{-3/5}`.

Screens are drawn in Fourier space (McGlamery 1976): random complex amplitudes of variance
:math:`\Phi(f)\,\Delta f^2` on the frequencies of a periodic grid, plus three levels of subharmonics that
restore the low frequencies the grid misses (Lane, Glindemann & Dainty 1992, *Waves in Random Media* 2, 209;
Johansson & Gavel 1994, Proc. SPIE 2200). A :class:`ScreenSpectrum` keeps the amplitudes, so the screen can be
evaluated at any translation, exactly (Taylor's frozen flow).
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


@dataclass(frozen=True)
class ScreenSpectrum:
    """Random Fourier amplitudes of a phase screen of ``n × n`` pixels of side ``dx`` (SI units).

    Build it with :meth:`draw`; :meth:`evaluate` returns the phase (rad) on the grid, translated by a shift.
    """

    n: int
    dx: float
    amplitudes: np.ndarray  # (n, n) complex, FFT ordering
    sub_frequencies: np.ndarray  # (K, 2) in m⁻¹
    sub_amplitudes: np.ndarray  # (K,) complex

    @classmethod
    def draw(
        cls,
        n: int,
        dx: float,
        r0: float,
        outer_scale: float,
        rng: np.random.Generator,
        *,
        subharmonics: int = 3,
    ) -> ScreenSpectrum:
        """Draw the amplitudes of a screen of Fried parameter ``r0`` and outer scale ``outer_scale``."""
        df = 1.0 / (n * dx)
        f = np.fft.fftfreq(n, dx)
        fx, fy = np.meshgrid(f, f)
        psd = von_karman_psd_si(np.hypot(fx, fy), r0, outer_scale)
        psd[0, 0] = 0.0
        noise = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        amplitudes = noise * np.sqrt(psd) * df

        frequencies, values = [], []
        for level in range(1, int(subharmonics) + 1):
            step = df / 3**level
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    if i == 0 and j == 0:
                        continue
                    frequency = (i * step, j * step)
                    density = von_karman_psd_si(math.hypot(*frequency), r0, outer_scale)
                    noise_value = rng.normal() + 1j * rng.normal()
                    frequencies.append(frequency)
                    values.append(noise_value * math.sqrt(float(density)) * step)
        return cls(
            n=int(n),
            dx=float(dx),
            amplitudes=amplitudes,
            sub_frequencies=np.asarray(frequencies, dtype=float).reshape(-1, 2),
            sub_amplitudes=np.asarray(values, dtype=complex),
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
        if len(self.sub_amplitudes):
            x = coordinates[None, :] - shift_x
            y = coordinates[:, None] - shift_y
            low = np.zeros((n, n))
            for (sfx, sfy), amplitude in zip(self.sub_frequencies, self.sub_amplitudes, strict=True):
                low += np.real(amplitude * np.exp(2j * math.pi * (sfx * x + sfy * y)))
            screen = screen + low
        return screen
