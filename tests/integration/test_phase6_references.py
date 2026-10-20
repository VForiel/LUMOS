"""Reference integration tests of the element library (step P6.10).

Complete chains of elements in a context are compared with analytic or published results (objective O1,
use case UC1). Independent references:

- chromatic PSF: in the focal plane of a uniform circular pupil each channel has its first dark ring at
  ``1.21967 λ f / D`` and the encircled energy ``1 - J0²(π r) - J1²(π r)`` within ``r λ f / D`` (Born & Wolf,
  *Principles of Optics*, §8.5.2; Rayleigh's formula), J0 and J1 from SciPy, taken at the second dark
  ring (2.233 λ f / D) where it does not depend on the pixelisation of the circle;
- ideal coronagraphy: an even-charge vortex removes an on-axis star from a circular pupil (Mawet et al. 2005,
  ApJ 633, 1191) and transmits a companion at 6 λ/D, which then dominates the focal plane;
- turbulence: the phase structure function of Kolmogorov turbulence is ``D(r) = 6.88 (r / r0)^(5/3)`` (Fried
  1965, JOSA 55, 1427); for a von Kármán spectrum of outer scale L0 ≫ r it is multiplied by
  ``1 - 1.485 (r / L0)^(1/3)`` (Voitsekhovich 1995, JOSA A 12, 1346), a 2 to 5 % correction here; the large
  scales make single screens very different, and 200 screens give D(r) within about 5 %, so 10 % is the
  tolerance;
- kernel nuller: with the nuller matrix of Martinache & Ireland (2018, A&A 619, A87), whose dark rows are
  ``(1, -1, 1, -1)/2``, ``(1, 1, -1, -1)/2`` and ``(1, -1, -1, 1)/2``, and the quadrature recombination of
  the pairs of dark outputs (p, q), the kernels are ``-Im(conj(d_p) d_q)`` for the planet (the star gives
  none): computed here with NumPy from the source flux and the telescope positions;
- Bracewell nuller: a two-telescope nuller of baseline B transmits ``2 P sin²(π B θ / λ)`` from a source at θ
  on its dark output (Bracewell 1978, Nature 274, 780).
"""

import math

import numpy as np
import pytest
from astropy import units as u
from scipy.special import j0, j1

from lumos import Context
from lumos.elements.atmosphere import Atmosphere
from lumos.elements.coronagraphs import VortexCoronagraph
from lumos.elements.detectors import PowerMeter
from lumos.elements.optics import FreeSpace, ThinLens
from lumos.elements.photonics import KERNEL_PAIRS, bracewell, kernel_nuller, kernels
from lumos.elements.sources import PointSource
from lumos.elements.telescopes import IdealInjectionArray, Telescope
from lumos.fields import Grid

H, C = 6.62607015e-34, 299_792_458.0


def band_context(low_um, high_um, n, seed=0):
    ctx = Context(seed=seed)
    ctx.observation.band = (low_um * u.um, high_um * u.um)
    ctx.observation.n_wavelengths = n
    return ctx


def local_minima(profile):
    return [
        i for i in range(1, len(profile) - 1) if profile[i] < profile[i - 1] and profile[i] <= profile[i + 1]
    ]


# --- Chromatic PSF ---------------------------------------------------------------------------------------


def test_chromatic_psf():
    d, f = 1.0, 10.0
    ctx = band_context(1.0, 1.6, 3)  # channels at 1.1, 1.3 and 1.5 um
    source = ctx.add(PointSource(name="src", flux_density=1 * u.Jy))
    scope = ctx.add(Telescope(name="scope", diameter=d * u.m, n_pixels=256))
    lens = ctx.add(ThinLens(name="lens", focal_length=f * u.m))
    step = 0.005 * 1.1e-6 * f / d
    cut = Grid(801, 1, dx=step * u.m, x0=400 * step * u.m)
    plane = Grid(128, dx=1.1e-6 * f / d / 8 * u.m)
    ctx.add(FreeSpace(name="cut", distance=f * u.m, method="fresnel_mft", output_grid=cut))
    ctx.add(FreeSpace(name="plane", distance=f * u.m, method="fresnel_mft", output_grid=plane))
    source >> scope >> lens
    ctx.connect(lens.port("out"), ctx["cut"].port("in"))
    ctx.connect(lens.port("out"), ctx["plane"].port("in"))
    result = ctx.run(["scope.out", "cut.out", "plane.out"])

    pupil, profile, image = result["scope.out"][0], result["cut.out"][0], result["plane.out"][0]
    x = cut.x.to_value(u.m)
    xx, yy = (np.asarray(c) for c in plane.coords(backend="numpy"))
    r = np.hypot(xx, yy)
    for k, lam in enumerate(pupil.wavelengths.to_value(u.m)):
        scale = lam * f / d
        ring = x[local_minima(np.asarray(profile.intensity)[k, 0])[0]]
        assert ring / scale == pytest.approx(1.21967, rel=5e-3)
        radius = 2.233  # second dark ring: the encircled energy is flat there, insensitive to pixels
        enclosed = float(np.sum(np.asarray(image.intensity)[k][r <= radius * scale])) * plane.dx_si**2
        expected = 1 - j0(math.pi * radius) ** 2 - j1(math.pi * radius) ** 2
        assert enclosed / pupil.power[k].value == pytest.approx(expected, rel=1e-2)


# --- Ideal coronagraphy ----------------------------------------------------------------------------------


def test_ideal_vortex_coronagraph_reveals_a_companion():
    d, f, lam = 1.0, 10.0, 1.0e-6
    ctx = band_context(0.9999, 1.0001, 1)
    star = ctx.add(PointSource(name="star", flux_density=1 * u.Jy))
    planet = ctx.add(PointSource(name="planet", flux_density=1e-4 * u.Jy, theta_x=6 * lam / d * u.rad))
    scope = ctx.add(Telescope(name="scope", diameter=d * u.m, n_pixels=256, oversize=1.1))
    ctx.connect(star.port("out"), scope.port("in"))
    ctx.connect(planet.port("out"), scope.port("in"))
    focal = Grid(81, dx=lam * f / d / 4 * u.m)
    chain = scope >> VortexCoronagraph(name="vortex", charge=2, pupil_diameter=d * u.m, lyot_stop=0.95)
    chain = chain >> ThinLens(name="lens", focal_length=f * u.m)
    chain >> FreeSpace(name="gap", distance=f * u.m, method="fresnel_mft", output_grid=focal)
    result = ctx.run(["scope.out", "vortex.out", "gap.out"])

    before, after = result["scope.out"].power.value, result["vortex.out"].power.value
    assert after[0] / before[0] < 1e-5  # star
    assert after[1] / before[1] > 0.8  # planet
    image = np.asarray(result["gap.out"].intensity)
    xx, yy = (np.asarray(c) for c in focal.coords(backend="numpy"))
    peak = np.unravel_index(np.argmax(image), image.shape)
    assert xx[peak] == pytest.approx(6 * lam * f / d, abs=focal.dx_si)
    assert yy[peak] == pytest.approx(0.0, abs=focal.dx_si)


# --- Turbulence ------------------------------------------------------------------------------------------


def test_kolmogorov_structure_function():
    grid = Grid(128, dx=1 * u.cm)
    r0, lam = 0.1, 500e-9
    lags = (4, 8, 16, 32)
    sums = np.zeros(len(lags))
    for seed in range(200):
        ctx = Context(seed=seed)
        atm = ctx.add(
            Atmosphere(name="atm", r0=r0 * u.m, reference_wavelength=lam * u.m, outer_scale=1e4 * u.m)
        )
        phase = 2 * math.pi * atm.opd(grid).to_value(u.m) / lam
        for i, lag in enumerate(lags):
            dx = phase[:, lag:] - phase[:, :-lag]
            dy = phase[lag:, :] - phase[:-lag, :]
            sums[i] += 0.5 * (np.mean(dx**2) + np.mean(dy**2))
    measured = sums / 200
    r = np.array(lags) * 0.01
    expected = 6.88 * (r / r0) ** (5 / 3) * (1 - 1.485 * (r / 1e4) ** (1 / 3))
    np.testing.assert_allclose(measured, expected, rtol=0.10)


# --- Kernel nuller -----------------------------------------------------------------------------------------


def test_kernel_nuller_response():
    ctx = band_context(3.5, 4.0, 2)
    positions = np.array([[0.0, 0.0], [30.0, 10.0], [55.0, -20.0], [80.0, 25.0]])
    tx, ty = 4e-8, -1.5e-8
    star = ctx.add(PointSource(name="star", flux_density=1 * u.Jy))
    planet = ctx.add(
        PointSource(name="planet", flux_density=0.01 * u.Jy, theta_x=tx * u.rad, theta_y=ty * u.rad)
    )
    array = ctx.add(
        IdealInjectionArray(name="array", frame="sky", positions=positions * u.m, diameter=2 * u.m)
    )
    ctx.connect(star.port("out"), array.port("in"))
    ctx.connect(planet.port("out"), array.port("in"))
    chain = array >> kernel_nuller(name="chip")
    chain >> PowerMeter(name="meter")
    result = ctx.run(["meter.out", "planet.out", "star.out"])
    power = result["meter.out"].data.to_value(u.W)

    sources = result["planet.out"]
    lam = sources.wavelengths.to_value(u.m)
    photons = (
        math.pi
        * 2.0**2
        / 4
        * sources.flux.to_value(u.photon / u.s / u.m**3)[0]
        * sources.bandwidths.to_value(u.m)
    )
    dark_rows = 0.5 * np.array([[1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]])
    expected = np.zeros(3)
    for k in range(len(lam)):
        phases = 2 * math.pi * (positions[:, 0] * tx + positions[:, 1] * ty) / lam[k]
        amplitudes = math.sqrt(photons[k]) * np.exp(1j * phases)
        d = dark_rows @ amplitudes
        pairs = [(0, 1), (0, 2), (1, 2)]
        expected += np.array([-np.imag(np.conj(d[p]) * d[q]) for p, q in pairs]) * H * C / lam[k]
    np.testing.assert_allclose(kernels(power), expected, rtol=1e-9)
    assert len(KERNEL_PAIRS) == 3

    star_flux = result["star.out"].flux.to_value(u.photon / u.s / u.m**3)[0]
    bright = np.sum(4 * math.pi * 2.0**2 / 4 * star_flux * sources.bandwidths.to_value(u.m) * H * C / lam)
    assert power[0] > 0.99 * bright  # the star goes to the bright output


# --- Bracewell transmission map ----------------------------------------------------------------------------


def test_bracewell_transmission_map():
    ctx = band_context(9.99, 10.01, 1)
    baseline = 20.0
    source = ctx.add(PointSource(name="src", flux_density=1 * u.Jy))
    array = ctx.add(
        IdealInjectionArray(
            name="array",
            frame="sky",
            positions=[[-baseline / 2, 0.0], [baseline / 2, 0.0]] * u.m,
            diameter=2 * u.m,
        )
    )
    source >> array >> bracewell(name="chip") >> PowerMeter(name="meter")
    lam = 10e-6
    thetas = np.linspace(0, lam / baseline, 9)
    for theta in thetas:
        source.theta_x = theta * u.rad
        result = ctx.run(["meter.out", "src.out"])
        sources = result["src.out"]
        per_telescope = (
            math.pi
            * 2.0**2
            / 4
            * sources.flux.to_value(u.photon / u.s / u.m**3)[0, 0]
            * sources.bandwidths.to_value(u.m)[0]
            * H
            * C
            / sources.wavelengths.to_value(u.m)[0]
        )
        lam_k = sources.wavelengths.to_value(u.m)[0]
        null = result["meter.out"].data.to_value(u.W)[0]
        expected = 2 * per_telescope * math.sin(math.pi * baseline * theta / lam_k) ** 2
        assert null == pytest.approx(expected, rel=1e-9, abs=1e-12 * per_telescope)
