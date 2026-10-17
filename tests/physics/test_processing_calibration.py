"""Tests for classical calibration (step P11.4, REQ-PIPE-5).

Requirements: phase fitting (gradient, trial and error, genetic algorithm), MTF measurement by the slanted
edge (ISO 12233), PSF measurement; MTF of a diffraction-limited optics compared with the analytic formula
(plan, tests of phase 11).

Independent references:

- the MTF of an aberration-free circular pupil is (2/π)[arccos ν - ν√(1 - ν²)], ν = f/f_c, f_c = D/λ
  (Goodman, *Introduction to Fourier Optics*, §6.3.3, eq. 6-31); a detector of square pixels multiplies it by
  |sinc(f p)| (pixel footprint; Boreman 2001, *Modulation Transfer Function in Optical and Electro-Optical
  Systems*, §2.3). The edge image is made here with NumPy FFTs of a sampled pupil, independently of LUMOS;
- the Airy pattern [2J₁(v)/v]² has a FWHM of 1.029 λ/D and holds 83.8 % of the energy within its first dark
  ring at 1.220 λ/D (Born & Wolf, *Principles of Optics*, §8.5.2);
- a Bracewell nuller with a phase error φ has the null depth tan²(φ/2), minimal when a correcting phase
  -φ is added (Serabyn 2000).
"""

import math

import numpy as np
import pytest
from astropy import units as u
from scipy.special import j1

from lumos import Context
from lumos.elements.detectors import PowerMeter
from lumos.elements.photonics import PhaseShifter, bracewell
from lumos.elements.sources import PointSource
from lumos.elements.telescopes import TelescopeArray
from lumos.fields import DataProduct
from lumos.processing import calibration
from lumos.processing.interferometry import null_depth

# --- Phase fitting ------------------------------------------------------------------------------------------


def nuller(errors):
    ctx = Context()
    ctx.observation.band = (9.99 * u.um, 10.01 * u.um)
    source = ctx.add(PointSource(name="src", flux_density=1 * u.Jy))
    array = ctx.add(
        TelescopeArray(name="array", frame="sky", positions=[[-5.0, 0.0], [5.0, 0.0]] * u.m, diameter=2 * u.m)
    )
    chain = source >> array
    for k, error in enumerate(errors):
        chain = chain >> PhaseShifter(name=f"error{k}", phase=error * u.rad, achromatic=True, channels=[1])
    correction = ctx.add(PhaseShifter(name="correction", phase=0 * u.rad, achromatic=True, channels=[1]))
    chain >> correction >> bracewell(name="chip") >> PowerMeter(name="meter")
    return ctx, correction


@pytest.mark.parametrize(("method", "tolerance"), [("gradient", 1e-3), ("scan", 2e-3), ("genetic", 2e-3)])
def test_phase_fitting_cancels_a_phase_error(method, tolerance):
    ctx, correction = nuller([0.3])

    def measure(phases):
        correction.phase = phases[0] * u.rad
        return float(null_depth(ctx.run(["meter.out"])["meter.out"], dark=0, bright=1).data[0].value)

    fit = calibration.fit_phases(measure, initial=[0.0] * u.rad, method=method, seed=3)
    assert fit.phases[0].to_value(u.rad) == pytest.approx(-0.3, abs=tolerance)
    assert fit.value < math.tan(tolerance) ** 2 + 1e-12
    assert fit.evaluations > 0
    assert fit.method == method


@pytest.mark.parametrize("method", ["gradient", "scan", "genetic"])
def test_phase_fitting_in_two_dimensions(method):
    target = np.array([0.4, -0.25])

    def measure(phases):  # two independent nulls, minimum at the target phases
        return float(np.sum(np.sin((np.asarray(phases) - target) / 2) ** 2))

    fit = calibration.fit_phases(measure, initial=[0.0, 0.0] * u.rad, method=method, seed=1)
    np.testing.assert_allclose(fit.phases.to_value(u.rad), target, atol=5e-3)


# --- MTF ----------------------------------------------------------------------------------------------------


def slanted_edge_image(angle_deg=5.0, fine=4, n_fine=1024, pupil_px=128):
    """Edge imaged by a diffraction-limited optics, detected by pixels of ``fine`` × ``fine`` fine samples."""
    yy, xx = np.mgrid[0:n_fine, 0:n_fine] - n_fine // 2
    pupil = (np.hypot(xx, yy) <= pupil_px / 2).astype(float)
    psf = np.abs(np.fft.fft2(pupil)) ** 2
    psf /= psf.sum()
    slope = math.tan(math.radians(angle_deg))
    sub = 8  # anti-aliasing of the edge itself
    offsets = (np.arange(sub) + 0.5) / sub - 0.5
    edge = np.zeros((n_fine, n_fine))
    for o in offsets:
        edge += ((xx + o) - slope * yy) > 0
    edge /= sub
    blurred = np.real(np.fft.ifft2(np.fft.fft2(edge) * np.fft.fft2(psf)))
    crop = blurred[n_fine // 4 : 3 * n_fine // 4, n_fine // 4 : 3 * n_fine // 4]  # away from wrap-around
    n = crop.shape[0] // fine
    detected = crop.reshape(n, fine, n, fine).sum(axis=(1, 3))
    cutoff = pupil_px / n_fine * fine  # cycles per detector pixel
    return detected, cutoff


def test_slanted_edge_mtf_of_a_diffraction_limited_optics():
    image, cutoff = slanted_edge_image()
    assert cutoff == pytest.approx(0.5)
    mtf = calibration.mtf_slanted_edge(DataProduct(image * u.adu))
    frequency = mtf.axes["frequency"].to_value(1 / u.pix)
    nu = frequency / cutoff
    inside = nu < 0.8
    optics = (2 / math.pi) * (np.arccos(nu[inside]) - nu[inside] * np.sqrt(1 - nu[inside] ** 2))
    expected = optics * np.abs(np.sinc(frequency[inside]))  # pixel footprint
    np.testing.assert_allclose(mtf.data.value[inside], expected, atol=0.02)
    assert mtf.data.value[0] == pytest.approx(1.0)
    assert mtf.meta["edge_angle_deg"] == pytest.approx(5.0, abs=0.1)


# --- PSF ----------------------------------------------------------------------------------------------------


def airy_image(samples=8.0, n=257, center=(0.37, -0.81)):
    yy, xx = np.mgrid[0:n, 0:n] - n // 2
    v = math.pi * np.hypot(xx - center[0], yy - center[1]) / samples  # λ/D = samples pixels
    safe = np.where(v > 0, v, 1.0)
    return np.where(v > 0, (2 * j1(safe) / safe) ** 2, 1.0)


def test_psf_measurement_of_an_airy_pattern():
    samples = 8.0
    image = airy_image(samples)
    n = image.shape[0]
    axis = (np.arange(n) - n // 2) * u.pix
    psf = calibration.measure_psf(
        DataProduct(image * u.adu, axes={"y": axis, "x": axis}), resolution=samples * u.pix
    )
    assert psf.centroid[0].to_value(u.pix) == pytest.approx(0.37, abs=0.02)
    assert psf.centroid[1].to_value(u.pix) == pytest.approx(-0.81, abs=0.02)
    assert psf.fwhm.to_value(u.pix) == pytest.approx(1.029 * samples, rel=0.01)
    # the image (±16 λ/D) holds 98.9 % of the energy: the encircled energy is relative to the total of the
    # Airy pattern, ∫[2J₁(v)/v]² dA = 4 s²/π for λ/D = s pixels
    total = 4 * samples**2 / math.pi * u.adu
    assert psf.encircled_energy(1.220 * samples * u.pix, total=total) == pytest.approx(0.838, abs=0.005)
    assert psf.encircled_energy(1.220 * samples * u.pix) == pytest.approx(0.838 / 0.98865, abs=0.005)
    assert psf.peak.to_value(u.adu) == pytest.approx(image.max())
    strehl = calibration.measure_psf(
        DataProduct(0.8 * image * u.adu, axes={"y": axis, "x": axis}),
        reference=DataProduct(image * u.adu, axes={"y": axis, "x": axis}),
    ).strehl
    assert strehl == pytest.approx(0.8, rel=1e-9)
