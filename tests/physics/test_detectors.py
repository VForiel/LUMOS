"""Tests for detectors: camera, photodiode and power meter (step P6.9).

Requirements: §5.9 (camera with QE(λ), photon, read and dark noises, gain, saturation, binning and timed
exposures; photodiode; power meter), REQ-EL-5/6 (timed actions), REQ-CTX-6 (reproducible random streams).

Independent references:

- a power P at wavelength λ carries ``P λ / (h c)`` photons per second (h and c from ``astropy.constants``);
- a camera of quantum efficiency η, dark current D and gain g (e⁻/ADU) exposed for t reads on average
  ``(η N t + D t) / g + bias`` ADU; the photon-transfer relation gives the variance
  ``(η N t + D t) / g² + σ_read² / g² + 1/12`` ADU² (Poisson statistics of photo- and dark electrons,
  read noise, uniform quantisation; Janesick, *Photon Transfer*, SPIE Press 2007, chap. 3 and 5). With
  16 384 pixels the relative standard error of a sample variance is about 1.1 %, so 5 % is a 4.5σ tolerance;
- charge binning of b × b pixels adds the charges before a single read: mean ``b²`` times larger, read noise
  counted once (Janesick 2007, chap. 11);
- full-well saturation clips the charge; the ADC clips at ``2^bits - 1``;
- a photodiode of quantum efficiency η has the responsivity ``η q λ / (h c)`` (A/W) and the shot noise
  ``σ_I = sqrt(2 q I Δf)`` over the bandwidth Δf (Schottky 1918); 400 samples estimate a standard deviation to
  about 3.5 %, so 12 % is a 3.4σ tolerance.
"""

import math

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u

from lumos import Context
from lumos.core.element import Element
from lumos.core.port import Output
from lumos.elements.detectors import Camera, Photodiode, PowerMeter
from lumos.fields import DataProduct, Grid, IncoherentBundle, ModalField, Wavecube, Wavefront

H, C, Q = const.h.value, const.c.value, const.e.value
LAM = 800e-9


def uniform(n=128, power=1e-12 * u.W, lam=LAM):
    grid = Grid(n, dx=10 * u.um)
    return Wavefront.plane(grid, lam * u.m, power=power)


def camera(**kw):
    defaults = {
        "quantum_efficiency": [0.8],
        "dark_current": 5 * u.electron / u.s,
        "read_noise": 3 * u.electron,
        "gain": 2 * u.electron / u.adu,
        "bias": 100 * u.adu,
        "full_well": 1e6 * u.electron,
        "exposure": 2 * u.s,
        "bits": 16,
    }
    defaults.update(kw)
    return Camera(**defaults)


def read(element, signal):
    return element.apply(**{"in": signal})["out"]


# --- Camera: mean signal ------------------------------------------------------------------------------


def test_mean_image():
    image = read(camera(noise=False), uniform(n=32))
    assert isinstance(image, DataProduct)
    assert image.unit == u.adu
    photons = 1e-12 / (32 * 32) * LAM / (H * C)  # per pixel per second
    expected = (0.8 * photons * 2 + 5 * 2) / 2 + 100
    np.testing.assert_allclose(image.data.to_value(u.adu), expected, rtol=1e-12)


def test_photon_rate_input_and_quantum_efficiency_curve():
    grid = Grid(16, dx=10 * u.um)
    cube = Wavecube.plane(grid, [600, 900] * u.nm, power=1e5 * u.photon / u.s, weights=[0.5, 2.0])
    cam = camera(
        noise=False,
        quantum_efficiency=[0.9, 0.5],
        qe_wavelengths=[500, 1000] * u.nm,
        dark_current=0 * u.electron / u.s,
        bias=0 * u.adu,
    )
    qe = np.interp([600, 900], [500, 1000], [0.9, 0.5])
    electrons = (0.5 * qe[0] + 2.0 * qe[1]) * 1e5 / 256 * 2
    np.testing.assert_allclose(read(cam, cube).data.to_value(u.adu), electrons / 2, rtol=1e-12)


def test_incoherent_modes_add_in_intensity():
    grid = Grid(16, dx=10 * u.um)
    one = Wavefront.plane(grid, LAM * u.m, power=1e-13 * u.W)
    bundle = IncoherentBundle.from_signals([one, one * 1j])
    cam = camera(noise=False, dark_current=0 * u.electron / u.s, bias=0 * u.adu)
    np.testing.assert_allclose(read(cam, bundle).data.value, 2 * read(cam, one).data.value, rtol=1e-12)


# --- Camera: noise ------------------------------------------------------------------------------------


def test_photon_transfer():
    ctx = Context(seed=1)
    cam = ctx.add(camera(name="cam"))
    image = read(cam, uniform(power=2e-11 * u.W)).data.to_value(u.adu)
    electrons = 0.8 * 2e-11 / 128**2 * LAM / (H * C) * 2 + 10
    assert image.mean() == pytest.approx(electrons / 2 + 100, rel=2e-3)
    expected_variance = electrons / 4 + 9 / 4 + 1 / 12
    assert image.var() == pytest.approx(expected_variance, rel=0.05)


def test_dark_frame():
    ctx = Context(seed=2)
    cam = ctx.add(camera(name="cam", dark_current=200 * u.electron / u.s, exposure=10 * u.s))
    image = read(cam, uniform(power=0 * u.W)).data.to_value(u.adu)
    assert image.mean() == pytest.approx(2000 / 2 + 100, rel=2e-3)
    assert image.var() == pytest.approx(2000 / 4 + 9 / 4 + 1 / 12, rel=0.05)


def test_saturation_and_adc_range():
    full = read(camera(noise=False, full_well=5e4 * u.electron), uniform(power=1e-6 * u.W))
    np.testing.assert_allclose(full.data.to_value(u.adu), 5e4 / 2 + 100)
    adc = read(camera(noise=False, full_well=1e9 * u.electron, bits=12), uniform(power=1e-6 * u.W))
    np.testing.assert_allclose(adc.data.to_value(u.adu), 2**12 - 1)


def test_binning():
    plain = read(camera(noise=False, bias=0 * u.adu), uniform(n=32)).data.value
    binned = read(camera(noise=False, bias=0 * u.adu, binning=2), uniform(n=32))
    assert binned.shape == (16, 16)
    np.testing.assert_allclose(binned.data.value, 4 * plain[:16, :16], rtol=1e-12)
    ctx = Context(seed=3)
    cam = ctx.add(camera(name="cam", binning=2))
    image = read(cam, uniform(n=256, power=2e-11 * u.W)).data.to_value(u.adu)
    electrons = 4 * (0.8 * 2e-11 / 256**2 * LAM / (H * C) * 2 + 10)
    assert image.var() == pytest.approx(electrons / 4 + 9 / 4 + 1 / 12, rel=0.05)


# --- Camera: timed exposures --------------------------------------------------------------------------


def exposures(seed):
    ctx = Context(seed=seed)
    source = ctx.add(_Constant(name="src"))
    cam = ctx.add(camera(name="cam", exposure=0.5 * u.s))
    source >> cam
    frames = []
    for _ in range(2):
        cam.expose()
        frames.append(ctx.run(["cam.out"])["cam.out"].data.value.copy())
    return ctx, frames


def test_exposures_take_time_and_give_new_frames():
    ctx, (first, second) = exposures(5)
    assert ctx.clock.elapsed.to_value(u.s) == pytest.approx(1.0)
    assert not np.array_equal(first, second)
    _, again = exposures(5)
    np.testing.assert_array_equal(again[0], first)
    np.testing.assert_array_equal(again[1], second)


class _Constant(Element):
    out = Output(Wavefront)

    @staticmethod
    def transfer(values, inputs):
        return {"out": uniform(n=16)}


# --- Photodiode and power meter -----------------------------------------------------------------------


@pytest.mark.parametrize("power", [1e-6 * u.W, 3e12 * u.photon / u.s])
def test_photodiode_current(power):
    diode = Photodiode(quantum_efficiency=0.7, dark_current=1 * u.nA, noise=False)
    current = read(diode, uniform(n=16, power=power)).data.to_value(u.A)
    watts = power.to_value(u.W) if power.unit == u.W else power.value * H * C / LAM
    assert float(current) == pytest.approx(0.7 * Q * LAM / (H * C) * watts + 1e-9, rel=1e-9)


def test_photodiode_shot_noise():
    ctx = Context(seed=6)
    diode = ctx.add(Photodiode(name="pd", quantum_efficiency=0.7, dark_current=0 * u.A, bandwidth=1 * u.MHz))
    signal = uniform(n=8, power=1e-6 * u.W)
    samples = []
    for _ in range(400):
        diode.acquire()
        samples.append(float(read(diode, signal).data.to_value(u.A)))
    mean = 0.7 * Q * LAM / (H * C) * 1e-6
    assert np.mean(samples) == pytest.approx(mean, rel=1e-3)
    assert np.std(samples) == pytest.approx(math.sqrt(2 * Q * mean * 1e6), rel=0.12)


def test_photodiodes_read_every_guided_channel():
    modal = ModalField(np.sqrt([[1e-6, 4e-6, 0.0]]), [LAM] * u.m)
    current = read(Photodiode(quantum_efficiency=1.0, noise=False), modal).data.to_value(u.A)
    np.testing.assert_allclose(current, Q * LAM / (H * C) * np.array([1e-6, 4e-6, 0.0]), rtol=1e-12)


def test_power_meter_reads_watts():
    meter = PowerMeter()
    photons = uniform(n=16, power=3e12 * u.photon / u.s)
    assert float(read(meter, photons).data.to_value(u.W)) == pytest.approx(3e12 * H * C / LAM, rel=1e-12)
    grid = Grid(8, dx=1 * u.um)
    cube = Wavecube.plane(grid, [1.0, 2.0] * u.um, power=1 * u.mW, weights=[0.25, 0.75])
    assert float(read(meter, cube).data.to_value(u.W)) == pytest.approx(1e-3, rel=1e-12)
    modal = ModalField(np.sqrt([[1e15, 2e15]]), [LAM] * u.m, power_unit=u.photon / u.s)
    np.testing.assert_allclose(read(meter, modal).data.to_value(u.W), np.array([1e15, 2e15]) * H * C / LAM)
