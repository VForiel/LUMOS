"""Tests for procedural scene components (step P8.6, REQ-SCN-8, §5.9 test targets).

Requirements: textures and random fields, for tests and generic imaging; laboratory test targets (Siemens
star, bars).

Independent references:

- a Gaussian random field with a power spectrum ``P(k) ∝ k^-β`` has a radially averaged periodogram
  ``|FFT|²`` of slope ``-β`` in log-log (Wiener-Khinchin theorem); measured here with ``numpy.fft``;
- a Siemens star of N spokes alternates 2N times between dark and bright around a circle (construction);
- a checkerboard of period p has a mean of 1/2 over whole periods and anti-symmetric neighbouring squares;
- the geometry of test targets is given in pixels of their grid, so that the same target serves sky scenes
  (``pixel_scale``, an angle) and laboratory scenes (``pixel_size``, a length);
- random fields are reproducible: same seed, same field; another seed, another field (ADR-0007).
"""

import numpy as np
import pytest
from astropy import units as u

from lumos import Context
from lumos.scenes.procedural import BarTarget, Checkerboard, GaussianRandomField, SiemensStar

RADIANCE = u.W / u.m**2 / u.sr / u.m
SB = u.Jy / u.arcsec**2


def run(component, seed=1):
    ctx = Context(seed=seed)
    ctx.observation.band = (1.0 * u.um, 1.2 * u.um)
    ctx.observation.n_wavelengths = 1
    ctx.add(component)
    return ctx.run([f"{ctx.id_of(component)}.out"])[f"{ctx.id_of(component)}.out"]


def brightness(field):
    """Map in Jy/arcsec² (channel 0)."""
    values = field.radiance.to(
        u.Jy / u.sr, equivalencies=u.spectral_density(field.wavelengths[:, None, None])
    )
    return values[0].to_value(SB)


# --- Random fields ----------------------------------------------------------------------------------------


def field(seed=1, **values):
    defaults = {"surface_brightness": 2e-3 * SB, "contrast": 0.2, "spectral_index": 3.0, "n_pixels": 256}
    return run(GaussianRandomField(name="rf", **{**defaults, **values}), seed=seed)


def test_random_field_mean_and_contrast():
    image = brightness(field())
    assert image.mean() == pytest.approx(2e-3, rel=1e-6)
    assert image.std() / image.mean() == pytest.approx(0.2, rel=1e-6)


def test_random_field_is_reproducible():
    a, b, c = brightness(field(seed=5)), brightness(field(seed=5)), brightness(field(seed=6))
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


@pytest.mark.parametrize("beta", [2.0, 3.5])
def test_random_field_power_spectrum_slope(beta):
    image = brightness(field(spectral_index=beta, contrast=0.05))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(image - image.mean()))) ** 2
    n = image.shape[0]
    ky, kx = np.indices(spectrum.shape) - n // 2
    k = np.hypot(kx, ky).astype(int)
    radial = np.bincount(k.ravel(), spectrum.ravel()) / np.bincount(k.ravel())
    ks = np.arange(4, n // 4)
    slope = np.polyfit(np.log(ks), np.log(radial[ks]), 1)[0]
    assert slope == pytest.approx(-beta, abs=0.2)


def test_random_field_stays_positive():
    image = brightness(field(contrast=0.5))
    assert image.min() >= 0.0


# --- Test targets -----------------------------------------------------------------------------------------


def test_siemens_star_alternates_twice_per_spoke():
    star = SiemensStar(name="star", n_spokes=12, n_pixels=256, pixel_scale=1 * u.mas, radius=120.0)
    image = brightness(run(star))
    angles = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    radius = 80  # pixels
    rows = np.round(128 + radius * np.sin(angles)).astype(int)
    columns = np.round(128 + radius * np.cos(angles)).astype(int)
    samples = image[rows, columns] > 0.5 * image.max()
    transitions = np.count_nonzero(samples != np.roll(samples, 1))
    assert transitions == 24


def test_siemens_star_is_dark_outside_its_radius():
    star = SiemensStar(name="star", n_spokes=8, n_pixels=128, pixel_scale=1 * u.mas, radius=40.0)
    image = brightness(run(star))
    assert image[0, 0] == 0.0
    assert image.max() == pytest.approx(1e-3, rel=1e-6)  # default surface brightness


def test_checkerboard():
    board = Checkerboard(name="board", period=8.0, n_pixels=64, pixel_scale=1 * u.mas)
    image = brightness(run(board))
    levels = image / image.max()
    assert levels.mean() == pytest.approx(0.5, abs=1e-12)
    # squares have a corner on the axis: pixel centres 1 and 5 lie inside neighbouring squares
    assert levels[1, 1] != levels[1, 5]
    assert levels[1, 1] == levels[5, 5]


def test_bar_target_has_the_requested_bars():
    bars = BarTarget(name="bars", n_bars=3, period=20.0, n_pixels=128, pixel_scale=0.5 * u.mas)
    image = brightness(run(bars))
    row = image[64] > 0.5 * image.max()
    starts = np.count_nonzero(row & ~np.roll(row, 1))
    assert starts == 3


def test_spatial_targets_for_laboratory_scenes():
    star = SiemensStar(name="star", n_spokes=8, n_pixels=64, pixel_size=10 * u.um, radius=30.0)
    result = run(star)
    assert result.grid.plane == "spatial"
    assert result.grid.dx.to_value(u.um) == pytest.approx(10.0)
