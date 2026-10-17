"""Tests for the reduction blocks (step P11.1, REQ-PIPE-1, REQ-PIPE-2).

Requirements: dark subtraction, flat field, stacking, cropping and resampling, aperture photometry, radial
profile, contrast and SNR, as functions on ``DataProduct`` and as elements with ``DataProduct`` ports in the
graph; uncertainties are propagated (plan, tests of phase 11).

Independent references:

- propagation of uncertainties of independent quantities (JCGM 100:2008, GUM, §5.1): σ(a - b)² = σa² + σb²,
  σ(a / b)² = (σa / b)² + (a σb / b²)², σ(mean of N) = σ / √N; checked here against the empirical scatter of
  Monte-Carlo draws (NumPy random generator);
- the efficiency of the median of N normal samples: σ_med ≈ √(π/2) σ / √N for large N (Kenney & Keeping
  1962, §13.13);
- a circular Gaussian spot of total flux F and standard deviation s holds F (1 - exp(-r²/2s²)) within r
  (checked with an aperture of 6 s, where the pixelation of the spot is negligible);
- the area of a disk of radius r is π r²: partial pixels count by their exact overlap with the aperture;
- the CCD equation SNR = S / √(S + n (B + D + R²)) (Howell 2006, *Handbook of CCD Astronomy*, eq. 4.4),
  checked against the scatter of simulated Poisson and Gaussian noises;
- the contrast limit with the small-sample penalty of Mawet et al. (2014, ApJ 792, 97, eq. 9): with n
  resolution elements at separation r, the threshold is σ_r t(1 - f, n - 1) √(1 + 1/n), f being the false
  positive fraction of a 5σ Gaussian threshold (1 - Φ(5)); computed here with scipy.stats.
"""

import math

import numpy as np
import pytest
from astropy import units as u
from scipy import stats

from lumos import Context
from lumos.core.element import Element
from lumos.core.port import Input, Output
from lumos.fields import DataProduct
from lumos.processing import reduction
from lumos.processing.photometry import aperture_photometry, ccd_snr, contrast_curve

RNG = np.random.default_rng(20261017)
E = u.electron


def frames(n, shape=(16, 16), mean=100.0, sigma=5.0):
    return [
        DataProduct(
            (mean + sigma * RNG.standard_normal(shape)) * E,
            uncertainty=np.full(shape, sigma) * E,
        )
        for _ in range(n)
    ]


# --- Uncertainties ------------------------------------------------------------------------------------------


def test_dark_subtraction_propagates_uncertainties():
    trials = 4000
    science = 1000 + 30 * RNG.standard_normal(trials)
    dark = 50 + 7 * RNG.standard_normal(trials)
    one = reduction.subtract_dark(
        DataProduct(np.array([1000.0]) * E, uncertainty=np.array([30.0]) * E),
        DataProduct(np.array([50.0]) * E, uncertainty=np.array([7.0]) * E),
    )
    assert one.data[0] == 950 * E
    assert one.uncertainty[0].to_value(E) == pytest.approx(math.hypot(30, 7), rel=1e-12)
    assert one.uncertainty[0].to_value(E) == pytest.approx(np.std(science - dark), rel=0.05)


def test_flat_field_propagates_uncertainties():
    trials = 20000
    signal, sigma_s = 500.0, 20.0
    flat, sigma_f = 0.8, 0.02
    draws = (signal + sigma_s * RNG.standard_normal(trials)) / (flat + sigma_f * RNG.standard_normal(trials))
    result = reduction.flat_field(
        DataProduct(np.array([signal]) * E, uncertainty=np.array([sigma_s]) * E),
        DataProduct(np.array([flat]) * u.one, uncertainty=np.array([sigma_f]) * u.one),
        normalise=False,
    )
    expected = math.sqrt((sigma_s / flat) ** 2 + (signal * sigma_f / flat**2) ** 2)
    assert result.data[0].to_value(E) == pytest.approx(signal / flat)
    assert result.uncertainty[0].to_value(E) == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(np.std(draws), rel=0.03)


def test_normalised_flat_keeps_the_mean_level():
    flat = DataProduct(np.array([[1.0, 3.0], [2.0, 2.0]]) * u.adu)
    science = DataProduct(np.array([[10.0, 30.0], [20.0, 20.0]]) * E)
    result = reduction.flat_field(science, flat)
    np.testing.assert_allclose(result.data.to_value(E), 20.0)


def test_mean_stack_uncertainty():
    stack = reduction.stack(frames(25), method="mean")
    assert stack.shape == (16, 16)
    np.testing.assert_allclose(stack.uncertainty.to_value(E), 5.0 / 5.0)
    # empirical: the scatter of many mean stacks
    means = [reduction.stack(frames(25, shape=(1,)), method="mean").data[0].value for _ in range(800)]
    assert np.std(means) == pytest.approx(1.0, rel=0.08)


def test_median_stack_uncertainty():
    stack = reduction.stack(frames(101), method="median")
    np.testing.assert_allclose(stack.uncertainty.to_value(E), math.sqrt(math.pi / 2) * 5 / math.sqrt(101))


def test_sigma_clipped_stack_rejects_outliers():
    data = frames(20, shape=(4, 4), sigma=1.0)
    hot = data[3].data.value.copy()
    hot[1, 2] = 1e6  # cosmic ray
    data[3] = DataProduct(hot * E, uncertainty=data[3].uncertainty)
    clipped = reduction.stack(data, method="sigma_clip", sigma=4.0)
    assert abs(clipped.data[1, 2].value - 100) < 2
    assert abs(reduction.stack(data, method="mean").data[1, 2].value - 100) > 1000


def test_stack_of_a_cube_along_its_first_axis():
    cube = DataProduct(np.arange(24.0).reshape(4, 2, 3) * E)
    result = reduction.stack(cube, method="mean")
    np.testing.assert_allclose(result.data.value, np.arange(24.0).reshape(4, 2, 3).mean(axis=0))
    assert result.axis_names == cube.axis_names[1:]


# --- Geometry -----------------------------------------------------------------------------------------------


def image_product(n=32, dx=2.0):
    axis = (np.arange(n) - n // 2) * dx * u.um
    data = RNG.random((n, n)) * E
    return DataProduct(data, axes={"y": axis, "x": axis}, uncertainty=np.full((n, n), 0.5) * E)


def test_crop_keeps_coordinates():
    image = image_product()
    cropped = reduction.crop(image, center=(4 * u.um, -2 * u.um), size=(8, 6))
    assert cropped.shape == (6, 8)
    np.testing.assert_allclose(cropped.axes["x"].to_value(u.um), (np.arange(8) - 4) * 2.0 + 4)
    np.testing.assert_allclose(cropped.axes["y"].to_value(u.um), (np.arange(6) - 3) * 2.0 - 2)
    np.testing.assert_array_equal(
        cropped.data.value, image.data.value[16 - 1 - 3 : 16 - 1 + 3, 18 - 4 : 18 + 4]
    )


def test_rebin_conserves_the_sum_and_adds_uncertainties_in_quadrature():
    image = image_product()
    binned = reduction.rebin(image, 4)
    assert binned.shape == (8, 8)
    assert binned.data.sum().value == pytest.approx(image.data.sum().value)
    np.testing.assert_allclose(binned.uncertainty.to_value(E), 0.5 * 4)  # √16 × 0.5
    np.testing.assert_allclose(
        binned.axes["x"].to_value(u.um), image.axes["x"].to_value(u.um).reshape(8, 4).mean(1)
    )


def test_resample_conserves_the_flux():
    image = image_product()
    new_axis = (np.arange(22) - 11) * 3.0 * u.um
    resampled = reduction.resample(image, x=new_axis, y=new_axis)
    assert resampled.shape == (22, 22)
    # new pixel edges -34.5 … 31.5 µm cover the old ones (-33 … 31 µm): every electron is kept
    assert resampled.data.sum().value == pytest.approx(image.data.sum().value, rel=1e-9)
    # a uniform image of 1 e/pixel of 4 µm² gives 9/4 e/pixel of 9 µm² where the new pixels are covered
    uniform = DataProduct(np.ones((32, 32)) * E, axes=image.axes)
    inner = reduction.resample(uniform, x=new_axis, y=new_axis).data.value[2:-2, 2:-2]
    np.testing.assert_allclose(inner, 9 / 4, rtol=1e-12)


# --- Photometry and SNR ------------------------------------------------------------------------------------


def spot(flux=1000.0, s=2.0, background=10.0, n=64, center=(0.3, -0.7)):
    yy, xx = np.mgrid[0:n, 0:n] - n // 2
    gauss = (
        flux / (2 * math.pi * s**2) * np.exp(-((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2 * s**2))
    )
    axis = (np.arange(n) - n // 2) * u.pix
    return DataProduct((gauss + background) * E, axes={"y": axis, "x": axis})


def test_aperture_photometry_with_background_annulus():
    image = spot()
    result = aperture_photometry(
        image, center=(0.3 * u.pix, -0.7 * u.pix), radius=12 * u.pix, annulus=(16 * u.pix, 24 * u.pix)
    )
    assert result.background.to_value(E) == pytest.approx(10.0, rel=1e-9)
    expected = 1000 * (1 - math.exp(-(12**2) / (2 * 2.0**2)))
    assert result.flux.to_value(E) == pytest.approx(expected, rel=1e-6)
    assert result.area.to_value(u.pix) == pytest.approx(math.pi * 144, rel=1e-3)


def test_partial_pixels_are_weighted_by_their_exact_overlap():
    n = 32
    axis = (np.arange(n) - n // 2) * u.pix
    uniform = DataProduct(np.ones((n, n)) * E, axes={"y": axis, "x": axis})
    for radius in (0.4, 1.5, 2.3, 3.7, 5.2):
        result = aperture_photometry(uniform, center=(0.3 * u.pix, -0.7 * u.pix), radius=radius * u.pix)
        assert result.flux.to_value(E) == pytest.approx(math.pi * radius**2, rel=1e-12)
    ring = aperture_photometry(
        uniform, center=(0.0 * u.pix, 0.0 * u.pix), radius=3 * u.pix, annulus=(4 * u.pix, 7 * u.pix)
    )
    assert ring.background_area.to_value(u.pix) == pytest.approx(math.pi * (49 - 16), rel=1e-12)


def test_ccd_equation_matches_simulated_noise():
    signal, n_pix, background, dark, read = 2000.0, 30, 15.0, 2.0, 4.0
    snr = ccd_snr(signal * E, n_pixels=n_pix, background=background * E, dark=dark * E, read_noise=read * E)
    assert snr == pytest.approx(signal / math.sqrt(signal + n_pix * (background + dark + read**2)))
    trials = 20000
    measured = (
        RNG.poisson(signal + n_pix * (background + dark), trials)
        + RNG.normal(0, read * math.sqrt(n_pix), trials)
        - n_pix * (background + dark)
    )
    assert signal / np.std(measured) == pytest.approx(snr, rel=0.03)


def test_contrast_curve_with_small_sample_penalty():
    n, resolution = 128, 4.0  # λ/D = 4 pixels
    noise = RNG.standard_normal((n, n))
    axis = (np.arange(n) - n // 2) * u.pix
    image = DataProduct(noise * E, axes={"y": axis, "x": axis})
    curve = contrast_curve(image, peak=1e4 * E, resolution=resolution * u.pix, sigma=5.0)
    separations = curve.axes["separation"].to_value(u.pix)
    yy, xx = np.mgrid[0:n, 0:n] - n // 2
    rr = np.hypot(xx, yy)
    fpf = stats.norm.sf(5.0)
    for k in (1, 3, 8):
        r = separations[k]
        ring = noise[(rr >= r - resolution / 2) & (rr < r + resolution / 2)]
        n_el = math.floor(2 * math.pi * r / resolution)
        expected = ring.std(ddof=1) * stats.t.ppf(1 - fpf, n_el - 1) * math.sqrt(1 + 1 / n_el) / 1e4
        assert curve.data[k].value == pytest.approx(expected, rel=1e-9)
    assert separations[0] == pytest.approx(resolution)


# --- Elements in the graph ----------------------------------------------------------------------------------


class FrameSource(Element):
    out = Output(DataProduct)

    @staticmethod
    def transfer(values, inputs):
        return {"out": DataProduct(np.full((4, 4), 120.0) * E, uncertainty=np.full((4, 4), 3.0) * E)}


class DarkSource(Element):
    out = Output(DataProduct)

    @staticmethod
    def transfer(values, inputs):
        return {"out": DataProduct(np.full((4, 4), 20.0) * E, uncertainty=np.full((4, 4), 4.0) * E)}


def test_reduction_elements_in_a_context():
    ctx = Context()
    science = ctx.add(FrameSource(name="science"))
    dark = ctx.add(DarkSource(name="dark"))
    subtract = ctx.add(reduction.DarkSubtraction(name="subtract"))
    ctx.connect(science.port("out"), subtract.port("in"))
    ctx.connect(dark.port("out"), subtract.port("dark"))
    binned = ctx.add(reduction.Rebin(name="bin", factor=2))
    subtract >> binned
    out = ctx.run(["bin.out"])["bin.out"]
    np.testing.assert_allclose(out.data.to_value(E), 400.0)
    np.testing.assert_allclose(out.uncertainty.to_value(E), 2 * 5.0)
    assert isinstance(reduction.Stack(name="s"), Element)
    assert isinstance(reduction.DarkSubtraction.dark, Input)


def test_ccd_equation_with_the_error_of_the_background_estimate():
    """Howell (2006), eq. 4.4 with the factor (1 + n/n_B) of a background measured over n_B pixels."""
    signal, n_pix, n_back, background, read = 3000.0, 40, 200, 20.0, 5.0
    snr = ccd_snr(
        signal * E, n_pixels=n_pix, n_background=n_back, background=background * E, read_noise=read * E
    )
    assert snr == pytest.approx(
        signal / math.sqrt(signal + n_pix * (1 + n_pix / n_back) * (background + read**2))
    )
    trials = 20000
    sky = RNG.poisson(background, (trials, n_back)) + RNG.normal(0, read, (trials, n_back))
    source = RNG.poisson(signal + n_pix * background, trials) + RNG.normal(0, read * math.sqrt(n_pix), trials)
    measured = source - n_pix * sky.mean(axis=1)
    assert signal / np.std(measured) == pytest.approx(snr, rel=0.03)
