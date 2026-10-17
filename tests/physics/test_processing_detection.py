"""Tests for detection statistics (step P11.3, REQ-PIPE-4).

Requirements: test statistics (energy, Neyman-Pearson), ROC curves, contrast limits and detection capability
versus separation and contrast; ROC of a Neyman-Pearson test on a Gaussian signal compared with the analytic
form (plan, tests of phase 11).

Independent references (Kay 1998, *Fundamentals of Statistical Signal Processing*, vol. II):

- known signal s in white Gaussian noise of standard deviation σ: the Neyman-Pearson test is the matched
  filter T = sᵀx / (σ‖s‖), distributed as N(0, 1) without signal and N(d, 1) with it, d = ‖s‖/σ; hence
  P_D = Q(Q⁻¹(P_FA) - d) (§4.3.1, eq. 4.14);
- energy detector T = Σ (x/σ)²: χ²_N without signal, non-central χ²_N(λ = Σ (s/σ)²) with it (§5.3);
- the distributions come from scipy.stats; the samples from NumPy's random generator.
"""

import numpy as np
import pytest
from astropy import units as u
from scipy import stats

from lumos.fields import DataProduct
from lumos.processing import detection

RNG = np.random.default_rng(7)


def test_matched_filter_roc_matches_the_analytic_form():
    n, sigma, trials = 50, 2.0, 20000
    template = np.sin(np.linspace(0, 3 * np.pi, n))
    signal = 0.6 * template  # d = 0.6 ‖t‖ / σ
    d = np.linalg.norm(signal) / sigma
    h0 = detection.matched_filter(sigma * RNG.standard_normal((trials, n)), template, sigma)
    h1 = detection.matched_filter(signal + sigma * RNG.standard_normal((trials, n)), template, sigma)
    assert np.mean(h0) == pytest.approx(0, abs=0.03)
    assert np.std(h0) == pytest.approx(1, rel=0.02)
    assert np.mean(h1) == pytest.approx(d, abs=0.03)
    roc = detection.roc_curve(h0, h1)
    pfa = roc.axes["false_alarm_probability"].value
    pd = roc.data.value
    analytic = stats.norm.sf(stats.norm.isf(pfa) - d)
    inside = (pfa > 0.01) & (pfa < 0.99)
    np.testing.assert_allclose(pd[inside], analytic[inside], atol=0.015)
    model = detection.roc_matched_filter(np.array([1e-3, 1e-2, 0.1]), deflection=d)
    np.testing.assert_allclose(model, stats.norm.sf(stats.norm.isf([1e-3, 1e-2, 0.1]) - d), rtol=1e-12)


def test_energy_detector_distributions():
    n, trials = 20, 20000
    signal = np.full(n, 0.5)
    h0 = detection.energy(RNG.standard_normal((trials, n)), sigma=1.0)
    h1 = detection.energy(signal + RNG.standard_normal((trials, n)), sigma=1.0)
    assert stats.kstest(h0, stats.chi2(n).cdf).pvalue > 1e-3
    assert stats.kstest(h1, stats.ncx2(n, np.sum(signal**2)).cdf).pvalue > 1e-3
    pfa = np.array([1e-2, 0.1])
    expected = stats.ncx2.sf(stats.chi2.isf(pfa, n), n, np.sum(signal**2))
    np.testing.assert_allclose(detection.roc_energy(pfa, dof=n, noncentrality=np.sum(signal**2)), expected)


@pytest.mark.parametrize("pfa", [1e-3, 1e-2, 0.05])
def test_thresholds_give_the_false_alarm_rate(pfa):
    trials = 200000
    h0 = RNG.standard_normal(trials)
    threshold = detection.threshold(pfa, statistic="matched_filter")
    assert np.mean(h0 > threshold) == pytest.approx(pfa, rel=0.1)
    energy = detection.energy(RNG.standard_normal((trials // 10, 5)), sigma=1.0)
    assert np.mean(energy > detection.threshold(pfa, statistic="energy", dof=5)) == pytest.approx(
        pfa, rel=0.25
    )


def test_contrast_limit_gives_the_requested_detection_probability():
    separations = np.array([1.0, 2.0, 4.0]) * u.arcsec
    signal_per_contrast = np.array([2e5, 5e5, 8e5])  # matched-filter deflection per unit contrast
    limit = detection.contrast_limit(
        separations, deflection_per_contrast=signal_per_contrast, pfa=1e-3, pd=0.5
    )
    assert limit.axes["separation"].unit == u.arcsec
    trials = 50000
    for k in range(3):
        d = limit.data[k].value * signal_per_contrast[k]
        statistic = d + RNG.standard_normal(trials)
        pd = np.mean(statistic > stats.norm.isf(1e-3))
        assert pd == pytest.approx(0.5, abs=0.01)


def test_detection_capability_map():
    separations = np.array([1.0, 2.0]) * u.arcsec
    contrasts = np.logspace(-6, -4, 5)
    capability = detection.detection_capability(
        separations, contrasts, deflection_per_contrast=np.array([1e5, 3e5]), pfa=1e-3
    )
    assert capability.shape == (5, 2)
    assert capability.axis_names == ("contrast", "separation")
    expected = stats.norm.sf(stats.norm.isf(1e-3) - contrasts[:, None] * np.array([1e5, 3e5])[None, :])
    np.testing.assert_allclose(capability.data.value, expected, rtol=1e-12)
    assert np.all(np.diff(capability.data.value, axis=0) >= 0)


def test_statistic_elements():
    data = DataProduct(np.array([3.0, 4.0]) * u.adu, uncertainty=np.array([1.0, 2.0]) * u.adu)
    out = detection.EnergyDetector().apply(**{"in": data})["out"]
    assert out.data[0].value == pytest.approx(9 + 4)
    template = DataProduct(np.array([1.0, 1.0]) * u.adu)
    matched = detection.MatchedFilter().apply(**{"in": data, "template": template})["out"]
    # sᵀ Σ⁻¹ x / √(sᵀ Σ⁻¹ s) with Σ = diag(σ²): (3/1 + 4/4) / √(1 + 1/4)
    assert matched.data[0].value == pytest.approx((3 + 1) / np.sqrt(1.25))
