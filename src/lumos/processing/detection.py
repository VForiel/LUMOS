r"""Detection statistics, ROC curves, contrast limits and detection capability (step P11.3, REQ-PIPE-4).

Test statistics for a measurement :math:`x` with independent Gaussian errors :math:`\sigma_i`
(Kay 1998, *Fundamentals of Statistical Signal Processing*, vol. II):

- :func:`matched_filter`, the Neyman-Pearson test of a known signal shape :math:`s`:
  :math:`T = \sum_i s_i x_i/\sigma_i^2 \,/\, \sqrt{\sum_i s_i^2/\sigma_i^2}`, distributed as
  :math:`\mathcal N(0, 1)` without signal and :math:`\mathcal N(d, 1)` with the signal :math:`a s`, of
  deflection :math:`d = a\sqrt{\sum s_i^2/\sigma_i^2}`. Its ROC is :math:`P_D = Q(Q^{-1}(P_{FA}) - d)`
  (:func:`roc_matched_filter`).
- :func:`energy`, for an unknown signal: :math:`T = \sum_i (x_i/\sigma_i)^2`, :math:`\chi^2_N` without
  signal and non-central :math:`\chi^2_N(\lambda)` with :math:`\lambda = \sum (s_i/\sigma_i)^2`
  (:func:`roc_energy`).

:func:`threshold` gives the threshold of a false-alarm probability; :func:`roc_curve` the empirical ROC of
samples of a statistic under both hypotheses. For a matched filter whose deflection grows linearly with the
contrast of a companion, :math:`d = c\,k(r)` (:math:`k` the deflection per unit contrast at separation
:math:`r`, from the throughput, the stellar flux and the noise), :func:`contrast_limit` returns the contrast
detected with probability :math:`P_D` at the false-alarm probability :math:`P_{FA}`,
:math:`c = (Q^{-1}(P_{FA}) - Q^{-1}(P_D))/k`, and :func:`detection_capability` the map
:math:`P_D(c, r)`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from astropy import units as u

from lumos.core.element import Element, Values
from lumos.core.port import Input, Output
from lumos.core.registry import register
from lumos.fields.dataproduct import DataProduct

__all__ = [
    "EnergyDetector",
    "MatchedFilter",
    "contrast_limit",
    "detection_capability",
    "energy",
    "matched_filter",
    "roc_curve",
    "roc_energy",
    "roc_matched_filter",
    "threshold",
]


def matched_filter(data: Any, template: Any, sigma: Any = 1.0) -> np.ndarray:
    """Return the matched-filter statistic of ``data`` (last axis: samples) for the signal ``template``."""
    x = np.asarray(data, dtype=float)
    s = np.asarray(template, dtype=float)
    weight = 1.0 / np.asarray(sigma, dtype=float) ** 2 * np.ones_like(s)
    return (x * s * weight).sum(axis=-1) / np.sqrt(np.sum(s**2 * weight))


def energy(data: Any, sigma: Any = 1.0) -> np.ndarray:
    """Return the energy statistic ``Σ (x/σ)²`` of ``data`` (last axis: samples)."""
    x = np.asarray(data, dtype=float) / np.asarray(sigma, dtype=float)
    return (x**2).sum(axis=-1)


def threshold(pfa: Any, *, statistic: str = "matched_filter", dof: int | None = None) -> Any:
    """Return the threshold of the false-alarm probability ``pfa`` (``"matched_filter"`` or ``"energy"``)."""
    from scipy import stats

    if statistic == "matched_filter":
        return stats.norm.isf(pfa)
    if statistic == "energy":
        if dof is None:
            raise ValueError("the energy statistic needs its number of degrees of freedom (dof)")
        return stats.chi2.isf(pfa, dof)
    raise ValueError(f"unknown statistic {statistic!r}: use 'matched_filter' or 'energy'")


def roc_matched_filter(pfa: Any, *, deflection: Any) -> Any:
    """Return the detection probability of a matched filter of ``deflection`` at the probabilities ``pfa``."""
    from scipy import stats

    return stats.norm.sf(stats.norm.isf(pfa) - deflection)


def roc_energy(pfa: Any, *, dof: int, noncentrality: float) -> Any:
    """Return the detection probability of an energy detector (``dof`` samples, non-centrality λ)."""
    from scipy import stats

    return stats.ncx2.sf(stats.chi2.isf(pfa, dof), dof, noncentrality)


def roc_curve(h0: Any, h1: Any) -> DataProduct:
    """Empirical ROC curve of samples of a statistic without (``h0``) and with (``h1``) signal.

    Returns
    -------
    DataProduct
        The detection probability on the axis ``false_alarm_probability`` (increasing), one point per
        threshold taken at every sample.
    """
    a = np.sort(np.ravel(np.asarray(h0, dtype=float)))
    b = np.sort(np.ravel(np.asarray(h1, dtype=float)))
    thresholds = np.unique(np.concatenate([a, b, [-np.inf]]))[::-1]
    pfa = 1.0 - np.searchsorted(a, thresholds, side="right") / a.size
    pd = 1.0 - np.searchsorted(b, thresholds, side="right") / b.size
    return DataProduct(
        pd * u.one, axes={"false_alarm_probability": pfa * u.one}, meta={"n_h0": a.size, "n_h1": b.size}
    )


def contrast_limit(
    separations: u.Quantity, *, deflection_per_contrast: Any, pfa: float = 1e-3, pd: float = 0.5
) -> DataProduct:
    """Return the contrast a matched filter detects with probability ``pd`` at the false-alarm ``pfa``."""
    from scipy import stats

    k = np.asarray(deflection_per_contrast, dtype=float)
    required = float(stats.norm.isf(pfa) - stats.norm.isf(pd))
    return DataProduct(
        required / k * u.one,
        axes={"separation": np.atleast_1d(separations)},
        meta={"pfa": float(pfa), "pd": float(pd)},
    )


def detection_capability(
    separations: u.Quantity, contrasts: Any, *, deflection_per_contrast: Any, pfa: float = 1e-3
) -> DataProduct:
    """Return the detection probability of a matched filter versus contrast (rows) and separation."""
    from scipy import stats

    c = np.atleast_1d(np.asarray(contrasts, dtype=float))
    k = np.atleast_1d(np.asarray(deflection_per_contrast, dtype=float))
    probability = stats.norm.sf(stats.norm.isf(pfa) - c[:, None] * k[None, :])
    return DataProduct(
        probability * u.one,
        axes={"contrast": c * u.one, "separation": np.atleast_1d(separations)},
        meta={"pfa": float(pfa)},
    )


def _sigma(product: DataProduct) -> np.ndarray:
    if product.uncertainty is None:
        raise ValueError("the statistic needs the uncertainties of the data")
    return np.asarray(product.uncertainty.to_value(product.unit), dtype=float)


@register("lumos.processing.EnergyDetector")
class EnergyDetector(Element):
    """Energy statistic ``Σ (x/σ)²`` of a product with uncertainties."""

    in_ = Input(DataProduct)
    out = Output(DataProduct)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the statistic and its number of degrees of freedom."""
        data = inputs["in"]
        x = np.asarray(data.data.value, dtype=float).ravel()
        statistic = energy(x, _sigma(data).ravel())
        return {"out": DataProduct(np.atleast_1d(statistic) * u.one, meta={"dof": int(x.size)})}


@register("lumos.processing.MatchedFilter")
class MatchedFilter(Element):
    """Matched-filter (Neyman-Pearson) statistic of a product for the signal shape given on ``template``."""

    in_ = Input(DataProduct)
    template = Input(DataProduct)
    out = Output(DataProduct)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the statistic, N(0, 1) without signal."""
        data, template = inputs["in"], inputs["template"]
        x = np.asarray(data.data.value, dtype=float).ravel()
        s = np.asarray(template.data.to_value(data.unit), dtype=float).ravel()
        statistic = matched_filter(x, s, _sigma(data).ravel())
        return {"out": DataProduct(np.atleast_1d(statistic) * u.one)}
