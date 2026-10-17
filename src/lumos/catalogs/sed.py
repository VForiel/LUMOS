r"""Fusion and completion of spectral energy distributions (step P9.7, REQ-CAT-4).

Fusion rules
------------
:func:`fuse` turns photometric points from many catalogues (:class:`~lumos.catalogs.vizier.PhotometricPoints`)
into one :class:`SpectralEnergyDistribution`, one point per filter:

1. points without a positive, finite flux density are dropped;
2. points are grouped by filter name;
3. within a filter, the points of the highest-priority catalogue present are kept (``priorities``, VizieR
   table prefixes, :data:`DEFAULT_PRIORITIES` by default);
4. among them, the point with the smallest relative uncertainty is chosen;
5. when none has an uncertainty, the median of their flux densities is used.

The result is sorted by wavelength and keeps the catalogue of each point.

Completion
----------
:func:`fit_blackbody` fits :math:`F_\nu = \Omega\,B_\nu(T)` to the SED (least squares on
:math:`\log F_\nu`): temperature :math:`T` and solid angle :math:`\Omega = \pi (\theta/2)^2` of the
photosphere of angular diameter :math:`\theta`. :meth:`SpectralEnergyDistribution.complete` adds points of
the fitted blackbody where the SED has none, and :func:`complete_star` fills the missing temperature,
distance (:math:`1/\varpi`) and radius (:math:`R = \theta d / 2`) of a star record. Completed values are
**flagged as estimated** (``source="lumos"``); measured values are never replaced.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from astropy import units as u

from lumos.catalogs.records import Measurement, StarRecord, record_from_json
from lumos.catalogs.vizier import PhotometricPoints

__all__ = ["DEFAULT_PRIORITIES", "SpectralEnergyDistribution", "complete_star", "fit_blackbody", "fuse"]

#: Default priority of catalogues (VizieR prefixes): homogeneous all-sky surveys first.
DEFAULT_PRIORITIES: tuple[str, ...] = (
    "I/355",  # Gaia DR3
    "II/246",  # 2MASS point-source catalogue
    "II/328",  # AllWISE
    "I/239",  # Hipparcos and Tycho
    "I/259",  # Tycho-2
    "II/237",  # Ducati 2002, Johnson UBVRIJKLMN
)

_H, _C, _K = 6.62607015e-34, 299_792_458.0, 1.380649e-23
_JY = 1e-26


def _scalar(quantity: Any, unit: Any) -> float:
    return float(cast(Any, quantity).to_value(unit))


def _planck_nu_jy(lam: np.ndarray, temperature: float, omega: float) -> np.ndarray:
    nu = _C / lam
    with np.errstate(over="ignore"):
        return omega * 2 * _H * nu**3 / _C**2 / np.expm1(_H * nu / (_K * temperature)) / _JY


@dataclass
class SpectralEnergyDistribution:
    """Fused SED: one flux density per filter.

    Attributes
    ----------
    wavelengths
        Effective wavelengths, increasing.
    flux_density
        Flux densities per unit frequency (Jy).
    uncertainty
        Uncertainties (NaN when unknown).
    filters
        Filter names.
    sources
        Catalogue of each point (``"lumos"`` for estimated points).
    estimated
        Whether each point was estimated by a completion.
    """

    wavelengths: u.Quantity
    flux_density: u.Quantity
    uncertainty: u.Quantity
    filters: list[str]
    sources: list[str]
    estimated: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    def __post_init__(self) -> None:
        """Default ``estimated`` to all ``False``."""
        if len(self.estimated) != len(self.filters):
            self.estimated = np.zeros(len(self.filters), dtype=bool)

    def __len__(self) -> int:
        """Return the number of points."""
        return len(self.filters)

    def complete(self, wavelengths: u.Quantity, **fit: Any) -> SpectralEnergyDistribution:
        """Return a copy with the fitted blackbody added at ``wavelengths`` (points flagged estimated)."""
        temperature, omega = fit_blackbody(self, **fit)
        lam = np.atleast_1d(np.asarray(cast(Any, wavelengths).to_value(u.m)))
        added = _planck_nu_jy(lam, _scalar(temperature, u.K), _scalar(omega, u.sr))
        all_lam = np.concatenate([np.asarray(cast(Any, self.wavelengths).to_value(u.m)), lam])
        order = np.argsort(all_lam)
        flux = np.concatenate([np.asarray(cast(Any, self.flux_density).to_value(u.Jy)), added])
        error = np.concatenate(
            [np.asarray(cast(Any, self.uncertainty).to_value(u.Jy)), np.full(lam.size, np.nan)]
        )
        filters = [*self.filters, *(["blackbody"] * lam.size)]
        sources = [*self.sources, *(["lumos"] * lam.size)]
        flags = np.concatenate([self.estimated, np.ones(lam.size, dtype=bool)])
        return SpectralEnergyDistribution(
            (all_lam[order] * u.m).to(u.um),
            flux[order] * u.Jy,
            error[order] * u.Jy,
            [filters[k] for k in order],
            [sources[k] for k in order],
            flags[order],
        )


def _priority(catalogue: str, priorities: Sequence[str]) -> int:
    for rank, prefix in enumerate(priorities):
        if catalogue.startswith(prefix):
            return rank
    return len(priorities)


def fuse(
    points: PhotometricPoints, *, priorities: Sequence[str] = DEFAULT_PRIORITIES
) -> SpectralEnergyDistribution:
    """Fuse photometric points into one point per filter (see the module documentation for the rules)."""
    lam = np.asarray(cast(Any, points.wavelengths).to_value(u.m), dtype=float)
    flux = np.asarray(cast(Any, points.flux_density).to_value(u.Jy), dtype=float)
    error = np.asarray(cast(Any, points.uncertainty).to_value(u.Jy), dtype=float)
    groups: dict[str, list[int]] = {}
    for k, name in enumerate(points.filters):
        if np.isfinite(flux[k]) and flux[k] > 0 and np.isfinite(lam[k]):
            groups.setdefault(name, []).append(k)
    chosen: list[tuple[float, float, float, str, str]] = []
    for name, members in groups.items():
        best_rank = min(_priority(points.catalogues[k], priorities) for k in members)
        candidates = [k for k in members if _priority(points.catalogues[k], priorities) == best_rank]
        with_error = [k for k in candidates if np.isfinite(error[k]) and error[k] > 0]
        if with_error:
            k = min(with_error, key=lambda index: error[index] / flux[index])
            chosen.append((lam[k], flux[k], error[k], name, points.catalogues[k]))
        else:
            values = flux[candidates]
            sources = sorted({points.catalogues[k] for k in candidates})
            source = (
                sources[0] if len(sources) == 1 else f"median of {len(candidates)} ({', '.join(sources)})"
            )
            chosen.append(
                (float(np.median(lam[candidates])), float(np.median(values)), math.nan, name, source)
            )
    chosen.sort(key=lambda item: item[0])
    return SpectralEnergyDistribution(
        (np.array([c[0] for c in chosen]) * u.m).to(u.um),
        np.array([c[1] for c in chosen]) * u.Jy,
        np.array([c[2] for c in chosen]) * u.Jy,
        [c[3] for c in chosen],
        [c[4] for c in chosen],
    )


def fit_blackbody(
    sed: SpectralEnergyDistribution,
    *,
    wavelength_range: tuple[u.Quantity, u.Quantity] | None = None,
    temperature: u.Quantity | None = None,
) -> tuple[u.Quantity, u.Quantity]:
    """Fit a blackbody to the measured points of ``sed``: return its temperature and solid angle.

    With ``temperature`` given, only the solid angle is fitted (mean of the logarithmic residuals).
    """
    from scipy.optimize import least_squares

    lam = np.asarray(cast(Any, sed.wavelengths).to_value(u.m), dtype=float)
    flux = np.asarray(cast(Any, sed.flux_density).to_value(u.Jy), dtype=float)
    keep = ~np.asarray(sed.estimated, dtype=bool) & (flux > 0)
    if wavelength_range is not None:
        low, high = (float(cast(Any, q).to_value(u.m)) for q in wavelength_range)
        keep &= (lam >= low) & (lam <= high)
    if keep.sum() < 2:
        raise ValueError("at least two measured points are needed to fit a blackbody")
    lam, log_flux = lam[keep], np.log(flux[keep])

    def log_omega(temperature: float) -> float:
        return float(np.mean(log_flux - np.log(_planck_nu_jy(lam, temperature, 1.0))))

    def cost(temperature: float) -> float:
        model = np.log(_planck_nu_jy(lam, temperature, 1.0)) + log_omega(temperature)
        return float(np.sum((log_flux - model) ** 2))

    if temperature is not None:
        fixed = float(cast(Any, temperature).to_value(u.K))
        return fixed * u.K, math.exp(log_omega(fixed)) * u.sr

    grid = np.geomspace(500.0, 100_000.0, 400)
    start = float(grid[int(np.argmin([cost(t) for t in grid]))])

    def residuals(p: np.ndarray) -> np.ndarray:
        temperature, log_scale = math.exp(p[0]), p[1]
        return log_flux - (np.log(_planck_nu_jy(lam, temperature, 1.0)) + log_scale)

    result = least_squares(residuals, x0=[math.log(start), log_omega(start)], x_scale="jac")
    return math.exp(result.x[0]) * u.K, math.exp(result.x[1]) * u.sr


def complete_star(
    record: StarRecord,
    sed: SpectralEnergyDistribution,
    *,
    wavelength_range: tuple[u.Quantity, u.Quantity] | None = None,
) -> StarRecord:
    """Return a copy of ``record`` with the missing temperature, distance and radius estimated from ``sed``.

    A measured temperature is kept and used to fit the solid angle; otherwise both are fitted.
    """
    completed = cast(StarRecord, record_from_json(record.to_json()))
    known = completed.teff if "teff" in completed else None
    temperature, omega = fit_blackbody(sed, wavelength_range=wavelength_range, temperature=known)
    reference = "blackbody fit of the SED"
    if "teff" not in completed:
        completed.set("teff", Measurement(temperature, None, "lumos", reference, estimated=True))
    if "distance" not in completed and "parallax" in completed:
        completed.set("distance", completed.distance_estimate())
    if "radius" not in completed and "distance" in completed:
        theta = 2 * math.sqrt(_scalar(omega, u.sr) / math.pi)  # angular diameter (rad)
        distance = float(cast(Any, completed.distance).to_value(u.m))
        radius = (theta / 2 * distance * u.m).to(u.R_sun)
        completed.set("radius", Measurement(radius, None, "lumos", reference, estimated=True))
    return completed
