"""VizieR photometry: the spectral energy distribution gathered by the CDS SED service.

The VizieR photometry viewer (Allen et al. 2014, ASPC 485, 193) collects, around a position or a name,
every photometric point of the VizieR catalogues: frequency, flux density, uncertainty, filter and
catalogue. They are returned as a :class:`PhotometricPoints` table, the raw material of the SED fusion
(:mod:`lumos.catalogs.sed`).
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from astropy import units as u

from lumos.catalogs.http import CatalogError, fetch
from lumos.catalogs.provider import Provider, register_provider

__all__ = ["PhotometricPoints", "VizierPhotometry"]


@dataclass
class PhotometricPoints:
    """Photometric measurements of one object.

    Attributes
    ----------
    wavelengths
        Effective wavelengths of the filters.
    flux_density
        Spectral flux densities per unit frequency (Jy).
    uncertainty
        Uncertainties (NaN when unknown).
    filters
        Names of the filters (``"Johnson:V"``, ``"2MASS:Ks"``...).
    catalogues
        VizieR tables of the measurements (``"II/246/out"``...).
    """

    wavelengths: u.Quantity
    flux_density: u.Quantity
    uncertainty: u.Quantity
    filters: list[str]
    catalogues: list[str]

    def __len__(self) -> int:
        """Return the number of points."""
        return len(self.filters)


@register_provider
class VizierPhotometry(Provider):
    """VizieR (CDS) photometry viewer: every photometric point of the VizieR catalogues."""

    name = "vizier"
    description = "VizieR photometry viewer (CDS)"
    url = "https://vizier.cds.unistra.fr/viz-bin/sed"
    reference = "Ochsenbein et al. 2000, A&AS 143, 23; Allen et al. 2014, ASPC 485, 193"

    def __init__(self, url: str | None = None) -> None:
        self.endpoint = url or type(self).url

    def sed(self, target: str, *, radius: u.Quantity = 1 * u.arcsec) -> PhotometricPoints:
        """Return the photometric points within ``radius`` of ``target`` (a name or "ra dec" in degrees)."""
        from astropy.io.votable import parse

        seconds = float(cast(Any, radius).to_value(u.arcsec))
        response = fetch(self.endpoint, params={"-c": target, "-c.rs": f"{seconds:g}"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                table: Any = parse(io.BytesIO(response.content), verify="ignore").get_first_table().to_table()
            except Exception as error:
                raise CatalogError(f"VizieR SED: unreadable answer for {target!r}: {error}") from error
        frequency = np.asarray(table["sed_freq"], dtype=float) * u.GHz
        flux = np.asarray(np.ma.filled(table["sed_flux"], np.nan), dtype=float)
        error = np.asarray(np.ma.filled(table["sed_eflux"], np.nan), dtype=float)
        keep = np.isfinite(flux) & (flux > 0) & np.isfinite(frequency.value) & (frequency.value > 0)
        wavelengths = frequency[keep].to(u.um, equivalencies=u.spectral())
        return PhotometricPoints(
            wavelengths,
            flux[keep] * u.Jy,
            error[keep] * u.Jy,
            [str(name) for name in np.asarray(table["sed_filter"])[keep]],
            [str(name) for name in np.asarray(table["_tabname"])[keep]],
        )
