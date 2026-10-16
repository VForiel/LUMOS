"""Observation archives (ESO, MAST) and infrared catalogues (IRSA), through their TAP services.

- :class:`Eso` and :class:`Mast` list the observations of a target in the IVOA ObsCore model (instrument,
  product type, position, spectral coverage, date, access URL): ESO Science Archive and the Mikulski Archive
  for Space Telescopes (HST, JWST, TESS...).
- :class:`Irsa` reads the 2MASS point-source catalogue (Skrutskie et al. 2006, AJ 131, 1163) at IPAC/IRSA.
"""

from __future__ import annotations

from typing import Any, cast

from astropy import units as u

from lumos.catalogs.provider import Provider, register_provider
from lumos.catalogs.records import Measurement
from lumos.catalogs.tap import adql_string, masked, tap_query

__all__ = ["Eso", "Irsa", "Mast"]

_OBSCORE = (
    "obs_id, target_name, instrument_name, dataproduct_type, s_ra, s_dec, em_min, em_max, t_min, access_url"
)


class _ObsCoreArchive(Provider):
    table = "ivoa.ObsCore"

    def __init__(self, url: str | None = None) -> None:
        self.endpoint = url or type(self).url

    def observations(self, target: str, *, limit: int = 100) -> Any:
        """Return the ObsCore rows of ``target`` (an :class:`astropy.table.Table`, at most ``limit`` rows)."""
        query = (
            f"SELECT TOP {int(limit)} {_OBSCORE} FROM {self.table} WHERE target_name = {adql_string(target)}"
        )
        return tap_query(self.endpoint, query)


@register_provider
class Eso(_ObsCoreArchive):
    """ESO Science Archive (ObsCore)."""

    name = "eso"
    description = "ESO Science Archive, observations (ObsCore)"
    url = "https://archive.eso.org/tap_obs/sync"
    reference = "ESO Science Archive Facility"
    table = "ivoa.ObsCore"


@register_provider
class Mast(_ObsCoreArchive):
    """Mikulski Archive for Space Telescopes (ObsCore of the CAOM database)."""

    name = "mast"
    description = "Mikulski Archive for Space Telescopes (STScI), observations (ObsCore)"
    url = "https://mast.stsci.edu/vo-tap/api/v0.1/caom/sync"
    reference = "Mikulski Archive for Space Telescopes (STScI)"
    table = "ivoa.obscore"


@register_provider
class Irsa(Provider):
    """NASA/IPAC Infrared Science Archive: 2MASS point sources."""

    name = "irsa"
    description = "NASA/IPAC Infrared Science Archive (2MASS)"
    url = "https://irsa.ipac.caltech.edu/TAP/sync"
    reference = "Skrutskie et al. 2006, AJ 131, 1163 (2MASS)"

    def __init__(self, url: str | None = None) -> None:
        self.endpoint = url or type(self).url

    def two_mass(
        self, ra: u.Quantity, dec: u.Quantity, *, radius: u.Quantity = 2 * u.arcsec
    ) -> dict[str, Measurement]:
        """Return the J, H and Ks magnitudes of the nearest 2MASS source within ``radius``."""
        ra_deg, dec_deg = float(cast(Any, ra).to_value(u.deg)), float(cast(Any, dec).to_value(u.deg))
        r_deg = float(cast(Any, radius).to_value(u.deg))
        table = tap_query(
            self.endpoint,
            "SELECT designation, ra, dec, j_m, j_msig, h_m, h_msig, k_m, k_msig FROM fp_psc WHERE "
            f"CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra_deg:.7f}, {dec_deg:.7f}, {r_deg:.6g})) = 1",
        )
        if len(table) == 0:
            return {}
        row = table[0]
        reference = f"2MASS J{row['designation']}"
        result: dict[str, Measurement] = {}
        for band, value, error in (("J", "j_m", "j_msig"), ("H", "h_m", "h_msig"), ("Ks", "k_m", "k_msig")):
            if masked(row[value]):
                continue
            sigma = None if masked(row[error]) else float(row[error]) * u.mag
            result[band] = Measurement(float(row[value]) * u.mag, sigma, self.name, reference)
        return result
