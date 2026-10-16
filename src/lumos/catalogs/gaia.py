"""Gaia Data Release 3 (Gaia Collaboration 2023, A&A 674, A1), queried through the ESA Gaia archive TAP."""

from __future__ import annotations

from typing import Any

from astropy import units as u

from lumos.catalogs.http import CatalogError
from lumos.catalogs.provider import Provider, register_provider
from lumos.catalogs.records import Measurement, StarRecord
from lumos.catalogs.tap import masked, tap_query

__all__ = ["GaiaDR3"]

SOURCE = "gaia_dr3"
REFERENCE = "Gaia Collaboration 2023, A&A 674, A1"


@register_provider
class GaiaDR3(Provider):
    """Gaia DR3: astrometry, photometry (G, BP, RP), radial velocities and astrophysical parameters."""

    name = "gaia_dr3"
    description = "Gaia Data Release 3 (ESA)"
    url = "https://gea.esac.esa.int/tap-server/tap/sync"
    reference = REFERENCE

    def __init__(self, url: str | None = None) -> None:
        self.endpoint = url or type(self).url

    def source(self, source_id: int) -> StarRecord:
        """Return the record of the Gaia DR3 source ``source_id``."""
        identifier = int(source_id)
        table = tap_query(
            self.endpoint,
            "SELECT source_id, ra, dec, parallax, parallax_error, pmra, pmdec, phot_g_mean_mag, "
            "phot_bp_mean_mag, phot_rp_mean_mag, teff_gspphot, radial_velocity FROM gaiadr3.gaia_source "
            f"WHERE source_id = {identifier}",
        )
        if len(table) == 0:
            raise CatalogError(f"Gaia DR3: source {identifier} not found")
        row = table[0]
        fields: dict[str, Measurement] = {}

        def put(key: str, column: str, unit: Any, error: str | None = None) -> None:
            if masked(row[column]):
                return
            uncertainty = None if error is None or masked(row[error]) else float(row[error]) * unit
            fields[key] = Measurement(float(row[column]) * unit, uncertainty, SOURCE, REFERENCE)

        put("ra", "ra", u.deg)
        put("dec", "dec", u.deg)
        put("parallax", "parallax", u.mas, "parallax_error")
        put("pm_ra", "pmra", u.mas / u.yr)
        put("pm_dec", "pmdec", u.mas / u.yr)
        put("mag_Gaia_G", "phot_g_mean_mag", u.mag)
        put("mag_Gaia_BP", "phot_bp_mean_mag", u.mag)
        put("mag_Gaia_RP", "phot_rp_mean_mag", u.mag)
        put("teff", "teff_gspphot", u.K)
        put("radial_velocity", "radial_velocity", u.km / u.s)
        return StarRecord(f"Gaia DR3 {identifier}", identifiers=[f"Gaia DR3 {identifier}"], **fields)
