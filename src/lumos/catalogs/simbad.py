"""SIMBAD astronomical database (CDS, Wenger et al. 2000, A&AS 143, 9), queried through its TAP service.

:meth:`Simbad.star` returns a :class:`~lumos.catalogs.records.StarRecord`: position, parallax, proper motion,
radial velocity and spectral type with their bibliographic references (table ``basic``), magnitudes (table
``flux``) and the most recent effective temperature, surface gravity and metallicity (table ``mesFe_h``, the
latest bibcode first). :meth:`Simbad.basic` returns the main fields of any object (type, position,
identifiers) and is used to resolve names (:func:`lumos.catalogs.resolve`).
"""

from __future__ import annotations

from typing import Any

from astropy import units as u

from lumos.catalogs.http import CatalogError
from lumos.catalogs.provider import Provider, register_provider
from lumos.catalogs.records import Measurement, StarRecord
from lumos.catalogs.tap import adql_string, masked, tap_query

__all__ = ["Simbad"]

SOURCE = "simbad"


def _value(row: Any, column: str) -> Any:
    value = row[column]
    return None if masked(value) else value


@register_provider
class Simbad(Provider):
    """SIMBAD (CDS): identifiers, positions, motions, spectral types, magnitudes, stellar parameters."""

    name = "simbad"
    description = "SIMBAD astronomical database (CDS)"
    url = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"
    reference = "Wenger et al. 2000, A&AS 143, 9"

    def __init__(self, url: str | None = None) -> None:
        self.endpoint = url or type(self).url

    def basic(self, name: str) -> dict[str, Any]:
        """Return the main fields of the object called ``name`` (table ``basic`` and its identifiers)."""
        query = (
            "SELECT b.main_id, b.otype, b.ra, b.dec, b.plx_value, b.plx_err, b.plx_bibcode, b.pmra, b.pmdec, "
            "b.pm_bibcode, b.rvz_radvel, b.rvz_err, b.rvz_bibcode, b.sp_type, b.sp_bibcode, i.ids "
            "FROM basic AS b JOIN ident AS n ON n.oidref = b.oid JOIN ids AS i ON i.oidref = b.oid "
            f"WHERE n.id = {adql_string(name)}"
        )
        table = tap_query(self.endpoint, query)
        if len(table) == 0:
            raise CatalogError(f"SIMBAD: object {name!r} not found")
        row = table[0]
        result = {column: _value(row, column) for column in table.colnames}
        result["object_type"] = result.pop("otype")
        ids = result.pop("ids") or ""
        result["identifiers"] = [
            identifier.strip() for identifier in str(ids).split("|") if identifier.strip()
        ]
        return result

    def star(self, name: str) -> StarRecord:
        """Return the :class:`~lumos.catalogs.records.StarRecord` of the star ``name``."""
        basic = self.basic(name)
        fields: dict[str, Measurement] = {}

        def put(key: str, value: Any, unit: Any, error: Any = None, reference: Any = None) -> None:
            if value is None:
                return
            uncertainty = None if error is None else float(error) * unit
            fields[key] = Measurement(float(value) * unit, uncertainty, SOURCE, reference or "")

        put("ra", basic["ra"], u.deg)
        put("dec", basic["dec"], u.deg)
        put("parallax", basic["plx_value"], u.mas, basic["plx_err"], basic["plx_bibcode"])
        put("pm_ra", basic["pmra"], u.mas / u.yr, reference=basic["pm_bibcode"])
        put("pm_dec", basic["pmdec"], u.mas / u.yr, reference=basic["pm_bibcode"])
        put("radial_velocity", basic["rvz_radvel"], u.km / u.s, basic["rvz_err"], basic["rvz_bibcode"])
        if basic["sp_type"]:
            fields["spectral_type"] = Measurement(
                str(basic["sp_type"]), None, SOURCE, basic["sp_bibcode"] or ""
            )
        if basic["object_type"]:
            fields["object_type"] = Measurement(str(basic["object_type"]), None, SOURCE)

        fluxes = tap_query(
            self.endpoint,
            "SELECT f.filter, f.flux, f.flux_err, f.bibcode FROM flux AS f JOIN ident AS n "
            f"ON n.oidref = f.oidref WHERE n.id = {adql_string(name)}",
        )
        for row in fluxes:
            band = str(row["filter"]).strip()
            if masked(row["flux"]) or not band.replace("_", "").isalnum():
                continue
            error = None if masked(row["flux_err"]) else float(row["flux_err"]) * u.mag
            reference = "" if masked(row["bibcode"]) else str(row["bibcode"])
            fields[f"mag_{band}"] = Measurement(float(row["flux"]) * u.mag, error, SOURCE, reference)

        parameters = tap_query(
            self.endpoint,
            "SELECT m.teff, m.log_g, m.fe_h, m.bibcode FROM mesFe_h AS m "
            "JOIN ident AS n ON n.oidref = m.oidref "
            f"WHERE n.id = {adql_string(name)} AND m.teff IS NOT NULL ORDER BY bibcode DESC",
        )
        if len(parameters):
            row = parameters[0]
            reference = "" if masked(row["bibcode"]) else str(row["bibcode"])
            fields["teff"] = Measurement(float(row["teff"]) * u.K, None, SOURCE, reference)
            if not masked(row["log_g"]):
                fields["logg"] = Measurement(
                    float(row["log_g"]) * u.dex(u.cm / u.s**2), None, SOURCE, reference
                )
            if not masked(row["fe_h"]):
                fields["metallicity"] = Measurement(float(row["fe_h"]) * u.dex, None, SOURCE, reference)

        return StarRecord(str(basic["main_id"]), identifiers=basic["identifiers"], **fields)
