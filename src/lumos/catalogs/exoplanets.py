"""Exoplanet catalogues: the NASA Exoplanet Archive and the Encyclopaedia of exoplanetary systems.

- :class:`NasaExoplanetArchive` (Akeson et al. 2013, PASP 125, 989) reads the *Planetary Systems Composite
  Parameters* table (``pscomppars``), one row per planet with a consistent set of parameters, through TAP.
- :class:`ExoplanetEU` (Schneider et al. 2011, A&A 532, A79) reads the catalogue of exoplanet.eu, downloaded
  as CSV (cached like every catalogue request).
"""

from __future__ import annotations

import csv
import io
from typing import Any

from astropy import units as u

from lumos.catalogs.http import CatalogError, fetch
from lumos.catalogs.provider import Provider, register_provider
from lumos.catalogs.records import ExoplanetRecord, Measurement
from lumos.catalogs.tap import adql_string, masked, tap_query

__all__ = ["ExoplanetEU", "NasaExoplanetArchive"]

_NASA_COLUMNS = (
    "pl_name, hostname, pl_orbper, pl_orbpererr1, pl_orbsmax, pl_orbeccen, pl_orbincl, pl_orblper, "
    "pl_rade, pl_radeerr1, pl_bmasse, pl_bmasseerr1, pl_eqt, discoverymethod, sy_dist, st_teff, st_rad, "
    "st_mass, ra, dec"
)
_NASA_FIELDS: tuple[tuple[str, str, Any, str | None], ...] = (
    ("orbital_period", "pl_orbper", u.day, "pl_orbpererr1"),
    ("semi_major_axis", "pl_orbsmax", u.au, None),
    ("eccentricity", "pl_orbeccen", u.dimensionless_unscaled, None),
    ("inclination", "pl_orbincl", u.deg, None),
    ("argument_periastron", "pl_orblper", u.deg, None),
    ("radius", "pl_rade", u.R_earth, "pl_radeerr1"),
    ("mass", "pl_bmasse", u.M_earth, "pl_bmasseerr1"),
    ("equilibrium_temperature", "pl_eqt", u.K, None),
    ("host_distance", "sy_dist", u.pc, None),
    ("host_teff", "st_teff", u.K, None),
    ("host_radius", "st_rad", u.R_sun, None),
    ("host_mass", "st_mass", u.M_sun, None),
    ("ra", "ra", u.deg, None),
    ("dec", "dec", u.deg, None),
)


@register_provider
class NasaExoplanetArchive(Provider):
    """NASA Exoplanet Archive (Caltech/IPAC), Planetary Systems Composite Parameters."""

    name = "nasa_exoplanet_archive"
    description = "NASA Exoplanet Archive, composite planetary parameters"
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    reference = "Akeson et al. 2013, PASP 125, 989 (table pscomppars)"

    def __init__(self, url: str | None = None) -> None:
        self.endpoint = url or type(self).url

    def _records(self, where: str) -> list[ExoplanetRecord]:
        table = tap_query(self.endpoint, f"SELECT {_NASA_COLUMNS} FROM pscomppars WHERE {where}")
        records = []
        for row in table:
            fields: dict[str, Any] = {}
            for key, column, unit, error in _NASA_FIELDS:
                if masked(row[column]):
                    continue
                uncertainty = None if error is None or masked(row[error]) else abs(float(row[error])) * unit
                fields[key] = Measurement(float(row[column]) * unit, uncertainty, self.name, "pscomppars")
            if not masked(row["discoverymethod"]):
                fields["discovery_method"] = Measurement(str(row["discoverymethod"]), None, self.name)
            records.append(ExoplanetRecord(str(row["pl_name"]), host=str(row["hostname"]), **fields))
        return records

    def planet(self, name: str) -> ExoplanetRecord:
        """Return the record of the planet ``name`` (e.g. ``"HD 209458 b"``)."""
        records = self._records(f"pl_name = {adql_string(name)}")
        if not records:
            raise CatalogError(f"NASA Exoplanet Archive: planet {name!r} not found")
        return records[0]

    def system(self, host: str) -> list[ExoplanetRecord]:
        """Return the planets of the star ``host``, sorted by name."""
        records = self._records(f"hostname = {adql_string(host)}")
        return sorted(records, key=lambda record: record.name)


_EU_FIELDS: tuple[tuple[str, str, Any, str | None], ...] = (
    ("orbital_period", "orbital_period", u.day, "orbital_period_error_max"),
    ("semi_major_axis", "semi_major_axis", u.au, None),
    ("eccentricity", "eccentricity", u.dimensionless_unscaled, None),
    ("inclination", "inclination", u.deg, None),
    ("argument_periastron", "omega", u.deg, None),
    ("radius", "radius", u.R_jup, "radius_error_max"),
    ("mass", "mass", u.M_jup, "mass_error_max"),
    ("equilibrium_temperature", "temp_calculated", u.K, None),
    ("host_distance", "star_distance", u.pc, None),
    ("host_teff", "star_teff", u.K, None),
    ("host_radius", "star_radius", u.R_sun, None),
    ("host_mass", "star_mass", u.M_sun, None),
    ("ra", "ra", u.deg, None),
    ("dec", "dec", u.deg, None),
)


def _number(text: str | None) -> float | None:
    if text is None or not text.strip():
        return None
    try:
        return float(text)
    except ValueError:
        return None


@register_provider
class ExoplanetEU(Provider):
    """The Encyclopaedia of exoplanetary systems (exoplanet.eu), whole catalogue as CSV."""

    name = "exoplanet_eu"
    description = "The Encyclopaedia of exoplanetary systems (exoplanet.eu)"
    url = "https://exoplanet.eu/catalog/csv/"
    reference = "Schneider et al. 2011, A&A 532, A79"

    def __init__(self, url: str | None = None) -> None:
        self.endpoint = url or type(self).url

    def rows(self) -> list[dict[str, str]]:
        """Return the rows of the catalogue (downloaded once, then cached)."""
        response = fetch(self.endpoint)
        return list(csv.DictReader(io.StringIO(response.text)))

    @staticmethod
    def _record(row: dict[str, str]) -> ExoplanetRecord:
        fields: dict[str, Any] = {}
        for key, column, unit, error in _EU_FIELDS:
            value = _number(row.get(column))
            if value is None:
                continue
            sigma = _number(row.get(error)) if error else None
            uncertainty = None if sigma is None else abs(sigma) * unit
            fields[key] = Measurement(
                value * unit, uncertainty, "exoplanet_eu", row.get("publication", "") or ""
            )
        method = (row.get("detection_type") or "").strip()
        if method:
            fields["discovery_method"] = Measurement(method, None, "exoplanet_eu")
        return ExoplanetRecord(row["name"], host=(row.get("star_name") or "").strip(), **fields)

    def planet(self, name: str) -> ExoplanetRecord:
        """Return the record of the planet ``name``."""
        for row in self.rows():
            if row.get("name", "").strip() == name:
                return self._record(row)
        raise CatalogError(f"exoplanet.eu: planet {name!r} not found")

    def system(self, host: str) -> list[ExoplanetRecord]:
        """Return the planets of the star ``host``, sorted by name."""
        found = [self._record(row) for row in self.rows() if (row.get("star_name") or "").strip() == host]
        return sorted(found, key=lambda record: record.name)
