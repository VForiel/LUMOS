"""Solar System: JPL Horizons ephemerides and the reference solar spectrum (step P9.4, REQ-CAT-2).

- :class:`Horizons` queries the JPL Horizons API (Giorgini et al. 1996, BAAS 28, 1158) for observer tables:
  apparent position (ICRF), apparent diameter, heliocentric distance ``r``, distance to the observer ``Δ`` and
  phase angle (Sun-Target-Observer), one :class:`~lumos.catalogs.records.SolarSystemRecord` per epoch.
- :func:`solar_spectrum` returns the solar reference spectrum at 1 au of CALSPEC (``sun_reference_stis``,
  Bohlin, Dickinson & Calzetti 2001, AJ 122, 2118): HST/STIS and ground measurements from 0.12 to 0.96 µm
  extended by a Kurucz model to 2.7 µm. Its integral (1326 W m⁻²) is the total solar irradiance minus the
  2.5 % emitted beyond 2.7 µm.
"""

from __future__ import annotations

import csv
import io
from typing import Any, cast

from astropy import units as u
from astropy.time import Time

from lumos.catalogs.calspec import Calspec
from lumos.catalogs.http import CatalogError, fetch
from lumos.catalogs.provider import Provider, register_provider
from lumos.catalogs.records import Measurement, SolarSystemRecord
from lumos.catalogs.spectra import Spectrum

__all__ = ["BODIES", "Horizons", "solar_spectrum"]

#: Horizons identifiers of the major bodies.
BODIES: dict[str, str] = {
    "sun": "10",
    "mercury": "199",
    "venus": "299",
    "earth": "399",
    "moon": "301",
    "mars": "499",
    "jupiter": "599",
    "saturn": "699",
    "uranus": "799",
    "neptune": "899",
    "pluto": "999",
}

_COLUMNS = {
    "R.A._(ICRF)": ("ra", u.deg),
    "DEC_(ICRF)": ("dec", u.deg),
    "Ang-diam": ("angular_diameter", u.arcsec),
    "r": ("heliocentric_distance", u.au),
    "delta": ("delta", u.au),
    "S-T-O": ("phase_angle", u.deg),
}


def _date(time: Time) -> str:
    """Return ``YYYY-MM-DD HH:MM`` in UTC."""
    return str(cast(Any, time).utc.strftime("%Y-%m-%d %H:%M"))


@register_provider
class Horizons(Provider):
    """JPL Horizons ephemerides of Solar-System bodies."""

    name = "jpl_horizons"
    description = "JPL Horizons ephemerides (NASA JPL Solar System Dynamics)"
    url = "https://ssd.jpl.nasa.gov/api/horizons.api"
    reference = "Giorgini et al. 1996, BAAS 28, 1158"

    def __init__(self, url: str | None = None) -> None:
        self.endpoint = url or type(self).url

    @staticmethod
    def command(body: str) -> str:
        """Return the Horizons command of ``body``: the identifier of a major body, else ``body`` itself."""
        return BODIES.get(body.strip().lower(), body.strip())

    def ephemeris(
        self,
        body: str,
        *,
        start: Time,
        stop: Time,
        step: str = "1 d",
        observer: str = "500@399",
    ) -> list[SolarSystemRecord]:
        """Return the apparent positions and distances of ``body`` seen from ``observer`` (geocentre)."""
        params = {
            "format": "json",
            "COMMAND": f"'{self.command(body)}'",
            "OBJ_DATA": "'YES'",
            "MAKE_EPHEM": "'YES'",
            "EPHEM_TYPE": "'OBSERVER'",
            "CENTER": f"'{observer}'",
            "START_TIME": f"'{_date(start)}'",
            "STOP_TIME": f"'{_date(stop)}'",
            "STEP_SIZE": f"'{step}'",
            "QUANTITIES": "'1,13,19,20,24'",
            "ANG_FORMAT": "'DEG'",
            "CSV_FORMAT": "'YES'",
        }
        answer: Any = fetch(self.endpoint, params=params).json()
        if "error" in answer:
            raise CatalogError(f"JPL Horizons: {answer['error']}")
        return self._parse(body, answer["result"])

    def _parse(self, body: str, text: str) -> list[SolarSystemRecord]:
        if "$$SOE" not in text:
            raise CatalogError(f"JPL Horizons: no ephemeris for {body!r}: {text[:300]}")
        header_block, _, rest = text.partition("$$SOE")
        rows_text = rest.partition("$$EOE")[0]
        header_lines = [line for line in header_block.splitlines() if line.strip().startswith("Date")]
        if not header_lines:
            raise CatalogError("JPL Horizons: unexpected table header")
        header = [cell.strip() for cell in next(csv.reader([header_lines[-1]]))]
        records = []
        for row in csv.reader(io.StringIO(rows_text.strip())):
            if not row:
                continue
            cells = dict(zip(header, (cell.strip() for cell in row), strict=False))
            epoch = Time.strptime(cells[header[0]], "%Y-%b-%d %H:%M", scale="utc")
            fields: dict[str, Any] = {
                "epoch": Measurement(float(cast(Any, epoch).jd) * u.day, None, self.name, "Julian date (UT)")
            }
            for column, (key, unit) in _COLUMNS.items():
                text_value = cells.get(column, "")
                try:
                    value = float(text_value)
                except ValueError:
                    continue
                fields[key] = Measurement(value * unit, None, self.name, "JPL Horizons")
            records.append(SolarSystemRecord(body, identifiers=[self.command(body)], **fields))
        return records


def solar_spectrum() -> Spectrum:
    """Return the CALSPEC solar reference spectrum at 1 au (F_λ in erg s⁻¹ cm⁻² Å⁻¹)."""
    spectrum = Calspec().spectrum("sun_reference")
    spectrum.name = "Sun"
    return spectrum
