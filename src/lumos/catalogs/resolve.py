"""Name resolution: from a name to a typed record (step P9.8, REQ-CAT-1).

:func:`resolve` decides which catalogue answers:

1. names of major Solar-System bodies (``"Mars"``, ``"Jupiter"``...) → JPL Horizons, ephemeris now;
2. names ending with a planet letter (``"HD 209458 b"``) → NASA Exoplanet Archive, then exoplanet.eu;
3. anything else → SIMBAD: objects whose SIMBAD type is a star (types containing ``*``, except clusters and
   associations) become a :class:`~lumos.catalogs.records.StarRecord` with photometry and parameters, other
   objects (nebulae, galaxies, clusters) an :class:`~lumos.catalogs.records.ExtendedObjectRecord`.
"""

from __future__ import annotations

import re
from typing import Any

from astropy import units as u

from lumos.catalogs.http import CatalogError
from lumos.catalogs.records import ExtendedObjectRecord, Measurement, Record

__all__ = ["resolve"]

_PLANET = re.compile(r"\s[b-z]$")
#: SIMBAD object types containing "*" that are not single stars.
_NOT_STARS = {"Cl*", "GlC", "OpC", "As*", "St*", "MGr"}


def _is_star(object_type: str | None) -> bool:
    return bool(object_type) and "*" in str(object_type) and str(object_type) not in _NOT_STARS


def resolve(name: str) -> Record:
    """Return the typed record of the object called ``name`` (see the module documentation)."""
    from lumos.catalogs.exoplanets import ExoplanetEU, NasaExoplanetArchive
    from lumos.catalogs.simbad import Simbad
    from lumos.catalogs.solar_system import BODIES, Horizons

    cleaned = name.strip()
    if cleaned.lower() in BODIES:
        from astropy.time import Time

        now = Time.now()
        return Horizons().ephemeris(cleaned, start=now, stop=now + 1 * u.day, step="1 d")[0]
    if _PLANET.search(cleaned):
        try:
            return NasaExoplanetArchive().planet(cleaned)
        except CatalogError:
            try:
                return ExoplanetEU().planet(cleaned)
            except CatalogError:
                pass  # not a known planet: ask SIMBAD
    simbad = Simbad()
    basic: dict[str, Any] = simbad.basic(cleaned)
    if _is_star(basic["object_type"]):
        return simbad.star(cleaned)
    fields: dict[str, Any] = {}
    for key in ("ra", "dec"):
        if basic[key] is not None:
            fields[key] = Measurement(float(basic[key]) * u.deg, None, "simbad")
    if basic["object_type"]:
        fields["object_type"] = Measurement(str(basic["object_type"]), None, "simbad")
    return ExtendedObjectRecord(str(basic["main_id"]), identifiers=basic["identifiers"], **fields)
