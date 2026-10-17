"""Astronomical catalogues (phase 9, REQ-CAT-*).

- :mod:`~lumos.catalogs.http`: HTTPS requests with verified TLS, timeouts, a user cache and an offline mode;
- :mod:`~lumos.catalogs.provider`: the registry of catalogue providers (pluggable);
- :mod:`~lumos.catalogs.records`: typed records with provenance per field;
- providers: :mod:`~lumos.catalogs.simbad`, :mod:`~lumos.catalogs.gaia`, :mod:`~lumos.catalogs.exoplanets`,
  :mod:`~lumos.catalogs.vizier`, :mod:`~lumos.catalogs.calspec`, :mod:`~lumos.catalogs.archives`,
  :mod:`~lumos.catalogs.solar_system`, :mod:`~lumos.catalogs.svo`, :mod:`~lumos.catalogs.models`,
  :mod:`~lumos.catalogs.surveys`.

Importing this package registers the providers of LUMOS.
"""

from lumos.catalogs import (
    archives,
    calspec,
    exoplanets,
    gaia,
    http,
    models,
    provider,
    records,
    simbad,
    solar_system,
    spectra,
    surveys,
    svo,
    tap,
    vizier,
)
from lumos.catalogs.http import CatalogError, OfflineError, settings
from lumos.catalogs.provider import Provider, providers, register_provider

__all__ = [
    "CatalogError",
    "OfflineError",
    "Provider",
    "archives",
    "calspec",
    "exoplanets",
    "gaia",
    "http",
    "models",
    "provider",
    "providers",
    "records",
    "register_provider",
    "settings",
    "simbad",
    "solar_system",
    "spectra",
    "surveys",
    "svo",
    "tap",
    "vizier",
]
