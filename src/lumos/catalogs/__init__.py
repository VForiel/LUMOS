"""Astronomical catalogues (phase 9, REQ-CAT-*).

- :mod:`~lumos.catalogs.http`: HTTPS requests with verified TLS, timeouts, a user cache and an offline mode;
- :mod:`~lumos.catalogs.provider`: the registry of catalogue providers (pluggable).
"""

from lumos.catalogs import http, provider
from lumos.catalogs.http import CatalogError, OfflineError, settings
from lumos.catalogs.provider import Provider, providers, register_provider

__all__ = [
    "CatalogError",
    "OfflineError",
    "Provider",
    "http",
    "provider",
    "providers",
    "register_provider",
    "settings",
]
