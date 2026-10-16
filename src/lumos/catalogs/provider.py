"""Catalogue providers and their registry (step P9.1, REQ-CAT-2, REQ-CAT-7).

A :class:`Provider` wraps one service (SIMBAD, Gaia, the NASA Exoplanet Archive...). Providers are registered
by name with :func:`register_provider`; third-party packages add theirs through the ``lumos.catalogs``
entry-point group (:func:`load_plugins`).
"""

from __future__ import annotations

import logging
from importlib import metadata
from typing import ClassVar

__all__ = ["Provider", "load_plugins", "providers", "register_provider"]

logger = logging.getLogger("lumos.catalogs")

PLUGIN_GROUP = "lumos.catalogs"


_REGISTRY: dict[str, type[Provider]] = {}


class Provider:
    """Base class of catalogue providers.

    Attributes
    ----------
    name
        Stable name of the provider (``"simbad"``, ``"gaia"``...).
    description
        One-line description.
    url
        Base URL of the service.
    reference
        Bibliographic reference to cite when using the data.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    url: ClassVar[str] = ""
    reference: ClassVar[str] = ""


def register_provider[P: type["Provider"]](cls: P) -> P:
    """Class decorator registering a provider under its ``name``."""
    name = getattr(cls, "name", "")
    if not name:
        raise ValueError(f"{cls.__name__} needs a non-empty name")
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(f"a provider named {name!r} is already registered ({existing.__qualname__})")
    _REGISTRY[name] = cls
    return cls


def providers() -> dict[str, type[Provider]]:
    """Return the registered providers by name."""
    return dict(_REGISTRY)


def load_plugins(group: str = PLUGIN_GROUP) -> list[str]:
    """Import the provider plugins declared in the entry-point ``group``; return the names loaded."""
    loaded = []
    for entry_point in metadata.entry_points(group=group):
        try:
            entry_point.load()
        except Exception as error:
            logger.warning("could not load catalogue plugin %r: %s", entry_point.name, error)
            continue
        loaded.append(entry_point.name)
    return loaded
