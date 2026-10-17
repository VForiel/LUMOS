"""Registry of element types (specs §10, NF-7).

Every element class that can be saved to or loaded from a ``.lumos`` file is registered under a **stable
identifier** and a **type version**::

    @register("lumos.optics.ThinLens", version=1)
    class ThinLens(Element): ...

Loading a file only instantiates registered classes: the identifier is looked up in the registry, never
imported. Identifiers starting with ``lumos.`` are reserved for classes defined in the ``lumos`` package;
third-party packages use their own prefix and declare a module to import in the ``lumos.elements``
entry-point group, loaded by :meth:`Registry.load_plugins`.

The type version is incremented when the serialised form of an element changes; migrations from older
versions are registered with the ``.lumos`` file format (step P1.6).
"""

from __future__ import annotations

import difflib
import logging
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from importlib import metadata
from typing import Any, TypeVar

from lumos.core.element import Element
from lumos.errors import FormatVersionError

__all__ = ["Registry", "UnknownTypeError", "register", "registry"]

logger = logging.getLogger(__name__)

_TYPE_ID = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*\.[A-Za-z][A-Za-z0-9_]*$")
_RESERVED_PREFIX = "lumos."
PLUGIN_GROUP = "lumos.elements"

E = TypeVar("E", bound=type[Element])
Migration = Callable[[dict[str, Any]], dict[str, Any]]


class UnknownTypeError(LookupError):
    """Raised when an element type identifier or class is not registered."""


@dataclass(frozen=True)
class RegistryEntry:
    """A registered element type."""

    type_id: str
    cls: type[Element]
    version: int


class Registry:
    """Mapping between stable type identifiers and element classes."""

    def __init__(self) -> None:
        self._by_id: dict[str, RegistryEntry] = {}
        self._by_class: dict[type[Element], RegistryEntry] = {}
        self._migrations: dict[tuple[str, int], Migration] = {}

    def register(self, type_id: str, version: int = 1) -> Callable[[E], E]:
        """Return a class decorator registering an element class under ``type_id``.

        Parameters
        ----------
        type_id
            Dotted identifier: lowercase package segments followed by a class name, for example
            ``"mylab.optics.ThinLens"``. The ``lumos.`` prefix is reserved for classes of LUMOS.
        version
            Version of the serialised form (an integer >= 1).
        """
        if not isinstance(type_id, str) or _TYPE_ID.match(type_id) is None:
            raise ValueError(
                f"invalid type identifier {type_id!r}: expected lowercase dotted segments followed by a "
                "class name, e.g. 'mylab.optics.ThinLens'"
            )
        if isinstance(version, bool) or not isinstance(version, int):
            raise TypeError(f"type version must be an integer, got {version!r}")
        if version < 1:
            raise ValueError(f"type version must be >= 1, got {version}")

        def decorator(cls: E) -> E:
            self._add(type_id, cls, version)
            return cls

        return decorator

    def _add(self, type_id: str, cls: type, version: int) -> None:
        if not (isinstance(cls, type) and issubclass(cls, Element)):
            raise TypeError(f"only Element subclasses can be registered, got {cls!r}")
        module = cls.__module__
        if type_id.startswith(_RESERVED_PREFIX) and not (module == "lumos" or module.startswith("lumos.")):
            raise ValueError(
                f"{type_id!r}: the 'lumos.' prefix is reserved for classes of the lumos package "
                f"({cls.__name__} is defined in {module!r})"
            )
        existing = self._by_id.get(type_id)
        if existing is not None:
            if existing.cls is cls and existing.version == version:
                return  # idempotent re-registration (e.g. module imported twice)
            raise ValueError(f"{type_id!r} is already registered for {existing.cls.__qualname__}")
        previous = self._by_class.get(cls)
        if previous is not None:
            raise ValueError(f"{cls.__qualname__} is already registered as {previous.type_id!r}")
        entry = RegistryEntry(type_id, cls, version)
        self._by_id[type_id] = entry
        self._by_class[cls] = entry

    def unregister(self, type_id: str) -> None:
        """Remove ``type_id`` (used to redefine a script block, e.g. when a notebook cell runs again)."""
        entry = self._entry(type_id)
        del self._by_id[type_id]
        self._by_class.pop(entry.cls, None)

    def _entry(self, type_id: str) -> RegistryEntry:
        try:
            return self._by_id[type_id]
        except KeyError:
            close = difflib.get_close_matches(type_id, list(self._by_id), n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            raise UnknownTypeError(f"unknown element type {type_id!r}{hint}") from None

    def get(self, type_id: str) -> type[Element]:
        """Return the class registered under ``type_id``."""
        return self._entry(type_id).cls

    def version(self, type_id: str) -> int:
        """Return the current type version of ``type_id``."""
        return self._entry(type_id).version

    def type_id(self, cls: type[Element]) -> str:
        """Return the identifier under which ``cls`` is registered."""
        try:
            return self._by_class[cls].type_id
        except KeyError:
            raise UnknownTypeError(f"{cls.__qualname__} is not registered") from None

    def add_migration(self, type_id: str, from_version: int) -> Callable[[Migration], Migration]:
        """Return a decorator registering the migration of ``type_id`` from ``from_version`` to the next one.

        A migration receives the serialised body of an element (``name``, ``params``, ``settings``,
        ``state``) written with type version ``from_version`` and returns the body for version
        ``from_version + 1``. Loading an old document applies the migrations one version at a time.
        """
        current = self.version(type_id)
        if isinstance(from_version, bool) or not isinstance(from_version, int):
            raise TypeError(f"from_version must be an integer, got {from_version!r}")
        if not 1 <= from_version < current:
            raise ValueError(
                f"{type_id!r} is at version {current}: migrations start from versions 1 to {current - 1}"
            )
        if (type_id, from_version) in self._migrations:
            raise ValueError(f"a migration of {type_id!r} from version {from_version} already exists")

        def decorator(function: Migration) -> Migration:
            self._migrations[(type_id, from_version)] = function
            return function

        return decorator

    def migrate(self, type_id: str, body: dict[str, Any], from_version: int) -> dict[str, Any]:
        """Upgrade ``body`` from ``from_version`` to the current version of ``type_id``."""
        current = self.version(type_id)
        if from_version > current:
            raise FormatVersionError(
                f"{type_id!r} version {from_version} is newer than the supported version {current}; "
                "upgrade LUMOS to read this document"
            )
        for version in range(from_version, current):
            migration = self._migrations.get((type_id, version))
            if migration is None:
                raise FormatVersionError(
                    f"no migration of {type_id!r} from version {version} to {version + 1}"
                )
            body = migration(body)
        return body

    def __contains__(self, type_id: object) -> bool:
        """Whether ``type_id`` is registered."""
        return type_id in self._by_id

    def __iter__(self) -> Iterator[str]:
        """Iterate over registered identifiers, in registration order."""
        return iter(self._by_id)

    def __len__(self) -> int:
        """Return the number of registered types."""
        return len(self._by_id)

    def load_plugins(self, group: str = PLUGIN_GROUP) -> list[str]:
        """Load the plugins declared in the entry-point ``group`` and return the names of those loaded.

        A plugin entry point designates a module (or object) whose import registers element types. A plugin
        that fails to load is logged as a warning and skipped, so that one broken plugin does not prevent
        LUMOS from working.
        """
        loaded = []
        for entry_point in metadata.entry_points(group=group):
            try:
                entry_point.load()
            except Exception as error:
                logger.warning(
                    "could not load LUMOS plugin %r (%s): %s", entry_point.name, entry_point.value, error
                )
                continue
            loaded.append(entry_point.name)
        return loaded


registry = Registry()
"""The global registry used by LUMOS."""

register = registry.register
"""Register an element class in the global registry (see :meth:`Registry.register`)."""
