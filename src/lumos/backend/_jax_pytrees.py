"""Registration of LUMOS containers as JAX pytrees, without importing JAX eagerly.

JAX can only trace, vectorise and differentiate containers registered in its pytree registry. LUMOS must
not import JAX when it is imported itself (NF-9), yet ``jax.grad(f)(element.values())`` must work as soon as
the user imports JAX. :func:`install` therefore registers the containers immediately if JAX is already
imported, and otherwise installs an import hook that registers them right after JAX is imported.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from collections.abc import Sequence
from types import ModuleType
from typing import Any

import numpy as np

__all__ = ["install", "register"]

_registered = False


def _is_leaf(value: Any) -> bool:
    """Numerical values are traced by JAX; settings (strings, integers, booleans) stay static."""
    if isinstance(value, bool | int | str) or value is None:
        return False
    return (
        isinstance(value, float | complex | np.ndarray | np.generic)
        or hasattr(value, "__jax_array__")
        or (type(value).__module__.split(".")[0] in ("jax", "jaxlib"))
    )


def register() -> None:
    """Register :class:`~lumos.core.element.Values` as a JAX pytree (idempotent)."""
    global _registered
    if _registered:
        return
    import jax

    from lumos.core.element import Values

    def flatten(values: Values) -> tuple[list[Any], tuple[Any, ...]]:
        data = dict(values.items())
        numeric = tuple(key for key, value in data.items() if _is_leaf(value))
        static = tuple((key, value) for key, value in data.items() if key not in numeric)
        return [data[key] for key in numeric], (tuple(data), numeric, static)

    def unflatten(aux: tuple[Any, ...], leaves: Sequence[Any]) -> Values:
        order, numeric, static = aux
        data = dict(static)
        data.update(zip(numeric, leaves, strict=True))
        return Values({key: data[key] for key in order})

    jax.tree_util.register_pytree_node(Values, flatten, unflatten)

    # Signals define _tree_flatten() -> (children, aux) and the classmethod _tree_unflatten(aux, children).
    from lumos.fields import pytree_classes

    for cls in pytree_classes():
        jax.tree_util.register_pytree_node(cls, cls._tree_flatten, cls._tree_unflatten)
    _registered = True


class _RegisteringLoader(importlib.abc.Loader):
    """Wrap the loader of the ``jax`` package to register LUMOS pytrees once JAX is imported."""

    def __init__(self, loader: importlib.abc.Loader) -> None:
        self._loader = loader

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        """Delegate to the original loader."""
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        """Import JAX, then register the LUMOS pytrees."""
        self._loader.exec_module(module)
        register()


class _JaxImportHook(importlib.abc.MetaPathFinder):
    def find_spec(
        self, fullname: str, path: Any, target: Any = None
    ) -> importlib.machinery.ModuleSpec | None:
        """Wrap the spec of the top-level ``jax`` package only."""
        if fullname != "jax" or _registered:
            return None
        # The hook stays installed until JAX is executed: importlib.util.find_spec("jax") (used to check
        # whether a backend is available) also calls this method without importing JAX.
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _RegisteringLoader(spec.loader)
        return spec


def install() -> None:
    """Register now if JAX is already imported, otherwise when it gets imported."""
    if "jax" in sys.modules:
        register()
    elif not any(isinstance(finder, _JaxImportHook) for finder in sys.meta_path):
        sys.meta_path.insert(0, _JaxImportHook())
