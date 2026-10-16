"""Backend objects, lazy loading and the current backend."""

from __future__ import annotations

import importlib.util
import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lumos._config import config

__all__ = [
    "Backend",
    "BackendUnavailableError",
    "available",
    "backend_of",
    "convert",
    "current",
    "get",
    "get_default",
    "known",
    "operand",
    "set_default",
    "to_numpy",
    "use",
]

_KNOWN = ("numpy", "jax", "cupy", "torch")


class BackendUnavailableError(ImportError):
    """Raised when a known backend is requested but its library is not installed."""


@dataclass(frozen=True)
class Backend:
    """An array backend.

    Attributes
    ----------
    name
        Backend name (``"numpy"``, ``"jax"``, ``"cupy"`` or ``"torch"``).
    xp
        Array API namespace of the backend (use it like ``numpy``: ``xp.exp``, ``xp.sum``...).
    """

    name: str
    xp: Any
    _is_array: Callable[[Any], bool] = field(repr=False)
    _to_numpy: Callable[[Any], np.ndarray] = field(repr=False)

    def asarray(self, obj: Any, dtype: Any = None) -> Any:
        """Create a backend array from ``obj`` (keeping its dtype unless ``dtype`` is given)."""
        return self.xp.asarray(obj, dtype=dtype)

    def is_array(self, obj: Any) -> bool:
        """Whether ``obj`` is an array of this backend."""
        return self._is_array(obj)

    def to_numpy(self, array: Any) -> np.ndarray:
        """Copy an array of this backend to a NumPy array (on the host)."""
        return self._to_numpy(array)


# --- Loaders (imports happen here, only when a backend is first requested) -----------------------


def _load_numpy() -> Backend:
    import array_api_compat.numpy as xp

    return Backend("numpy", xp, lambda a: isinstance(a, np.ndarray), np.asarray)


def prepare_jax() -> None:
    """Let JAX share the GPU: no preallocation of 75 % of its memory unless the user chose otherwise.

    JAX reads ``XLA_PYTHON_CLIENT_PREALLOCATE`` when it creates its GPU client, at the first use of a device,
    so LUMOS sets it to ``false`` (if unset) before it loads JAX or lists its devices (ADR-0008). With
    preallocation, cuFFT and cuBLAS run out of memory on GPUs shared with the display or with CuPy.
    """
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _load_jax() -> Backend:
    prepare_jax()
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)  # LUMOS works in double precision by default
    return Backend("jax", jnp, lambda a: isinstance(a, jax.Array), np.asarray)


def _load_cupy() -> Backend:
    import array_api_compat.cupy as xp
    import cupy  # pyright: ignore[reportMissingImports]  (optional backend)

    return Backend("cupy", xp, lambda a: isinstance(a, cupy.ndarray), cupy.asnumpy)


def _load_torch() -> Backend:
    import array_api_compat.torch as xp
    import torch  # pyright: ignore[reportMissingImports]  (optional backend)

    return Backend("torch", xp, torch.is_tensor, lambda a: a.detach().cpu().numpy())


_LOADERS: dict[str, Callable[[], Backend]] = {
    "numpy": _load_numpy,
    "jax": _load_jax,
    "cupy": _load_cupy,
    "torch": _load_torch,
}
_loaded: dict[str, Backend] = {}


def known() -> tuple[str, ...]:
    """Return the names of the backends LUMOS knows about."""
    return _KNOWN


def available() -> tuple[str, ...]:
    """Return the names of the installed backends, ``"numpy"`` first (nothing is imported)."""
    return tuple(name for name in _KNOWN if name == "numpy" or importlib.util.find_spec(name) is not None)


def _check_name(name: str) -> None:
    if name not in _KNOWN:
        raise ValueError(f"unknown backend {name!r}; known backends: {', '.join(_KNOWN)}")


def get(name: str | None = None) -> Backend:
    """Return the backend ``name`` (the current backend if ``name`` is ``None``).

    Raises
    ------
    ValueError
        If the name is unknown.
    BackendUnavailableError
        If the backend library is not installed.
    """
    if name is None:
        return current()
    _check_name(name)
    if name not in _loaded:
        if name != "numpy" and importlib.util.find_spec(name) is None:
            raise BackendUnavailableError(
                f"the {name!r} backend needs the {name} package: pip install 'lumos-maxima[{_EXTRAS[name]}]'"
            )
        _loaded[name] = _LOADERS[name]()
    return _loaded[name]


_EXTRAS = {"jax": "jax", "cupy": "gpu", "torch": "torch"}

# --- Current and default backends -------------------------------------------------------------------

logger = logging.getLogger(__name__)
_current: ContextVar[str | None] = ContextVar("lumos_backend", default=None)
_auto_choice: str | None = None


def get_default() -> str:
    """Return the default backend setting: ``"auto"`` or a backend name (same as ``lumos.config.backend``)."""
    return config.backend


def set_default(name: str) -> None:
    """Set the default backend for the whole process (``"auto"`` or a backend name)."""
    config.backend = name


def _resolve_default() -> str:
    """Return the default backend name, chosen from the hardware when the setting is ``"auto"``."""
    global _auto_choice
    if config.backend != "auto":
        return config.backend
    if _auto_choice is None:
        from lumos.backend.hardware import hardware_report, select_backend  # deferred: probes the GPUs

        _auto_choice = select_backend(hardware_report(probe_gpus=True))
        logger.info("automatically selected the %r backend", _auto_choice)
    return _auto_choice


def current() -> Backend:
    """Return the backend active in the current thread or task."""
    return get(_current.get() or _resolve_default())


@contextmanager
def use(name: str) -> Iterator[Backend]:
    """Use backend ``name`` inside a ``with`` block (local to the current thread or task)."""
    selected = get(name)
    token = _current.set(name)
    try:
        yield selected
    finally:
        _current.reset(token)


# --- Conversions ------------------------------------------------------------------------------------------


def backend_of(array: Any) -> str:
    """Return the name of the backend owning ``array``.

    Raises
    ------
    TypeError
        If ``array`` is not an array of a known, installed backend.
    """
    if isinstance(array, np.ndarray):
        return "numpy"
    root = type(array).__module__.split(".")[0]
    if root in ("jax", "jaxlib") and "jax" in available():
        root = "jax"
    if root in _KNOWN and root != "numpy" and get(root).is_array(array):
        return root
    raise TypeError(f"{type(array).__name__} is not an array of a known backend")


def operand(value: Any, name: str) -> Any:
    """Return ``value`` ready to combine with arrays of backend ``name``.

    NumPy arrays (masks, transmission maps, factors) are converted to the backend; scalars and arrays of the
    backend are returned unchanged. CuPy, unlike JAX and PyTorch, refuses to mix its arrays with NumPy arrays.
    """
    if isinstance(value, np.ndarray) and name != "numpy":
        return get(name).asarray(value)
    return value


def to_numpy(array: Any) -> np.ndarray:
    """Return ``array`` as a NumPy array (a copy on the host for device arrays)."""
    if isinstance(array, np.ndarray):
        return array
    try:
        name = backend_of(array)
    except TypeError:
        return np.asarray(array)
    return get(name).to_numpy(array)


def convert(array: Any, name: str) -> Any:
    """Convert ``array`` to an array of backend ``name``, keeping its dtype."""
    target = get(name)
    if target.is_array(array):
        return array
    return target.asarray(to_numpy(array))
