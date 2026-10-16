"""LUMOS: Lumos is a Universal and Modular Optical Simulator & Operating System."""

from importlib.metadata import PackageNotFoundError, version
from typing import Any

try:
    __version__: str = version("lumos-maxima")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled source tree
    __version__ = "0+unknown"

from lumos._config import config
from lumos.backend import jit
from lumos.backend.hardware import hardware_report
from lumos.core.context import Context


def load_archive(path: Any) -> Any:
    """Read a run archive back (see :func:`lumos.results.load_archive`)."""
    from lumos.results.reader import load_archive as _load

    return _load(path)


#: Subpackages imported on first access (``lumos.io``, ``lumos.plotting``...), to keep ``import lumos`` fast.
_LAZY_SUBMODULES = frozenset(
    {
        "backend",
        "catalogs",
        "core",
        "elements",
        "fields",
        "hardware",
        "io",
        "physics",
        "plotting",
        "processing",
        "propagation",
        "results",
        "scenes",
        "service",
        "symbolic",
        "units",
    }
)


def __getattr__(name: str) -> Any:
    """Import the subpackages of LUMOS on first access."""
    if name in _LAZY_SUBMODULES:
        import importlib

        return importlib.import_module(f"lumos.{name}")
    raise AttributeError(f"module 'lumos' has no attribute {name!r}")


__all__ = ["Context", "__version__", "config", "hardware_report", "jit", "load_archive"]
