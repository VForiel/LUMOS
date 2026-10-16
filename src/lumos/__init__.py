"""LUMOS: Lumos is a Universal and Modular Optical Simulator & Operating System."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("lumos-maxima")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled source tree
    __version__ = "0+unknown"

from lumos._config import config
from lumos.backend.hardware import hardware_report

__all__ = ["__version__", "config", "hardware_report"]
