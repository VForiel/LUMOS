"""LUMOS: Lumos is a Universal and Modular Optical Simulator & Operating System."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("lumos-maxima")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled source tree
    __version__ = "0+unknown"

__all__ = ["__version__"]
