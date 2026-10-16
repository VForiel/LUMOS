"""Global configuration of LUMOS, exposed as ``lumos.config``."""

from __future__ import annotations

import os
from typing import Any

__all__ = ["Config", "config"]


class Config:
    """Process-wide settings of LUMOS.

    Every setting is validated on assignment, and unknown settings are refused (to catch typos).

    Attributes
    ----------
    backend
        Default array backend: ``"auto"`` (chosen from the hardware, ADR-0004) or a backend name
        (``"numpy"``, ``"jax"``, ``"cupy"``, ``"torch"``).
    precision
        Floating-point precision of new arrays: ``"double"`` (float64 / complex128, default) or
        ``"single"`` (float32 / complex64).
    fft_workers
        Number of CPU threads used by FFTs on the NumPy backend; ``-1`` uses every core.
    archive_root
        Directory in which run archives are created (``None``: an ``archives`` directory next to the
        executed script or notebook, see :func:`lumos.results.default_archive_root`).
    """

    __slots__ = ("_archive_root", "_backend", "_fft_workers", "_precision")

    def __init__(self) -> None:
        self._backend = "auto"
        self._precision = "double"
        self._fft_workers = -1
        self._archive_root: str | None = None

    @property
    def archive_root(self) -> str | None:
        """Directory of the run archives (``None``: next to the executed script or notebook)."""
        return self._archive_root

    @archive_root.setter
    def archive_root(self, value: Any) -> None:
        if value is not None and not isinstance(value, str | os.PathLike):
            raise TypeError(f"archive_root must be a path or None, got {type(value).__name__}")
        self._archive_root = None if value is None else os.fspath(value)

    @property
    def precision(self) -> str:
        """Floating-point precision of new arrays: ``"double"`` or ``"single"``."""
        return self._precision

    @precision.setter
    def precision(self, value: str) -> None:
        if value not in ("double", "single"):
            raise ValueError(f"precision must be 'double' or 'single', got {value!r}")
        self._precision = value

    @property
    def backend(self) -> str:
        """Default array backend: ``"auto"`` or a backend name."""
        return self._backend

    @backend.setter
    def backend(self, value: str) -> None:
        if value != "auto":
            from lumos.backend._core import get  # deferred: the backend package reads this configuration

            get(value)  # raises for unknown or missing backends
        self._backend = value

    @property
    def fft_workers(self) -> int:
        """Number of CPU threads used by FFTs on the NumPy backend (``-1``: every core)."""
        return self._fft_workers

    @fft_workers.setter
    def fft_workers(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"fft_workers must be an integer, got {value!r}")
        if value == 0 or value < -1:
            raise ValueError(f"fft_workers must be -1 (every core) or a positive integer, got {value}")
        self._fft_workers = value

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse unknown settings."""
        if name not in Config.__slots__ and not isinstance(getattr(Config, name, None), property):
            raise AttributeError(f"lumos.config has no setting {name!r}")
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        """List the settings."""
        return (
            f"lumos.config(backend={self.backend!r}, precision={self.precision!r}, "
            f"fft_workers={self.fft_workers}, archive_root={self.archive_root!r})"
        )


config = Config()
