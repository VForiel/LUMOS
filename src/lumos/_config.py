"""Global configuration of LUMOS, exposed as ``lumos.config``."""

from __future__ import annotations

from typing import Any

__all__ = ["Config", "config"]


class Config:
    """Process-wide settings of LUMOS.

    Every setting is validated on assignment, and unknown settings are refused (to catch typos).

    Attributes
    ----------
    fft_workers
        Number of CPU threads used by FFTs on the NumPy backend; ``-1`` uses every core.
    """

    __slots__ = ("_fft_workers",)

    def __init__(self) -> None:
        self._fft_workers = -1

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
        return f"lumos.config(fft_workers={self.fft_workers})"


config = Config()
