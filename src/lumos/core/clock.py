"""Clock of a context: absolute start date and elapsed time (REQ-CTX-5).

The absolute time is an :class:`astropy.time.Time` in UTC; durations are quantities. Time advances only when
an action completes or when the context waits, so a simulation is reproducible whatever the speed of the
computer.
"""

from __future__ import annotations

import math
from typing import Any, cast

from astropy import units as u
from astropy.time import Time

from lumos.units import require_unit, to_si

__all__ = ["Clock", "duration_si"]


def duration_si(value: Any, *, name: str = "duration") -> float:
    """Return a duration in seconds, checking that it is a non-negative finite time quantity."""
    seconds = float(to_si(require_unit(value, u.s, name=name)))
    if not (math.isfinite(seconds) and seconds >= 0):
        raise ValueError(f"{name} must be non-negative and finite, got {value}")
    return seconds


class Clock:
    """Start date and elapsed time.

    Parameters
    ----------
    start
        Absolute date of the beginning of the context (UTC); defaults to now.
    """

    def __init__(self, start: Time | None = None) -> None:
        if start is None:
            start = Time.now()
        if not isinstance(start, Time):
            raise TypeError(f"start must be an astropy Time, got {type(start).__name__}")
        self._start: Time = cast(Time, start.utc)
        self._elapsed = 0.0

    @property
    def start(self) -> Time:
        """Date of the beginning of the context (UTC)."""
        return self._start

    @property
    def elapsed(self) -> u.Quantity:
        """Time elapsed since the start."""
        return self._elapsed * u.s

    @property
    def elapsed_si(self) -> float:
        """Time elapsed since the start, in seconds."""
        return self._elapsed

    @property
    def time(self) -> Time:
        """Current absolute date."""
        return cast(Time, cast(Any, self._start) + self._elapsed * u.s)

    def advance(self, seconds: float) -> None:
        """Advance the clock by ``seconds`` (non-negative)."""
        if seconds < 0:
            raise ValueError("the clock cannot go backwards")
        self._elapsed += seconds

    def reset(self, start: Time | None = None, elapsed: u.Quantity | None = None) -> None:
        """Set the start date and elapsed time (used when loading a context)."""
        if start is not None:
            self._start = cast(Time, start.utc)
        self._elapsed = 0.0 if elapsed is None else duration_si(elapsed, name="elapsed")
