"""Timed actions of elements (REQ-EL-5, REQ-EL-6).

An action is a method of an element that changes the state of the equipment it represents: move a stage,
open a shutter, take an exposure. It is declared with the :func:`action` decorator and a duration, fixed or
computed from the element properties before the action runs::

    class Stage(Element):
        velocity = Param(unit=u.mm / u.s, default=2 * u.mm / u.s)
        position = State(unit=u.mm, default=0 * u.mm)

        @action(duration=lambda self, target: abs(target - self.position) / self.velocity)
        def move(self, target):
            self.position = target

When the element belongs to a context, calling the action goes through the context: in simulation the body
updates the virtual state, the command is logged and the clock advances by the duration; in operation the
command is sent to the driver (see :mod:`lumos.core.driver`). Outside a context the body simply runs.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from astropy import units as u

from lumos.core.clock import duration_si

__all__ = ["ActionSpec", "action", "actions_of"]

Duration = u.Quantity | Callable[..., u.Quantity] | None


@dataclass(frozen=True)
class ActionSpec:
    """Description of an action: its name, its duration and the undecorated method."""

    name: str
    duration: Duration
    method: Callable[..., Any]
    doc: str

    def duration_si(self, element: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> float:
        """Duration of a call, in seconds (computed before the action runs)."""
        if self.duration is None:
            return 0.0
        value = self.duration(element, *args, **kwargs) if callable(self.duration) else self.duration
        return duration_si(value, name=f"duration of {self.name}")


def action(duration: Duration = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Declare a method as a timed action.

    Parameters
    ----------
    duration
        A time quantity, a function ``(self, *args, **kwargs) -> quantity`` evaluated before the action, or
        ``None`` for an instantaneous action.
    """

    def decorate(method: Callable[..., Any]) -> Callable[..., Any]:
        spec = ActionSpec(method.__name__, duration, method, method.__doc__ or "")

        @functools.wraps(method)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            context = getattr(self, "context", None)
            if context is None:
                spec.duration_si(self, args, kwargs)  # validate the duration even without a clock
                return method(self, *args, **kwargs)
            return context._perform(self, spec, args, kwargs)

        wrapper.__lumos_action__ = spec  # type: ignore[attr-defined]
        return wrapper

    return decorate


def actions_of(cls: type) -> dict[str, ActionSpec]:
    """Return the actions declared by ``cls`` and its bases, by name."""
    found: dict[str, ActionSpec] = {}
    for klass in reversed(cls.__mro__):
        for value in vars(klass).values():
            spec = getattr(value, "__lumos_action__", None)
            if isinstance(spec, ActionSpec):
                found[spec.name] = spec
    return found
