"""Recorded sequences of actions (step P12.8, REQ-HW-7, ADR-0014).

``with ctx.record() as sequence:`` records every action of the elements of a context as a
:class:`~lumos.core.driver.Command`, with the time of the clock at which it started (from the start of the
recording). A sequence recorded in simulation is replayed in operation (``ctx.replay(sequence)``): the
commands sent are the same. Sequences are saved as JSON documents of kind ``"sequence"``; quantities keep
their units.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astropy import units as u

from lumos.core.driver import Command
from lumos.units import quantity_from_json, quantity_to_json

__all__ = ["Sequence", "Step"]

FORMAT = "1.0"


@dataclass(frozen=True)
class Step:
    """A command of a sequence and the time it started, from the start of the recording."""

    time: u.Quantity
    command: Command


class Sequence:
    """An ordered list of :class:`Step` objects."""

    def __init__(self, steps: list[Step] | None = None) -> None:
        self.steps: list[Step] = list(steps or [])

    def append(self, time: u.Quantity, command: Command) -> None:
        """Add a step."""
        self.steps.append(Step(time, command))

    def __iter__(self) -> Iterator[Step]:
        """Return an iterator over the steps."""
        return iter(self.steps)

    def __len__(self) -> int:
        """Return the number of steps."""
        return len(self.steps)

    def __getitem__(self, index: int) -> Step:
        """Return a step."""
        return self.steps[index]

    def __repr__(self) -> str:
        """Summarise the sequence."""
        return f"Sequence({len(self.steps)} steps)"

    # --- JSON -------------------------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        """Return the JSON document of the sequence."""
        return {
            "kind": "sequence",
            "format": FORMAT,
            "steps": [{"time": quantity_to_json(s.time), "command": s.command.to_json()} for s in self.steps],
        }

    @classmethod
    def from_json(cls, document: Mapping[str, Any]) -> Sequence:
        """Read a document written by :meth:`to_json`."""
        if document.get("kind") != "sequence":
            raise ValueError("not a LUMOS sequence document (kind must be 'sequence')")
        steps = [
            Step(quantity_from_json(step["time"]), Command.from_json(step["command"]))
            for step in document["steps"]
        ]
        return cls(steps)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the sequence as JSON."""
        Path(path).write_text(json.dumps(self.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Sequence:
        """Read a sequence written by :meth:`save`."""
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))
