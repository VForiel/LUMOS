"""Structured journal of a context (REQ-CTX-9).

Every action, wait and numerical choice of a context is recorded as a :class:`JournalEntry` in
``ctx.journal`` and emitted through the standard :mod:`logging` module on the ``lumos.context`` logger, with
the entry attached to the record as ``record.lumos`` (a JSON-compatible dictionary). :class:`JsonFormatter`
writes such records as one JSON object per line, ready for log files and the web interface.

Each entry records the **origin** of the action: ``"python"`` (scripts and notebooks), ``"web"`` (the visual
interface) or ``"mcp"`` (an AI assistant), set with ``ctx.as_origin(...)`` by the service layer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from lumos.core.driver import Command

__all__ = ["ORIGINS", "Journal", "JournalEntry", "JsonFormatter"]

#: Allowed origins of actions.
ORIGINS: tuple[str, ...] = ("python", "web", "mcp")


@dataclass
class JournalEntry:
    """One event of a context.

    Attributes
    ----------
    kind
        ``"action"``, ``"wait"``, ``"run"`` (outputs computed, in ``args``, with the duration of the
        computation in ``measured``) or ``"log"`` (records of other LUMOS loggers during a run).
    elapsed
        Elapsed time of the context clock when the event started, in seconds.
    date
        Absolute date of the clock when the event started (ISO 8601, UTC).
    origin
        ``"python"``, ``"web"`` or ``"mcp"``.
    element, action, args, command, mode, driver
        For actions: element path, action name, readable arguments, the :class:`~lumos.core.driver.Command`,
        effective mode and driver class name (operation only).
    predicted, measured
        Predicted and measured durations in seconds (``measured`` in operation only).
    logger, level, message
        For log entries: logger name, level name and message.
    """

    kind: str
    elapsed: float
    date: str
    origin: str
    element: str | None = None
    action: str | None = None
    args: str = ""
    command: Command | None = None
    mode: str | None = None
    driver: str | None = None
    predicted: float | None = None
    measured: float | None = None
    logger: str | None = None
    level: str | None = None
    message: str = ""
    wall_time: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="milliseconds"))

    def to_json(self) -> dict[str, Any]:
        """Return the entry as a JSON-compatible dictionary."""
        data = {key: value for key, value in vars(self).items() if key != "command"}
        data["command"] = None if self.command is None else self.command.to_json()
        return data


class Journal:
    """Ordered list of :class:`JournalEntry` objects."""

    def __init__(self) -> None:
        self._entries: list[JournalEntry] = []

    def append(self, entry: JournalEntry) -> None:
        """Add an entry."""
        self._entries.append(entry)

    def __iter__(self) -> Iterator[JournalEntry]:
        """Iterate over the entries in order."""
        return iter(list(self._entries))

    def __len__(self) -> int:
        """Return the number of entries."""
        return len(self._entries)

    def entries(self, kind: str | None = None) -> list[JournalEntry]:
        """Return the entries, optionally of one kind only."""
        return [e for e in self._entries if kind is None or e.kind == kind]

    def commands(self) -> list[Command]:
        """Return the commands of the actions, in order (identical in simulation and operation)."""
        return [e.command for e in self._entries if e.kind == "action" and e.command is not None]

    def clear(self) -> None:
        """Remove every entry."""
        self._entries.clear()

    def to_json(self) -> list[dict[str, Any]]:
        """Return the entries as JSON-compatible dictionaries."""
        return [e.to_json() for e in self._entries]


class JsonFormatter(logging.Formatter):
    """Format log records as one JSON object per line, including the structured ``record.lumos`` entry."""

    def format(self, record: logging.LogRecord) -> str:
        """Return the JSON line of ``record``."""
        data = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
            "lumos": getattr(record, "lumos", None),
        }
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False, default=str)


class _Capture(logging.Handler):
    """Copy the records of other LUMOS loggers into a journal while a context runs."""

    def __init__(self, journal: Journal, entry_factory: Any) -> None:
        super().__init__(logging.INFO)
        self._journal = journal
        self._entry_factory = entry_factory

    def emit(self, record: logging.LogRecord) -> None:
        if hasattr(record, "lumos"):
            return  # already a journal entry
        entry = self._entry_factory("log")
        entry.logger = record.name
        entry.level = record.levelname
        entry.message = record.getMessage()
        self._journal.append(entry)
