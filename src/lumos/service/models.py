"""Typed messages of the service layer (Pydantic models, step P13.1, ADR-0015).

Every value with a unit is written as a string ``"<value> <unit>"`` (``"0.5 mm"``), so that the messages read
well in JSON, in the web interface and in the answers of an AI assistant.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ActionResult",
    "ArchiveInfo",
    "ConnectionInfo",
    "ContextSummary",
    "DeviceModel",
    "DocHit",
    "ElementSummary",
    "ElementTypeInfo",
    "Issue",
    "JobInfo",
    "Page",
    "PortSummary",
    "PresetInfo",
    "ServiceConfig",
    "SessionInfo",
]


class ServiceConfig(BaseModel):
    """Configuration of a service (deployment profiles set it, step P14.3)."""

    operation_enabled: bool = Field(default=False, description="allow actions on real equipment")
    max_contexts_per_session: int = Field(default=20, ge=1)
    max_jobs_per_session: int = Field(default=2, ge=1, description="runs in progress at the same time")
    max_workers: int = Field(default=4, ge=1, description="threads executing the runs")
    max_run_seconds: float = Field(default=600.0, gt=0, description="time allowed to a run or an inspection")
    max_grid_pixels: int | None = Field(
        default=None, ge=1, description="largest grid side, in pixels, of the elements of a computed context"
    )
    max_memory_mb: float | None = Field(
        default=None, gt=0, description="largest estimated memory of the fields of a computation, in MB"
    )
    max_concurrent_runs: int | None = Field(
        default=None, ge=1, description="runs queued or in progress at the same time, all sessions together"
    )
    archive_root: str | None = Field(default=None, description="directory of the archives of the runs")
    presets_dir: str | None = Field(default=None, description="directory of the preset .lumos files")


class SessionInfo(BaseModel):
    """A session: its identifier and the origin of its actions (``python``, ``web`` or ``mcp``)."""

    id: str
    origin: str
    user: str | None = None


class Page[T](BaseModel):
    """One page of a long list."""

    items: list[T]
    total: int
    page: int
    page_size: int
    next_page: int | None = None


class ElementTypeInfo(BaseModel):
    """An element type of the registry."""

    type_id: str
    title: str
    summary: str
    inputs: list[str]
    outputs: list[str]


class DocHit(BaseModel):
    """A documentation page matching a search."""

    title: str
    path: str
    excerpt: str
    score: float


class ElementSummary(BaseModel):
    """An element of a context, with its values in their units."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    type_id: str
    params: dict[str, Any]
    settings: dict[str, Any]
    state: dict[str, Any]
    inputs: list[str]
    outputs: list[str]
    actions: list[str]


class ConnectionInfo(BaseModel):
    """A connection ``source`` (``"id.port"``) → ``target``."""

    source: str
    target: str


class ContextSummary(BaseModel):
    """A context: its elements, connections, mode and observation band."""

    id: str
    name: str
    description: str = ""
    mode: str
    twin: bool
    band: list[str] | None = None
    n_wavelengths: int = 1
    elements: list[ElementSummary]
    connections: list[ConnectionInfo]
    outputs: list[str] = []


class PresetInfo(BaseModel):
    """A preset context."""

    name: str
    description: str
    outputs: list[str]


class Issue(BaseModel):
    """A problem found in a context, with a suggested fix."""

    severity: str
    element: str | None
    message: str
    hint: str


class JobInfo(BaseModel):
    """A run: its status (``queued``, ``running``, ``done``, ``failed``), progress and outputs."""

    id: str
    context: str
    outputs: list[str]
    status: str
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    hint: str | None = None
    seconds: float | None = None
    archive: str | None = None


class PortSummary(BaseModel):
    """Compact description of a signal: kind, unit, shape, a few values, statistics, optional figure."""

    port: str
    kind: str
    unit: str | None = None
    shape: list[int] = []
    values: list[Any] | None = None
    statistics: dict[str, float] = {}
    details: dict[str, str] = {}
    figure_png: bytes | None = None


class ArchiveInfo(BaseModel):
    """An archive of a run."""

    path: str
    name: str
    created_at: str | None = None
    outputs: list[str] = []


class DeviceModel(BaseModel):
    """A discovered device."""

    vendor: str
    model: str
    serial: str
    firmware: str
    url: str


class ActionResult(BaseModel):
    """Outcome of an action: mode, predicted and measured durations, returned value."""

    element: str
    action: str
    mode: str
    predicted_s: float
    measured_s: float | None = None
    value: str | None = None
