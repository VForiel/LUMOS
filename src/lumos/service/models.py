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
    "FeedbackInfo",
    "FocusInfo",
    "GroupInfo",
    "Issue",
    "JobInfo",
    "Page",
    "PortFile",
    "PortSummary",
    "PresetInfo",
    "ServiceConfig",
    "SessionInfo",
    "StorageInfo",
    "StudyPoint",
    "StudyResult",
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
    max_study_points: int = Field(
        default=1000, ge=1, description="points of a sweep or draws of a Monte-Carlo run"
    )
    archive_root: str | None = Field(default=None, description="directory of the archives of the runs")
    session_storage: bool = Field(
        default=False,
        description="keep the archives of each session in a directory of its own, listed to it only and "
        "deleted with it (public instances)",
    )
    max_session_bytes: int | None = Field(
        default=None, ge=1, description="bytes kept per session (session_storage): the oldest data go beyond"
    )
    min_free_disk_bytes: int | None = Field(
        default=None, ge=0, description="free disk below which archiving is refused (warning below twice)"
    )
    max_recorded_figures: int = Field(
        default=2000, ge=0, description="figures recorded per study (record=True, Recorder blocks)"
    )
    max_jobs_kept: int = Field(
        default=50, ge=1, description="finished runs and studies of a session whose results stay in memory"
    )
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
    free: dict[str, dict[str, str] | None] = Field(
        default={}, description="free parameters and their priors, values in the unit of the parameter"
    )


class ConnectionInfo(BaseModel):
    """A connection ``source`` (``"id.port"``) → ``target``."""

    source: str
    target: str


class FeedbackInfo(BaseModel):
    """A feedback of a loop: the port ``source`` drives the field ``target`` (``mode`` ``set`` or ``add``)."""

    source: str
    target: str
    mode: str


class GroupInfo(BaseModel):
    """The inside of a group: connections of its elements and exposed ports (``port: "child.port"``)."""

    connections: list[ConnectionInfo]
    exposed: dict[str, str]


class ContextSummary(BaseModel):
    """A context: its elements, connections, mode and observation band."""

    id: str
    name: str
    description: str = ""
    mode: str
    twin: bool
    seed: int = 0
    band: list[str] | None = None
    n_wavelengths: int = 1
    elements: list[ElementSummary]
    connections: list[ConnectionInfo]
    feedbacks: list[FeedbackInfo] = []
    outputs: list[str] = []
    groups: dict[str, GroupInfo] = {}
    ui: dict[str, Any] = {}
    can_undo: bool = False
    can_redo: bool = False
    revision: int = Field(default=0, description="grows at every change of the context")


class FocusInfo(BaseModel):
    """The context in the foreground of a session, shared by the web interface and the assistant (P14.18).

    ``origin`` tells who focused it last (``web``, ``mcp`` or ``python``); ``count`` grows at every change of
    the focus, so that a client sees a change even when the same context comes back.
    """

    context: str | None = None
    origin: str | None = None
    count: int = 0


class PresetInfo(BaseModel):
    """A preset context: file name, title, description, outputs and the types of its elements."""

    name: str
    title: str = ""
    description: str
    outputs: list[str]
    elements: list[str] = []


class Issue(BaseModel):
    """A problem found in a context, with a suggested fix."""

    severity: str
    element: str | None
    message: str
    hint: str


class JobInfo(BaseModel):
    """A run or a study: its status (``queued``, ``running``, ``done``, ``failed``), progress and outputs."""

    id: str
    kind: str = Field(default="run", description="run, sweep, sample (Monte-Carlo run) or loop")
    context: str
    outputs: list[str]
    status: str
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    hint: str | None = None
    seconds: float | None = None
    archive: str | None = None
    warnings: list[str] = Field(default=[], description="e.g. old data deleted to stay within the storage")


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


class StudyPoint(BaseModel):
    """A point of a study: values of the parameters, noise realization and summaries of the outputs."""

    params: dict[str, str]
    realization: int = 0
    outputs: dict[str, PortSummary]


class StudyResult(BaseModel):
    """The points of a sweep or of a Monte-Carlo run, or the steps of a loop (parameter ``time``)."""

    job: str
    context: str
    kind: str
    mode: str | None = None
    noise: str
    seed: int
    parameters: list[str]
    outputs: list[str]
    points: list[StudyPoint]
    recorded: dict[str, list[int]] = Field(
        default={}, description="points with a recorded figure, by port (results browser, P14.20)"
    )


class PortFile(BaseModel):
    """A file made from the signal of a port: its name, media type and content."""

    filename: str
    media_type: str
    content: bytes


class ArchiveInfo(BaseModel):
    """An archive of a run."""

    path: str
    name: str
    created_at: str | None = None
    outputs: list[str] = []
    size_bytes: int = 0


class StorageInfo(BaseModel):
    """Disk used by the data of a session (archives, recordings), its limit and the free disk (P14.21)."""

    used_bytes: int = 0
    limit_bytes: int | None = None
    disk_free_bytes: int = 0
    disk_total_bytes: int = 0
    warnings: list[str] = []


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
