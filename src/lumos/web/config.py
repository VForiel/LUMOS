"""Configuration of the web instance (steps P14.1 and P14.3, REQ-WEB-9, REQ-WEB-10)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from lumos.service.models import ServiceConfig

__all__ = ["LOOPBACK_HOSTS", "WebConfig"]

#: Host names of the local machine.
LOOPBACK_HOSTS: tuple[str, ...] = ("localhost", "127.0.0.1", "::1", "[::1]")


class WebConfig(BaseModel):
    """Settings of a web instance of LUMOS.

    The ``local`` profile serves one user on their own machine: it listens on ``localhost`` and only answers
    requests addressed to a local host name. The ``public`` profile serves anyone (P14.3).
    """

    profile: Literal["local", "public"] = "local"
    host: str = Field(default="127.0.0.1", description="address to listen on")
    port: int = Field(default=8000, ge=0, le=65535)
    allowed_hosts: list[str] = Field(
        default_factory=list, description="host names accepted in requests, besides the local ones"
    )
    allowed_origins: list[str] = Field(
        default_factory=list, description="origins allowed to open WebSockets, besides the instance itself"
    )
    session_idle_minutes: float = Field(
        default=12 * 60, gt=0, description="sessions expire after this idle time"
    )
    service: ServiceConfig = Field(default_factory=ServiceConfig)
