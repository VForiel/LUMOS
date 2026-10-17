"""Configuration of the web instance (steps P14.1 to P14.3, REQ-WEB-9, REQ-WEB-10, ADR-0019).

Two profiles:

- ``local``: one user on their own machine. The server listens on ``localhost`` only; its user has every role
  (the operation of equipment, when enabled, still asks a confirmation for each action).
- ``public``: anyone on the internet. Visitors are anonymous simulators by default, with tight quotas; the
  operation of equipment is disabled and can only be enabled together with TLS and registered operators.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from lumos.service.models import ServiceConfig

__all__ = ["LOOPBACK_HOSTS", "PROFILE_DEFAULTS", "ConfigError", "WebConfig", "load_config"]

#: Host names of the local machine.
LOOPBACK_HOSTS: tuple[str, ...] = ("localhost", "127.0.0.1", "::1", "[::1]")

Role = Literal["viewer", "simulator", "operator"]

#: Settings that depend on the profile when they are not given.
PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "local": {
        "anonymous_role": "operator",
        "requests_per_minute": None,
        "login_attempts_per_minute": 20,
        "max_sessions": None,
    },
    "public": {
        "anonymous_role": "simulator",
        "requests_per_minute": 600,
        "login_attempts_per_minute": 5,
        "max_sessions": 500,
    },
}

#: Quotas of the service on a public instance, unless the configuration sets them (REQ-WEB-10).
PUBLIC_SERVICE_DEFAULTS: dict[str, Any] = {
    "max_contexts_per_session": 5,
    "max_jobs_per_session": 1,
    "max_run_seconds": 60.0,
    "max_grid_pixels": 1024,
    "max_memory_mb": 2048.0,
    "max_concurrent_runs": 4,
}


class ConfigError(ValueError):
    """A configuration of the web instance is invalid or unsafe."""


def _is_loopback(host: str) -> bool:
    return host.lower() in LOOPBACK_HOSTS


class WebConfig(BaseModel):
    """Settings of a web instance of LUMOS.

    Settings left out take the default of the profile (:data:`PROFILE_DEFAULTS`; on a public instance, the
    quotas of the service of ``PUBLIC_SERVICE_DEFAULTS`` too).
    """

    profile: Literal["local", "public"] = "local"
    host: str = Field(default="127.0.0.1", description="address to listen on")
    port: int = Field(default=8000, ge=0, le=65535)
    listen_all: bool = Field(
        default=False,
        description="in a container whose port is published on the local machine only: listen on 0.0.0.0",
    )
    allowed_hosts: list[str] = Field(
        default_factory=list, description="host names accepted in requests, besides the local ones"
    )
    allowed_origins: list[str] = Field(
        default_factory=list, description="origins allowed to open WebSockets, besides the instance itself"
    )
    trust_proxy: bool = Field(
        default=False,
        description="behind a reverse proxy terminating TLS: trust its X-Forwarded-For and X-Forwarded-Proto",
    )
    tls_certfile: str | None = Field(default=None, description="certificate (PEM) to serve HTTPS directly")
    tls_keyfile: str | None = Field(default=None, description="private key (PEM) of the certificate")
    users_file: str | None = Field(default=None, description="TOML file of the users (lumos users add)")
    anonymous_role: Role | None = Field(
        default="operator",
        description='role of visitors who did not log in (None or "none": they must log in)',
    )
    session_idle_minutes: float = Field(
        default=12 * 60, gt=0, description="sessions expire after this idle time"
    )
    session_max_hours: float = Field(
        default=12.0, gt=0, description="authenticated sessions expire after this time, even when used"
    )
    challenge_seconds: float = Field(default=120.0, gt=0, description="validity of an SSH challenge")
    requests_per_minute: int | None = Field(default=None, ge=1, description="requests per client address")
    login_attempts_per_minute: int = Field(default=20, ge=1, description="per client address and per user")
    max_sessions: int | None = Field(default=None, ge=1, description="sessions open at the same time")
    audit_log: str | None = Field(default=None, description="JSON-lines file of the security events")
    assistant: bool = Field(default=True, description="offer the assistant panel (MCP over HTTP)")
    service: ServiceConfig = Field(default_factory=ServiceConfig)

    @field_validator("anonymous_role", mode="before")
    @classmethod
    def _no_role(cls, value: Any) -> Any:
        return None if value == "none" else value  # TOML has no null: "none" requires a login

    @model_validator(mode="after")
    def _profile_defaults(self) -> WebConfig:
        for key, value in PROFILE_DEFAULTS[self.profile].items():
            if key not in self.model_fields_set:
                setattr(self, key, value)
        if self.profile == "public":
            given = self.service.model_fields_set
            for key, value in PUBLIC_SERVICE_DEFAULTS.items():
                if key not in given:
                    setattr(self.service, key, value)
        return self

    @property
    def local_only(self) -> bool:
        """Whether the instance only serves its own machine (local profile on a loopback address)."""
        return self.profile == "local" and (_is_loopback(self.host) or self.listen_all)

    @property
    def tls(self) -> bool:
        """Whether requests reach the instance over TLS (certificate, or a proxy terminating TLS)."""
        return bool(self.tls_certfile and self.tls_keyfile) or self.trust_proxy

    def check(self) -> None:
        """Refuse unsafe combinations of settings.

        Raises
        ------
        ConfigError
            The local profile on a non-local address; the operation of equipment outside the local machine
            without TLS, without a registered operator or with anonymous operators; a missing users file.
        """
        from lumos.web.auth import load_users

        if self.profile == "local" and not _is_loopback(self.host) and not self.listen_all:
            raise ConfigError(
                f"the local profile only listens on the local machine, not on {self.host}; "
                "use the public profile (--profile public) to serve other machines"
            )
        if self.profile == "public" and self.anonymous_role == "operator":
            raise ConfigError("anonymous visitors of a public instance cannot be operators")
        users = load_users(self.users_file) if self.users_file else {}
        if self.service.operation_enabled and not self.local_only:
            if not self.tls:
                raise ConfigError(
                    "the operation of equipment outside the local machine needs TLS: give tls_certfile and "
                    "tls_keyfile, or serve the instance behind a reverse proxy terminating TLS (trust_proxy)"
                )
            if not any(user.role == "operator" for user in users.values()):
                raise ConfigError(
                    "the operation of equipment outside the local machine needs at least one operator in the "
                    "users file (lumos users add NAME --role operator)"
                )
        if (self.tls_certfile is None) != (self.tls_keyfile is None):
            raise ConfigError("give both tls_certfile and tls_keyfile")


def load_config(path: str | Path, **overrides: Any) -> WebConfig:
    """Read a TOML configuration file (sections at the top level and ``[service]``).

    Parameters
    ----------
    path
        TOML file, e.g. ``profile = "public"``, ``users_file = "users.toml"``, ``[service]
        max_run_seconds = 30``.
    overrides
        Settings given on the command line; they win over the file.

    Raises
    ------
    ConfigError
        The file is not valid TOML or has unknown or invalid settings.
    """
    try:
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot read the configuration {path}: {error}") from error
    service = {**data.pop("service", {}), **overrides.pop("service", {})}
    data.update(overrides)
    known = set(WebConfig.model_fields)
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"unknown settings in {path}: {', '.join(unknown)}")
    try:
        return WebConfig(**data, service=ServiceConfig(**service))
    except ValidationError as error:
        raise ConfigError(f"invalid configuration {path}: {error}") from error
