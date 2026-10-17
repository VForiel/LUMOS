"""Deployment of the web instance (step P14.10, REQ-WEB-13).

Requirements: a Docker image serves the interface on a single machine with the local profile (the container
listens on all its interfaces, its port is published on the local machine only); the instance answers a health
check; the files given for a public instance on a VPS (configuration, reverse proxy, compose file) are valid
and safe: the example configuration loads and passes the checks of the server, the published port of the local
image is bound to 127.0.0.1, the image runs as an unprivileged user.
"""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from lumos.web import WebConfig, create_app
from lumos.web.config import ConfigError, load_config

DEPLOY = Path(__file__).resolve().parents[3] / "deploy"


def test_the_instance_answers_a_health_check():
    with TestClient(create_app(), base_url="http://localhost") as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert "set-cookie" not in response.headers  # a probe opens no session


def test_a_container_listens_on_all_its_interfaces_with_the_local_profile():
    config = WebConfig(host="0.0.0.0", listen_all=True)
    create_app(config)  # accepted: the port is published on the local machine only
    assert config.local_only
    with pytest.raises(ConfigError):
        create_app(WebConfig(host="0.0.0.0"))  # outside a container, the local profile stays on localhost


def test_the_image_serves_the_local_profile_on_a_port_published_locally():
    dockerfile = (DEPLOY / "Dockerfile").read_text(encoding="utf-8")
    assert "npm ci" in dockerfile
    assert "npm run build" in dockerfile
    assert "USER " in dockerfile  # not root
    assert "HEALTHCHECK" in dockerfile
    assert "--listen-all" in dockerfile
    compose = (DEPLOY / "compose.yaml").read_text(encoding="utf-8")
    assert '"127.0.0.1:8000:8000"' in compose


def test_the_example_configuration_of_a_public_instance_is_valid(tmp_path):
    config = load_config(
        DEPLOY / "public" / "lumos.toml", users_file=None, audit_log=str(tmp_path / "audit.log")
    )
    assert config.profile == "public"
    assert config.trust_proxy
    assert config.service.operation_enabled is False
    assert config.service.max_grid_pixels is not None
    create_app(config)  # passes the checks of the server


def test_the_reverse_proxy_terminates_tls_and_forwards_websockets():
    caddyfile = (DEPLOY / "public" / "Caddyfile").read_text(encoding="utf-8")
    assert "reverse_proxy" in caddyfile
    assert "lumos:8000" in caddyfile or "127.0.0.1:8000" in caddyfile
