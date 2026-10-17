"""Deployment profiles and configuration files of the web instance (step P14.3, REQ-WEB-9, REQ-WEB-10).

Requirements: two profiles, ``local`` (localhost, its user has every right) and ``public`` (anonymous
simulators, quotas, operation forbidden by default); a TOML file configures an instance and the command line
overrides it; an unsafe configuration stops the server at start-up with an explanation.
"""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from lumos.cli import main
from lumos.service import ServiceConfig
from lumos.web import WebConfig, create_app
from lumos.web.config import ConfigError, load_config


def test_the_local_profile_is_the_default():
    config = WebConfig()
    assert config.profile == "local"
    assert config.host == "127.0.0.1"
    assert config.anonymous_role == "operator"
    assert config.service.operation_enabled is False
    assert config.requests_per_minute is None
    assert config.service.max_grid_pixels is None


def test_the_public_profile_serves_anonymous_simulators_with_quotas_and_no_operation():
    config = WebConfig(profile="public")
    assert config.anonymous_role == "simulator"
    assert config.service.operation_enabled is False
    assert config.requests_per_minute is not None
    assert config.max_sessions is not None
    service = config.service
    assert service.max_run_seconds <= 120
    assert service.max_grid_pixels is not None
    assert service.max_grid_pixels <= 2048
    assert service.max_memory_mb is not None
    assert service.max_concurrent_runs is not None
    assert service.max_contexts_per_session <= 10


def test_settings_given_explicitly_win_over_the_profile():
    config = WebConfig(
        profile="public",
        anonymous_role="viewer",
        service=ServiceConfig(max_run_seconds=10, max_grid_pixels=4096),
    )
    assert config.anonymous_role == "viewer"
    assert config.service.max_run_seconds == 10
    assert config.service.max_grid_pixels == 4096
    assert config.service.max_memory_mb is not None  # not given: default of the profile


def test_a_configuration_file(tmp_path):
    path = tmp_path / "lumos-web.toml"
    path.write_text(
        'profile = "public"\nhost = "0.0.0.0"\nallowed_hosts = ["lumos.example.org"]\ntrust_proxy = true\n'
        "\n[service]\nmax_run_seconds = 30\nmax_grid_pixels = 512\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.profile == "public"
    assert config.allowed_hosts == ["lumos.example.org"]
    assert config.service.max_run_seconds == 30
    assert config.service.max_grid_pixels == 512
    assert config.service.max_concurrent_runs is not None
    overridden = load_config(path, port=9000, service={"max_run_seconds": 5})
    assert overridden.port == 9000
    assert overridden.service.max_run_seconds == 5
    assert overridden.service.max_grid_pixels == 512


@pytest.mark.parametrize(
    "content",
    ['profile = "remote"\n', 'colour = "blue"\n', "[service]\nmax_run_seconds = -1\n", "not = toml = 1\n"],
)
def test_invalid_configuration_files_are_explained(tmp_path, content):
    path = tmp_path / "lumos-web.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_the_local_profile_refuses_to_listen_on_other_interfaces():
    with pytest.raises(ConfigError, match="public"):
        create_app(WebConfig(host="0.0.0.0"))


def test_the_command_line_stops_on_an_unsafe_configuration(capsys):
    assert main(["web", "--host", "0.0.0.0"]) == 2
    assert "public" in capsys.readouterr().err


def test_the_operation_is_forbidden_by_default_on_a_public_instance():
    with TestClient(create_app(WebConfig(profile="public")), base_url="http://testserver") as client:
        assert client.get("/api/info").json()["operation"] is False
        response = client.post("/api/devices/discover", json={"tcp": ["tcp://127.0.0.1:1"]})
        assert response.status_code == 403


def test_the_number_of_sessions_is_bounded():
    app = create_app(WebConfig(profile="public", max_sessions=2))
    statuses = []
    for _ in range(3):
        with TestClient(app, base_url="http://testserver") as visitor:
            statuses.append(visitor.get("/api/info").status_code)
            visitor.cookies.clear()
    assert statuses[:2] == [200, 200]
    assert statuses[2] == 429


def test_the_quotas_of_a_public_instance_reach_the_service():
    app = create_app(WebConfig(profile="public", service=ServiceConfig(max_grid_pixels=64)))
    with TestClient(app, base_url="http://testserver") as client:
        cid = client.post("/api/contexts", json={}).json()["id"]
        client.post(
            f"/api/contexts/{cid}/elements", json={"type_id": "lumos.sources.GaussianLaser", "name": "laser"}
        )
        response = client.get(f"/api/contexts/{cid}/ports/laser.out")
        assert response.status_code == 429
        assert response.json()["code"] == "quota"
