"""Durable connector links (fix of the connector of 20/10/2026, ADR-0019 amendment).

A connector added to Claude.ai or ChatGPT keeps its URL for months, while a browser session expires after
some idle time and the server restarts at every update: a link that died with its session left the connector
broken ("unknown or revoked connector link"). Requirements:

- a link outlives its browser session: used after the session expired, it opens a new session (same user and
  role) instead of being refused;
- with ``connector_links_file``, links survive a restart of the server; the file holds digests of the
  secrets, never the secrets;
- a revoked or replaced link stops working, also after a restart; a link expires after
  ``connector_link_days``.
"""

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mcp")

from fastapi.testclient import TestClient

from lumos.web import WebConfig, create_app

VERSION = "2025-06-18"
HEADERS = {"accept": "application/json, text/event-stream", "content-type": "application/json"}


def initialize(client, path):
    info = {"name": "connector-test", "version": "1"}
    message = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": VERSION, "capabilities": {}, "clientInfo": info},
    }
    return client.post(path, content=json.dumps(message), headers=HEADERS)


def make_link(app):
    with TestClient(app, base_url="http://localhost") as browser:
        browser.get("/api/info")
        return browser.post("/api/assistant/link").json()["url"].removeprefix("http://localhost")


def test_a_link_outlives_its_browser_session(tmp_path):
    app = create_app(WebConfig(archive_root=str(tmp_path)))
    path = make_link(app)
    sessions = app.state.sessions
    for token in [s.token for s in list(sessions._sessions.values())]:
        sessions.close(token)  # the browser session expired
    with TestClient(app, base_url="http://localhost") as claude:
        assert initialize(claude, path).status_code == 200


def test_links_survive_a_restart_and_only_their_digests_are_stored(tmp_path):
    store = tmp_path / "links.json"
    config = WebConfig(archive_root=str(tmp_path), connector_links_file=str(store))
    path = make_link(create_app(config))
    secret = path.removeprefix("/mcp/")
    assert store.is_file()
    assert secret not in store.read_text(encoding="utf-8")
    with TestClient(create_app(config), base_url="http://localhost") as claude:  # a new server process
        assert initialize(claude, path).status_code == 200


def test_revoked_links_stay_revoked_after_a_restart(tmp_path):
    config = WebConfig(archive_root=str(tmp_path), connector_links_file=str(tmp_path / "links.json"))
    app = create_app(config)
    with TestClient(app, base_url="http://localhost") as browser:
        browser.get("/api/info")
        first = browser.post("/api/assistant/link").json()["url"].removeprefix("http://localhost")
        second = browser.post("/api/assistant/link").json()["url"].removeprefix("http://localhost")
        browser.delete("/api/assistant/link")
    with TestClient(create_app(config), base_url="http://localhost") as claude:
        assert initialize(claude, first).status_code == 404
        assert initialize(claude, second).status_code == 404


def test_links_expire(tmp_path):
    config = WebConfig(
        archive_root=str(tmp_path), connector_links_file=str(tmp_path / "links.json"), connector_link_days=0
    )
    path = make_link(create_app(config))
    with TestClient(create_app(config), base_url="http://localhost") as claude:
        assert initialize(claude, path).status_code == 404
