"""Storage of the sessions of a public instance over HTTP (step P14.21, review of 19/10/2026).

Requirements: a public instance keeps the archives of each visitor apart (``session_storage``), at most 1 GB
per session (``max_session_bytes`` = 10⁹ bytes), and refuses to archive when less than 2 GB stay free on the
disk; ``/api/archives`` lists the archives of the session only; ``/api/storage`` gives the use of the
session and its warnings. The local profile keeps its archives (no per-session storage, no limit).
"""

import time

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from lumos.web import WebConfig, create_app


def test_a_public_instance_limits_and_separates_the_storage_of_its_sessions():
    service = WebConfig(profile="public", allowed_hosts=["lumos.example.org"]).service
    assert service.session_storage is True
    assert service.max_session_bytes == 10**9
    assert service.min_free_disk_bytes == 2 * 10**9
    local = WebConfig().service
    assert local.session_storage is False
    assert local.max_session_bytes is None


def run_archived(client):
    cid = client.post("/api/contexts", json={"name": "bench"}).json()["id"]
    laser = client.post(f"/api/contexts/{cid}/elements", json={"type_id": "lumos.sources.GaussianLaser"})
    meter = client.post(f"/api/contexts/{cid}/elements", json={"type_id": "lumos.detectors.PowerMeter"})
    source, target = laser.json()["id"], meter.json()["id"]
    client.post(
        f"/api/contexts/{cid}/connections", json={"source": f"{source}.out", "target": f"{target}.in"}
    )
    job = client.post(
        f"/api/contexts/{cid}/runs", json={"outputs": [f"{target}.out"], "archive": True}
    ).json()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        info = client.get(f"/api/jobs/{job['id']}").json()
        if info["status"] not in ("queued", "running"):
            return info
        time.sleep(0.05)
    pytest.fail("the run did not end")


def test_the_archives_and_the_storage_are_those_of_the_session(tmp_path):
    config = WebConfig(archive_root=str(tmp_path))
    config.service.session_storage = True
    config.service.max_session_bytes = 10**8
    app = create_app(config)
    with (
        TestClient(app, base_url="http://localhost") as alice,
        TestClient(app, base_url="http://localhost") as bob,
    ):
        info = run_archived(alice)
        assert info["status"] == "done"
        bob.get("/api/info")
        assert [a["path"] for a in alice.get("/api/archives").json()] == [info["archive"]]
        assert bob.get("/api/archives").json() == []
        refused = bob.get("/api/archives/download", params={"path": info["archive"]})
        assert refused.status_code == 403

        storage = alice.get("/api/storage").json()
        assert storage["used_bytes"] > 0
        assert storage["limit_bytes"] == 10**8
        assert storage["disk_free_bytes"] > 0
        assert storage["warnings"] == []
        assert bob.get("/api/storage").json()["used_bytes"] == 0
