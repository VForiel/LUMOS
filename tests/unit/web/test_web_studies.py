"""Studies over HTTP (step P14.13, REQ-WEB-1, REQ-CTX-10).

Requirements: the seed of a context is changed with the other properties of the context; a parameter is freed
with a prior; a study (sweep or Monte-Carlo run) starts like a run, is followed as a job, and its points are
read as JSON or downloaded as CSV; a viewer cannot start one.

Expected values: a 2 mW laser through a top-hat filter of power transmission ``T`` centred on its wavelength
gives ``2 mW × T`` on a power meter (truncation of the beam by the grid below 1e-3).
"""

import time

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from lumos.web import WebConfig, create_app


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(WebConfig(archive_root=str(tmp_path))), base_url="http://localhost") as client:
        yield client


def filter_bench(client):
    cid = client.post("/api/contexts", json={"name": "filter"}).json()["id"]
    for type_id, name, params in [
        ("lumos.sources.GaussianLaser", "laser", {"power": "2 mW"}),
        ("lumos.optics.SpectralFilter", "filt", {"center": "633 nm", "width": "10 nm"}),
        ("lumos.detectors.PowerMeter", "meter", {}),
    ]:
        client.post(
            f"/api/contexts/{cid}/elements", json={"type_id": type_id, "name": name, "params": params}
        )
    client.post(f"/api/contexts/{cid}/connections", json={"source": "laser.out", "target": "filt.in"})
    client.post(f"/api/contexts/{cid}/connections", json={"source": "filt.out", "target": "meter.in"})
    return cid


def wait(client, job):
    deadline = time.monotonic() + 60
    while True:
        info = client.get(f"/api/jobs/{job}").json()
        if info["status"] not in ("queued", "running") or time.monotonic() > deadline:
            return info
        time.sleep(0.05)


def test_seed_free_parameters_and_studies(client):
    cid = filter_bench(client)
    assert client.patch(f"/api/contexts/{cid}", json={"seed": 5}).json()["seed"] == 5

    summary = client.put(
        f"/api/contexts/{cid}/free",
        json={
            "path": "filt.transmission",
            "free": True,
            "prior": {"kind": "uniform", "low": 0.2, "high": 0.8},
        },
    ).json()
    filt = next(e for e in summary["elements"] if e["id"] == "filt")
    assert filt["free"]["transmission"] == {"kind": "uniform", "low": "0.2", "high": "0.8"}

    job = client.post(
        f"/api/contexts/{cid}/studies",
        json={
            "kind": "sweep",
            "parameters": {"filt.transmission": {"start": 0.5, "stop": 1, "num": 2}},
            "outputs": ["meter.out"],
        },
    )
    assert job.status_code == 202
    info = wait(client, job.json()["id"])
    assert info["status"] == "done"
    assert info["kind"] == "sweep"
    study = client.get(f"/api/jobs/{info['id']}/study").json()
    assert [p["params"]["filt.transmission"] for p in study["points"]] == ["0.5", "1"]
    assert study["points"][1]["outputs"]["meter.out"]["values"][0] == pytest.approx(2e-3, rel=1e-3)
    table = client.get(f"/api/jobs/{info['id']}/study.csv")
    assert table.headers["content-type"].startswith("text/csv")
    assert "attachment" in table.headers["content-disposition"]
    assert table.text.splitlines()[0].startswith("point,realization,filt.transmission")

    sample = client.post(
        f"/api/contexts/{cid}/studies", json={"kind": "sample", "n": 5, "outputs": ["meter.out"]}
    )
    info = wait(client, sample.json()["id"])
    assert len(client.get(f"/api/jobs/{info['id']}/study").json()["points"]) == 5


def test_errors_and_rights(client, tmp_path):
    cid = filter_bench(client)
    refused = client.post(
        f"/api/contexts/{cid}/studies", json={"kind": "sample", "n": 3, "outputs": ["meter.out"]}
    )
    assert refused.status_code == 400
    assert "free" in refused.json()["hint"]

    config = WebConfig(
        profile="public",
        allowed_hosts=["lumos.example"],
        trust_proxy=True,
        anonymous_role="viewer",
        audit_log=str(tmp_path / "audit.log"),
    )
    with TestClient(create_app(config), base_url="https://lumos.example") as viewer:
        cid = viewer.post("/api/presets/first_simulation/load").json()["id"]
        refused = viewer.post(f"/api/contexts/{cid}/studies", json={"kind": "sample", "n": 2})
        assert refused.status_code == 403
        assert refused.json()["code"] == "forbidden"
