"""Routes of runs, inspection and downloads of the web interface (step P14.6, REQ-WEB-5).

Requirements: the outputs of a run and of any port are downloaded as figures (``.png``, ``.plt``), data
(``.npz``) or JSON with a file name; the figure of a computed output is shown without computing again; an
archive is downloaded as a ``.zip``; a run can be cancelled.
"""

import io
import time
import zipfile

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from lumos.service import ServiceConfig
from lumos.web import WebConfig, create_app


@pytest.fixture
def client(tmp_path):
    config = WebConfig(service=ServiceConfig(archive_root=str(tmp_path / "archives")))
    with TestClient(create_app(config), base_url="http://localhost") as client:
        yield client


def bench(client):
    cid = client.post("/api/contexts", json={"name": "bench"}).json()["id"]
    client.post(
        f"/api/contexts/{cid}/elements", json={"type_id": "lumos.sources.GaussianLaser", "name": "laser"}
    )
    client.post(
        f"/api/contexts/{cid}/elements", json={"type_id": "lumos.detectors.PowerMeter", "name": "meter"}
    )
    client.post(f"/api/contexts/{cid}/connections", json={"source": "laser.out", "target": "meter.in"})
    return cid


def finished(client, job):
    for _ in range(600):
        info = client.get(f"/api/jobs/{job}").json()
        if info["status"] not in ("queued", "running"):
            return info
        time.sleep(0.05)
    raise TimeoutError(job)


@pytest.mark.parametrize(
    ("fmt", "media"), [("png", "image/png"), ("plt", None), ("npz", None), ("json", "application/json")]
)
def test_any_port_is_downloaded(client, fmt, media):
    cid = bench(client)
    response = client.get(f"/api/contexts/{cid}/ports/laser.out/download", params={"format": fmt})
    assert response.status_code == 200, response.text
    assert response.headers["content-disposition"].endswith(f'laser.out.{fmt}"')
    if media:
        assert response.headers["content-type"].startswith(media)


def test_the_outputs_of_a_run_are_downloaded_and_shown(client):
    cid = bench(client)
    job = client.post(f"/api/contexts/{cid}/runs", json={"outputs": ["meter.out"], "archive": True}).json()[
        "id"
    ]
    info = finished(client, job)
    assert info["status"] == "done"
    response = client.get(f"/api/jobs/{job}/results/meter.out/download", params={"format": "npz"})
    assert response.status_code == 200
    assert response.headers["content-disposition"].endswith('meter.out.npz"')
    archive = client.get("/api/archives/download", params={"path": info["archive"]})
    assert archive.status_code == 200
    assert archive.headers["content-disposition"].endswith('.zip"')
    assert any(n.endswith("metadata.yml") for n in zipfile.ZipFile(io.BytesIO(archive.content)).namelist())


def test_unknown_formats_are_refused(client):
    cid = bench(client)
    response = client.get(f"/api/contexts/{cid}/ports/laser.out/download", params={"format": "docx"})
    assert response.status_code == 400


def test_cancelling_a_finished_run_is_explained(client):
    cid = bench(client)
    job = client.post(f"/api/contexts/{cid}/runs", json={"outputs": ["meter.out"]}).json()["id"]
    finished(client, job)
    response = client.post(f"/api/jobs/{job}/cancel")
    assert response.status_code == 400
    assert response.json()["hint"]
