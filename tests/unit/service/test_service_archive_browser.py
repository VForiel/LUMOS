"""Browsing the archives of a session (step P15.17, review of 20/10/2026).

Requirements: the files of an archive are listed (data, figures, context, journal, metadata); a data file is
summarised with a figure, as an output of a run; a single file is downloaded; an archive is deleted; nothing
outside the archives of the session is reachable (another session, a path outside the archive directory).

Expected values: the data read back are the outputs the run archived (same summary as the run itself).
"""

import time

import pytest

from lumos.service import Service, ServiceConfig, ServiceError


@pytest.fixture
def archived(tmp_path):
    service = Service(ServiceConfig(archive_root=str(tmp_path / "archives"), session_storage=True))
    session = service.create_session().id
    ctx = service.load_preset(session, "first_simulation").id
    job = service.wait_job(session, service.run(session, ctx, ["camera.out"], archive=True).id, timeout=120)
    assert job.status == "done", job.message
    yield service, session, job
    service.shutdown()


def test_the_files_of_an_archive_are_listed(archived):
    service, session, _ = archived
    (archive,) = service.list_archives(session=session)
    files = {f.name: f.kind for f in service.archive_files(archive.path, session=session)}
    assert files["data/camera.out.npz"] == "data"
    assert files["context.lumos"] == "context"
    assert files["metadata.yml"] == "metadata"


def test_a_data_file_is_summarised_with_a_figure(archived):
    service, session, job = archived
    (archive,) = service.list_archives(session=session)
    summary = service.archive_data(archive.path, "data/camera.out.npz", session=session)
    original = service.get_result(session, job.id, "camera.out")
    assert summary.kind == original.kind == "DataProduct"
    assert summary.statistics == pytest.approx(original.statistics)
    assert summary.figure_png


def test_a_file_is_downloaded_and_an_archive_deleted(archived):
    service, session, _ = archived
    (archive,) = service.list_archives(session=session)
    content = service.archive_file(archive.path, "metadata.yml", session=session)
    assert b"created_at" in content
    service.delete_archive(archive.path, session=session)
    assert service.list_archives(session=session) == []


def test_nothing_outside_the_archives_of_the_session(archived):
    service, session, _ = archived
    (archive,) = service.list_archives(session=session)
    other = service.create_session().id
    with pytest.raises(ServiceError):
        service.archive_files(archive.path, session=other)
    with pytest.raises(ServiceError):
        service.archive_file(archive.path, "../../metadata.yml", session=session)
    with pytest.raises(ServiceError):
        service.delete_archive(archive.path, session=other)


def test_the_web_api(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from lumos.web import WebConfig, create_app

    config = WebConfig(service=ServiceConfig(archive_root=str(tmp_path)))
    with TestClient(create_app(config), base_url="http://localhost") as client:
        client.get("/api/info")
        cid = client.post("/api/presets/first_simulation/load").json()["id"]
        job = client.post(
            f"/api/contexts/{cid}/runs", json={"outputs": ["camera.out"], "archive": True}
        ).json()
        for _ in range(600):
            if client.get(f"/api/jobs/{job['id']}").json()["status"] in ("done", "failed"):
                break
            time.sleep(0.1)
        (archive,) = client.get("/api/archives").json()
        files = client.get("/api/archives/files", params={"path": archive["path"]}).json()
        assert any(f["name"] == "data/camera.out.npz" for f in files)
        summary = client.get(
            "/api/archives/data", params={"path": archive["path"], "name": "data/camera.out.npz"}
        )
        assert summary.json()["kind"] == "DataProduct"
        raw = client.get("/api/archives/file", params={"path": archive["path"], "name": "metadata.yml"})
        assert raw.status_code == 200
        assert client.delete("/api/archives", params={"path": archive["path"]}).status_code == 204
