"""Recordings of the points of a study, for the results browser (step P14.20, review of 19/10/2026).

Requirements: to browse the results of a study point by point (a slider over the steps of a loop or the points
of a sweep), the figures of its outputs must be kept:

- a study started with ``record=True`` (the *archive* switch of the interface) keeps the figure of every
  drawable output at every point; ``StudyResult.recorded`` gives, per port, the points that have one, and
  ``Service.get_recording`` returns it as PNG;
- without ``record``, only the ports tapped by a ``Recorder`` block are recorded, every ``every`` points; a
  ``Recorder`` passes its signal unchanged;
- recordings belong to the session: counted in its storage, deleted with it, and with their job when the job
  is forgotten (``max_jobs_kept``).
"""

import time
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from lumos.service import Service, ServiceConfig, ServiceError

PNG = b"\x89PNG\r\n\x1a\n"


def camera_bench(service, session, recorder_every=None):
    ctx = service.create_context(session, name="camera").id
    laser = {"power": "1 mW", "n_pixels": 32, "pixel_scale": "0.25 mm"}  # 8 mm: the 1 mm beam fits
    service.add_element(session, ctx, "lumos.sources.GaussianLaser", name="laser", params=laser)
    service.add_element(session, ctx, "lumos.detectors.Camera", name="camera")
    service.connect(session, ctx, "laser.out", "camera.in")
    if recorder_every is not None:
        service.add_element(
            session, ctx, "lumos.processing.Recorder", name="rec", params={"every": recorder_every}
        )
        service.connect(session, ctx, "laser.out", "rec.in")
    return ctx


def sweep(service, session, ctx, **options):
    job = service.run_study(
        session,
        ctx,
        kind="sweep",
        parameters={"laser.power": {"start": "1 mW", "stop": "4 mW", "num": 4}},
        outputs=["camera.out"],
        **options,
    )
    info = service.wait_job(session, job.id, timeout=120)
    assert info.status == "done", info.error
    return info, service.get_study(session, job.id)


@pytest.fixture
def service(tmp_path):
    service = Service(ServiceConfig(archive_root=str(tmp_path / "archives"), session_storage=True))
    yield service
    service.shutdown()


def test_a_recorded_study_keeps_a_figure_per_point(service):
    session = service.create_session(origin="web").id
    ctx = camera_bench(service, session)
    job, study = sweep(service, session, ctx, record=True)
    assert study.recorded == {"camera.out": [0, 1, 2, 3]}
    for k in range(4):
        assert service.get_recording(session, job.id, k, "camera.out").startswith(PNG)
    with pytest.raises(ServiceError) as missing:
        service.get_recording(session, job.id, 9, "camera.out")
    assert missing.value.code == "not_found"
    assert service.storage(session).used_bytes > 0


def test_without_record_nothing_is_kept(service):
    session = service.create_session(origin="web").id
    ctx = camera_bench(service, session)
    job, study = sweep(service, session, ctx)
    assert study.recorded == {}
    with pytest.raises(ServiceError):
        service.get_recording(session, job.id, 0, "camera.out")


def test_a_recorder_block_records_its_port_every_n_points(service):
    session = service.create_session(origin="web").id
    ctx = camera_bench(service, session, recorder_every=2)
    job, study = sweep(service, session, ctx)
    assert "rec.out" in study.outputs
    assert study.recorded == {"rec.out": [0, 2]}
    assert service.get_recording(session, job.id, 2, "rec.out").startswith(PNG)
    # the recorder passes the signal unchanged: same power as the laser at each point
    for point, milliwatts in zip(study.points, [1, 2, 3, 4], strict=True):
        summary = point.outputs["rec.out"]
        assert summary.kind == "Wavefront"
        assert u.Quantity(summary.details["power"]).to_value(u.W) == pytest.approx(
            milliwatts * 1e-3, rel=1e-6
        )


def test_recordings_go_with_their_job_and_their_session(tmp_path):
    service = Service(
        ServiceConfig(archive_root=str(tmp_path / "archives"), session_storage=True, max_jobs_kept=1)
    )
    try:
        session = service.create_session(origin="web").id
        ctx = camera_bench(service, session)
        first, _ = sweep(service, session, ctx, record=True)
        root = Path(tmp_path / "archives")
        assert any(p.name == first.id for p in root.rglob("*") if p.is_dir())
        second, _ = sweep(service, session, ctx, record=True)
        deadline = time.monotonic() + 5
        while any(p.name == first.id for p in root.rglob("*") if p.is_dir()) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not any(p.name == first.id for p in root.rglob("*") if p.is_dir())
        assert service.get_recording(session, second.id, 0, "camera.out").startswith(PNG)
        service.close_session(session)
        assert not any(p.name == second.id for p in root.rglob("*") if p.is_dir())
    finally:
        service.shutdown()


def test_the_recorder_is_the_identity():
    from lumos.elements.sources import GaussianLaser
    from lumos.processing import Recorder

    beam = GaussianLaser(n_pixels=16).apply()["out"]
    out = Recorder().apply(**{"in": beam})["out"]
    assert out is beam or np.array_equal(np.asarray(out.field), np.asarray(beam.field))
