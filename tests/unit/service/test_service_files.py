"""Files of the outputs of a context, archives and cancellation of runs (step P14.6, REQ-WEB-5, REQ-RES-4).

Requirements: the signal of any port, inspected or computed by a run, can be downloaded as a figure (``.png``,
and ``.plt`` for the PltEdit editor), as data (``.npz`` without pickled objects) and as the JSON form of the
signal (as in archives); an archive is downloaded as a ``.zip``, only from the archive directory; a run can
be cancelled.

Expected values are physical: a power meter behind a laser of power P reads P, and the beam of a laser of
power P carries P (energy conservation, within the 1e-4 lost by the truncation of the Gaussian on the grid).
"""

import io
import json
import time
import zipfile

import numpy as np
import pytest
from astropy import units as u

from lumos.core.context import Context
from lumos.fields import DataProduct, Wavefront
from lumos.results import load_data
from lumos.service import Service, ServiceConfig, ServiceError

LASER = "lumos.sources.GaussianLaser"
METER = "lumos.detectors.PowerMeter"


@pytest.fixture
def service(tmp_path):
    service = Service(ServiceConfig(archive_root=str(tmp_path / "archives")))
    yield service
    service.shutdown()


@pytest.fixture
def session(service):
    return service.create_session(origin="web").id


def bench(service, session):
    ctx = service.create_context(session, name="bench").id
    service.add_element(session, ctx, LASER, name="laser", params={"power": "2 mW"})
    service.add_element(session, ctx, METER, name="meter")
    service.connect(session, ctx, "laser.out", "meter.in")
    return ctx


def test_a_figure_of_any_port(service, session, tmp_path):
    ctx = bench(service, session)
    png = service.port_file(session, ctx, "laser.out", "png")
    assert png.filename == "laser.out.png"
    assert png.media_type == "image/png"
    assert png.content.startswith(b"\x89PNG\r\n\x1a\n")
    plt = service.port_file(session, ctx, "laser.out", "plt")
    assert plt.filename.endswith(".plt")
    from lumos._vendor import pltedit

    path = tmp_path / plt.filename
    path.write_bytes(plt.content)
    assert pltedit.load(path) is not None  # reopens in PltEdit


def test_the_data_of_a_field_and_of_a_data_product(service, session, tmp_path):
    ctx = bench(service, session)
    field = service.port_file(session, ctx, "laser.out", "npz")
    (tmp_path / "field.npz").write_bytes(field.content)
    data = load_data(tmp_path / "field.npz")
    dx = np.diff(data["x"].to_value(u.m))[0]
    power = (data["intensity"].sum() * (dx * u.m) ** 2).to(u.mW)
    assert power.value == pytest.approx(2.0, rel=1e-4)
    with np.load(tmp_path / "field.npz", allow_pickle=False) as archive:  # no pickled object
        assert "intensity" in archive.files
    reading = service.port_file(session, ctx, "meter.out", "npz")
    (tmp_path / "reading.npz").write_bytes(reading.content)
    product = DataProduct.from_npz(tmp_path / "reading.npz")
    assert u.Quantity(product.data).to_value(u.mW) == pytest.approx(2.0, rel=1e-4)


def test_the_json_form_of_a_signal(service, session):
    ctx = bench(service, session)
    document = json.loads(service.port_file(session, ctx, "laser.out", "json").content)
    assert document["signal"] == "Wavefront"
    wavefront = Wavefront.from_json(document["data"])
    assert wavefront.power.to_value(u.mW) == pytest.approx(2.0, rel=1e-4)


def test_the_outputs_of_a_run_are_downloaded_without_computing_again(service, session):
    ctx = bench(service, session)
    job = service.wait_job(session, service.run(session, ctx, ["meter.out"]).id, timeout=60)
    file = service.result_file(session, job.id, "meter.out", "json")
    assert json.loads(file.content)["signal"] == "DataProduct"
    with pytest.raises(ServiceError):
        service.result_file(session, job.id, "laser.out", "json")  # not computed by this run


def test_unknown_formats_and_figureless_signals_are_explained(service, session):
    ctx = bench(service, session)
    with pytest.raises(ServiceError) as error:
        service.port_file(session, ctx, "laser.out", "docx")
    assert "png" in error.value.hint


def test_archives_are_downloaded_as_zip_files(service, session):
    ctx = bench(service, session)
    job = service.wait_job(session, service.run(session, ctx, ["meter.out"], archive=True).id, timeout=60)
    assert job.archive
    names = zipfile.ZipFile(io.BytesIO(service.archive_zip(job.archive))).namelist()
    assert any(name.endswith("metadata.yml") for name in names)
    with pytest.raises(ServiceError) as error:
        service.archive_zip(str(service.archive_root().parent))
    assert error.value.code == "forbidden"


def test_a_run_can_be_cancelled(service, session, monkeypatch):
    original = Context.run

    def slow(self, outputs, *args, progress=None, **kwargs):
        for _ in range(100):
            time.sleep(0.05)
            if progress is not None:
                progress(0, 1, "waiting")
        return original(self, outputs, *args, progress=progress, **kwargs)

    monkeypatch.setattr(Context, "run", slow)
    ctx = bench(service, session)
    job = service.run(session, ctx, ["meter.out"])
    time.sleep(0.3)
    assert service.cancel_job(session, job.id).id == job.id
    final = service.wait_job(session, job.id, timeout=30)
    assert final.status == "cancelled"
    with pytest.raises(ServiceError):
        service.cancel_job(session, job.id)  # already over
