"""Tests for the service layer (step P13.1, REQ-SVC-1, ADR-0015).

Requirements: one service layer exposing sessions, contexts, editing, asynchronous execution with progress,
inspection, catalogues, archives and operation, with a typed (Pydantic) API; the web backend and the MCP
server are thin adapters of it. Test of the plan: parity between the service and the Python API (same
``.lumos`` document and same result).
"""

import json
import time

import numpy as np
import pytest
from astropy import units as u
from pydantic import BaseModel

import lumos.io
from lumos import Context
from lumos.elements.detectors import PowerMeter
from lumos.elements.sources import GaussianLaser
from lumos.service import Service, ServiceConfig, ServiceError


@pytest.fixture
def service(tmp_path):
    return Service(ServiceConfig(archive_root=str(tmp_path / "archives")))


@pytest.fixture
def session(service):
    return service.create_session(origin="mcp").id


def laser_bench(service, session):
    ctx = service.create_context(session, name="bench").id
    service.add_element(session, ctx, "lumos.sources.GaussianLaser", name="laser", params={"power": "2 mW"})
    service.add_element(session, ctx, "lumos.detectors.PowerMeter", name="meter")
    service.connect(session, ctx, "laser.out", "meter.in")
    return ctx


# --- Discovery --------------------------------------------------------------------------------------------


def test_element_types_are_listed_with_pagination(service):
    page = service.list_element_types(query="laser", page_size=5)
    assert isinstance(page, BaseModel)
    assert any(item.type_id == "lumos.sources.GaussianLaser" for item in page.items)
    assert all("laser" in (item.type_id + item.summary).lower() for item in page.items)
    everything = service.list_element_types(page_size=10)
    assert len(everything.items) == 10
    assert everything.total > 10
    assert everything.next_page == 2


def test_element_schema(service):
    schema = service.get_element_schema("lumos.sources.GaussianLaser")
    assert schema["properties"]["params"]["properties"]["power"]["x-lumos-unit"] == "mW"
    with pytest.raises(ServiceError) as error:
        service.get_element_schema("lumos.sources.Laser")
    assert "GaussianLaser" in error.value.hint  # actionable: the closest type


def test_docs_search(service):
    hits = service.search_docs("emergency stop")
    assert hits
    assert all(hit.excerpt for hit in hits)


# --- Contexts and editing ---------------------------------------------------------------------------------


def test_build_edit_and_describe_a_context(service, session):
    ctx = laser_bench(service, session)
    service.set_param(session, ctx, "laser.waist", "0.5 mm")
    service.set_param(session, ctx, "laser.wavelength", {"value": 1.064, "unit": "um"})
    summary = service.describe_context(session, ctx)
    laser = next(e for e in summary.elements if e.id == "laser")
    assert laser.params["waist"] == "0.5 mm"
    assert laser.params["wavelength"] == "1064 nm"  # in the unit of the parameter
    assert ("laser.out", "meter.in") in [(c.source, c.target) for c in summary.connections]
    with pytest.raises(ServiceError) as error:
        service.set_param(session, ctx, "laser.waist", "3 kg")
    assert "length" in error.value.hint or "mm" in error.value.hint
    with pytest.raises(ServiceError):
        service.connect(session, ctx, "meter.out", "laser.out")


def test_validation_reports_problems_with_hints(service, session):
    ctx = service.create_context(session).id
    service.add_element(session, ctx, "lumos.detectors.PowerMeter", name="meter")
    issues = service.validate_context(session, ctx)
    assert any(issue.element == "meter" and "not connected" in issue.message for issue in issues)
    assert all(issue.hint for issue in issues)


def test_presets_and_lumos_documents(service, session):
    presets = service.list_presets()
    assert "coronagraph" in [p.name for p in presets]
    loaded = service.load_preset(session, "first_simulation")
    assert {e.id for e in loaded.elements} >= {"laser", "lens", "camera"}
    text = service.export_lumos(session, loaded.id)
    again = service.import_lumos(session, text)
    assert {e.id for e in again.elements} == {e.id for e in loaded.elements}
    code = service.export_python(session, loaded.id)
    assert "GaussianLaser" in code


# --- Execution and inspection -----------------------------------------------------------------------------


def test_asynchronous_run_with_progress(service, session):
    ctx = laser_bench(service, session)
    job = service.run(session, ctx, ["meter.out"])
    assert job.status in ("queued", "running", "done")
    final = service.wait_job(session, job.id, timeout=60)
    assert final.status == "done"
    assert final.progress == pytest.approx(1.0)
    summary = service.get_result(session, job.id, "meter.out")
    assert summary.kind == "DataProduct"
    assert summary.unit == "W"
    assert summary.values[0] == pytest.approx(2e-3, rel=1e-3)


def test_inspection_returns_a_summary_and_a_figure(service, session):
    loaded = service.load_preset(session, "first_simulation")
    service.wait_job(session, service.run(session, loaded.id, ["camera.out", "focus.out"]).id, timeout=120)
    focus = service.inspect_port(session, loaded.id, "focus.out", figure=True)
    assert focus.kind == "Wavefront"
    assert "wavelength" in focus.details
    assert focus.figure_png.startswith(b"\x89PNG")
    camera = service.inspect_port(session, loaded.id, "camera.out", figure=False)
    assert camera.figure_png is None
    assert camera.unit == "adu"


def test_failed_jobs_report_actionable_errors(service, session):
    ctx = service.create_context(session).id
    service.add_element(session, ctx, "lumos.detectors.PowerMeter", name="meter")
    final = service.wait_job(session, service.run(session, ctx, ["meter.out"]).id, timeout=30)
    assert final.status == "failed"
    assert "not connected" in final.error
    assert final.hint


def test_quotas(tmp_path):
    service = Service(ServiceConfig(max_contexts_per_session=2))
    session = service.create_session().id
    service.create_context(session)
    service.create_context(session)
    with pytest.raises(ServiceError, match="quota"):
        service.create_context(session)


# --- Parity -----------------------------------------------------------------------------------------------


def test_parity_with_the_python_api(service, session):
    ctx = laser_bench(service, session)
    python = Context(name="bench")
    laser = python.add(GaussianLaser(name="laser", power=2 * u.mW))
    python.add(PowerMeter(name="meter"))
    laser >> python["meter"]
    via_service = json.loads(service.export_lumos(session, ctx))
    via_python = lumos.io.to_document(python)
    for document in (via_service, via_python):
        document.pop("created_at", None)
        document["meta"].pop("name", None)
    assert via_service["elements"] == via_python["elements"]
    assert via_service["connections"] == via_python["connections"]
    result = service.get_result(
        session, service.wait_job(session, service.run(session, ctx, ["meter.out"]).id).id, "meter.out"
    )
    expected = python.run(["meter.out"])["meter.out"].data.to_value(u.W)
    np.testing.assert_allclose(result.values, expected)


# --- Origin, archives, operation --------------------------------------------------------------------------


def test_actions_are_traced_with_their_origin(service, session):
    ctx = service.create_context(session).id
    service.add_element(session, ctx, "lumos.lab.LinearStage", name="stage")
    service.execute_action(session, ctx, "stage", "home", confirmed=True)
    entries = service.journal(session, ctx, kind="action")
    assert entries[-1]["origin"] == "mcp"
    assert entries[-1]["action"] == "home"


def test_archives_are_listed(service, session, tmp_path):
    ctx = laser_bench(service, session)
    service.wait_job(session, service.run(session, ctx, ["meter.out"], archive="service-run").id, timeout=60)
    archives = service.list_archives()
    assert len(archives) == 1
    metadata = service.read_archive_metadata(archives[0].path)
    assert metadata["origin"] == "mcp"


def test_operation_is_disabled_by_default(service, session):
    ctx = service.create_context(session, mode="operation").id
    service.add_element(session, ctx, "lumos.lab.LinearStage", name="stage")
    with pytest.raises(ServiceError, match="disabled"):
        service.execute_action(session, ctx, "stage", "home", confirmed=True)
    service.stop_all(session, ctx)  # the emergency stop is always allowed


def test_operation_needs_a_confirmation(tmp_path):
    from lumos.hardware.emulators import EmulatorServer, create_emulator

    server = EmulatorServer(create_emulator("LL-LS100", latency=0.0)).start()
    try:
        service = Service(ServiceConfig(operation_enabled=True))
        session = service.create_session(origin="mcp").id
        ctx = service.create_context(session, mode="operation").id
        service.add_element(
            session,
            ctx,
            "lumos.lab.LinearStage",
            name="stage",
            params={"velocity": "50 mm/s", "acceleration": "500 mm/s2"},
        )
        devices = service.discover_devices(tcp=[server.url])
        assert devices[0].model == "LL-LS100"
        service.associate_devices(session, ctx)
        with pytest.raises(ServiceError, match="confirm"):
            service.execute_action(session, ctx, "stage", "home", confirmed=False)
        service.execute_action(session, ctx, "stage", "home", confirmed=True)
        result = service.execute_action(session, ctx, "stage", "move", args=["5 mm"], confirmed=True)
        assert result.measured_s > 0
        status = service.device_status(session, ctx, "stage")
        assert status["model"] == "LL-LS100"
        with pytest.raises(ServiceError, match="limits"):
            service.execute_action(session, ctx, "stage", "move", args=["500 mm"], confirmed=True)
        assert server.emulator.position == pytest.approx(5)
        service.stop_all(session, ctx)
        assert server.emulator.stopped
    finally:
        server.stop()
        time.sleep(0.05)
