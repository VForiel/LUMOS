"""Procedures through the service, the web API and the MCP server (step P15.11, ADR-0024).

Requirements: a procedure of a context is created, listed, replaced and removed; it runs as a job of kind
``procedure`` (progress, cancellation, journal, quotas) whose outputs are its named results; its problems are
reported before it runs and by the validation of the context; a procedure acting on equipment needs one human
confirmation. The web API and the MCP server offer the same operations.

Expected values: a power meter behind a laser reads the power of the laser (truncation of the beam by the grid
below 1e-3 with a 0.25 mm pixel and 32 pixels).
"""

import anyio
import pytest

from lumos.service import Service, ServiceConfig, ServiceError

POWERS = {
    "name": "powers",
    "steps": [
        {
            "do": "for_each",
            "values": ["1 mW", "3 mW"],
            "as": "power",
            "steps": [
                {"do": "set", "target": "laser.power", "value": "$power"},
                {"do": "measure", "outputs": ["meter.out"], "as": "readings", "collect": True},
            ],
        },
        {"do": "measure", "outputs": ["meter.out"], "as": "last"},
        {"do": "show", "results": ["last"]},
    ],
}


@pytest.fixture
def setup(tmp_path):
    service = Service(ServiceConfig(archive_root=str(tmp_path / "archives")))
    session = service.create_session().id
    ctx = service.create_context(session).id
    service.add_element(
        session,
        ctx,
        "lumos.sources.GaussianLaser",
        name="laser",
        params={"pixel_scale": "0.25 mm", "n_pixels": 32},
    )
    service.add_element(session, ctx, "lumos.detectors.PowerMeter", name="meter")
    service.connect(session, ctx, "laser.out", "meter.in")
    yield service, session, ctx
    service.shutdown()


def test_procedures_are_created_listed_and_removed(setup):
    service, session, ctx = setup
    info = service.set_procedure(session, ctx, POWERS)
    assert info.name == "powers"
    assert info.steps == 3
    assert info.problems == []
    assert [p.name for p in service.list_procedures(session, ctx)] == ["powers"]
    assert service.get_procedure(session, ctx, "powers")["steps"][0]["do"] == "for_each"
    service.remove_procedure(session, ctx, "powers")
    assert service.list_procedures(session, ctx) == []


def test_a_procedure_runs_as_a_job_whose_outputs_are_its_results(setup):
    service, session, ctx = setup
    service.set_procedure(session, ctx, POWERS)
    job = service.run_procedure(session, ctx, "powers")
    assert job.kind == "procedure"
    done = service.wait_job(session, job.id, timeout=60)
    assert done.status == "done", done.message
    assert set(done.outputs) == {"readings", "last"}
    last = service.get_result(session, job.id, "last")
    assert last.values[0] == pytest.approx(3e-3, rel=1e-3)
    readings = service.get_result(session, job.id, "readings")
    assert readings.kind == "list"
    assert readings.shape == [2]
    entries = service.journal(session, ctx)
    assert any(entry["kind"] == "procedure" for entry in entries)


def test_problems_are_reported_before_running_and_by_the_validation(setup):
    service, session, ctx = setup
    wrong = {"name": "wrong", "steps": [{"do": "action", "element": "nowhere", "action": "close"}]}
    info = service.set_procedure(session, ctx, wrong)
    assert any("nowhere" in p for p in info.problems)
    with pytest.raises(ServiceError, match="nowhere"):
        service.run_procedure(session, ctx, "wrong")
    issues = service.validate_context(session, ctx)
    assert any("wrong" in i.message and "nowhere" in i.message for i in issues)


def test_an_input_fed_by_a_procedure_is_not_reported_as_unconnected(setup):
    service, session, ctx = setup
    service.add_element(session, ctx, "lumos.processing.DarkSubtraction", name="subtract")
    service.connect(session, ctx, "meter.out", "subtract.in")
    feeding = {
        "name": "feed",
        "steps": [
            {"do": "measure", "outputs": ["meter.out"], "as": "dark"},
            {"do": "use", "result": "dark", "port": "subtract.dark"},
            {"do": "measure", "outputs": ["subtract.out"], "as": "zero"},
        ],
    }
    service.set_procedure(session, ctx, feeding)
    issues = service.validate_context(session, ctx)
    assert not [i for i in issues if i.severity == "error" and "subtract.dark" in i.message]


def test_a_long_procedure_is_cancelled(setup):
    service, session, ctx = setup
    slow = {
        "name": "slow",
        "steps": [
            {"do": "repeat", "times": 5000, "steps": [{"do": "measure", "outputs": ["meter.out"], "as": "x"}]}
        ],
    }
    service.set_procedure(session, ctx, slow)
    job = service.run_procedure(session, ctx, "slow")
    service.cancel_job(session, job.id)
    done = service.wait_job(session, job.id, timeout=60)
    assert done.status == "cancelled"


def test_procedures_are_in_the_exported_document(setup):
    service, session, ctx = setup
    service.set_procedure(session, ctx, POWERS)
    text = service.export_lumos(session, ctx)
    reloaded = service.import_lumos(session, text)
    assert [p.name for p in service.list_procedures(session, reloaded.id)] == ["powers"]


def test_the_web_api(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from lumos.web import WebConfig, create_app

    with TestClient(create_app(WebConfig(archive_root=str(tmp_path))), base_url="http://localhost") as client:
        client.get("/api/info")
        cid = client.post("/api/contexts", json={"name": "p"}).json()["id"]
        client.post(
            f"/api/contexts/{cid}/elements",
            json={"type_id": "lumos.sources.GaussianLaser", "name": "laser", "params": {"n_pixels": 32}},
        )
        client.post(
            f"/api/contexts/{cid}/elements", json={"type_id": "lumos.detectors.PowerMeter", "name": "meter"}
        )
        client.post(f"/api/contexts/{cid}/connections", json={"source": "laser.out", "target": "meter.in"})
        assert client.put(f"/api/contexts/{cid}/procedures/powers", json=POWERS).status_code == 200
        assert [p["name"] for p in client.get(f"/api/contexts/{cid}/procedures").json()] == ["powers"]
        job = client.post(f"/api/contexts/{cid}/procedures/powers/run", json={}).json()
        assert job["kind"] == "procedure"
        assert client.delete(f"/api/contexts/{cid}/procedures/powers").status_code == 204


def test_the_mcp_tools(tmp_path):
    pytest.importorskip("mcp")
    from mcp import Client

    from lumos.mcp import create_server

    service = Service(ServiceConfig(archive_root=str(tmp_path / "archives")))
    server = create_server(service)

    def data(result):
        assert not result.is_error, result.content
        content = result.structured_content
        return content["result"] if set(content) == {"result"} else content

    async def scenario(client):
        names = {tool.name for tool in (await client.list_tools()).tools}
        assert {"set_procedure", "list_procedures", "run_procedure", "remove_procedure"} <= names
        ctx = data(await client.call_tool("create_context", {"name": "p"}))["id"]
        for type_id, name, params in (
            ("lumos.sources.GaussianLaser", "laser", {"n_pixels": 32, "pixel_scale": "0.25 mm"}),
            ("lumos.detectors.PowerMeter", "meter", {}),
        ):
            await client.call_tool(
                "add_element", {"context": ctx, "type_id": type_id, "name": name, "params": params}
            )
        await client.call_tool("connect", {"context": ctx, "source": "laser.out", "target": "meter.in"})
        data(await client.call_tool("set_procedure", {"context": ctx, "procedure": POWERS}))
        job = data(await client.call_tool("run_procedure", {"context": ctx, "name": "powers"}))
        return job

    async def main():
        async with Client(server) as client:
            return await scenario(client)

    try:
        job = anyio.run(main)
        assert job["status"] == "done"
        assert "last" in job["outputs"]
    finally:
        service.shutdown()
