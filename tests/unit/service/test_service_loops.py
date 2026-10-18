"""Feedback loops in the service layer, the web API and the MCP server (step P14.14, ADR-0020).

Requirements: feedbacks are added, listed in the description of a context, removed and undone like any edit,
and saved in its ``.lumos`` document; a loop runs as a job (kind ``loop``) whose steps are read like the
points of a study (parameter ``time``); the adaptive optics preset closes its loop and its residual decreases.
The same operations exist over HTTP and as MCP tools.

Expected behaviour: in closed loop, an integrator of gain 0.5 on a turbulent wavefront lowers the residual
optical path by more than half within a few frames (Hardy 1998, §8.2; see tests/physics/test_adaptive_optics).
"""

import json
import time

import pytest

from lumos.service import Service, ServiceConfig, ServiceError


@pytest.fixture
def service(tmp_path):
    service = Service(ServiceConfig(archive_root=str(tmp_path / "archives")))
    yield service
    service.shutdown()


@pytest.fixture
def session(service):
    return service.create_session(origin="web").id


def residuals(study):
    return [p.outputs["wfs.residual"].values[0] for p in study.points]


def test_the_adaptive_optics_preset_closes_its_loop(service, session):
    ctx = service.load_preset(session, "adaptive_optics")
    assert [(f.source, f.target, f.mode) for f in ctx.feedbacks] == [("ctrl.commands", "dm.commands", "add")]
    job = service.run_loop(session, ctx.id, steps=10, period="2 ms", outputs=["wfs.residual"])
    assert job.kind == "loop"
    job = service.wait_job(session, job.id, timeout=300)
    assert job.status == "done", job.error
    study = service.get_study(session, job.id)
    assert study.kind == "loop"
    assert study.parameters == ["time"]
    assert [p.params["time"] for p in study.points[:3]] == ["0 s", "0.002 s", "0.004 s"]
    values = residuals(study)
    assert values[-1] < 0.5 * values[0]
    assert service.study_csv(session, job.id).splitlines()[0].startswith("point,realization,time [s]")
    # the loop has run: the mirror keeps its commands, the clock has advanced; undo puts them back
    dm = next(e for e in service.describe_context(session, ctx.id).elements if e.id == "dm")
    assert dm.state["commands"] != "0 nm"
    service.undo(session, ctx.id)
    dm = next(e for e in service.describe_context(session, ctx.id).elements if e.id == "dm")
    assert dm.state["commands"] in ("0 nm", "[0] nm")


def test_feedbacks_are_edited_saved_and_undone(service, session):
    ctx = service.load_preset(session, "adaptive_optics").id
    summary = service.remove_feedback(session, ctx, "ctrl.commands", "dm.commands")
    assert summary.feedbacks == []
    assert json.loads(service.export_lumos(session, ctx)).get("feedbacks", []) == []
    assert service.undo(session, ctx).feedbacks[0].target == "dm.commands"
    with pytest.raises(ServiceError) as error:
        service.add_feedback(session, ctx, "ctrl.commands", "dm.commands", "add")
    assert "already" in error.value.message
    with pytest.raises(ServiceError):
        service.add_feedback(session, ctx, "ctrl.commands", "dm.nothing", "add")
    with pytest.raises(ServiceError):
        service.remove_feedback(session, ctx, "wfs.slopes", "dm.commands")
    with pytest.raises(ServiceError):
        service.run_loop(session, ctx, steps=0, period="1 ms", outputs=["wfs.residual"])
    with pytest.raises(ServiceError):
        service.run_loop(session, ctx, steps=3, period="1 m", outputs=["wfs.residual"])


def test_a_loop_needs_a_feedback_and_respects_the_quota(tmp_path):
    service = Service(ServiceConfig(max_study_points=5))
    session = service.create_session().id
    ctx = service.load_preset(session, "adaptive_optics").id
    with pytest.raises(ServiceError) as error:
        service.run_loop(session, ctx, steps=6, period="1 ms", outputs=["wfs.residual"])
    assert error.value.code == "quota"
    service.remove_feedback(session, ctx, "ctrl.commands", "dm.commands")
    with pytest.raises(ServiceError) as error:
        service.run_loop(session, ctx, steps=3, period="1 ms", outputs=["wfs.residual"])
    assert "feedback" in error.value.hint
    service.shutdown()


# --- Web API and MCP --------------------------------------------------------------------------------------


def test_loops_over_http(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from lumos.web import WebConfig, create_app

    with TestClient(create_app(WebConfig(archive_root=str(tmp_path))), base_url="http://localhost") as client:
        cid = client.post("/api/presets/adaptive_optics/load").json()["id"]
        removed = client.request(
            "DELETE",
            f"/api/contexts/{cid}/feedbacks",
            params={"source": "ctrl.commands", "target": "dm.commands"},
        )
        assert removed.json()["feedbacks"] == []
        added = client.post(
            f"/api/contexts/{cid}/feedbacks",
            json={"source": "ctrl.commands", "target": "dm.commands", "mode": "add"},
        )
        assert added.status_code == 201
        assert added.json()["feedbacks"] == [
            {"source": "ctrl.commands", "target": "dm.commands", "mode": "add"}
        ]
        job = client.post(
            f"/api/contexts/{cid}/loops", json={"steps": 3, "period": "2 ms", "outputs": ["wfs.residual"]}
        )
        assert job.status_code == 202
        deadline = time.monotonic() + 120
        while (info := client.get(f"/api/jobs/{job.json()['id']}").json())["status"] in ("queued", "running"):
            assert time.monotonic() < deadline
            time.sleep(0.1)
        assert info["kind"] == "loop"
        assert len(client.get(f"/api/jobs/{info['id']}/study").json()["points"]) == 3


def test_loops_through_mcp():
    pytest.importorskip("mcp")
    import anyio
    from mcp import Client

    from lumos.mcp import create_server

    service = Service()
    server = create_server(service)

    async def scenario():
        async with Client(server) as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            assert {"add_feedback", "remove_feedback", "run_loop"} <= names
            ctx = (await client.call_tool("load_preset", {"name": "adaptive_optics"})).structured_content[
                "id"
            ]
            await client.call_tool(
                "remove_feedback", {"source": "ctrl.commands", "target": "dm.commands", "context": ctx}
            )
            added = await client.call_tool(
                "add_feedback",
                {"source": "ctrl.commands", "target": "dm.commands", "mode": "add", "context": ctx},
            )
            assert not added.is_error
            result = await client.call_tool(
                "run_loop", {"steps": 3, "period": "2 ms", "outputs": ["wfs.residual"]}
            )
            assert result.structured_content["status"] == "done"
            study = await client.call_tool("get_study", {"job": result.structured_content["id"]})
            assert "time [s]" in study.content[0].text

    anyio.run(scenario)
    service.shutdown()
