"""Studies on procedures through the service and the MCP server (step P15.13, ADR-0024 §7).

Requirements: a sweep or a Monte-Carlo run takes a procedure as its body; the outputs of its points are the
named results of the procedure; the points run in parallel on the workers of the service.

Expected values: the power read by a power meter is proportional to the power of the laser.
"""

import anyio
import pytest

from lumos.service import Service, ServiceConfig

PROCEDURE = {
    "name": "measure",
    "steps": [
        {"do": "measure", "outputs": ["meter.out"], "as": "reading"},
        {"do": "process", "function": "total", "args": {"product": "$reading"}, "as": "watts"},
    ],
}


def bench(service):
    session = service.create_session().id
    ctx = service.create_context(session).id
    service.add_element(
        session,
        ctx,
        "lumos.sources.GaussianLaser",
        name="laser",
        params={"n_pixels": 32, "pixel_scale": "0.25 mm"},
    )
    service.add_element(session, ctx, "lumos.detectors.PowerMeter", name="meter")
    service.connect(session, ctx, "laser.out", "meter.in")
    service.set_procedure(session, ctx, PROCEDURE)
    return session, ctx


def test_a_sweep_on_a_procedure(tmp_path):
    service = Service(ServiceConfig(archive_root=str(tmp_path), max_workers=3))
    session, ctx = bench(service)
    job = service.run_study(
        session, ctx, kind="sweep", parameters={"laser.power": ["1 mW", "2 mW", "4 mW"]}, procedure="measure"
    )
    done = service.wait_job(session, job.id, timeout=120)
    assert done.status == "done", done.message
    study = service.get_study(session, job.id)
    assert set(study.outputs) == {"reading", "watts"}
    watts = [p.outputs["watts"].values[0] for p in study.points]
    assert watts[1] / watts[0] == pytest.approx(2, rel=1e-9)
    assert watts[2] / watts[0] == pytest.approx(4, rel=1e-9)
    service.shutdown()


def test_the_mcp_tool_runs_a_sweep_on_a_procedure(tmp_path):
    pytest.importorskip("mcp")
    from mcp import Client

    from lumos.mcp import create_server

    service = Service(ServiceConfig(archive_root=str(tmp_path)))  # the MCP server builds its own bench

    async def scenario(client):
        def data(result):
            assert not result.is_error, result.content
            content = result.structured_content
            return content["result"] if set(content) == {"result"} else content

        cid = data(await client.call_tool("create_context", {"name": "p"}))["id"]
        for type_id, name, params in (
            ("lumos.sources.GaussianLaser", "laser", {"n_pixels": 32, "pixel_scale": "0.25 mm"}),
            ("lumos.detectors.PowerMeter", "meter", {}),
        ):
            await client.call_tool(
                "add_element", {"context": cid, "type_id": type_id, "name": name, "params": params}
            )
        await client.call_tool("connect", {"context": cid, "source": "laser.out", "target": "meter.in"})
        data(await client.call_tool("set_procedure", {"context": cid, "procedure": PROCEDURE}))
        return data(
            await client.call_tool(
                "run_sweep",
                {"context": cid, "parameters": {"laser.power": ["1 mW", "2 mW"]}, "procedure": "measure"},
            )
        )

    async def main():
        async with Client(create_server(service)) as client:
            return await scenario(client)

    try:
        job = anyio.run(main)
        assert job["status"] == "done"
        assert set(job["outputs"]) == {"reading", "watts"}
    finally:
        service.shutdown()
