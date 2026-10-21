"""No code runs on the public instance, by any route (step P15.4, ADR-0024 §6, ADR-0025 §5, NF-7).

Requirements (docs/specs/pistes_evolution.md §2-3, decision of 20/10): the code zones (procedures written as
Python scripts, components with code) run only on local and private instances. On a public instance open to
anonymous visitors they are refused through every route: the service, the web API, the MCP server, studies
on procedures and documents opened by a visitor; such an instance cannot even be configured to run them.
Procedures keep the quotas of the instance (time of a run, runs at a time).
"""

import json

import pytest

from lumos.service import Service, ServiceConfig, ServiceError

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from lumos.web import WebConfig, create_app
from lumos.web.config import ConfigError

HOST = "https://lumos.example"
SCRIPT = {"name": "script", "script": "import os\nresults['pid'] = os.getpid()\n"}
CODE = """
from lumos.core.registry import register
from lumos.core.element import Element

@register("plugins.attacker.Probe")
class Probe(Element):
    '''Probe.'''
"""


def with_script():
    """A document whose context holds a script procedure (as a visitor could upload)."""
    import lumos.io
    from lumos import Context
    from lumos.core.procedure import Procedure

    ctx = Context(name="trap")
    ctx.procedures["script"] = Procedure.from_json(SCRIPT)
    return lumos.io.to_document(ctx)


def public_config(tmp_path, **service):
    return WebConfig(
        profile="public",
        allowed_hosts=["lumos.example"],
        trust_proxy=True,
        audit_log=str(tmp_path / "audit.log"),
        service=ServiceConfig(archive_root=str(tmp_path / "archives"), **service),
    )


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(public_config(tmp_path)), base_url=HOST) as client:
        assert client.get("/api/info").json()["scripts"] is False
        yield client


def test_a_public_instance_open_to_visitors_cannot_be_configured_to_run_code(tmp_path):
    with pytest.raises(ConfigError, match="scripts"):
        public_config(tmp_path, scripts_enabled=True).check()


def test_the_web_api_refuses_code_components(client):
    response = client.post("/api/components/code", json={"source": CODE})
    assert response.status_code == 403
    assert "plugins.attacker.Probe" not in json.dumps(client.get("/api/palette").json())


def test_the_web_api_refuses_to_run_a_script_procedure(client):
    cid = client.post("/api/contexts", json={}).json()["id"]
    saved = client.put(f"/api/contexts/{cid}/procedures/script", json=SCRIPT)
    assert saved.status_code == 200
    assert any("script" in problem for problem in saved.json()["problems"])
    run = client.post(f"/api/contexts/{cid}/procedures/script/run", json={"confirmed": True})
    assert run.status_code >= 400
    assert "script" in run.text


def test_an_opened_document_does_not_run_its_script(client):
    opened = client.post("/api/contexts/import", content=json.dumps(with_script()))
    assert opened.status_code == 201, opened.text
    cid = opened.json()["id"]
    run = client.post(f"/api/contexts/{cid}/procedures/script/run", json={"confirmed": True})
    assert run.status_code >= 400


def test_a_study_on_a_script_procedure_is_refused(tmp_path):
    service = Service(ServiceConfig(archive_root=str(tmp_path), scripts_enabled=False))
    session = service.create_session().id
    ctx = service.create_context(session).id
    service.set_procedure(session, ctx, SCRIPT)
    with pytest.raises(ServiceError, match="script"):
        service.run_study(session, ctx, kind="sweep", parameters={}, procedure="script")
    service.shutdown()


def test_the_mcp_server_refuses_scripts(tmp_path):
    pytest.importorskip("mcp")
    import asyncio

    from mcp import Client

    from lumos.mcp.server import create_server

    service = Service(ServiceConfig(archive_root=str(tmp_path), scripts_enabled=False))
    server = create_server(service)

    async def scenario():
        async with Client(server) as mcp:
            tools = {tool.name for tool in (await mcp.list_tools()).tools}
            assert not any("code" in name for name in tools)  # no tool runs Python code
            await mcp.call_tool("create_context", {"name": "bench"})
            await mcp.call_tool("set_procedure", {"procedure": SCRIPT})
            result = await mcp.call_tool("run_procedure", {"name": "script"})
            return result

    result = asyncio.run(scenario())
    assert result.is_error
    assert "script" in " ".join(getattr(block, "text", "") for block in result.content)
    service.shutdown()


def test_procedures_keep_the_time_quota_of_the_instance(tmp_path):
    service = Service(ServiceConfig(archive_root=str(tmp_path), max_run_seconds=0.2))
    session = service.create_session().id
    ctx = service.create_context(session).id
    service.add_element(session, ctx, "lumos.sources.GaussianLaser", name="laser")
    service.add_element(session, ctx, "lumos.detectors.PowerMeter", name="meter")
    service.connect(session, ctx, "laser.out", "meter.in")
    long = {
        "name": "long",
        "steps": [
            {
                "do": "repeat",
                "times": 10000,
                "steps": [{"do": "measure", "outputs": ["meter.out"], "as": "p"}],
            }
        ],
    }
    service.set_procedure(session, ctx, long)
    job = service.run_procedure(session, ctx, "long")
    done = service.wait_job(session, job.id, timeout=60)
    assert done.status == "failed"
    assert "exceeded" in (done.error or "")
    service.shutdown()
