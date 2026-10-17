"""Tests for the MCP server over stdio (step P13.2, REQ-MCP-1, REQ-MCP-2).

Requirements: an MCP server (``lumos.mcp``, official MCP Python SDK) usable over stdio (``lumos mcp``), whose
tools cover discovery (``list_element_types``, ``get_element_schema``, ``search_docs``), construction
(``create_context``, ``add_element``, ``connect``, ``set_param``, ``create_group``, ``load_preset``,
``import_lumos``, ``export_lumos``), verification (``validate_context``, ``explain_sampling``,
``plan_propagation``), execution (``run`` with progress, ``get_result``, ``inspect_port``), catalogues
(``resolve_object``), processing (``build_pipeline``, ``evaluate_detection``), results (``list_archives``,
``read_archive_metadata``) and code (``export_python``). Tests of the plan: every tool is called through an
MCP test client; parity between the service, MCP and the Python API (same ``.lumos`` and same result).

The expected numbers are physical: a power meter behind a 2 mW laser reads 2 mW (energy conservation).
"""

import json
import sys

import anyio
import numpy as np
import pytest
from astropy import units as u

pytest.importorskip("mcp")

from mcp import Client, StdioServerParameters

import lumos.io
from lumos import Context
from lumos.elements.detectors import PowerMeter
from lumos.elements.sources import GaussianLaser
from lumos.mcp import create_server
from lumos.service import Service, ServiceConfig

REQUIRED_TOOLS = {
    "list_element_types",
    "get_element_schema",
    "search_docs",
    "create_context",
    "list_contexts",
    "describe_context",
    "add_element",
    "remove_element",
    "connect",
    "disconnect",
    "set_param",
    "set_observation",
    "create_group",
    "list_presets",
    "load_preset",
    "import_lumos",
    "export_lumos",
    "validate_context",
    "explain_sampling",
    "plan_propagation",
    "run",
    "get_job",
    "get_result",
    "inspect_port",
    "resolve_object",
    "build_pipeline",
    "evaluate_detection",
    "list_archives",
    "read_archive_metadata",
    "export_python",
}


@pytest.fixture
def service(tmp_path):
    service = Service(ServiceConfig(archive_root=str(tmp_path / "archives")))
    yield service
    service.shutdown()


@pytest.fixture
def server(service):
    return create_server(service)


def session_of(server):
    """Run ``scenario(client)`` against ``server`` through an in-process MCP client."""

    def run(scenario, **options):
        async def main():
            async with Client(server, **options) as client:
                return await scenario(client)

        return anyio.run(main)

    return run


def data(result):
    """Structured content of a successful tool result."""
    assert not result.is_error, result.content
    content = result.structured_content
    if content is None:
        return json.loads(result.content[0].text)
    if set(content) == {"result"}:
        return content["result"]
    return content


def text(result):
    return " ".join(getattr(block, "text", "") for block in result.content)


async def laser_bench(client):
    created = data(await client.call_tool("create_context", {"name": "bench"}))
    ctx = created["id"]
    await client.call_tool(
        "add_element",
        {
            "context": ctx,
            "type_id": "lumos.sources.GaussianLaser",
            "name": "laser",
            "params": {"power": "2 mW"},
        },
    )
    await client.call_tool(
        "add_element", {"context": ctx, "type_id": "lumos.detectors.PowerMeter", "name": "meter"}
    )
    data(await client.call_tool("connect", {"context": ctx, "source": "laser.out", "target": "meter.in"}))
    return ctx


# --- Listing ----------------------------------------------------------------------------------------------


def test_the_server_lists_the_required_tools(server):
    async def scenario(client):
        return await client.list_tools()

    tools = session_of(server)(scenario).tools
    names = {tool.name for tool in tools}
    assert REQUIRED_TOOLS <= names
    for tool in tools:
        assert tool.description, f"{tool.name} has no description"
        assert tool.input_schema["type"] == "object"


def test_the_server_has_instructions(server):
    async def scenario(client):
        return client.instructions

    instructions = session_of(server)(scenario)
    assert instructions
    assert "unit" in instructions.lower()


# --- Discovery --------------------------------------------------------------------------------------------


def test_discovery_tools(server):
    async def scenario(client):
        types = data(await client.call_tool("list_element_types", {"query": "laser"}))
        schema = data(
            await client.call_tool("get_element_schema", {"type_id": "lumos.sources.GaussianLaser"})
        )
        docs = data(await client.call_tool("search_docs", {"query": "emergency stop"}))
        return types, schema, docs

    types, schema, docs = session_of(server)(scenario)
    assert "lumos.sources.GaussianLaser" in [item["type_id"] for item in types["items"]]
    assert schema["properties"]["params"]["properties"]["power"]["x-lumos-unit"] == "mW"
    assert docs
    assert all(hit["excerpt"] for hit in docs)


# --- Construction -----------------------------------------------------------------------------------------


def test_construction_tools(server):
    async def scenario(client):
        ctx = await laser_bench(client)
        await client.call_tool("set_param", {"context": ctx, "path": "laser.waist", "value": "0.5 mm"})
        described = data(await client.call_tool("describe_context", {"context": ctx}))
        contexts = data(await client.call_tool("list_contexts", {}))
        return ctx, described, contexts

    ctx, described, contexts = session_of(server)(scenario)
    laser = next(e for e in described["elements"] if e["id"] == "laser")
    assert laser["params"]["waist"] == "0.5 mm"
    assert {"source": "laser.out", "target": "meter.in"} in described["connections"]
    assert ctx in [c["id"] for c in contexts]


def test_the_context_defaults_to_the_current_one(server):
    """After create_context or load_preset, the other tools work on that context without naming it."""

    async def scenario(client):
        await client.call_tool("create_context", {"name": "bench"})
        await client.call_tool(
            "add_element",
            {"type_id": "lumos.sources.GaussianLaser", "name": "laser", "params": {"power": "2 mW"}},
        )
        await client.call_tool("add_element", {"type_id": "lumos.detectors.PowerMeter", "name": "meter"})
        await client.call_tool("connect", {"source": "laser.out", "target": "meter.in"})
        return data(await client.call_tool("describe_context", {}))

    described = session_of(server)(scenario)
    assert described["name"] == "bench"
    assert {e["id"] for e in described["elements"]} == {"laser", "meter"}


def test_editing_tools_remove_and_disconnect(server):
    async def scenario(client):
        ctx = await laser_bench(client)
        data(
            await client.call_tool(
                "disconnect", {"context": ctx, "source": "laser.out", "target": "meter.in"}
            )
        )
        data(await client.call_tool("remove_element", {"context": ctx, "element": "meter"}))
        data(
            await client.call_tool("set_observation", {"context": ctx, "minimum": "1 um", "maximum": "2 um"})
        )
        return data(await client.call_tool("describe_context", {"context": ctx}))

    described = session_of(server)(scenario)
    assert [e["id"] for e in described["elements"]] == ["laser"]
    assert described["connections"] == []
    assert described["band"] == ["1 um", "2 um"]


def test_create_group(server):
    """A group is built from its elements, connections and exposed ports, then used as one element."""

    async def scenario(client):
        ctx = data(await client.call_tool("create_context", {"name": "grouped"}))["id"]
        group = data(
            await client.call_tool(
                "create_group",
                {
                    "context": ctx,
                    "name": "source",
                    "elements": [
                        {"type": "lumos.sources.GaussianLaser", "name": "laser", "params": {"power": "2 mW"}},
                    ],
                    "connections": [],
                    "expose": {"out": "laser.out"},
                },
            )
        )
        await client.call_tool(
            "add_element", {"context": ctx, "type_id": "lumos.detectors.PowerMeter", "name": "meter"}
        )
        data(
            await client.call_tool("connect", {"context": ctx, "source": "source.out", "target": "meter.in"})
        )
        job = data(await client.call_tool("run", {"context": ctx, "outputs": ["meter.out"]}))
        result = data(await client.call_tool("get_result", {"job": job["id"], "port": "meter.out"}))
        return group, result

    group, result = session_of(server)(scenario)
    assert group["id"] == "source"
    assert group["outputs"] == ["out"]
    assert result["values"][0] == pytest.approx(2e-3, rel=1e-3)


def test_presets_and_documents(server):
    async def scenario(client):
        presets = data(await client.call_tool("list_presets", {}))
        loaded = data(await client.call_tool("load_preset", {"name": "first_simulation"}))
        document = data(await client.call_tool("export_lumos", {"context": loaded["id"]}))
        again = data(await client.call_tool("import_lumos", {"document": document}))
        code = data(await client.call_tool("export_python", {"context": loaded["id"]}))
        return presets, loaded, again, code

    presets, loaded, again, code = session_of(server)(scenario)
    assert "coronagraph" in [p["name"] for p in presets]
    assert {e["id"] for e in again["elements"]} == {e["id"] for e in loaded["elements"]}
    assert "GaussianLaser" in code


# --- Verification -----------------------------------------------------------------------------------------


def test_verification_tools(server):
    async def scenario(client):
        ctx = data(await client.call_tool("create_context", {}))["id"]
        await client.call_tool(
            "add_element", {"context": ctx, "type_id": "lumos.detectors.PowerMeter", "name": "meter"}
        )
        issues = data(await client.call_tool("validate_context", {"context": ctx}))
        loaded = data(await client.call_tool("load_preset", {"name": "propagation_methods"}))
        free_space = next(e["id"] for e in loaded["elements"] if e["type_id"] == "lumos.optics.FreeSpace")
        explained = data(
            await client.call_tool("explain_sampling", {"context": loaded["id"], "element": free_space})
        )
        planned = data(
            await client.call_tool("plan_propagation", {"context": loaded["id"], "element": free_space})
        )
        return issues, explained, planned

    issues, explained, planned = session_of(server)(scenario)
    assert any("not connected" in issue["message"] and issue["hint"] for issue in issues)
    assert explained["method"]
    assert explained["reason"]
    assert planned["method"] == explained["method"]


# --- Execution --------------------------------------------------------------------------------------------


def test_run_reports_progress_and_returns_the_result(server):
    progress = []

    async def scenario(client):
        ctx = await laser_bench(client)

        async def on_progress(value, total, message):
            progress.append((value, total, message))

        job = data(
            await client.call_tool(
                "run", {"context": ctx, "outputs": ["meter.out"]}, progress_callback=on_progress
            )
        )
        status = data(await client.call_tool("get_job", {"job": job["id"]}))
        result = data(await client.call_tool("get_result", {"job": job["id"], "port": "meter.out"}))
        return job, status, result

    job, status, result = session_of(server)(scenario)
    assert job["status"] == "done"
    assert status["status"] == "done"
    assert result["unit"] == "W"
    assert result["values"][0] == pytest.approx(2e-3, rel=1e-3)
    assert progress
    assert progress[-1][0] == pytest.approx(progress[-1][1])  # ends at 100 %


def test_run_defaults_to_the_outputs_of_a_preset(server):
    async def scenario(client):
        loaded = data(await client.call_tool("load_preset", {"name": "first_simulation"}))
        job = data(await client.call_tool("run", {}))
        return loaded, job

    loaded, job = session_of(server)(scenario)
    assert job["status"] == "done"
    assert job["outputs"] == loaded["outputs"]


def test_inspect_port(server):
    async def scenario(client):
        await client.call_tool("load_preset", {"name": "first_simulation"})
        return data(await client.call_tool("inspect_port", {"port": "focus.out"}))

    summary = session_of(server)(scenario)
    assert summary["kind"] == "Wavefront"
    assert "wavelength" in summary["details"]


def test_errors_are_tool_errors_with_a_hint(server):
    async def scenario(client):
        ctx = await laser_bench(client)
        wrong_unit = await client.call_tool(
            "set_param", {"context": ctx, "path": "laser.waist", "value": "3 kg"}
        )
        wrong_type = await client.call_tool("get_element_schema", {"type_id": "lumos.sources.Laser"})
        return wrong_unit, wrong_type

    wrong_unit, wrong_type = session_of(server)(scenario)
    assert wrong_unit.is_error
    assert "hint" in text(wrong_unit).lower()
    assert wrong_type.is_error
    assert "GaussianLaser" in text(wrong_type)  # the closest type is suggested


# --- Processing, catalogues, archives ---------------------------------------------------------------------


def test_build_pipeline_and_archives(server, service):
    async def scenario(client):
        loaded = data(await client.call_tool("load_preset", {"name": "first_simulation"}))
        steps = data(
            await client.call_tool(
                "build_pipeline",
                {
                    "context": loaded["id"],
                    "source": "camera.out",
                    "steps": [
                        {"type": "lumos.processing.Crop", "name": "cropped", "params": {"nx": 16, "ny": 16}}
                    ],
                },
            )
        )
        job = data(
            await client.call_tool(
                "run", {"context": loaded["id"], "outputs": ["cropped.out"], "archive": "mcp-run"}
            )
        )
        archives = data(await client.call_tool("list_archives", {}))
        metadata = data(await client.call_tool("read_archive_metadata", {"path": archives[0]["path"]}))
        return steps, job, archives, metadata

    steps, job, archives, metadata = session_of(server)(scenario)
    assert steps[0]["id"] == "cropped"
    assert job["status"] == "done"
    assert job["archive"]
    assert len(archives) == 1
    assert metadata["origin"] == "mcp"


def test_evaluate_detection(server):
    async def scenario(client):
        loaded = data(await client.call_tool("load_preset", {"name": "first_simulation"}))
        return data(
            await client.call_tool(
                "evaluate_detection",
                {"context": loaded["id"], "port": "camera.out", "peak": "4e6 adu", "resolution": "20 um"},
            )
        )

    curve = session_of(server)(scenario)
    assert len(curve["separation"]) == len(curve["contrast"]) > 0
    assert curve["sigma"] == 5


def test_resolve_object_reports_unknown_names_as_errors(server, monkeypatch):
    import lumos.catalogs

    def unknown(name, **_):
        raise LookupError(f"{name} not found")

    monkeypatch.setattr(lumos.catalogs, "resolve", unknown)

    async def scenario(client):
        return await client.call_tool("resolve_object", {"name": "Nowhere"})

    result = session_of(server)(scenario)
    assert result.is_error
    assert "Nowhere" in text(result)


# --- Parity -----------------------------------------------------------------------------------------------


def test_parity_between_mcp_service_and_python(server, service):
    """The same bench built through MCP, the service and Python gives the same .lumos and the same reading."""

    async def scenario(client):
        ctx = await laser_bench(client)
        document = json.loads(data(await client.call_tool("export_lumos", {"context": ctx})))
        job = data(await client.call_tool("run", {"context": ctx, "outputs": ["meter.out"]}))
        result = data(await client.call_tool("get_result", {"job": job["id"], "port": "meter.out"}))
        return document, result

    via_mcp, mcp_result = session_of(server)(scenario)

    session = service.create_session(origin="python").id
    ctx = service.create_context(session, name="bench").id
    service.add_element(session, ctx, "lumos.sources.GaussianLaser", name="laser", params={"power": "2 mW"})
    service.add_element(session, ctx, "lumos.detectors.PowerMeter", name="meter")
    service.connect(session, ctx, "laser.out", "meter.in")
    via_service = json.loads(service.export_lumos(session, ctx))

    python = Context(name="bench")
    laser = python.add(GaussianLaser(name="laser", power=2 * u.mW))
    python.add(PowerMeter(name="meter"))
    laser >> python["meter"]
    via_python = lumos.io.to_document(python)

    for document in (via_mcp, via_service, via_python):
        assert document["elements"] == via_python["elements"]
        assert document["connections"] == via_python["connections"]
    expected = python.run(["meter.out"])["meter.out"].data.to_value(u.W)
    np.testing.assert_allclose(mcp_result["values"], expected)


# --- Command line over stdio ------------------------------------------------------------------------------


def test_lumos_mcp_serves_over_stdio(tmp_path):
    """``lumos mcp`` starts a server that a client reaches over the standard streams."""
    parameters = StdioServerParameters(
        command=sys.executable, args=["-m", "lumos.cli", "mcp", "--archive-root", str(tmp_path)]
    )

    async def main():
        async with Client(parameters, read_timeout_seconds=120) as client:
            tools = await client.list_tools()
            ctx = await laser_bench(client)
            job = data(await client.call_tool("run", {"context": ctx, "outputs": ["meter.out"]}))
            result = data(await client.call_tool("get_result", {"job": job["id"], "port": "meter.out"}))
            return tools, result

    tools, result = anyio.run(main)
    assert REQUIRED_TOOLS <= {tool.name for tool in tools.tools}
    assert result["values"][0] == pytest.approx(2e-3, rel=1e-3)


def test_lumos_mcp_help():
    from lumos.cli import main

    with pytest.raises(SystemExit) as stopped:
        main(["mcp", "--help"])
    assert stopped.value.code == 0
