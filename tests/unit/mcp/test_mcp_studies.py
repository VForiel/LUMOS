"""Studies through the MCP server (step P14.13, REQ-MCP-1, REQ-CTX-10).

Requirements: an assistant sets the seed of a context (``set_seed``), frees a parameter with a prior
(``set_free_parameter``), runs a sweep (``run_sweep``) or a Monte-Carlo run (``run_monte_carlo``) and reads
its points (``get_study``), as the web interface and the Python API do.

Expected values: a 2 mW laser through a top-hat filter of power transmission ``T`` centred on its wavelength
gives ``2 mW × T`` on a power meter (truncation of the beam by the grid below 1e-3).
"""

import anyio
import pytest
from astropy import units as u

pytest.importorskip("mcp")

from mcp import Client

from lumos.mcp import create_server
from lumos.service import Service, ServiceConfig

STUDY_TOOLS = {"set_seed", "set_free_parameter", "run_sweep", "run_monte_carlo", "get_study"}


def data(result):
    """Structured content of a successful tool result."""
    assert not result.is_error, result.content
    content = result.structured_content
    return content["result"] if set(content) == {"result"} else content


@pytest.fixture
def server(tmp_path):
    service = Service(ServiceConfig(archive_root=str(tmp_path / "archives")))
    yield create_server(service)
    service.shutdown()


def run(server, scenario):
    async def main():
        async with Client(server) as client:
            return await scenario(client)

    return anyio.run(main)


async def filter_bench(client):
    ctx = data(await client.call_tool("create_context", {"name": "filter"}))["id"]
    elements = [
        ("lumos.sources.GaussianLaser", "laser", {"power": "2 mW"}),
        ("lumos.optics.SpectralFilter", "filt", {"center": "633 nm", "width": "10 nm"}),
        ("lumos.detectors.PowerMeter", "meter", {}),
    ]
    for type_id, name, params in elements:
        await client.call_tool(
            "add_element", {"context": ctx, "type_id": type_id, "name": name, "params": params}
        )
    await client.call_tool("connect", {"context": ctx, "source": "laser.out", "target": "filt.in"})
    await client.call_tool("connect", {"context": ctx, "source": "filt.out", "target": "meter.in"})
    return ctx


def test_the_study_tools_are_listed(server):
    async def scenario(client):
        return {tool.name for tool in (await client.list_tools()).tools}

    assert run(server, scenario) >= STUDY_TOOLS


def test_an_assistant_sweeps_and_samples_a_context(server):
    async def scenario(client):
        ctx = await filter_bench(client)
        assert data(await client.call_tool("set_seed", {"context": ctx, "seed": 99}))["seed"] == 99
        sweep = data(
            await client.call_tool(
                "run_sweep",
                {"context": ctx, "parameters": {"filt.transmission": [0.5, 1.0]}, "outputs": ["meter.out"]},
            )
        )
        assert sweep["status"] == "done"
        points = data(await client.call_tool("get_study", {"job": sweep["id"]}))["points"]
        powers = [
            (p["outputs"]["meter.out"]["values"][0] * u.Unit(p["outputs"]["meter.out"]["unit"])).to_value(
                u.mW
            )
            for p in points
        ]
        assert powers == pytest.approx([1.0, 2.0], rel=1e-3)

        await client.call_tool(
            "set_free_parameter",
            {
                "context": ctx,
                "path": "filt.transmission",
                "free": True,
                "prior": {"kind": "uniform", "low": 0.2, "high": 0.8},
            },
        )
        sample = data(
            await client.call_tool("run_monte_carlo", {"context": ctx, "n": 20, "outputs": ["meter.out"]})
        )
        assert sample["status"] == "done"
        result = await client.call_tool("get_study", {"job": sample["id"]})
        study = data(result)
        assert study["kind"] == "sample"
        assert study["seed"] == 99
        assert len(study["points"]) == 20
        assert all(0.2 <= float(p["params"]["filt.transmission"]) <= 0.8 for p in study["points"])
        assert "meter.out" in result.content[0].text  # a readable table for the assistant

    run(server, scenario)
