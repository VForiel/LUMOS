"""Tests for the resources and prompts of the MCP server (step P13.3, REQ-MCP-3).

Requirements: resources for the documentation pages, the element schemas, the presets, the current context,
the journals and the metadata of archives; prompts guided by use case: design an instrument, estimate a
detection capability, calibrate a bench, diagnose a simulation. Each resource and prompt is reached through an
MCP test client (plan, tests of phase 13).
"""

import json

import anyio
import pytest

pytest.importorskip("mcp")

from mcp import Client

from lumos.mcp import create_server
from lumos.service import Service, ServiceConfig

PROMPTS = {"design_instrument", "estimate_detection", "calibrate_bench", "diagnose_simulation"}


@pytest.fixture
def server(tmp_path):
    service = Service(ServiceConfig(archive_root=str(tmp_path / "archives")))
    yield create_server(service)
    service.shutdown()


def call(server, scenario):
    async def main():
        async with Client(server) as client:
            return await scenario(client)

    return anyio.run(main)


async def read_text(client, uri):
    result = await client.read_resource(uri)
    return result.contents[0].text


async def read_json(client, uri):
    return json.loads(await read_text(client, uri))


def test_resources_and_templates_are_listed(server):
    async def scenario(client):
        return await client.list_resources(), await client.list_resource_templates()

    resources, templates = call(server, scenario)
    uris = {str(resource.uri) for resource in resources.resources}
    assert {
        "lumos://docs",
        "lumos://elements",
        "lumos://presets",
        "lumos://context/current",
        "lumos://archives",
    } <= uris
    patterns = {template.uri_template for template in templates.resource_templates}
    assert {
        "lumos://docs/{+path}",
        "lumos://elements/{type_id}",
        "lumos://presets/{name}",
        "lumos://contexts/{id}",
        "lumos://contexts/{id}/journal",
        "lumos://archives/{+path}",
    } <= patterns
    for item in [*resources.resources, *templates.resource_templates]:
        assert item.description, f"{item.name} has no description"


def test_documentation_pages(server):
    async def scenario(client):
        index = await read_json(client, "lumos://docs")
        page = next(p for p in index if p["path"] == "hardware/operation.md")
        return page, await read_text(client, page["uri"])

    page, text = call(server, scenario)
    assert page["uri"] == "lumos://docs/hardware/operation.md"
    assert page["title"]
    assert page["title"] in text


def test_element_schemas(server):
    async def scenario(client):
        listed = await read_json(client, "lumos://elements")
        schema = await read_json(client, "lumos://elements/lumos.sources.GaussianLaser")
        return listed, schema

    listed, schema = call(server, scenario)
    assert "lumos.sources.GaussianLaser" in [item["type_id"] for item in listed]
    assert schema["properties"]["params"]["properties"]["power"]["x-lumos-unit"] == "mW"


def test_presets(server):
    async def scenario(client):
        listed = await read_json(client, "lumos://presets")
        document = await read_json(client, "lumos://presets/coronagraph")
        return listed, document

    listed, document = call(server, scenario)
    assert "coronagraph" in [item["name"] for item in listed]
    assert document["kind"] == "context"
    assert document["meta"]["name"] == "Vortex coronagraph"


def test_current_context_and_journal(server):
    async def scenario(client):
        loaded = (await client.call_tool("load_preset", {"name": "first_simulation"})).structured_content
        await client.call_tool("run", {"outputs": ["camera.out"]})
        current = await read_json(client, "lumos://context/current")
        named = await read_json(client, f"lumos://contexts/{loaded['id']}")
        journal = await read_json(client, f"lumos://contexts/{loaded['id']}/journal")
        return loaded, current, named, journal

    loaded, current, named, journal = call(server, scenario)
    assert current["id"] == loaded["id"]
    assert {e["id"] for e in current["elements"]} >= {"laser", "lens", "camera"}
    assert named == current
    assert journal
    assert all(entry["origin"] == "mcp" for entry in journal)


def test_archive_metadata(server):
    async def scenario(client):
        await client.call_tool("load_preset", {"name": "first_simulation"})
        await client.call_tool("run", {"outputs": ["camera.out"], "archive": "resource-run"})
        listed = await read_json(client, "lumos://archives")
        metadata = await read_json(client, listed[0]["uri"])
        return listed, metadata

    listed, metadata = call(server, scenario)
    assert len(listed) == 1
    assert listed[0]["uri"].startswith("lumos://archives/")
    assert metadata["origin"] == "mcp"


def test_prompts_guide_the_use_cases(server):
    arguments = {
        "design_instrument": {"goal": "image an exoplanet at 3 lambda/D"},
        "estimate_detection": {"target": "HR 8799 e", "instrument": "coronagraph"},
        "calibrate_bench": {"bench": "a laser, a linear stage and a camera"},
        "diagnose_simulation": {"problem": "the PSF looks aliased"},
    }
    tools = {
        "design_instrument": ["list_element_types", "validate_context", "run"],
        "estimate_detection": ["resolve_object", "evaluate_detection"],
        "calibrate_bench": ["device_status", "execute_action"],
        "diagnose_simulation": ["validate_context", "explain_sampling"],
    }

    async def scenario(client):
        listed = await client.list_prompts()
        texts = {}
        for name, args in arguments.items():
            prompt = await client.get_prompt(name, args)
            texts[name] = " ".join(message.content.text for message in prompt.messages)
        return listed, texts

    listed, texts = call(server, scenario)
    assert PROMPTS <= {prompt.name for prompt in listed.prompts}
    for prompt in listed.prompts:
        assert prompt.description
    for name, text in texts.items():
        for value in arguments[name].values():
            assert value in text
        for tool in tools[name]:
            assert tool in text, f"{name} does not mention {tool}"
