"""Tests for the responses of the MCP server, tailored to language models (step P13.5, REQ-MCP-5).

Requirements: compact summaries, explicit units, actionable errors (cause and suggested fix), stable
identifiers, pagination of long lists, figures returned as images.
"""

import base64
import json

import anyio
import pytest

pytest.importorskip("mcp")

from mcp import Client

from lumos.mcp import create_server
from lumos.service import Service, ServiceConfig

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def server(tmp_path):
    service = Service(ServiceConfig(archive_root=str(tmp_path)))
    yield create_server(service)
    service.shutdown()


def call(server, scenario):
    async def main():
        async with Client(server) as client:
            return await scenario(client)

    return anyio.run(main)


def texts(result):
    return [block.text for block in result.content if block.type == "text"]


def images(result):
    return [block for block in result.content if block.type == "image"]


# --- Figures and summaries --------------------------------------------------------------------------------


def test_inspect_port_returns_a_figure_and_a_compact_summary(server):
    async def scenario(client):
        await client.call_tool("load_preset", {"name": "first_simulation"})
        return await client.call_tool("inspect_port", {"port": "focus.out"})

    result = call(server, scenario)
    assert not result.is_error, result.content
    [image] = images(result)
    assert image.mime_type == "image/png"
    assert base64.b64decode(image.data).startswith(PNG)
    summary = texts(result)[0]
    assert len(summary) < 800  # a summary, not a dump
    assert "focus.out" in summary
    assert "Wavefront" in summary
    assert " nm" in summary  # wavelength with its unit
    structured = result.structured_content
    assert structured["kind"] == "Wavefront"
    assert not structured.get("figure_png")  # the figure travels as an image, not as base64 in the JSON


def test_get_result_can_return_a_figure(server):
    async def scenario(client):
        await client.call_tool("load_preset", {"name": "first_simulation"})
        job = (await client.call_tool("run", {"outputs": ["camera.out"]})).structured_content
        without = await client.call_tool("get_result", {"job": job["id"], "port": "camera.out"})
        with_figure = await client.call_tool(
            "get_result", {"job": job["id"], "port": "camera.out", "figure": True}
        )
        return without, with_figure

    without, with_figure = call(server, scenario)
    assert images(without) == []
    assert len(images(with_figure)) == 1
    assert "adu" in texts(with_figure)[0]
    assert with_figure.structured_content["unit"] == "adu"


def test_incoherent_bundles_are_summarised_with_units(server):
    """The output of a telescope observing a star and a planet: two mutually incoherent modes."""

    async def scenario(client):
        await client.call_tool("load_preset", {"name": "coronagraph"})
        return await client.call_tool("inspect_port", {"port": "telescope.out"})

    result = call(server, scenario)
    assert not result.is_error, result.content
    summary = result.structured_content
    assert summary["kind"] == "IncoherentBundle"
    assert summary["unit"]
    assert summary["statistics"]["max"] > 0
    assert summary["details"]["modes"] == "2"
    assert "nm" in summary["details"]["wavelength"]
    assert "grid" in summary["details"]
    assert len(images(result)) == 1


def test_run_answers_with_a_short_status(server):
    async def scenario(client):
        await client.call_tool("load_preset", {"name": "first_simulation"})
        return await client.call_tool("run", {})

    result = call(server, scenario)
    summary = texts(result)[0]
    assert len(summary) < 400
    assert "done" in summary
    assert "camera.out" in summary
    assert result.structured_content["id"] in summary  # the identifier to reuse


# --- Units ------------------------------------------------------------------------------------------------


def test_quantities_need_their_unit(server):
    async def scenario(client):
        await client.call_tool("create_context", {})
        await client.call_tool("add_element", {"type_id": "lumos.sources.GaussianLaser", "name": "laser"})
        bare = await client.call_tool("set_param", {"path": "laser.waist", "value": 0.5})
        await client.call_tool("add_element", {"type_id": "lumos.photonics.Waveguide", "name": "guide"})
        dimensionless = await client.call_tool("set_param", {"path": "guide.effective_index", "value": 1.6})
        return bare, dimensionless

    bare, dimensionless = call(server, scenario)
    assert bare.is_error  # a length without unit is ambiguous
    assert "unit" in texts(bare)[0]
    assert "mm" in texts(bare)[0]  # an example in the unit of the parameter
    assert not dimensionless.is_error, dimensionless.content


@pytest.mark.parametrize("preset", ["first_simulation", "coronagraph", "kernel_nuller"])
def test_the_values_shown_can_be_given_back(server, preset):
    """Every value written by describe_context is accepted by set_param and gives the same description."""

    async def scenario(client):
        before = (await client.call_tool("load_preset", {"name": preset})).structured_content
        failures = []
        for element in before["elements"]:
            for group in ("params", "settings"):
                for name, value in element[group].items():
                    if isinstance(value, str) and "..." in value:
                        continue  # long arrays are abbreviated in the summary
                    result = await client.call_tool(
                        "set_param", {"path": f"{element['id']}.{name}", "value": value}
                    )
                    if result.is_error:
                        failures.append((element["id"], name, value, texts(result)))
        after = (await client.call_tool("describe_context", {})).structured_content
        return before, after, failures

    before, after, failures = call(server, scenario)
    assert failures == []
    assert after["elements"] == before["elements"]


# --- Errors, identifiers, pagination ----------------------------------------------------------------------


def test_errors_say_what_to_do(server):
    requests = [
        ("describe_context", {"context": "nope"}),
        ("load_preset", {"name": "nope"}),
        ("get_job", {"job": "nope"}),
        ("add_element", {"type_id": "lumos.sources.GausianLaser"}),
        ("connect", {"source": "laser", "target": "meter.in"}),
        ("set_param", {"path": "laser.nope", "value": "1 mm"}),
    ]

    async def scenario(client):
        await client.call_tool("create_context", {})
        await client.call_tool("add_element", {"type_id": "lumos.sources.GaussianLaser", "name": "laser"})
        return [await client.call_tool(name, args) for name, args in requests]

    for (name, _), result in zip(requests, call(server, scenario), strict=True):
        assert result.is_error, name
        lines = texts(result)[0].splitlines()
        hints = [line for line in lines if line.startswith("hint: ")]
        assert hints, f"{name}: no hint in {lines}"
        assert len(hints[0]) > len("hint: ") + 5
        assert any(line.startswith("code: ") for line in lines)
    assert "GaussianLaser" in texts(call(server, scenario)[3])[0]  # the closest type is suggested


def test_identifiers_are_stable(server):
    async def scenario(client):
        created = (await client.call_tool("create_context", {"name": "a"})).structured_content
        listed = (await client.call_tool("list_contexts", {})).structured_content["result"]
        described = (
            await client.call_tool("describe_context", {"context": created["id"]})
        ).structured_content
        return created, listed, described

    created, listed, described = call(server, scenario)
    assert [c["id"] for c in listed] == [created["id"]]
    assert described["id"] == created["id"]


def test_long_lists_are_paginated(server):
    async def scenario(client):
        first = (await client.call_tool("list_element_types", {"page_size": 5})).structured_content
        second = (
            await client.call_tool("list_element_types", {"page": 2, "page_size": 5})
        ).structured_content
        return first, second

    first, second = call(server, scenario)
    assert len(first["items"]) == 5
    assert first["next_page"] == 2
    assert first["total"] > 10
    assert not {i["type_id"] for i in first["items"]} & {i["type_id"] for i in second["items"]}


def test_structured_results_are_json(server):
    """Every structured result is plain JSON (no bytes, no NaN), readable by any client."""

    async def scenario(client):
        await client.call_tool("load_preset", {"name": "coronagraph"})
        return [
            await client.call_tool(name, args)
            for name, args in [
                ("describe_context", {}),
                ("inspect_port", {"port": "image.out"}),
                ("validate_context", {}),
                ("list_presets", {}),
            ]
        ]

    for result in call(server, scenario):
        assert not result.is_error, result.content
        json.dumps(result.structured_content, allow_nan=False)
