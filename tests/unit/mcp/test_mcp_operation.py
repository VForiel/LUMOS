"""Tests for the operation tools of the MCP server (step P13.4, REQ-MCP-4).

Requirements: the operation tools are disabled by default; once enabled by the configuration, every action on
real equipment asks a human confirmation (MCP elicitation); limits and the emergency stop are applied by the
server whatever the request of the language model; actions of MCP origin are journaled as such. Tests of the
plan: refusal of a hardware action without confirmation; refusal in disabled mode.

The equipment is the LumosLabs LL-LS100 emulator (travel 0-100 mm, ADR-0014), reached over TCP.
"""

import time

import anyio
import mcp.types as mcp_types
import pytest

pytest.importorskip("mcp")

from mcp import Client

from lumos.hardware.emulators import EmulatorServer, create_emulator
from lumos.mcp import create_server
from lumos.service import Service, ServiceConfig

OPERATION_TOOLS = {"list_devices", "device_status", "execute_action", "stop_all"}
ENABLED_ONLY = {"discover_devices", "associate_devices"}


@pytest.fixture
def stage():
    server = EmulatorServer(create_emulator("LL-LS100", latency=0.0)).start()
    yield server
    server.stop()
    time.sleep(0.05)


def make_server(enabled):
    return create_server(Service(ServiceConfig(operation_enabled=enabled)))


def call(server, scenario, **options):
    async def main():
        async with Client(server, **options) as client:
            return await scenario(client)

    return anyio.run(main)


def answering(confirm, asked):
    async def callback(context, params):
        asked.append(params.message)
        return mcp_types.ElicitResult(action="accept", content={"confirm": confirm})

    return callback


async def declining(context, params):
    return mcp_types.ElicitResult(action="decline")


async def operation_bench(client, url):
    await client.call_tool("create_context", {"name": "bench", "mode": "operation"})
    await client.call_tool(
        "add_element",
        {
            "type_id": "lumos.lab.LinearStage",
            "name": "stage",
            "params": {"velocity": "50 mm/s", "acceleration": "500 mm/s2"},
        },
    )
    found = await client.call_tool("discover_devices", {"tcp": [url]})
    assert not found.is_error, found.content
    associated = await client.call_tool("associate_devices", {})
    assert not associated.is_error, associated.content


def text(result):
    return " ".join(getattr(block, "text", "") for block in result.content)


# --- Disabled by default ----------------------------------------------------------------------------------


def test_operation_is_disabled_by_default():
    server = create_server(Service())

    async def scenario(client):
        tools = {tool.name for tool in (await client.list_tools()).tools}
        await client.call_tool("create_context", {"mode": "operation"})
        await client.call_tool("add_element", {"type_id": "lumos.lab.LinearStage", "name": "stage"})
        refused = await client.call_tool("execute_action", {"element": "stage", "action": "home"})
        stopped = await client.call_tool("stop_all", {})
        return tools, refused, stopped

    tools, refused, stopped = call(server, scenario, elicitation_callback=answering(True, []))
    assert tools >= OPERATION_TOOLS
    assert not tools & ENABLED_ONLY  # no access to equipment at all
    assert refused.is_error
    assert "disabled" in text(refused)
    assert "--enable-operation" in text(refused)
    assert not stopped.is_error  # the emergency stop is always allowed


def test_simulated_actions_need_no_confirmation():
    asked = []
    server = make_server(enabled=False)

    async def scenario(client):
        await client.call_tool("create_context", {})
        await client.call_tool("add_element", {"type_id": "lumos.lab.LinearStage", "name": "stage"})
        moved = await client.call_tool(
            "execute_action", {"element": "stage", "action": "move", "args": ["5 mm"]}
        )
        journal = (await client.call_tool("get_journal", {"kind": "action"})).structured_content["result"]
        return moved, journal

    moved, journal = call(server, scenario, elicitation_callback=answering(True, asked))
    assert not moved.is_error, moved.content
    assert moved.structured_content["mode"] == "simulation"
    assert moved.structured_content["predicted_s"] > 0
    assert asked == []
    assert journal[-1]["origin"] == "mcp"


# --- Enabled: confirmations, limits, emergency stop -------------------------------------------------------


@pytest.mark.parametrize("mode", ["auto", "legacy"])  # protocol 2026-07-28 and the handshake-era revisions
def test_actions_on_equipment_are_confirmed_by_a_human(stage, mode):
    asked = []
    server = make_server(enabled=True)

    async def scenario(client):
        tools = {tool.name for tool in (await client.list_tools()).tools}
        await operation_bench(client, stage.url)
        homed = await client.call_tool("execute_action", {"element": "stage", "action": "home"})
        moved = await client.call_tool(
            "execute_action", {"element": "stage", "action": "move", "args": ["5 mm"]}
        )
        status = await client.call_tool("device_status", {"element": "stage"})
        devices = await client.call_tool("list_devices", {})
        journal = (await client.call_tool("get_journal", {"kind": "action"})).structured_content["result"]
        return tools, homed, moved, status, devices, journal

    tools, homed, moved, status, devices, journal = call(
        server, scenario, elicitation_callback=answering(True, asked), mode=mode
    )
    assert tools >= OPERATION_TOOLS | ENABLED_ONLY
    assert not homed.is_error, homed.content
    assert not moved.is_error, moved.content
    assert moved.structured_content["mode"] == "operation"
    assert moved.structured_content["measured_s"] > 0
    assert stage.emulator.position == pytest.approx(5, abs=0.01)
    assert len(asked) == 2  # one confirmation per action
    assert "stage" in asked[1]
    assert "move" in asked[1]
    assert "5 mm" in asked[1]
    assert "LL-LS100" in text(status)
    assert "stage" in text(devices)
    assert [entry["origin"] for entry in journal[-2:]] == ["mcp", "mcp"]


def test_a_refused_confirmation_blocks_the_action(stage):
    async def scenario(client):
        await operation_bench(client, stage.url)
        return await client.call_tool("execute_action", {"element": "stage", "action": "home"})

    refused = call(make_server(enabled=True), scenario, elicitation_callback=answering(False, []))
    assert refused.is_error
    assert "confirm" in text(refused)
    declined = call(make_server(enabled=True), scenario, elicitation_callback=declining)
    assert declined.is_error
    assert not stage.emulator.homed


def test_without_elicitation_the_action_is_refused(stage):
    """A client that cannot ask the user cannot act on equipment."""
    server = make_server(enabled=True)

    async def scenario(client):
        await operation_bench(client, stage.url)
        return await client.call_tool("execute_action", {"element": "stage", "action": "home"})

    refused = call(server, scenario)
    assert refused.is_error
    assert not stage.emulator.homed


def test_limits_are_enforced_by_the_server_even_when_confirmed(stage):
    server = make_server(enabled=True)

    async def scenario(client):
        await operation_bench(client, stage.url)
        await client.call_tool("execute_action", {"element": "stage", "action": "home"})
        await client.call_tool("execute_action", {"element": "stage", "action": "move", "args": ["5 mm"]})
        return await client.call_tool(
            "execute_action", {"element": "stage", "action": "move", "args": ["500 mm"]}
        )

    beyond = call(server, scenario, elicitation_callback=answering(True, []))
    assert beyond.is_error
    assert "limit" in text(beyond)
    assert stage.emulator.position == pytest.approx(5, abs=0.01)


def test_the_emergency_stop_needs_no_confirmation(stage):
    asked = []
    server = make_server(enabled=True)

    async def scenario(client):
        await operation_bench(client, stage.url)
        return await client.call_tool("stop_all", {})

    stopped = call(server, scenario, elicitation_callback=answering(True, asked))
    assert not stopped.is_error, stopped.content
    assert stage.emulator.stopped
    assert asked == []


def test_lumos_mcp_enable_operation_option(capsys):
    from lumos.cli import main

    with pytest.raises(SystemExit):
        main(["mcp", "--help"])
    assert "--enable-operation" in capsys.readouterr().out
