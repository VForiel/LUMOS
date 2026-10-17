"""Tests for the streamable HTTP transport of the MCP server (step P13.6, REQ-MCP-1).

Requirements: the MCP server is also served over streamable HTTP, ready to be mounted in the web instance
(authentication comes with it in P14). Until then it listens on the local machine by default; listening on
other interfaces is an explicit choice and never allowed together with the operation of equipment, since
nobody would be authenticated to confirm the actions.

The expected reading is physical: a power meter behind a 2 mW laser reads 2 mW.
"""

import socket
import subprocess
import sys
import time

import anyio
import pytest

pytest.importorskip("mcp")

from mcp import Client

from lumos.cli import main
from lumos.mcp import create_http_app, create_server


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def wait_for(port, process, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"the server stopped: {process.stderr.read()}")
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise TimeoutError("the server did not start")


@pytest.fixture
def http_server(tmp_path):
    port = free_port()
    process = subprocess.Popen(
        [sys.executable, "-m", "lumos.cli", "mcp", "--transport", "http", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=tmp_path,
    )
    try:
        wait_for(port, process)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        process.terminate()
        process.wait(timeout=30)


def test_lumos_mcp_serves_over_streamable_http(http_server):
    async def scenario():
        async with Client(http_server) as client:
            tools = {tool.name for tool in (await client.list_tools()).tools}
            await client.call_tool("create_context", {"name": "bench"})
            await client.call_tool(
                "add_element",
                {"type_id": "lumos.sources.GaussianLaser", "name": "laser", "params": {"power": "2 mW"}},
            )
            await client.call_tool("add_element", {"type_id": "lumos.detectors.PowerMeter", "name": "meter"})
            await client.call_tool("connect", {"source": "laser.out", "target": "meter.in"})
            job = (await client.call_tool("run", {"outputs": ["meter.out"]})).structured_content
            result = await client.call_tool("get_result", {"job": job["id"], "port": "meter.out"})
            return tools, result.structured_content

    tools, result = anyio.run(scenario)
    assert {"run", "get_result", "inspect_port"} <= tools
    assert result["values"][0] == pytest.approx(2e-3, rel=1e-3)


def test_the_http_app_can_be_mounted():
    """The ASGI application is what the web backend mounts (P14)."""
    app = create_http_app(create_server(), path="/mcp")
    assert callable(app)
    assert "/mcp" in [getattr(route, "path", None) for route in app.routes]


def test_http_listens_on_the_local_machine_by_default(capsys):
    with pytest.raises(SystemExit):
        main(["mcp", "--help"])
    help_text = capsys.readouterr().out
    assert "--transport" in help_text
    assert "127.0.0.1" in help_text


def test_remote_listening_is_explicit_and_never_with_operation(capsys):
    assert main(["mcp", "--transport", "http", "--host", "0.0.0.0"]) == 2
    assert "--allow-remote" in capsys.readouterr().err
    refused = main(
        ["mcp", "--transport", "http", "--host", "0.0.0.0", "--allow-remote", "--enable-operation"]
    )
    assert refused == 2
    assert "operation" in capsys.readouterr().err
