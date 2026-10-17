"""Generic conformance of ``lumos mcp`` to the Model Context Protocol over stdio (step P13.7, REQ-MCP-1).

Any MCP client, not only the official SDK, must be able to use the server. This test speaks raw JSON-RPC 2.0
to ``lumos mcp`` and checks the behaviour required by the specification, revision 2025-06-18
(https://modelcontextprotocol.io/specification/2025-06-18):

- transports: messages are newline-delimited JSON-RPC on stdin/stdout, and the server writes nothing else to
  stdout;
- lifecycle: ``initialize`` answers with the protocol version, the server information and its capabilities;
  then ``notifications/initialized``; ``ping`` answers an empty result;
- tools, resources and prompts: ``*/list`` and ``tools/call``, ``resources/read``, ``prompts/get``; tool names
  use the characters ``[A-Za-z0-9_.-]`` and every tool has an input schema of type ``object``;
- errors: an unknown method is the JSON-RPC error -32601; a failing tool call returns ``isError: true``.
"""

import json
import re
import subprocess
import sys

import pytest

pytest.importorskip("mcp")

VERSION = "2025-06-18"


class RawClient:
    """A minimal MCP client written from the specification, without the SDK."""

    def __init__(self, tmp_path):
        self.process = subprocess.Popen(
            [sys.executable, "-m", "lumos.cli", "mcp", "--archive-root", str(tmp_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        self.next_id = 0
        self.lines = []

    def send(self, message):
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def notify(self, method, params=None):
        self.send({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})})

    def request(self, method, params=None):
        self.next_id += 1
        identifier = self.next_id
        self.send(
            {"jsonrpc": "2.0", "id": identifier, "method": method, **({"params": params} if params else {})}
        )
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"the server closed stdout: {self.process.stderr.read()}")
            self.lines.append(line)
            message = json.loads(line)  # every line of stdout is a JSON-RPC message
            assert message["jsonrpc"] == "2.0"
            if message.get("id") == identifier:
                return message

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=30)
        finally:
            if self.process.poll() is None:
                self.process.kill()
            self.process.communicate()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    raw = RawClient(tmp_path_factory.mktemp("archives"))
    raw.initialized = raw.request(
        "initialize",
        {"protocolVersion": VERSION, "capabilities": {}, "clientInfo": {"name": "raw-test", "version": "1"}},
    )
    raw.notify("notifications/initialized")
    yield raw
    raw.close()


def test_initialize(client):
    result = client.initialized["result"]
    assert result["protocolVersion"] == VERSION
    assert result["serverInfo"]["name"] == "lumos"
    assert result["serverInfo"]["version"]
    assert {"tools", "resources", "prompts"} <= set(result["capabilities"])
    assert result.get("instructions")


def test_ping(client):
    assert client.request("ping")["result"] == {}


def test_tools(client):
    tools = client.request("tools/list")["result"]["tools"]
    assert tools
    for tool in tools:
        assert re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", tool["name"])
        assert tool["description"]
        assert tool["inputSchema"]["type"] == "object"
    called = client.request("tools/call", {"name": "list_presets", "arguments": {}})["result"]
    assert called["isError"] is False
    assert all(block["type"] == "text" for block in called["content"])
    assert "coronagraph" in " ".join(block["text"] for block in called["content"])


def test_a_failing_tool_call_is_a_tool_error(client):
    result = client.request("tools/call", {"name": "load_preset", "arguments": {"name": "nope"}})["result"]
    assert result["isError"] is True
    assert "hint" in result["content"][0]["text"]


def test_resources_and_prompts(client):
    resources = client.request("resources/list")["result"]["resources"]
    assert all({"uri", "name"} <= set(resource) for resource in resources)
    read = client.request("resources/read", {"uri": "lumos://presets"})["result"]["contents"]
    assert read[0]["uri"] == "lumos://presets"
    assert "coronagraph" in read[0]["text"]
    templates = client.request("resources/templates/list")["result"]["resourceTemplates"]
    assert all("uriTemplate" in template for template in templates)
    prompts = client.request("prompts/list")["result"]["prompts"]
    assert any(prompt["name"] == "diagnose_simulation" for prompt in prompts)
    prompt = client.request(
        "prompts/get", {"name": "diagnose_simulation", "arguments": {"problem": "aliasing"}}
    )
    messages = prompt["result"]["messages"]
    assert messages[0]["role"] == "user"
    assert "aliasing" in messages[0]["content"]["text"]


def test_unknown_methods_are_json_rpc_errors(client):
    answer = client.request("lumos/unknown")
    assert answer["error"]["code"] == -32601


def test_stdout_carries_only_protocol_messages(client):
    client.request("tools/call", {"name": "load_preset", "arguments": {"name": "first_simulation"}})
    client.request("tools/call", {"name": "run", "arguments": {"outputs": ["camera.out"]}})
    assert client.lines
    for line in client.lines:
        assert json.loads(line)["jsonrpc"] == "2.0"
