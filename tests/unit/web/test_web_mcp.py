"""The MCP server mounted in the web instance (step P14.9, REQ-MCP-1, REQ-WEB-12, ADR-0019).

Requirements: the web instance serves the MCP server over streamable HTTP at ``/mcp`` with the same
authentication as the interface; an assistant connected there works in the session of the user, on the same
contexts as the interface; viewers and visitors who did not log in (when the instance requires it) are
refused; only operators get the tools that drive equipment; the assistant panel can be turned off.

The client speaks raw JSON-RPC 2.0 as written in the MCP specification, revision 2025-06-18 (streamable HTTP
transport: POST of a message, answer as JSON or as server-sent events, ``Mcp-Session-Id`` header).
"""

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mcp")

from fastapi.testclient import TestClient

from lumos.service import ServiceConfig
from lumos.web import WebConfig, create_app

VERSION = "2025-06-18"
HEADERS = {"accept": "application/json, text/event-stream", "content-type": "application/json"}


class Mcp:
    """A minimal MCP client over the streamable HTTP transport, sharing the cookies of ``client``."""

    def __init__(self, client):
        self.client = client
        self.session = None
        self.next_id = 0

    def post(self, message):
        headers = dict(HEADERS)
        if self.session:
            headers["mcp-session-id"] = self.session
            headers["mcp-protocol-version"] = VERSION
        return self.client.post("/mcp", content=json.dumps(message), headers=headers)

    def request(self, method, params=None):
        self.next_id += 1
        response = self.post({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params or {}})
        assert response.status_code == 200, response.text
        self.session = response.headers.get("mcp-session-id", self.session)
        if response.headers["content-type"].startswith("text/event-stream"):
            for line in response.text.splitlines():
                if line.startswith("data:"):
                    message = json.loads(line[5:])
                    if message.get("id") == self.next_id:
                        return message
            raise AssertionError(f"no answer in {response.text!r}")
        return response.json()

    def start(self):
        info = {"name": "raw-test", "version": "1"}
        result = self.request(
            "initialize", {"protocolVersion": VERSION, "capabilities": {}, "clientInfo": info}
        )
        self.post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return result

    def tools(self):
        return {tool["name"] for tool in self.request("tools/list")["result"]["tools"]}

    def call(self, name, arguments):
        return self.request("tools/call", {"name": name, "arguments": arguments})["result"]


def local(**service):
    return TestClient(create_app(WebConfig(service=ServiceConfig(**service))), base_url="http://localhost")


def test_the_assistant_works_on_the_contexts_of_the_interface():
    with local() as client:
        client.get("/api/info")  # opens the session of the browser
        mcp = Mcp(client)
        assert mcp.start()["result"]["serverInfo"]["name"] == "lumos"
        assert {"create_context", "add_element", "run", "inspect_port"} <= mcp.tools()
        result = mcp.call("create_context", {"name": "made by the assistant"})
        assert not result.get("isError"), result
        names = [c["name"] for c in client.get("/api/contexts").json()]
        assert "made by the assistant" in names
        cid = client.post("/api/contexts", json={"name": "made in the interface"}).json()["id"]
        described = mcp.call("describe_context", {"context": cid})
        assert not described.get("isError"), described
        assert "made in the interface" in json.dumps(described)


def test_the_actions_of_the_assistant_are_journaled_as_mcp():
    with local() as client:
        client.get("/api/info")
        mcp = Mcp(client)
        mcp.start()
        cid = client.post("/api/contexts", json={}).json()["id"]
        client.post(
            f"/api/contexts/{cid}/elements", json={"type_id": "lumos.lab.LinearStage", "name": "stage"}
        )
        result = mcp.call("execute_action", {"element": "stage", "action": "home", "context": cid})
        assert not result.get("isError"), result
        entries = client.get(f"/api/contexts/{cid}/journal", params={"kind": "action"}).json()
        assert entries[-1]["origin"] == "mcp"


def test_visitors_who_must_log_in_and_viewers_are_refused():
    config = WebConfig(profile="public", anonymous_role=None)
    with TestClient(create_app(config), base_url="http://testserver") as client:
        assert Mcp(client).post({"jsonrpc": "2.0", "id": 1, "method": "ping"}).status_code == 401
    config = WebConfig(profile="public", anonymous_role="viewer")
    with TestClient(create_app(config), base_url="http://testserver") as client:
        assert Mcp(client).post({"jsonrpc": "2.0", "id": 1, "method": "ping"}).status_code == 403


def test_only_operators_get_the_tools_that_drive_equipment(tmp_path):
    with local(operation_enabled=True) as client:  # local profile: the user is an operator
        client.get("/api/info")
        mcp = Mcp(client)
        mcp.start()
        assert {"discover_devices", "associate_devices"} <= mcp.tools()
    config = WebConfig(
        profile="public",
        trust_proxy=True,
        users_file=str(operators(tmp_path)),
        service=ServiceConfig(operation_enabled=True),
    )
    with TestClient(create_app(config), base_url="https://testserver") as client:  # anonymous simulator
        client.get("/api/info")
        mcp = Mcp(client)
        mcp.start()
        assert not {"discover_devices", "associate_devices"} & mcp.tools()


def operators(tmp_path):
    from argon2 import PasswordHasher

    path = tmp_path / "users.toml"
    path.write_text(
        f'[users.olga]\nrole = "operator"\npassword = "{PasswordHasher().hash("x" * 12)}"\n', encoding="utf-8"
    )
    return path


def test_the_assistant_can_be_turned_off():
    config = WebConfig(assistant=False)
    with TestClient(create_app(config), base_url="http://localhost") as client:
        assert client.get("/api/info").json()["assistant"] is False
        assert Mcp(client).post({"jsonrpc": "2.0", "id": 1, "method": "ping"}).status_code == 404
