"""Connector links: the MCP server of a browser session for Claude, ChatGPT and other MCP clients (P14.15).

Requirements (review of 18/10/2026: "use my Claude, GPT or Gemini account without an API key"): a
subscription of Claude.ai or ChatGPT cannot be used by another application, but these applications accept
remote MCP servers as connectors. A user of the interface (simulator or operator) creates a *connector link*:
a secret URL ``/mcp/<token>`` of the MCP server of their session, needing no cookie or header, so that an
assistant added as a connector works on the contexts shown in their browser. The link is revocable, outlives
its session (test_web_connector_links), uses HTTPS behind a TLS proxy, and is refused to viewers and when the
assistant is turned off.

The client speaks raw JSON-RPC 2.0 over the streamable HTTP transport (MCP specification 2025-06-18).
"""

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mcp")

from fastapi.testclient import TestClient

from lumos.web import WebConfig, create_app

VERSION = "2025-06-18"
HEADERS = {"accept": "application/json, text/event-stream", "content-type": "application/json"}


def rpc(client, path, method, params, session=None, identifier=1):
    headers = dict(HEADERS)
    if session:
        headers.update({"mcp-session-id": session, "mcp-protocol-version": VERSION})
    message = {"jsonrpc": "2.0", "id": identifier, "method": method, "params": params}
    response = client.post(path, content=json.dumps(message), headers=headers)
    return response


def answer(response):
    if response.headers["content-type"].startswith("text/event-stream"):
        for line in response.text.splitlines():
            if line.startswith("data:"):
                return json.loads(line[5:])
    return response.json()


def connect(client, path):
    """Initialise an MCP session at ``path`` with a client holding no cookie; return its session id."""
    info = {"name": "connector-test", "version": "1"}
    response = rpc(
        client, path, "initialize", {"protocolVersion": VERSION, "capabilities": {}, "clientInfo": info}
    )
    assert response.status_code == 200, response.text
    session = response.headers["mcp-session-id"]
    client.post(
        path,
        content=json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        headers={**HEADERS, "mcp-session-id": session, "mcp-protocol-version": VERSION},
    )
    return session


def test_a_connector_link_works_on_the_session_of_the_browser(tmp_path):
    app = create_app(WebConfig(archive_root=str(tmp_path)))
    with (
        TestClient(app, base_url="http://localhost") as browser,
        TestClient(app, base_url="http://localhost") as claude,
    ):
        browser.get("/api/info")
        assert browser.get("/api/assistant/link").json() == {"url": None}
        link = browser.post("/api/assistant/link").json()["url"]
        assert link.startswith("http://localhost/mcp/")
        assert browser.get("/api/assistant/link").json() == {"url": link}
        path = link.removeprefix("http://localhost")

        session = connect(claude, path)  # no cookie: the token of the link is the credential
        created = answer(
            rpc(
                claude,
                path,
                "tools/call",
                {"name": "create_context", "arguments": {"name": "from Claude"}},
                session,
                2,
            )
        )
        assert not created["result"].get("isError")
        names = [c["name"] for c in browser.get("/api/contexts").json()]
        assert "from Claude" in names  # the browser shows what the connector did

        again = browser.post("/api/assistant/link").json()["url"]
        assert again != link  # a new link replaces the old one
        assert rpc(claude, path, "tools/list", {}, session, 3).status_code == 404
        browser.delete("/api/assistant/link")
        assert (
            rpc(claude, again.removeprefix("http://localhost"), "tools/list", {}, None, 4).status_code == 404
        )
        assert rpc(claude, "/mcp/not-a-link", "tools/list", {}, None, 5).status_code == 404


def test_links_use_https_behind_a_tls_proxy_and_are_refused_to_viewers(tmp_path):
    def public(role):
        return WebConfig(
            profile="public",
            allowed_hosts=["lumos.example"],
            trust_proxy=True,
            anonymous_role=role,
            audit_log=str(tmp_path / "audit.log"),
        )

    headers = {"x-forwarded-proto": "https", "x-forwarded-for": "203.0.113.5"}
    with TestClient(create_app(public("simulator")), base_url="https://lumos.example") as client:
        client.get("/api/info", headers=headers)
        assert (
            client.post("/api/assistant/link", headers=headers)
            .json()["url"]
            .startswith("https://lumos.example/mcp/")
        )
    with TestClient(create_app(public("viewer")), base_url="https://lumos.example") as client:
        client.get("/api/info", headers=headers)
        refused = client.post("/api/assistant/link", headers=headers)
        assert refused.status_code == 403


def test_no_link_when_the_assistant_is_turned_off(tmp_path):
    with TestClient(create_app(WebConfig(assistant=False)), base_url="http://localhost") as client:
        client.get("/api/info")
        assert client.post("/api/assistant/link").status_code == 404
