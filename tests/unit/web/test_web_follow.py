"""The interface follows what an assistant does through a connector link (step P14.18, review of 19/10/2026).

Requirements: a user added LUMOS to Claude.ai as a connector; the assistant worked, but nothing appeared in
the open page. The interface and the assistant share the session, so:

- the tools of the assistant work by default on the context *focused* in the session, i.e. the one open in
  the interface (``PUT /api/focus``); a context created or loaded by the assistant takes the focus;
- the WebSocket of the interface sends ``{"type": "context", "context", "revision"}`` when a context changes
  and ``{"type": "focus", "context", "origin", "count"}`` when the focus changes, so that the page shows the
  changes and switches to the context of the assistant;
- the tool ``get_view_link`` gives the user a link opening the context in a browser, also when the page was
  closed or never opened: ``/view/<secret>?context=<id>`` attaches the browser to the session (cookie) and
  leads to ``/?context=<id>``. The secret is as powerful as the connector link that asked for it, expires,
  and an unknown secret is refused.
"""

import json
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mcp")

from fastapi.testclient import TestClient

from lumos.web import WebConfig, create_app

VERSION = "2025-06-18"
HEADERS = {"accept": "application/json, text/event-stream", "content-type": "application/json"}


def rpc(client, path, session, method, params, identifier=2):
    headers = {**HEADERS, "mcp-session-id": session, "mcp-protocol-version": VERSION}
    message = {"jsonrpc": "2.0", "id": identifier, "method": method, "params": params}
    response = client.post(path, content=json.dumps(message), headers=headers)
    if response.headers["content-type"].startswith("text/event-stream"):
        for line in response.text.splitlines():
            if line.startswith("data:"):
                return json.loads(line[5:])
    return response.json()


def connect(client, path):
    info = {"name": "follow-test", "version": "1"}
    message = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": VERSION, "capabilities": {}, "clientInfo": info},
    }
    response = client.post(path, content=json.dumps(message), headers=HEADERS)
    assert response.status_code == 200, response.text
    session = response.headers["mcp-session-id"]
    client.post(
        path,
        content=json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        headers={**HEADERS, "mcp-session-id": session, "mcp-protocol-version": VERSION},
    )
    return session


def call(client, path, session, tool, **arguments):
    answer = rpc(client, path, session, "tools/call", {"name": tool, "arguments": arguments})
    result = answer["result"]
    assert not result.get("isError"), result
    return result


def ws_headers(client):
    return {"origin": "http://localhost", "cookie": f"lumos_session={client.cookies['lumos_session']}"}


@pytest.fixture
def instance(tmp_path):
    app = create_app(WebConfig(archive_root=str(tmp_path)))
    with (
        TestClient(app, base_url="http://localhost") as browser,
        TestClient(app, base_url="http://localhost") as claude,
    ):
        browser.get("/api/info")
        link = browser.post("/api/assistant/link").json()["url"]
        path = link.removeprefix("http://localhost")
        yield browser, claude, path, connect(claude, path)


def test_the_tools_work_on_the_context_open_in_the_interface(instance):
    browser, claude, path, session = instance
    cid = browser.post("/api/contexts", json={"name": "mine"}).json()["id"]
    browser.post("/api/contexts", json={"name": "other"})
    assert browser.put("/api/focus", json={"context": cid}).json()["context"] == cid

    described = call(claude, path, session, "describe_context")  # no context given
    assert described["structuredContent"]["id"] == cid
    call(claude, path, session, "add_element", type_id="lumos.sources.GaussianLaser")
    elements = browser.get(f"/api/contexts/{cid}").json()["elements"]
    assert [e["type_id"] for e in elements] == ["lumos.sources.GaussianLaser"]


def test_the_websocket_tells_the_changes_and_the_focus_of_the_assistant(instance):
    browser, claude, path, session = instance
    cid = browser.post("/api/contexts", json={"name": "mine"}).json()["id"]
    browser.put("/api/focus", json={"context": cid})
    before = browser.get(f"/api/contexts/{cid}").json()["revision"]
    with browser.websocket_connect("/api/ws", headers=ws_headers(browser)) as socket_:
        call(claude, path, session, "add_element", type_id="lumos.sources.GaussianLaser")
        loaded = call(claude, path, session, "load_preset", name="first_simulation")
        new = loaded["structuredContent"]["id"]
        changed = focused = None
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not (changed and focused):
            message = socket_.receive_json()
            if message["type"] == "context" and message["context"] == cid and message["revision"] > before:
                changed = message
            if message["type"] == "focus" and message["context"] == new:
                focused = message
        assert changed is not None, "no change of the context received"
        assert focused is not None
        assert focused["origin"] == "mcp"
        assert browser.get("/api/focus").json()["context"] == new


def test_the_view_link_opens_the_context_in_a_new_browser(instance, tmp_path):
    browser, claude, path, session = instance
    loaded = call(claude, path, session, "load_preset", name="first_simulation")
    cid = loaded["structuredContent"]["id"]
    result = call(claude, path, session, "get_view_link")
    url = result["structuredContent"]["url"]
    assert url.startswith("http://localhost/view/")
    assert url.endswith(f"?context={cid}")
    text = result["content"][0]["text"]
    assert url in text

    with TestClient(browser.app, base_url="http://localhost") as stranger:  # another browser, no cookie
        response = stranger.get(url.removeprefix("http://localhost"), follow_redirects=False)
        assert response.status_code in (302, 303, 307)
        assert response.headers["location"] == f"/?context={cid}"
        assert "lumos_session" in response.cookies
        names = [c["id"] for c in stranger.get("/api/contexts").json()]
        assert cid in names  # the same session as the connector

    with TestClient(browser.app, base_url="http://localhost") as stranger:
        refused = stranger.get("/view/not-a-secret", follow_redirects=False)
        assert refused.status_code == 404


def test_a_view_link_without_web_interface_says_how_to_get_one(tmp_path):
    import anyio
    from mcp import Client

    from lumos.mcp import create_server
    from lumos.service import Service, ServiceConfig

    service = Service(ServiceConfig(archive_root=str(tmp_path)))
    server = create_server(service)  # stdio server of a single user: no web interface

    async def main():
        async with Client(server) as client:
            result = await client.call_tool("get_view_link", {})
            assert result.is_error
            assert "lumos web" in " ".join(getattr(block, "text", "") for block in result.content)

    try:
        anyio.run(main)
    finally:
        service.shutdown()
