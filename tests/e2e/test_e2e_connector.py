"""Connector link in the assistant panel (step P14.15).

Requirement: the settings of the assistant offer to use a Claude or ChatGPT subscription through a connector
link: the panel creates the secret MCP URL of the session, shows it with a copy button and the steps to add it
to Claude or ChatGPT, and revokes it.
"""

import re

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def test_a_connector_link_is_created_shown_and_revoked(page):
    page.goto("/")
    page.get_by_test_id("tab-assistant").click()
    panel = page.get_by_test_id("assistant")
    panel.get_by_test_id("assistant-connector-create").click()
    url = panel.get_by_test_id("assistant-connector-url")
    expect(url).to_have_value(re.compile(r"^http://localhost:\d+/mcp/[A-Za-z0-9_-]{20,}$"))
    expect(panel).to_contain_text("Claude")
    expect(panel).to_contain_text("ChatGPT")
    panel.get_by_test_id("assistant-connector-revoke").click()
    expect(panel.get_by_test_id("assistant-connector-url")).to_have_count(0)


# --- Following the assistant (step P14.18, review of 19/10/2026) ---------------------------------------

MCP_HEADERS = {"accept": "application/json, text/event-stream", "content-type": "application/json"}


def mcp_call(url, name, arguments):
    """Call a tool at a connector link like Claude.ai does: JSON-RPC over HTTP, no cookie."""
    import json

    import httpx

    with httpx.Client(timeout=60) as client:
        init = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "e2e", "version": "1"},
            },
        }
        response = client.post(url, content=json.dumps(init), headers=MCP_HEADERS)
        headers = {
            **MCP_HEADERS,
            "mcp-session-id": response.headers["mcp-session-id"],
            "mcp-protocol-version": "2025-06-18",
        }
        client.post(
            url,
            content=json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            headers=headers,
        )
        message = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        response = client.post(url, content=json.dumps(message), headers=headers)
        for line in response.text.splitlines():
            if line.startswith("data:"):
                return json.loads(line[5:])["result"]
        return response.json()["result"]


def test_the_page_follows_the_assistant_and_its_view_link_opens_the_context(page, browser, base_url):
    page.goto("/")
    page.get_by_test_id("new").click()
    expect(page.locator(".react-flow__node")).to_have_count(0)
    link = page.request.post("/api/assistant/link").json()["url"]

    # the assistant edits the open context: the block appears
    mcp_call(link, "add_element", {"type_id": "lumos.detectors.PowerMeter"})
    expect(page.locator(".react-flow__node")).to_have_count(1, timeout=10_000)

    # the assistant loads a preset: the page switches to it
    mcp_call(link, "load_preset", {"name": "first_simulation"})
    expect(page.get_by_test_id("context-name")).to_contain_text("First simulation", timeout=10_000)
    expect(page.locator(".react-flow__node")).to_have_count(4)

    # another browser opens the link given to the user
    view = mcp_call(link, "get_view_link", {})["structuredContent"]["url"]
    other = browser.new_context(base_url=base_url)
    fresh = other.new_page()
    fresh.goto(view)
    expect(fresh.locator(".react-flow__node")).to_have_count(4, timeout=10_000)
    other.close()
