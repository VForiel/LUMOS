"""The assistant panel in the browser (step P14.9, REQ-WEB-12).

Requirements: the panel asks for the Anthropic API key of the user, kept in the browser tab and sent to the
Claude API only (never to the LUMOS server); the assistant reads the tools of the MCP server of the instance,
calls them on the context open in the interface, and the interface shows the result.

The Claude API is replaced by a scripted stand-in (Playwright route): it asks for one tool call, then answers
with text, as the Messages API does (``stop_reason`` ``tool_use`` then ``end_turn``).
"""

import json
import re

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def message(content, stop_reason):
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 10},
    }


def test_the_assistant_edits_the_open_context_through_the_mcp_server(page):
    calls = []
    leaked = []

    def claude(route):
        body = route.request.post_data_json
        calls.append({"headers": route.request.headers, "body": body})
        if len(calls) == 1:
            system = body["system"] if isinstance(body["system"], str) else json.dumps(body["system"])
            context = re.search(r'context "([0-9a-f]+)"', system).group(1)
            content = [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "add_element",
                    "input": {"type_id": "lumos.sources.GaussianLaser", "name": "laser", "context": context},
                }
            ]
            reply = message(content, "tool_use")
        else:
            reply = message([{"type": "text", "text": "I added a laser."}], "end_turn")
        route.fulfill(status=200, content_type="application/json", body=json.dumps(reply))

    page.route(re.compile(r"https://api\.anthropic\.com/v1/messages.*"), claude)
    page.on(
        "request",
        lambda r: (
            leaked.append(r.url)
            if "sk-ant-test" in json.dumps(r.headers) and "anthropic.com" not in r.url
            else None
        ),
    )
    page.goto("/")
    page.get_by_test_id("new").click()
    page.get_by_test_id("tab-assistant").click()
    panel = page.get_by_test_id("assistant")
    panel.get_by_test_id("assistant-key").fill("sk-ant-test")
    panel.get_by_test_id("assistant-key-save").click()
    panel.get_by_test_id("assistant-input").fill("Add a HeNe laser")
    panel.get_by_test_id("assistant-send").click()
    expect(panel).to_contain_text("I added a laser.", timeout=30_000)
    expect(panel.get_by_test_id("tool-call")).to_contain_text("add_element")
    expect(page.locator('.react-flow__node[data-id="laser"]')).to_be_visible()
    first = calls[0]
    assert first["headers"]["x-api-key"] == "sk-ant-test"
    assert first["body"]["model"] == "claude-opus-5-5"
    assert "add_element" in {tool["name"] for tool in first["body"]["tools"]}
    second = calls[1]["body"]["messages"]
    assert second[-1]["content"][0]["type"] == "tool_result"
    assert second[-1]["content"][0]["tool_use_id"] == "toolu_1"
    assert leaked == []  # the key never reached the LUMOS server


def test_the_key_stays_in_the_tab(page):
    page.goto("/")
    page.get_by_test_id("tab-assistant").click()
    panel = page.get_by_test_id("assistant")
    panel.get_by_test_id("assistant-key").fill("sk-ant-test")
    panel.get_by_test_id("assistant-key-save").click()
    assert page.evaluate("sessionStorage.getItem('lumos.anthropicKey')") == "sk-ant-test"
    assert page.evaluate("localStorage.getItem('lumos.anthropicKey')") is None
    panel.get_by_test_id("assistant-key-forget").click()
    assert page.evaluate("sessionStorage.getItem('lumos.anthropicKey')") is None
