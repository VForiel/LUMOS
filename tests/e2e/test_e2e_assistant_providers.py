"""Providers of the assistant panel (step P14.15, review of 18/10/2026).

Requirements: besides Claude (Anthropic API key), the assistant works with any provider of the OpenAI chat
completions API with tools: OpenAI, Google Gemini (its OpenAI-compatible endpoint), OpenRouter and any
compatible endpoint (base URL, optional key: Mistral, Groq, a local Ollama or vLLM...). OpenRouter offers a
login without copying a key: an OAuth PKCE flow (``code_challenge`` = base64url of the SHA-256 of the
verifier, RFC 7636) in a popup, whose code is exchanged for a key created on the account of the user. Keys
stay in the browser tab and go to their provider only, never to the LUMOS server.

The providers are replaced by scripted stand-ins (Playwright routes) that ask for one tool call, then answer
with text, as the chat completions API does (``finish_reason`` ``tool_calls`` then ``stop``).
"""

import base64
import hashlib
import json
import re
from urllib.parse import parse_qs, urlparse

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def completion(message, finish_reason):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }


def scripted(calls):
    """A chat completions stand-in: one call of add_element on the open context, then a text answer."""

    def answer(route):
        body = route.request.post_data_json
        calls.append({"url": route.request.url, "headers": route.request.headers, "body": body})
        if len(calls) == 1:
            system = next(m["content"] for m in body["messages"] if m["role"] == "system")
            context = re.search(r'context "([0-9a-f]+)"', system).group(1)
            arguments = {"type_id": "lumos.sources.GaussianLaser", "name": "laser", "context": context}
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "add_element", "arguments": json.dumps(arguments)},
                    }
                ],
            }
            reply = completion(message, "tool_calls")
        else:
            reply = completion({"role": "assistant", "content": "I added a laser."}, "stop")
        route.fulfill(status=200, content_type="application/json", body=json.dumps(reply))

    return answer


def leak_watcher(page, secret):
    leaked = []
    page.on(
        "request",
        lambda r: (
            leaked.append(r.url)
            if secret in json.dumps(r.headers) and r.url.startswith(page.url[:21])
            else None
        ),
    )
    return leaked


def ask(page):
    panel = page.get_by_test_id("assistant")
    panel.get_by_test_id("assistant-input").fill("Add a HeNe laser")
    panel.get_by_test_id("assistant-send").click()
    expect(panel).to_contain_text("I added a laser.", timeout=30_000)
    expect(panel.get_by_test_id("tool-call")).to_contain_text("add_element")
    expect(page.locator('.react-flow__node[data-id="laser"]')).to_be_visible()


def check_protocol(calls):
    first = calls[0]["body"]
    assert any(t["type"] == "function" and t["function"]["name"] == "add_element" for t in first["tools"])
    assert first["messages"][0]["role"] == "system"
    second = calls[1]["body"]["messages"]
    assert second[-2]["tool_calls"][0]["id"] == "call_1"
    assert second[-1]["role"] == "tool"
    assert second[-1]["tool_call_id"] == "call_1"


def open_panel(page):
    page.goto("/")
    page.get_by_test_id("new").click()
    page.get_by_test_id("tab-assistant").click()
    return page.get_by_test_id("assistant")


def test_openai_with_an_api_key(page):
    calls = []
    page.route(re.compile(r"https://api\.openai\.com/v1/chat/completions"), scripted(calls))
    leaked = leak_watcher(page, "sk-openai-test")
    panel = open_panel(page)
    panel.get_by_test_id("assistant-provider").select_option("openai")
    expect(panel.get_by_test_id("assistant-model")).not_to_have_value("")
    panel.get_by_test_id("assistant-key").fill("sk-openai-test")
    panel.get_by_test_id("assistant-key-save").click()
    ask(page)
    assert calls[0]["headers"]["authorization"] == "Bearer sk-openai-test"
    check_protocol(calls)
    assert leaked == []
    assert page.evaluate("localStorage.getItem('lumos.assistant.provider')") == "openai"
    assert "sk-openai-test" not in json.dumps(page.evaluate("Object.assign({}, localStorage)"))


def test_gemini_uses_its_openai_compatible_endpoint(page):
    calls = []
    page.route(
        re.compile(r"https://generativelanguage\.googleapis\.com/v1beta/openai/chat/completions"),
        scripted(calls),
    )
    panel = open_panel(page)
    panel.get_by_test_id("assistant-provider").select_option("gemini")
    panel.get_by_test_id("assistant-key").fill("gemini-test-key")
    panel.get_by_test_id("assistant-key-save").click()
    ask(page)
    assert calls[0]["headers"]["authorization"] == "Bearer gemini-test-key"
    assert calls[0]["body"]["model"].startswith("gemini")


def test_any_compatible_endpoint_without_a_key(page):
    calls = []
    page.route(re.compile(r"https://llm\.example\.org/v1/chat/completions"), scripted(calls))
    panel = open_panel(page)
    panel.get_by_test_id("assistant-provider").select_option("custom")
    panel.get_by_test_id("assistant-base-url").fill("https://llm.example.org/v1")
    panel.get_by_test_id("assistant-model").fill("local-model")
    panel.get_by_test_id("assistant-key-save").click()
    ask(page)
    assert calls[0]["body"]["model"] == "local-model"
    assert "authorization" not in calls[0]["headers"]
    check_protocol(calls)


def test_openrouter_login_with_oauth_pkce(page, context):
    exchanged = []

    def authorize(route):
        # the OpenRouter login page: the user accepts, OpenRouter returns to the callback with a code
        query = parse_qs(urlparse(route.request.url).query)
        callback = query["callback_url"][0]
        state = query.get("state", [""])[0]
        assert query["code_challenge_method"] == ["S256"]
        exchanged.append({"challenge": query["code_challenge"][0], "state": state})
        target = f"{callback}?code=one-time-code&state={state}"
        route.fulfill(
            status=200,
            content_type="text/html",
            body=f"<script>location.href = {json.dumps(target)}</script>",
        )

    def keys(route):
        body = route.request.post_data_json
        assert body["code"] == "one-time-code"
        digest = hashlib.sha256(body["code_verifier"].encode()).digest()
        assert base64.urlsafe_b64encode(digest).rstrip(b"=").decode() == exchanged[0]["challenge"]
        assert body["code_challenge_method"] == "S256"
        route.fulfill(status=200, content_type="application/json", body=json.dumps({"key": "sk-or-test"}))

    context.route(re.compile(r"https://openrouter\.ai/auth\?.*"), authorize)
    context.route(re.compile(r"https://openrouter\.ai/api/v1/auth/keys"), keys)
    calls = []
    context.route(re.compile(r"https://openrouter\.ai/api/v1/chat/completions"), scripted(calls))

    panel = open_panel(page)
    panel.get_by_test_id("assistant-provider").select_option("openrouter")
    with page.expect_popup() as popup:
        panel.get_by_test_id("assistant-openrouter-login").click()
    popup.value.wait_for_event("close", timeout=15_000)
    expect(panel.get_by_test_id("assistant-input")).to_be_visible()
    ask(page)
    assert calls[0]["headers"]["authorization"] == "Bearer sk-or-test"
    assert exchanged[0]["state"]
