"""Tests for the documentation "AI assistant" (step P13.7, REQ-MCP-6).

Requirements: a step-by-step configuration of Claude Code, Claude Desktop and other MCP clients, and example
conversations. The reference page must list every tool, resource and prompt that the server really offers, so
that the documentation cannot drift from the code.
"""

import json
import re
from pathlib import Path

import anyio
import pytest

pytest.importorskip("mcp")

from mcp import Client

from lumos.mcp import create_server
from lumos.service import Service, ServiceConfig

DOCS = Path(__file__).resolve().parents[3] / "docs" / "assistant"


def read(name):
    return (DOCS / name).read_text(encoding="utf-8")


def json_blocks(text):
    return [json.loads(block) for block in re.findall(r"```json\n(.*?)```", text, re.DOTALL)]


def test_the_assistant_section_has_its_pages():
    index = read("index.md")
    for page in ("setup", "conversations", "reference"):
        assert (DOCS / f"{page}.md").is_file()
        assert page in index


def test_setup_of_claude_code_claude_desktop_and_other_clients():
    text = read("setup.md")
    assert 'pip install "lumos-maxima[mcp]"' in text
    assert "claude mcp add lumos -- lumos mcp" in text  # Claude Code, stdio
    assert "claude mcp add --transport http lumos http://127.0.0.1:8765/mcp" in text  # Claude Code, HTTP
    assert "claude_desktop_config.json" in text
    desktop = [block for block in json_blocks(text) if "mcpServers" in block]
    assert desktop, "no Claude Desktop configuration"
    assert desktop[0]["mcpServers"]["lumos"]["command"] == "lumos"
    assert desktop[0]["mcpServers"]["lumos"]["args"] == ["mcp"]
    assert "lumos mcp --transport http" in text
    assert "--enable-operation" in text


def test_example_conversations_cover_the_use_cases():
    text = read("conversations.md")
    sections = re.split(r"\n## ", text)[1:]
    assert len(sections) >= 3
    lowered = text.lower()
    assert "3 λ/d" in lowered  # UC9: contrast of a coronagraph at 3 λ/D
    assert "mtf" in lowered  # UC9: a bench to measure the MTF of a lens
    assert "sampl" in lowered  # UC9: why is my simulation under-sampled?


def test_the_reference_lists_everything_the_server_offers():
    server = create_server(Service(ServiceConfig(operation_enabled=True)))

    async def scenario():
        async with Client(server) as client:
            tools = [tool.name for tool in (await client.list_tools()).tools]
            resources = [str(r.uri) for r in (await client.list_resources()).resources]
            templates = [t.uri_template for t in (await client.list_resource_templates()).resource_templates]
            prompts = [prompt.name for prompt in (await client.list_prompts()).prompts]
            return tools, resources, templates, prompts

    tools, resources, templates, prompts = anyio.run(scenario)
    text = read("reference.md")
    missing = [name for name in [*tools, *resources, *templates, *prompts] if f"`{name}`" not in text]
    assert missing == []
    conversations = read("conversations.md")
    used = set(re.findall(r"`(\w+)\(", conversations))
    assert used, "the conversations show no tool call"
    assert used <= set(tools), f"unknown tools in the conversations: {used - set(tools)}"
