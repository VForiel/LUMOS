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
