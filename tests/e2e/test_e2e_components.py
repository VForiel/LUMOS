"""The component editor (step P15.15, ADR-0025; docs/specs/pistes_evolution.md §2).

Requirements: "New component" in the palette opens a form (identifier, texts, license, template, parameters,
behaviour, checks); the problems of the component are listed while it is typed and saving is possible only
without problem; a saved component appears in the category "Components" of the palette and is added like any
block.
"""

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def test_a_component_is_made_in_the_form_and_used(page):
    page.goto("/")
    expect(page.get_by_test_id("palette")).to_be_visible()
    page.get_by_test_id("new").click()
    page.get_by_test_id("new-component").click()
    editor = page.get_by_test_id("component-editor")
    expect(editor).to_be_visible()
    editor.get_by_test_id("component-author").fill("e2e")
    editor.get_by_test_id("component-name").fill("Dimmer")
    editor.get_by_test_id("component-title").fill("Dimmer")
    editor.get_by_test_id("component-template").select_option("gain_phase")
    editor.get_by_test_id("component-gain").fill("0.25")
    # no description, no license, no check yet: problems are listed and saving is refused
    expect(editor.get_by_test_id("component-problems")).to_contain_text("license", timeout=10_000)
    expect(editor.get_by_test_id("component-save")).to_be_disabled()
    editor.get_by_test_id("component-description").fill("Divides the power by four.")
    editor.get_by_test_id("component-license").fill("MIT")
    editor.get_by_test_id("component-add-check").click()
    editor.get_by_test_id("check-wavelength-0").fill("1 um")
    editor.get_by_test_id("check-power-0").fill("4 mW")
    editor.get_by_test_id("check-expect-0").fill("1 mW")
    expect(editor.get_by_test_id("component-problems")).to_have_count(0, timeout=10_000)
    editor.get_by_test_id("component-save").click()
    expect(editor).to_have_count(0, timeout=10_000)
    page.get_by_test_id("palette-search").fill("Dimmer")
    item = page.get_by_test_id("palette-item").filter(has=page.get_by_text("Dimmer", exact=True))
    expect(item).to_be_visible(timeout=10_000)
    item.first.click()
    expect(page.locator('.react-flow__node[data-id="dimmer1"]')).to_be_visible()
