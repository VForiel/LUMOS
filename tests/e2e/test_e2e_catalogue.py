"""The Catalogue tab of the palette (step P15.16, ADR-0026; docs/specs/pistes_evolution.md §2).

Requirements: the palette has a "Catalogue" tab listing the contributions of the community (title, author,
version, license); a declarative component is imported with one click, after which it is an element of the
palette; a shared preset opens as a new context.
"""

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def test_a_component_of_the_catalogue_is_imported_in_one_click(page):
    page.goto("/")
    expect(page.get_by_test_id("palette")).to_be_visible()
    page.get_by_test_id("palette-tab-catalogue").click()
    entry = page.get_by_test_id("catalogue-entry").filter(has=page.get_by_text("Neutral density", exact=True))
    expect(entry).to_be_visible(timeout=10_000)
    expect(entry).to_contain_text("MIT")
    entry.get_by_test_id("catalogue-use").click()
    page.get_by_test_id("palette-tab-elements").click()
    page.get_by_test_id("palette-search").fill("Neutral density")
    item = page.get_by_test_id("palette-item").filter(has=page.get_by_text("Neutral density", exact=True))
    expect(item).to_be_visible(timeout=10_000)


def test_a_shared_preset_opens_as_a_context(page):
    page.goto("/")
    expect(page.get_by_test_id("palette")).to_be_visible()
    page.get_by_test_id("palette-tab-catalogue").click()
    entry = page.get_by_test_id("catalogue-entry").filter(has_text="bench").first
    expect(entry).to_be_visible(timeout=10_000)
    entry.get_by_test_id("catalogue-use").click()
    expect(page.locator(".react-flow__node")).to_have_count(3, timeout=10_000)
