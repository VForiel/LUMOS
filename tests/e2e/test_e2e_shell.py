"""The shell of the web interface in a browser (step P14.4, REQ-WEB-1, REQ-WEB-2, REQ-WEB-8).

Requirements: the interface loads without errors; its palette is generated from the registry of elements
(one entry per type, grouped by category, searchable); a block added from the palette shows the ports of its
type; the language follows the browser and can be changed (English, French), and the choice is remembered.
"""

import re

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def test_the_interface_loads_without_errors(page, errors):
    page.goto("/")
    expect(page).to_have_title(re.compile("LUMOS"))
    expect(page.get_by_test_id("palette")).to_be_visible()
    expect(page.get_by_test_id("canvas")).to_be_visible()
    page.wait_for_load_state("networkidle")
    assert errors == []


def test_the_palette_is_generated_from_the_registry(page):
    page.goto("/")
    palette = page.request.get("/api/palette").json()
    categories = {item["category"] for item in palette["types"]}
    expect(page.get_by_test_id("palette-category")).to_have_count(len(categories))
    expect(page.get_by_test_id("palette-item")).to_have_count(0)  # categories start collapsed (P14.16)
    for header in page.get_by_test_id("palette-category").all():
        header.locator("button").first.click()
    expect(page.get_by_test_id("palette-item")).to_have_count(len(palette["types"]))
    page.get_by_test_id("palette-search").fill("laser")
    items = page.get_by_test_id("palette-item")
    expect(items.filter(has_text="GaussianLaser")).to_be_visible()
    expect(items.filter(has_text="LaserSource")).to_be_visible()
    expect(items.filter(has_text="CircularAperture")).to_have_count(0)


def test_a_block_added_from_the_palette_shows_the_ports_of_its_type(page):
    page.goto("/")
    page.get_by_test_id("palette-search").fill("PowerMeter")
    page.get_by_test_id("palette-item").filter(has_text="PowerMeter").click()
    node = page.locator(".react-flow__node").filter(has_text="powermeter1")
    expect(node).to_be_visible()
    expect(node.locator('.react-flow__handle[data-handleid="in"]')).to_have_count(1)
    expect(node.locator('.react-flow__handle[data-handleid="out"]')).to_have_count(1)


def test_the_language_follows_the_browser_and_can_be_changed(browser, base_url):
    context = browser.new_context(base_url=base_url, locale="fr-FR")
    page = context.new_page()
    page.goto("/")
    expect(page.locator("html")).to_have_attribute("lang", "fr")
    expect(page.get_by_test_id("palette-search")).to_have_attribute("placeholder", re.compile("Rechercher"))
    page.get_by_test_id("language").select_option("en")
    expect(page.locator("html")).to_have_attribute("lang", "en")
    expect(page.get_by_test_id("palette-search")).to_have_attribute("placeholder", re.compile("Search"))
    page.reload()
    expect(page.locator("html")).to_have_attribute("lang", "en")
    context.close()
