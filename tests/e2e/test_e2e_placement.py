"""Blocks added in quick succession never overlap (step P15.4; failure seen in CI run 39511601539).

Requirements (REQ-WEB-2, P14.12): a block added from the palette takes a free spot of the visible canvas. When
the server is slow, a second block may be added before the first one is shown: it must still take another
spot (the CI saw a lens placed on a power meter, its ports covering those of the meter).
"""

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def boxes_overlap(a, b):
    return (
        a["x"] < b["x"] + b["width"]
        and b["x"] < a["x"] + a["width"]
        and a["y"] < b["y"] + b["height"]
        and b["y"] < a["y"] + a["height"]
    )


def test_blocks_added_while_the_server_is_slow_take_different_spots(page):
    page.goto("/")
    expect(page.get_by_test_id("palette")).to_be_visible()
    page.keyboard.press("Escape")  # the gallery of presets shown on a first visit

    def slow(route):
        page.wait_for_timeout(700)
        route.continue_()

    page.route("**/api/contexts/*/elements", slow)
    items = page.get_by_test_id("palette-item")
    page.get_by_test_id("palette-search").fill("PowerMeter")
    meter = items.filter(has_text="PowerMeter").first
    page.get_by_test_id("palette-search").fill("ThinLens")
    expect(items.filter(has_text="ThinLens").first).to_be_visible()
    page.get_by_test_id("palette-search").fill("")
    for category in page.get_by_test_id("palette-category").all():
        if category.locator("button[aria-expanded='false']").count():
            category.locator("button").first.click()
    meter = items.filter(has_text="PowerMeter").first
    lens = items.filter(has_text="ThinLens").first
    meter.click()
    lens.click()  # before the meter is shown
    first = page.locator('.react-flow__node[data-id="powermeter1"]')
    second = page.locator('.react-flow__node[data-id="thinlens1"]')
    expect(first).to_be_visible(timeout=10_000)
    expect(second).to_be_visible(timeout=10_000)
    assert not boxes_overlap(first.bounding_box(), second.bounding_box())
