"""A top bar that fits every window (step P15.19, review of 20/10/2026).

Requirements: whatever the width of the window, every control of the top bar shown is inside the window
(none cut off); the mode switch and Run stay visible; the controls without room go into a "More" menu, from
which they work the same; on a wide window everything is in the bar and there is no "More" menu.
"""

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def controls_inside(page, width):
    """Right edges of the visible controls of the top bar that go past the window."""
    return page.evaluate(
        """(width) => [...document.querySelectorAll('header button, header select')]
            .filter((e) => e.offsetParent !== null)
            .map((e) => e.getBoundingClientRect())
            .filter((r) => r.width > 0 && r.right > width + 0.5)
            .map((r) => r.right)""",
        width,
    )


@pytest.mark.parametrize("width", [360, 640, 800, 1100])
def test_the_top_bar_fits_and_the_rest_is_in_more(page, width):
    page.set_viewport_size({"width": width, "height": 700})
    page.goto("/")
    page.get_by_test_id("presets").first.get_by_test_id("preset-dark_then_measure").click()
    header = page.locator("header")
    expect(header.get_by_test_id("run")).to_be_visible()
    expect(header.get_by_test_id("view-procedures")).to_be_visible()
    assert controls_inside(page, width) == []
    header.get_by_test_id("more").click()
    menu = page.get_by_test_id("more-menu")
    expect(menu).to_be_visible()
    expect(menu.get_by_test_id("more-language")).to_be_visible()
    if width < 1024:
        with page.expect_download() as download:
            menu.get_by_test_id("more-save").click()
        assert download.value.suggested_filename.endswith(".lumos")
    else:
        menu.get_by_test_id("more-shortcuts").click()
        expect(page.get_by_role("dialog")).to_be_visible()


def test_a_wide_window_has_everything_in_the_bar(page):
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto("/")
    header = page.locator("header")
    expect(header.get_by_test_id("save")).to_be_visible()
    expect(header.get_by_test_id("language")).to_be_visible()
    expect(header.get_by_test_id("more")).to_be_hidden()
    assert controls_inside(page, 1440) == []
