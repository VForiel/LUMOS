"""Review of 20/10/2026 (step P15.17): narrow windows, the Outputs tab, the archives of the session.

Requirements (todo.md, "Action côté IA", 20/10/2026):

- in a narrow window the two side columns (palette of blocks, inspector) are closed when the page loads, and
  opening one closes the other; the tabs of the bottom panel collapse into a menu (three lines) showing only
  the open tab;
- the bottom tab "Results" is named "Outputs" (it shows the outputs of the last run, unlike the Study tab);
- the archive button opens a window with the switch "archive each run" and a browser of the archives of the
  session: download, and a view of their data.
"""

import re

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def test_a_narrow_window_keeps_one_side_column(browser, server):
    page = browser.new_page(viewport={"width": 1000, "height": 760}, base_url=server)
    page.goto("/")
    expect(page.get_by_test_id("show-palette")).to_be_visible()
    expect(page.get_by_test_id("show-inspector")).to_be_visible()
    page.get_by_test_id("show-palette").click()
    expect(page.get_by_test_id("hide-palette")).to_be_visible()
    page.get_by_test_id("show-inspector").click()
    expect(page.get_by_test_id("hide-inspector")).to_be_visible()
    expect(page.get_by_test_id("show-palette")).to_be_visible()  # the palette closed
    page.close()


def test_the_bottom_tabs_collapse_into_a_menu_in_a_narrow_window(browser, server):
    page = browser.new_page(viewport={"width": 1000, "height": 760}, base_url=server)
    page.goto("/")
    expect(page.get_by_test_id("tab-journal")).to_have_count(0)
    page.get_by_test_id("bottom-menu").click()
    page.get_by_test_id("menu-tab-journal").click()
    expect(page.get_by_test_id("bottom-current")).to_contain_text("Journal")
    page.close()


def test_wide_windows_keep_both_columns_and_every_tab(page):
    page.set_viewport_size({"width": 1500, "height": 900})
    page.goto("/")
    expect(page.get_by_test_id("hide-palette")).to_be_visible()
    expect(page.get_by_test_id("hide-inspector")).to_be_visible()
    expect(page.get_by_test_id("tab-journal")).to_be_visible()
    expect(page.get_by_test_id("tab-results")).to_have_text("Outputs")


def test_the_archives_of_the_session_are_browsed(page):
    page.set_viewport_size({"width": 1500, "height": 900})
    page.goto("/")
    page.get_by_test_id("presets").first.get_by_test_id("preset-first_simulation").click()
    page.get_by_test_id("archive-toggle").click()
    dialog = page.get_by_test_id("archives")
    expect(dialog).to_be_visible()
    dialog.get_by_test_id("archive-switch").check()
    page.keyboard.press("Escape")
    page.get_by_test_id("run").click()
    expect(page.get_by_test_id("output-card").first).to_be_visible(timeout=60_000)
    page.get_by_test_id("archive-toggle").click()
    dialog = page.get_by_test_id("archives")
    expect(dialog.get_by_test_id("archive-item")).to_have_count(1, timeout=10_000)
    dialog.get_by_test_id("archive-item").first.click()
    expect(dialog.get_by_test_id("archive-download")).to_have_attribute(
        "href", re.compile("/api/archives/download")
    )
    files = dialog.get_by_test_id("archive-file")
    expect(files.first).to_be_visible()
    dialog.get_by_test_id("archive-file").filter(has_text="camera.out").click()
    expect(dialog.get_by_test_id("archive-preview")).to_contain_text("DataProduct", timeout=20_000)
    expect(dialog.get_by_test_id("archive-preview").locator("img")).to_be_visible()
