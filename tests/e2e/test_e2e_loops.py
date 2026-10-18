"""Feedback loops in the browser (step P14.14, REQ-CTX-2, ADR-0020; review of 18/10/2026).

Requirements: a feedback is drawn on the canvas as a dashed link going back from its port to the block it
drives; the inspector of the source block lists its feedbacks, removes them and adds new ones (port, field,
mode); the study dialog runs a loop (number of steps and period) whose steps are plotted against time in the
study tab; the adaptive optics preset opens with its loop ready, and its residual decreases.
"""

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def open_preset(page, name):
    page.goto("/")
    expect(page.get_by_test_id("palette")).to_be_visible()
    page.get_by_test_id("open-presets").click()
    page.get_by_role("dialog").get_by_test_id(f"preset-{name}").click()
    expect(page.locator(".react-flow__node").first).to_be_visible()


def select(page, element):
    page.locator(f'.react-flow__node[data-id="{element}"]').locator("span.font-semibold").first.click()


def test_the_adaptive_optics_preset_runs_its_loop(page):
    open_preset(page, "adaptive_optics")
    expect(page.locator(".react-flow__edge.feedback")).to_have_count(1)
    page.get_by_test_id("open-study").click()
    expect(page.get_by_test_id("study-kind-loop")).to_have_attribute("aria-pressed", "true")
    page.get_by_test_id("study-steps").fill("8")
    expect(page.get_by_test_id("study-period")).to_have_value("2 ms")
    page.get_by_test_id("study-run").click()
    rows = page.get_by_test_id("study-table").locator("tbody tr")
    expect(rows).to_have_count(8, timeout=120_000)
    expect(page.get_by_test_id("study-chart").locator("path.curve")).to_be_visible()
    page.get_by_test_id("study-output").select_option("wfs.residual")

    headers = page.get_by_test_id("study-table").locator("thead th").all_inner_texts()
    column = next(i for i, text in enumerate(headers) if "wfs.residual" in text)

    def residual(row):
        return float(rows.nth(row).locator("td").nth(column).inner_text().split()[0])

    assert residual(7) < 0.5 * residual(0)


def test_feedbacks_are_removed_and_added_in_the_inspector(page):
    open_preset(page, "adaptive_optics")
    select(page, "ctrl")
    section = page.get_by_test_id("feedbacks")
    expect(section).to_contain_text("dm.commands")
    section.get_by_test_id("remove-feedback-0").click()
    expect(page.locator(".react-flow__edge.feedback")).to_have_count(0)
    expect(page.get_by_test_id("unsaved")).to_be_visible()

    section.get_by_test_id("feedback-port").select_option("commands")
    section.get_by_test_id("feedback-target").select_option("dm.commands")
    section.get_by_test_id("feedback-mode").select_option("add")
    section.get_by_test_id("feedback-add").click()
    expect(page.locator(".react-flow__edge.feedback")).to_have_count(1)
    expect(section).to_contain_text("dm.commands")
