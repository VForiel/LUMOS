"""Studies in the browser: seed, free parameters, sweeps and Monte-Carlo runs (step P14.13, REQ-CTX-10).

Requirements (review of 18/10/2026): the seed of every random process of a context is set (or drawn) in the
inspector of the context; a parameter is freed with a prior from the inspector of its element; the study
dialog starts a sweep or a Monte-Carlo run; the study tab of the bottom panel shows its points as a chart
(curve for a sweep, histogram for a Monte-Carlo run) and a table, downloaded as CSV; a Monte-Carlo preset
opens with its study ready.
"""

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def select(page, element):
    page.locator(f'.react-flow__node[data-id="{element}"]').locator("span.font-semibold").first.click()


def open_preset(page, name):
    page.goto("/")
    expect(page.get_by_test_id("palette")).to_be_visible()
    page.get_by_test_id("open-presets").click()
    page.get_by_role("dialog").get_by_test_id(f"preset-{name}").click()  # not the gallery of a first visit
    expect(page.locator(".react-flow__node").first).to_be_visible()


def test_the_seed_is_set_and_drawn_in_the_inspector(page):
    open_preset(page, "first_simulation")
    expect(page.get_by_test_id("context-title")).to_be_visible()  # nothing selected: the context
    seed = page.get_by_test_id("seed")
    seed.fill("42")
    seed.press("Enter")
    expect(page.get_by_test_id("unsaved")).to_be_visible()
    page.reload()
    expect(page.get_by_test_id("seed")).to_have_value("42")
    page.get_by_test_id("new-seed").click()
    expect(page.get_by_test_id("seed")).not_to_have_value("42")


def test_a_parameter_is_freed_with_a_prior(page):
    open_preset(page, "first_simulation")
    select(page, "lens")
    inspector = page.get_by_test_id("inspector")
    inspector.get_by_test_id("free-focal_length").click()
    inspector.get_by_test_id("prior-kind-focal_length").select_option("normal")
    inspector.get_by_test_id("prior-a-focal_length").fill("200")
    inspector.get_by_test_id("prior-b-focal_length").fill("2")
    inspector.get_by_test_id("prior-apply-focal_length").click()
    expect(inspector.get_by_test_id("prior-summary-focal_length")).to_contain_text("200 mm")
    expect(page.locator('.react-flow__node[data-id="lens"]').get_by_test_id("free-badge")).to_be_visible()

    page.get_by_test_id("open-study").click()
    page.get_by_test_id("study-kind-sample").click()
    expect(page.get_by_test_id("study-free")).to_contain_text("lens.focal_length")


def test_a_monte_carlo_preset_runs_its_study(page):
    open_preset(page, "monte_carlo")
    page.get_by_test_id("open-study").click()
    expect(page.get_by_test_id("study-kind-sample")).to_have_attribute("aria-pressed", "true")
    page.get_by_test_id("study-n").fill("12")
    page.get_by_test_id("study-run").click()
    expect(page.get_by_test_id("study-table").locator("tbody tr")).to_have_count(12, timeout=60_000)
    expect(page.get_by_test_id("study-chart").locator("rect.bar").first).to_be_visible()
    with page.expect_download() as download:
        page.get_by_test_id("study-csv").click()
    assert download.value.suggested_filename.endswith(".csv")


def test_a_sweep_draws_a_curve(page):
    open_preset(page, "monte_carlo")
    page.get_by_test_id("open-study").click()
    page.get_by_test_id("study-kind-sweep").click()
    page.get_by_test_id("study-param-0").select_option("focus.distance")
    page.get_by_test_id("study-start-0").fill("190")
    page.get_by_test_id("study-stop-0").fill("210")
    page.get_by_test_id("study-num-0").fill("5")
    expect(page.get_by_test_id("study-count")).to_contain_text("5")
    page.get_by_test_id("study-run").click()
    expect(page.get_by_test_id("study-table").locator("tbody tr")).to_have_count(5, timeout=60_000)
    expect(page.get_by_test_id("study-chart").locator("path.curve")).to_be_visible()
    page.get_by_test_id("study-metric").select_option("max")
    expect(page.get_by_test_id("study-chart").locator("circle.point")).to_have_count(5)
