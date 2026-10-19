"""Studies of any numeric value, intermediate outputs and help on feedbacks (P14.17, review of 19/10/2026).

Requirements:

- a numeric setting (``n_pixels`` of a laser) has the dice button freeing it with a prior, like a parameter;
- an array parameter (``transmissions`` of a filter wheel) frees one of its components, chosen in the
  editor of the prior; the summary names the component;
- the study dialog sweeps settings and components, and offers every output of every block, also those
  connected to another block (intermediate outputs);
- the feedback section of the inspector has an information button explaining how loops work.
"""

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def add(page, title):
    page.get_by_test_id("palette-search").fill(title)
    page.get_by_test_id("palette-item").filter(has_text=title).first.click()
    page.get_by_test_id("palette-search").fill("")


def node(page, element):
    return page.locator(f'.react-flow__node[data-id="{element}"]')


@pytest.fixture
def bench(page):
    page.goto("/")
    expect(page.get_by_test_id("palette")).to_be_visible()
    page.get_by_test_id("new").click()
    summary = page.request.get("/api/focus").json()
    cid = summary["context"]
    for type_id, name in (
        ("lumos.sources.GaussianLaser", "laser"),
        ("lumos.lab.FilterWheel", "wheel"),
        ("lumos.detectors.PowerMeter", "meter"),
    ):
        page.request.post(f"/api/contexts/{cid}/elements", data={"type_id": type_id, "name": name})
    page.request.post(f"/api/contexts/{cid}/connections", data={"source": "laser.out", "target": "wheel.in"})
    page.request.post(f"/api/contexts/{cid}/connections", data={"source": "wheel.out", "target": "meter.in"})
    expect(page.locator(".react-flow__node")).to_have_count(3, timeout=10_000)  # the page follows the changes
    return page


def test_a_numeric_setting_is_freed_with_a_prior(bench):
    node(bench, "laser").click()
    inspector = bench.get_by_test_id("inspector")
    inspector.get_by_test_id("free-n_pixels").click()
    inspector.get_by_test_id("prior-kind-n_pixels").select_option("uniform")
    inspector.get_by_test_id("prior-a-n_pixels").fill("31.5")
    inspector.get_by_test_id("prior-b-n_pixels").fill("64.5")
    inspector.get_by_test_id("prior-apply-n_pixels").click()
    expect(inspector.get_by_test_id("prior-summary-n_pixels")).to_contain_text("U[31.5, 64.5]")


def test_a_component_of_an_array_is_freed(bench):
    node(bench, "wheel").click()
    inspector = bench.get_by_test_id("inspector")
    inspector.get_by_test_id("free-transmissions").click()
    inspector.get_by_test_id("prior-index-transmissions").select_option("transmissions[2]")
    inspector.get_by_test_id("prior-apply-transmissions").click()
    summary = inspector.get_by_test_id("prior-summary-transmissions[2]")
    expect(summary).to_contain_text("transmissions[2]")
    expect(summary).to_contain_text("N(")


def test_the_study_dialog_sweeps_them_and_offers_intermediate_outputs(bench):
    bench.get_by_test_id("open-study").click()
    dialog = bench.get_by_test_id("study-dialog")
    dialog.get_by_test_id("study-kind-sweep").click()
    choices = dialog.get_by_test_id("study-param-0").locator("option").all_inner_texts()
    assert any(c.startswith("laser.n_pixels") for c in choices)
    assert any(c.startswith("wheel.transmissions[0]") for c in choices)
    dialog.get_by_test_id("study-param-0").select_option("wheel.transmissions[0]")
    dialog.get_by_test_id("study-start-0").fill("0.2")
    dialog.get_by_test_id("study-stop-0").fill("0.6")
    dialog.get_by_test_id("study-num-0").fill("3")
    dialog.get_by_test_id("study-intermediate").click()
    dialog.get_by_test_id("study-output-wheel.out").check()
    dialog.get_by_test_id("study-run").click()
    table = bench.get_by_test_id("study-table")
    expect(table).to_be_visible(timeout=60_000)
    expect(table).to_contain_text("wheel.transmissions[0]")
    expect(table).to_contain_text("wheel.out")
    expect(table.locator("tbody tr")).to_have_count(3)


def test_the_feedback_section_explains_the_loops(bench):
    node(bench, "wheel").click()
    inspector = bench.get_by_test_id("inspector")
    inspector.get_by_test_id("feedback-help").click()
    help_text = inspector.get_by_test_id("feedback-help-text")
    expect(help_text).to_be_visible()
    expect(help_text).to_contain_text("loop")
    expect(help_text).to_contain_text("Study")
