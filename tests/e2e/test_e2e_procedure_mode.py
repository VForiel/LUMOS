"""The Graph / Procedures mode and the export of the procedures (step P15.18, review of 20/10/2026).

Requirements: the switch between the graph and the procedures is a mode of the whole interface, in the top
bar: in the procedures mode the editor of procedures takes the whole working area, without the palette and
the inspector, which come back with the graph. The Python export of a context includes its procedures (one
function each), an option on by default that can be turned off.
"""

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


@pytest.fixture
def bench(page):
    page.goto("/")
    expect(page.get_by_test_id("palette")).to_be_visible()
    # the gallery of presets is shown on a first visit
    page.get_by_test_id("presets").first.get_by_test_id("preset-dark_then_measure").click()
    expect(page.locator(".react-flow__node").first).to_be_visible()
    return page


def test_the_procedures_mode_takes_the_whole_working_area(bench):
    switch = bench.locator("header").get_by_test_id("view-procedures")
    expect(switch).to_be_visible()
    switch.click()
    expect(bench.get_by_test_id("procedures")).to_be_visible()
    expect(bench.get_by_test_id("palette")).to_have_count(0)
    expect(bench.get_by_test_id("show-palette")).to_have_count(0)
    expect(bench.get_by_test_id("inspector")).to_have_count(0)
    expect(bench.get_by_test_id("show-inspector")).to_have_count(0)
    bench.locator("header").get_by_test_id("view-graph").click()
    expect(bench.get_by_test_id("palette")).to_be_visible()
    expect(bench.get_by_test_id("canvas")).to_be_visible()


def test_the_python_export_includes_the_procedures(bench):
    bench.get_by_test_id("python").click()
    code = bench.get_by_test_id("python-code")
    option = bench.get_by_test_id("python-procedures")
    expect(option).to_be_checked()
    expect(code).to_contain_text("def dark_then_measure(ctx):")
    option.uncheck()
    expect(code).not_to_contain_text("def dark_then_measure")
    expect(code).to_contain_text("Context(")
