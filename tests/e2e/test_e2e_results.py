"""Running a context and inspecting its outputs in the browser (step P14.6, REQ-WEB-5).

Requirements: a run shows its progress and ends with the outputs, their figures and statistics; any port is
inspected from its block; figures and data are downloaded; the journal shows the runs live; problems found by
the validation are listed and lead to the element concerned.
"""

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def load(page, preset):
    summary = page.request.post(f"/api/presets/{preset}/load").json()
    page.goto("/")
    page.evaluate(f"localStorage.setItem('lumos.context', '{summary['id']}')")
    page.reload()
    expect(page.locator(".react-flow__node").first).to_be_visible()


def test_a_run_shows_its_outputs_with_figures(page):
    load(page, "vlt_psf")
    page.get_by_test_id("run").click()
    results = page.get_by_test_id("results")
    expect(results).to_be_visible()
    expect(results.get_by_test_id("output-card")).to_have_count(2, timeout=60_000)
    figure = results.locator("img").first
    expect(figure).to_be_visible(timeout=30_000)
    page.wait_for_function("(img) => img.complete && img.naturalWidth > 0", arg=figure.element_handle())
    expect(results).to_contain_text("image.out")


def test_the_run_starts_from_the_keyboard_and_the_journal_shows_it(page):
    load(page, "first_simulation")
    page.locator(".react-flow__pane").click()
    page.keyboard.press("Control+Enter")
    page.get_by_test_id("tab-journal").click()
    expect(page.get_by_test_id("journal")).to_contain_text("run", timeout=60_000)


def test_any_port_is_inspected_and_downloaded(page):
    load(page, "first_simulation")
    node = (
        page.locator(".react-flow__node")
        .filter(has=page.locator('.react-flow__handle[data-handleid="out"]'))
        .first
    )
    node.get_by_test_id("node-inspect").click()  # the labels of the ports start links (P14.16)
    inspected = page.get_by_test_id("inspected")
    expect(inspected).to_be_visible()
    expect(inspected.get_by_test_id("statistics")).to_be_visible(timeout=30_000)
    with page.expect_download() as download:
        inspected.get_by_test_id("download-png").click()
    path = download.value.path()
    with open(path, "rb") as stream:
        assert stream.read(8) == b"\x89PNG\r\n\x1a\n"


def test_problems_lead_to_the_element(page):
    page.goto("/")
    page.get_by_test_id("new").click()
    page.get_by_test_id("palette-search").fill("PowerMeter")
    page.get_by_test_id("palette-item").filter(has_text="PowerMeter").click()
    page.get_by_test_id("tab-problems").click()
    problems = page.get_by_test_id("problems")
    expect(problems).to_contain_text("powermeter1.in")
    problems.get_by_role("button").filter(has_text="powermeter1").first.click()
    expect(page.get_by_test_id("element-name")).to_have_value("powermeter1")
