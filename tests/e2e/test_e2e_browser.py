"""The results browser (step P14.20, review of 19/10/2026; REQ-WEB-5).

Requirements: a window lists the runs and studies of the session, newest first; a run shows the figures of its
outputs; a study recorded (switch *Record the figures* of the study dialog) is browsed point by point with a
slider (and a play button), the figure and the values of the chosen port following the slider.
"""

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


@pytest.fixture
def bench(page):
    page.goto("/")
    expect(page.get_by_test_id("palette")).to_be_visible()
    page.get_by_test_id("new").click()
    cid = page.request.get("/api/focus").json()["context"]
    laser = {"power": "1 mW", "n_pixels": 32, "pixel_scale": "0.1 mm"}
    page.request.post(
        f"/api/contexts/{cid}/elements",
        data={"type_id": "lumos.sources.GaussianLaser", "name": "laser", "params": laser},
    )
    page.request.post(
        f"/api/contexts/{cid}/elements", data={"type_id": "lumos.detectors.Camera", "name": "camera"}
    )
    page.request.post(f"/api/contexts/{cid}/connections", data={"source": "laser.out", "target": "camera.in"})
    expect(page.locator(".react-flow__node")).to_have_count(2, timeout=10_000)
    return page


def loaded(page, image):
    page.wait_for_function("(img) => img.complete && img.naturalWidth > 0", arg=image.element_handle())


def test_a_recorded_sweep_is_browsed_point_by_point(bench):
    bench.get_by_test_id("run").click()
    expect(bench.get_by_test_id("output-card")).to_have_count(1, timeout=60_000)

    bench.get_by_test_id("open-study").click()
    dialog = bench.get_by_test_id("study-dialog")
    dialog.get_by_test_id("study-kind-sweep").click()
    dialog.get_by_test_id("study-param-0").select_option("laser.power")
    dialog.get_by_test_id("study-start-0").fill("1")
    dialog.get_by_test_id("study-stop-0").fill("4")
    dialog.get_by_test_id("study-num-0").fill("4")
    dialog.get_by_test_id("study-record").check()
    dialog.get_by_test_id("study-run").click()
    expect(bench.get_by_test_id("study-table").locator("tbody tr")).to_have_count(4, timeout=60_000)

    bench.get_by_test_id("open-browser").click()
    browser = bench.get_by_test_id("results-browser")
    expect(browser).to_be_visible()
    jobs = browser.get_by_test_id("browser-job")
    expect(jobs).to_have_count(2)
    expect(jobs.first).to_contain_text("sweep")  # newest first

    jobs.first.click()
    slider = browser.get_by_test_id("browser-slider")
    expect(slider).to_be_visible()
    figure = browser.get_by_test_id("browser-figure")
    loaded(bench, figure)
    first = figure.get_attribute("src")
    slider.fill("2")
    expect(browser.get_by_test_id("browser-step")).to_contain_text("3 / 4")
    expect(browser.get_by_test_id("browser-step")).to_contain_text("3 mW")
    expect(figure).not_to_have_attribute("src", first)
    loaded(bench, figure)

    browser.get_by_test_id("browser-play").click()
    expect(browser.get_by_test_id("browser-step")).to_contain_text("4 / 4", timeout=10_000)

    jobs.nth(1).click()  # the run
    expect(browser.get_by_test_id("browser-output").first).to_contain_text("camera.out")
    loaded(bench, browser.get_by_test_id("browser-output").first.locator("img"))
