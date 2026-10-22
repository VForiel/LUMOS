"""Presets and the step-by-step tutorial in the browser (step P14.7, REQ-WEB-7, REQ-WEB-8, UC5).

Requirements: a first visit offers the presets (the same ``.lumos`` files as the notebooks) and the tour;
the gallery is also reachable from the toolbar; the tour is overlaid on the interface, step by step, in the
language of the user, and walks through a preset from loading to results. UC5: a visitor of the public
instance loads a preset and follows the tutorial without an account.
"""

import re

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def test_a_first_visit_offers_the_presets_and_the_tour(page):
    page.goto("/")
    welcome = page.get_by_test_id("presets")
    expect(welcome).to_be_visible()
    expect(welcome.get_by_test_id("start-tour")).to_be_visible()
    welcome.get_by_test_id("preset-coronagraph").click()
    expect(page.locator(".react-flow__node")).to_have_count(6)
    expect(page.get_by_test_id("context-name")).to_have_text("Vortex coronagraph")
    page.reload()  # the next visit goes back to the work in progress
    expect(page.get_by_test_id("presets")).to_have_count(0)


def test_the_gallery_opens_from_the_toolbar(page):
    page.goto("/")
    page.get_by_test_id("presets").get_by_test_id("preset-first_simulation").click()
    expect(page.get_by_test_id("presets")).to_have_count(0)  # the preset opened, the first gallery closed
    page.get_by_test_id("open-presets").click()
    gallery = page.get_by_test_id("presets")
    cards = gallery.get_by_test_id(re.compile(r"^preset-"))
    expect(cards).to_have_count(len(page.request.get("/api/presets").json()))
    gallery.get_by_test_id("preset-vlt_psf").click()
    expect(page.get_by_test_id("context-name")).to_contain_text("VLT")


def tour(page):
    """Press Next until the tour ends; return the titles of its steps."""
    popover = page.locator(".driver-popover")
    expect(popover).to_be_visible()
    titles = []
    for _ in range(30):
        title = popover.locator(".driver-popover-title").inner_text()
        titles.append(title)
        popover.locator(".driver-popover-next-btn").click()
        # the next step (or the end of the tour) may take a run: wait for one of them, not a fixed delay
        page.wait_for_function(
            """(previous) => {
                const current = document.querySelector('.driver-popover-title');
                return !current || current.textContent !== previous;
            }""",
            arg=title,
            timeout=60_000,
        )
        if page.locator(".driver-popover").count() == 0:
            break
    return titles


def test_the_tour_walks_through_a_preset_to_its_results(page):
    page.goto("/")
    page.get_by_test_id("presets").get_by_test_id("start-tour").click()
    titles = tour(page)
    assert len(titles) >= 6
    expect(page.get_by_test_id("output-card").first).to_be_visible(timeout=60_000)  # the tour ran it
    expect(page.locator(".driver-popover")).to_have_count(0)


def test_the_tour_speaks_french(browser, base_url):
    context = browser.new_context(base_url=base_url, locale="fr-FR", viewport={"width": 1440, "height": 900})
    page = context.new_page()
    page.goto("/")
    page.get_by_test_id("presets").get_by_test_id("start-tour").click()
    expect(page.locator(".driver-popover-title")).to_contain_text("Bienvenue")
    context.close()


def test_uc5_a_visitor_of_the_public_instance_follows_the_tutorial(browser, public_server):
    context = browser.new_context(base_url=public_server, viewport={"width": 1440, "height": 900})
    page = context.new_page()
    page.goto("/")
    assert page.request.get("/api/info").json()["role"] == "simulator"
    page.get_by_test_id("presets").get_by_test_id("start-tour").click()
    tour(page)
    expect(page.get_by_test_id("output-card").first).to_be_visible(timeout=60_000)
    context.close()
