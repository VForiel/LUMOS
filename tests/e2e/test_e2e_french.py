"""The main path of the interface in French (step P14.11, REQ-WEB-8).

Requirements: a French-speaking user builds, edits and runs a context with every text of the interface in
French: palette, inspector, units, toolbar, results, journal and problems.
"""

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


@pytest.fixture
def browser_context_args(browser_context_args):
    return {**browser_context_args, "locale": "fr-FR"}


def test_build_edit_and_run_in_french(page):
    page.goto("/")
    expect(page.locator("html")).to_have_attribute("lang", "fr")
    expect(page.get_by_test_id("palette")).to_contain_text("Éléments")
    expect(page.get_by_test_id("presets")).to_contain_text("Bienvenue dans LUMOS")
    page.get_by_test_id("new").click()
    page.get_by_test_id("palette-search").fill("PowerMeter")
    page.get_by_test_id("palette-item").filter(has_text="PowerMeter").click()
    page.get_by_test_id("tab-problems").click()
    expect(page.get_by_test_id("tab-problems")).to_contain_text("Problèmes")
    page.get_by_test_id("palette-search").fill("GaussianLaser")
    page.get_by_test_id("palette-item").filter(has_text="GaussianLaser").click()
    inspector = page.get_by_test_id("inspector")
    expect(inspector).to_contain_text("Paramètres")
    expect(inspector).to_contain_text("Réglages")
    inspector.get_by_test_id("value-power").fill("-2")
    inspector.get_by_test_id("value-power").press("Enter")
    expect(inspector.get_by_test_id("bounds-power")).to_contain_text("Doit")
    expect(page.get_by_test_id("run")).to_contain_text("Lancer")
    page.get_by_test_id("tab-journal").click()
    expect(page.get_by_test_id("journal")).to_contain_text("Rien pour l'instant")


def test_results_in_french(page):
    summary = page.request.post("/api/presets/first_simulation/load").json()
    page.goto("/")
    page.evaluate(f"localStorage.setItem('lumos.context', '{summary['id']}')")
    page.reload()
    page.get_by_test_id("run").click()
    results = page.get_by_test_id("results")
    expect(results).to_contain_text("Dernier calcul")
    expect(results).to_contain_text("terminé", timeout=60_000)
    expect(page.get_by_test_id("toast-success")).to_contain_text("Calcul terminé")
    expect(page.get_by_test_id("tab-results")).to_contain_text("Sorties")
