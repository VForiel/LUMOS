"""The Procedures tab (step P15.12, ADR-0024; docs/specs/pistes_evolution.md §3).

Requirements: a "Procedures" tab edits the procedures of the context with blocks (Blockly); the preset "dark
then measure" opens with its procedure drawn as blocks; a procedure is run from the tab and its named results
are listed; a new procedure starts from an empty workspace and blocks dragged from the toolbox are saved with
the context; "Export to Python" gives an equivalent script.
"""

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e

#: Blocks of the workspace (the toolbox flyout holds blocks too, in another SVG).
BLOCKS = "svg.blocklySvg .blocklyBlockCanvas .blocklyDraggable"


@pytest.fixture
def procedures(page):
    page.goto("/")
    expect(page.get_by_test_id("palette")).to_be_visible()
    # the gallery of presets is shown on a first visit
    page.get_by_test_id("presets").first.get_by_test_id("preset-dark_then_measure").click()
    page.get_by_test_id("view-procedures").click()
    expect(page.get_by_test_id("procedures")).to_be_visible()
    return page


def test_the_preset_procedure_is_drawn_as_blocks_and_runs(procedures):
    tab = procedures.get_by_test_id("procedures")
    expect(tab.get_by_test_id("procedure-select")).to_have_value("dark then measure")
    expect(tab.locator(BLOCKS).first).to_be_visible(timeout=10_000)
    assert tab.locator(BLOCKS).count() >= 8
    expect(tab.get_by_test_id("procedure-problems")).to_have_count(0)
    tab.get_by_test_id("procedure-run").click()
    results = tab.get_by_test_id("procedure-results")
    expect(results).to_contain_text("science", timeout=60_000)
    expect(results).to_contain_text("dark")


def test_a_procedure_is_exported_to_python(procedures):
    tab = procedures.get_by_test_id("procedures")
    tab.get_by_test_id("procedure-python").click()
    code = procedures.get_by_test_id("procedure-code")
    expect(code).to_contain_text('ctx["shutter"].close()')
    expect(code).to_contain_text("for ")


def test_a_new_procedure_is_saved_with_the_context(procedures):
    tab = procedures.get_by_test_id("procedures")
    tab.get_by_test_id("procedure-new").click()
    procedures.get_by_test_id("dialog-input").fill("check")
    procedures.get_by_test_id("dialog-confirm").click()
    expect(tab.get_by_test_id("procedure-select")).to_have_value("check")
    # one "wait" block from the toolbox, dropped inside the procedure block
    tab.get_by_test_id("procedure-add-wait").click()
    expect(tab.get_by_test_id("procedure-saved")).to_be_visible(timeout=10_000)
    procedures.reload()
    procedures.get_by_test_id("view-procedures").click()
    procedures.get_by_test_id("procedures").get_by_test_id("procedure-select").select_option("check")
    expect(procedures.get_by_test_id("procedures").locator(BLOCKS)).to_have_count(2, timeout=10_000)
