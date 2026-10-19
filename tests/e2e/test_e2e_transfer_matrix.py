"""Editor of the transfer matrix of a simplified MMI (step P14.19, review of 19/10/2026).

Requirements: the inspector of a ``TransferMatrix`` block shows its matrix as a table of cells, an amplitude
and a phase per cell (output ``i`` in rows, input ``j`` in columns); a cell typed changes the block; inputs
and outputs are added and removed; the table says whether the matrix is lossless, lossy or amplifying
(refused). Expected values: the default matrix is the ideal 3-dB coupler (1/√2 = 0.707107, phases 0 and π/2).
"""

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


@pytest.fixture
def editor(page):
    page.goto("/")
    expect(page.get_by_test_id("palette")).to_be_visible()
    page.get_by_test_id("new").click()
    page.get_by_test_id("palette-search").fill("TransferMatrix")
    page.get_by_test_id("palette-item").filter(has_text="TransferMatrix").first.click()
    expect(page.get_by_test_id("matrix-editor")).to_be_visible()
    return page


def test_the_matrix_is_shown_and_edited_cell_by_cell(editor):
    matrix = editor.get_by_test_id("matrix-editor")
    expect(matrix.get_by_test_id("matrix-a-0-1")).to_have_value("0.707107")
    expect(matrix.get_by_test_id("matrix-p-0-1")).to_have_value("1.5708")
    expect(matrix.get_by_test_id("matrix-status")).to_contain_text("lossless")

    matrix.get_by_test_id("matrix-a-0-1").fill("0.5")
    matrix.get_by_test_id("matrix-a-0-1").press("Enter")
    expect(matrix.get_by_test_id("matrix-status")).to_contain_text("lossy", timeout=10_000)
    matrix.get_by_test_id("matrix-a-0-1").fill("1.5")
    matrix.get_by_test_id("matrix-a-0-1").press("Enter")
    expect(matrix.get_by_test_id("matrix-status")).to_contain_text("amplif", timeout=10_000)
    editor.get_by_test_id("tab-problems").click()
    expect(editor.get_by_test_id("problems")).to_contain_text("amplif")


def test_inputs_and_outputs_are_added_and_removed(editor):
    matrix = editor.get_by_test_id("matrix-editor")
    matrix.get_by_test_id("matrix-add-input").click()
    expect(matrix.get_by_test_id("matrix-a-1-2")).to_be_visible(timeout=10_000)
    matrix.get_by_test_id("matrix-add-output").click()
    expect(matrix.get_by_test_id("matrix-a-2-2")).to_be_visible(timeout=10_000)
    expect(matrix.get_by_test_id("matrix-shape")).to_contain_text("3 × 3")
    matrix.get_by_test_id("matrix-remove-output").click()
    matrix.get_by_test_id("matrix-remove-input").click()
    expect(matrix.get_by_test_id("matrix-a-2-2")).to_have_count(0, timeout=10_000)
    expect(matrix.get_by_test_id("matrix-shape")).to_contain_text("2 × 2")
