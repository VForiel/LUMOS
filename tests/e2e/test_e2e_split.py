"""Splitting a bus port in the editor (step P15.10; docs/specs/pistes_evolution.md §1, step 4).

Requirements: the inspector of a block offers "Split" on its bus ports of known width; the Mux (inputs) or
Demux (outputs) inserted is drawn against the block, on the side of the port, moves with the block and is
deleted with it.
"""

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def node(page, element):
    return page.locator(f'.react-flow__node[data-id="{element}"]')


@pytest.fixture
def editor(page):
    page.goto("/")
    expect(page.get_by_test_id("palette")).to_be_visible()
    page.get_by_test_id("new").click()
    page.get_by_test_id("palette-search").fill("MMI")
    page.get_by_test_id("palette-item").filter(has=page.get_by_text("MMI", exact=True)).first.click()
    page.get_by_test_id("palette-search").fill("")
    expect(node(page, "mmi1")).to_be_visible()
    return page


def test_a_split_port_is_drawn_against_its_block_and_follows_it(editor):
    inspector = editor.get_by_test_id("inspector")
    inspector.get_by_test_id("split-in").click()
    attached = node(editor, "mmi1_in")
    expect(attached).to_be_visible(timeout=10_000)
    block = node(editor, "mmi1").bounding_box()
    mux = attached.bounding_box()
    assert mux["x"] + mux["width"] <= block["x"] + 2  # on the left of the block
    assert mux["x"] + mux["width"] >= block["x"] - 40  # against it

    header = node(editor, "mmi1").locator("text=mmi1").first
    box = header.bounding_box()
    editor.mouse.move(box["x"] + 5, box["y"] + 5)
    editor.mouse.down()
    editor.mouse.move(box["x"] + 125, box["y"] + 65, steps=10)
    editor.mouse.up()
    editor.wait_for_timeout(300)
    moved, after = attached.bounding_box(), node(editor, "mmi1").bounding_box()
    assert after["x"] - block["x"] > 60  # the block moved (snapped to the grid of the canvas)
    assert moved["x"] - mux["x"] == pytest.approx(after["x"] - block["x"], abs=1)
    assert moved["y"] - mux["y"] == pytest.approx(after["y"] - block["y"], abs=1)

    node(editor, "mmi1").click()
    editor.keyboard.press("Delete")
    expect(node(editor, "mmi1")).to_have_count(0, timeout=10_000)
    expect(attached).to_have_count(0)


def test_ports_of_free_width_offer_no_split(editor):
    editor.get_by_test_id("palette-search").fill("Waveguide")
    editor.get_by_test_id("palette-item").filter(
        has=editor.get_by_text("Waveguide", exact=True)
    ).first.click()
    expect(node(editor, "waveguide1")).to_be_visible()
    expect(editor.get_by_test_id("inspector").get_by_test_id("split-in")).to_have_count(0)
