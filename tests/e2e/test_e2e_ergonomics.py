"""Ergonomics of the editor (step P14.12, review of 18/10/2026; REQ-WEB-1, REQ-WEB-4).

Requirements:

- a pointer mode chooses what a drag on the empty canvas does: move the view (default) or draw a selection
  box; a button of the canvas and the keys H and V switch it;
- an edit marks the context as not saved (indicator next to its name); replacing the context (new context,
  file, preset) asks to confirm the loss of the changes, and so does closing the page; saving clears the mark;
- a link is removed on its own, its blocks stay;
- the palette (left), the inspector (right) and the bottom panel are resized by dragging their edge, hidden and
  shown again; their sizes are kept by the browser.
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


def handle(page, element, port):
    return node(page, element).locator(f'.react-flow__handle[data-handleid="{port}"]')


def drag(page, start, end, steps=12):
    page.mouse.move(*start)
    page.mouse.down()
    page.mouse.move(*end, steps=steps)
    page.wait_for_timeout(100)
    page.mouse.up()


def centre(locator):
    box = locator.bounding_box()
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


def connect(page, source, target):
    target.hover()
    source.hover()
    drag(page, centre(source), centre(target))


def transform(page):
    return page.locator(".react-flow__viewport").get_attribute("style")


@pytest.fixture
def editor(page):
    page.goto("/")
    expect(page.get_by_test_id("palette")).to_be_visible()
    page.get_by_test_id("new").click()
    expect(page.locator(".react-flow__node")).to_have_count(0)
    return page


def empty_corner(page):
    """A point of the empty canvas, away from the blocks, the controls and the minimap."""
    box = page.get_by_test_id("canvas").bounding_box()
    return box["x"] + box["width"] * 0.75, box["y"] + box["height"] * 0.3


def test_the_pointer_mode_moves_the_view_by_default_and_selects_a_zone_on_demand(editor):
    add(editor, "GaussianLaser")
    add(editor, "PowerMeter")
    expect(node(editor, "powermeter1")).to_be_visible()
    editor.mouse.click(*empty_corner(editor))  # the block just added is selected: deselect it
    expect(editor.locator(".react-flow__node.selected")).to_have_count(0)
    expect(editor.get_by_test_id("pointer-pan")).to_have_attribute("aria-pressed", "true")

    before = transform(editor)
    x, y = empty_corner(editor)
    drag(editor, (x, y), (x - 120, y + 60))
    expect(editor.locator(".react-flow__viewport")).not_to_have_attribute("style", before)
    expect(editor.locator(".react-flow__node.selected")).to_have_count(0)

    editor.get_by_test_id("pointer-select").click()
    expect(editor.get_by_test_id("pointer-select")).to_have_attribute("aria-pressed", "true")
    moved = transform(editor)
    boxes = [node(editor, n).bounding_box() for n in ("gaussianlaser1", "powermeter1")]
    left = min(b["x"] for b in boxes) - 30
    top = min(b["y"] for b in boxes) - 30
    right = max(b["x"] + b["width"] for b in boxes) + 30
    bottom = max(b["y"] + b["height"] for b in boxes) + 30
    drag(editor, (left, top), (right, bottom))
    expect(editor.locator(".react-flow__node.selected")).to_have_count(2)
    expect(editor.locator(".react-flow__viewport")).to_have_attribute("style", moved)

    editor.locator(".react-flow__pane").click(position={"x": 5, "y": 5})
    editor.keyboard.press("h")
    expect(editor.get_by_test_id("pointer-pan")).to_have_attribute("aria-pressed", "true")
    editor.keyboard.press("v")
    expect(editor.get_by_test_id("pointer-select")).to_have_attribute("aria-pressed", "true")


def test_unsaved_changes_are_flagged_and_their_loss_is_confirmed(editor):
    expect(editor.get_by_test_id("unsaved")).to_have_count(0)
    add(editor, "GaussianLaser")
    expect(editor.get_by_test_id("unsaved")).to_be_visible()

    editor.get_by_test_id("new").click()
    dialog = editor.get_by_test_id("discard-dialog")
    expect(dialog).to_be_visible()
    dialog.get_by_test_id("discard-cancel").click()
    expect(dialog).to_have_count(0)
    expect(node(editor, "gaussianlaser1")).to_be_visible()

    editor.get_by_test_id("open-presets").click()
    editor.get_by_test_id("preset-first_simulation").click()
    expect(editor.get_by_test_id("discard-dialog")).to_be_visible()
    editor.get_by_test_id("discard-cancel").click()
    expect(node(editor, "gaussianlaser1")).to_be_visible()

    with editor.expect_download():
        editor.get_by_test_id("save").click()
    expect(editor.get_by_test_id("unsaved")).to_have_count(0)
    editor.get_by_test_id("new").click()  # saved: no question
    expect(editor.get_by_test_id("discard-dialog")).to_have_count(0)
    expect(editor.locator(".react-flow__node")).to_have_count(0)

    add(editor, "PowerMeter")
    expect(editor.get_by_test_id("unsaved")).to_be_visible()
    editor.get_by_test_id("new").click()
    editor.get_by_test_id("discard-confirm").click()
    expect(editor.locator(".react-flow__node")).to_have_count(0)
    expect(editor.get_by_test_id("unsaved")).to_have_count(0)


def test_closing_the_page_with_unsaved_changes_asks_first(editor):
    add(editor, "GaussianLaser")
    expect(editor.get_by_test_id("unsaved")).to_be_visible()
    asked = []

    def answer(dialog):
        asked.append(dialog.type)
        dialog.dismiss()

    editor.on("dialog", answer)
    editor.close(run_before_unload=True)
    editor.wait_for_timeout(500)
    assert asked == ["beforeunload"]


def test_a_link_is_removed_and_its_blocks_stay(editor):
    add(editor, "GaussianLaser")
    add(editor, "PowerMeter")
    expect(node(editor, "powermeter1")).to_be_visible()
    connect(editor, handle(editor, "gaussianlaser1", "out"), handle(editor, "powermeter1", "in"))
    edge = editor.locator(".react-flow__edge")
    expect(edge).to_have_count(1)

    edge.locator(".react-flow__edge-interaction").click(force=True)
    remove = editor.get_by_test_id("remove-link")
    expect(remove).to_be_visible()
    remove.click()
    expect(editor.locator(".react-flow__edge")).to_have_count(0)
    expect(editor.locator(".react-flow__node")).to_have_count(2)

    editor.keyboard.press("Control+z")  # an edit like the others
    expect(editor.locator(".react-flow__edge")).to_have_count(1)


def width(locator):
    return locator.bounding_box()["width"]


def test_the_side_and_bottom_panels_are_resized_hidden_and_remembered(editor):
    palette = editor.get_by_test_id("palette")
    inspector = editor.get_by_test_id("inspector")
    start = width(palette)
    drag(editor, centre(editor.get_by_test_id("resize-palette")), (centre(editor.get_by_test_id("resize-palette"))[0] + 100, 400))
    assert width(palette) == pytest.approx(start + 100, abs=4)

    start_inspector = width(inspector)
    grip = centre(editor.get_by_test_id("resize-inspector"))
    drag(editor, grip, (grip[0] - 80, grip[1]))
    assert width(inspector) == pytest.approx(start_inspector + 80, abs=4)

    editor.get_by_test_id("tab-journal").click()
    bottom = editor.get_by_test_id("bottom-panel")
    start_height = bottom.bounding_box()["height"]
    grip = centre(editor.get_by_test_id("resize-bottom"))
    drag(editor, grip, (grip[0], grip[1] - 100))
    assert bottom.bounding_box()["height"] == pytest.approx(start_height + 100, abs=4)

    editor.get_by_test_id("hide-palette").click()
    expect(palette).to_have_count(0)
    editor.get_by_test_id("show-palette").click()
    expect(editor.get_by_test_id("palette")).to_be_visible()
    assert width(editor.get_by_test_id("palette")) == pytest.approx(start + 100, abs=4)

    editor.reload()
    expect(editor.get_by_test_id("palette")).to_be_visible()
    assert width(editor.get_by_test_id("palette")) == pytest.approx(start + 100, abs=4)
    assert width(editor.get_by_test_id("inspector")) == pytest.approx(start_inspector + 80, abs=4)
