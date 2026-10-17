"""Editing a context in the browser (step P14.5, REQ-WEB-1, REQ-WEB-3, REQ-WEB-4, REQ-WEB-6).

Requirements: blocks are connected with the mouse, and incompatible ports refuse the connection; parameters
are edited with their unit (conversion when the unit changes, bounds checked); every edit can be undone and
redone from the keyboard; a selection becomes a group and back; blocks are copied, pasted and deleted from the
keyboard, or added by name (Ctrl+K); a context is saved as a ``.lumos`` file, opened again, and exported as a
Python script.
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


def drag(page, source, target):
    source.hover()
    page.mouse.down()
    box = target.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=12)
    page.mouse.up()


def select(page, element, additive=False):
    header = node(page, element).locator("span.font-semibold").first
    if additive:
        header.click(modifiers=["Shift"])
    else:
        header.click()


@pytest.fixture
def editor(page):
    page.goto("/")
    expect(page.get_by_test_id("palette")).to_be_visible()
    page.get_by_test_id("new").click()
    expect(page.locator(".react-flow__node")).to_have_count(0)
    return page


def test_blocks_are_connected_with_the_mouse(editor):
    add(editor, "GaussianLaser")
    add(editor, "PowerMeter")
    expect(node(editor, "powermeter1")).to_be_visible()
    drag(editor, handle(editor, "gaussianlaser1", "out"), handle(editor, "powermeter1", "in"))
    expect(editor.locator(".react-flow__edge")).to_have_count(1)


def test_incompatible_ports_refuse_the_connection(editor):
    add(editor, "PowerMeter")
    add(editor, "ThinLens")
    expect(node(editor, "thinlens1")).to_be_visible()
    drag(editor, handle(editor, "powermeter1", "out"), handle(editor, "thinlens1", "in"))  # data into optics
    editor.wait_for_timeout(500)
    expect(editor.locator(".react-flow__edge")).to_have_count(0)


def test_a_parameter_is_edited_with_its_unit(editor):
    add(editor, "GaussianLaser")
    select(editor, "gaussianlaser1")
    inspector = editor.get_by_test_id("inspector")
    expect(inspector.get_by_test_id("unit-power")).to_have_value("mW")
    inspector.get_by_test_id("value-power").fill("3")
    inspector.get_by_test_id("value-power").press("Enter")
    expect(node(editor, "gaussianlaser1")).to_contain_text("3 mW")
    inspector.get_by_test_id("unit-power").select_option("W")  # same power, shown in another unit
    expect(inspector.get_by_test_id("value-power")).to_have_value("0.003")
    expect(node(editor, "gaussianlaser1")).to_contain_text("3 mW")
    inspector.get_by_test_id("value-power").fill("0.005")
    inspector.get_by_test_id("value-power").press("Enter")
    expect(node(editor, "gaussianlaser1")).to_contain_text("5 mW")


def test_values_out_of_bounds_are_flagged_and_not_applied(editor):
    add(editor, "GaussianLaser")
    select(editor, "gaussianlaser1")
    inspector = editor.get_by_test_id("inspector")
    inspector.get_by_test_id("value-power").fill("-1")
    inspector.get_by_test_id("value-power").press("Enter")
    expect(inspector.get_by_test_id("bounds-power")).to_be_visible()
    expect(node(editor, "gaussianlaser1")).to_contain_text("1 mW")


def test_edits_are_undone_and_redone_from_the_keyboard(editor):
    add(editor, "GaussianLaser")
    select(editor, "gaussianlaser1")
    value = editor.get_by_test_id("inspector").get_by_test_id("value-power")
    value.fill("7")
    value.press("Enter")
    expect(node(editor, "gaussianlaser1")).to_contain_text("7 mW")
    editor.locator(".react-flow__pane").click()
    editor.keyboard.press("Control+z")
    expect(node(editor, "gaussianlaser1")).to_contain_text("1 mW")
    editor.keyboard.press("Control+Shift+z")
    expect(node(editor, "gaussianlaser1")).to_contain_text("7 mW")
    editor.get_by_test_id("undo").click()
    expect(node(editor, "gaussianlaser1")).to_contain_text("1 mW")


def test_a_selection_becomes_a_group_and_back(editor):
    add(editor, "ThinLens")
    add(editor, "FreeSpace")
    expect(node(editor, "freespace1")).to_be_visible()
    drag(editor, handle(editor, "thinlens1", "out"), handle(editor, "freespace1", "in"))
    expect(editor.locator(".react-flow__edge")).to_have_count(1)
    select(editor, "thinlens1")
    select(editor, "freespace1", additive=True)
    editor.keyboard.press("Control+g")
    editor.get_by_test_id("dialog-input").fill("relay")
    editor.get_by_test_id("dialog-confirm").click()
    expect(node(editor, "relay")).to_be_visible()
    expect(node(editor, "thinlens1")).to_have_count(0)
    expect(handle(editor, "relay", "in")).to_have_count(1)
    select(editor, "relay")
    editor.get_by_test_id("ungroup").click()
    expect(node(editor, "thinlens1")).to_be_visible()
    expect(node(editor, "relay")).to_have_count(0)


def test_blocks_are_copied_pasted_and_deleted_from_the_keyboard(editor):
    add(editor, "ThinLens")
    select(editor, "thinlens1")
    editor.keyboard.press("Control+c")
    editor.keyboard.press("Control+v")
    expect(node(editor, "thinlens2")).to_be_visible()
    select(editor, "thinlens2")
    editor.keyboard.press("Delete")
    expect(node(editor, "thinlens2")).to_have_count(0)


def test_blocks_are_added_by_name(editor):
    editor.locator(".react-flow__pane").click()
    editor.keyboard.press("Control+k")
    editor.get_by_test_id("quick-add").fill("thin lens")
    editor.get_by_test_id("quick-add").press("Enter")
    expect(node(editor, "thinlens1")).to_be_visible()


def test_a_context_is_saved_opened_and_exported_as_python(editor, tmp_path):
    add(editor, "GaussianLaser")
    expect(node(editor, "gaussianlaser1")).to_be_visible()
    with editor.expect_download() as download:
        editor.get_by_test_id("save").click()
    path = tmp_path / download.value.suggested_filename
    download.value.save_as(path)
    assert path.suffix == ".lumos"
    editor.get_by_test_id("new").click()
    expect(editor.locator(".react-flow__node")).to_have_count(0)
    editor.get_by_test_id("open-file").set_input_files(str(path))
    expect(node(editor, "gaussianlaser1")).to_be_visible()
    editor.get_by_test_id("python").click()
    expect(editor.get_by_test_id("python-code")).to_contain_text("Context(")
