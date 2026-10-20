"""Buses in the editor (step P15.7, ADR-0023; docs/specs/pistes_evolution.md §1.1).

Requirements: a port carrying a bus of known width draws one pin per lane, grouped by a brace, and a single
handle (a link carries the whole bus); the ports of a Demux or a Mux follow its number of lanes; a Demux and a
Mux are wired port by port like any block, and a link carrying a bus is drawn thicker.
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
    target.hover()
    source.hover()
    page.mouse.down()
    box = target.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=12)
    page.wait_for_timeout(150)
    page.mouse.up()


@pytest.fixture
def editor(page):
    page.goto("/")
    expect(page.get_by_test_id("palette")).to_be_visible()
    page.get_by_test_id("new").click()
    return page


def test_the_ports_of_a_mux_follow_its_number_of_lanes(editor):
    add(editor, "Mux")
    mux = node(editor, "mux1")
    expect(mux.locator(".react-flow__handle")).to_have_count(3)  # in0, in1, out
    expect(mux.get_by_test_id("bus-pins").last).to_have_attribute("data-lanes", "2")
    value = editor.get_by_test_id("inspector").get_by_test_id("value-n_lanes")
    value.fill("4")
    value.press("Enter")
    expect(handle(editor, "mux1", "in3")).to_be_visible(timeout=10_000)
    expect(mux.get_by_test_id("bus-pins").last).to_have_attribute("data-lanes", "4")
    expect(mux.locator('[data-testid="bus-pins"]').last.locator("line")).to_have_count(4)


def test_a_demux_and_a_mux_are_wired_lane_by_lane(editor):
    add(editor, "Demux")
    add(editor, "Mux")
    drag(editor, handle(editor, "demux1", "out0"), handle(editor, "mux1", "in1"))
    drag(editor, handle(editor, "demux1", "out1"), handle(editor, "mux1", "in0"))
    expect(editor.locator(".react-flow__edge")).to_have_count(2, timeout=10_000)
    add(editor, "Waveguide")
    drag(editor, handle(editor, "mux1", "out"), handle(editor, "waveguide1", "in"))
    edges = editor.locator(".react-flow__edge-path")
    expect(edges).to_have_count(3, timeout=10_000)
    widths = edges.evaluate_all("(paths) => paths.map((p) => p.style.strokeWidth)")
    assert widths.count("3.5") == 1  # the bus out of the Mux; the lanes of the Demux are single
