"""Blocks of the editor (step P14.16, review of 19/10/2026; REQ-WEB-1, REQ-WEB-4, REQ-WEB-5).

Requirements:

- adding or removing a block leaves the other blocks where they are, also in a context whose positions were
  laid out automatically (presets without saved positions); no two blocks end up on top of each other;
- the handle of a port holds its name: the whole label starts or ends a connection, and it grows sideways
  in place when the pointer is over it (it does not slide off the edge of the block);
- the header of a block has an *Inspect* button showing its first output in the results;
- the categories of the palette are collapsed until opened (a search shows its matches);
- the content of a result card never spills out of the card;
- selecting a block shows the inspector when it was hidden.
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


def load(page, preset):
    summary = page.request.post(f"/api/presets/{preset}/load").json()
    page.goto("/")
    page.evaluate(f"localStorage.setItem('lumos.context', '{summary['id']}')")
    page.reload()
    expect(page.locator(".react-flow__node").first).to_be_visible()
    return summary


def drag(page, start, end, steps=12):
    page.mouse.move(*start)
    page.mouse.down()
    page.mouse.move(*end, steps=steps)
    page.wait_for_timeout(100)
    page.mouse.up()


def centre(box):
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


def flow_positions(page):
    """Positions of the blocks in the coordinates of the graph (independent of the view)."""
    return page.evaluate(
        """() => Object.fromEntries([...document.querySelectorAll('.react-flow__node')].map((n) => {
            const m = /translate\\(([-0-9.]+)px, ?([-0-9.]+)px\\)/.exec(n.style.transform);
            return [n.dataset.id, [Number(m[1]), Number(m[2])]];
        }))"""
    )


def overlapping(page):
    boxes = page.evaluate(
        """() => [...document.querySelectorAll('.react-flow__node')].map((n) => {
            const r = n.getBoundingClientRect();
            return [n.dataset.id, r.left, r.top, r.right, r.bottom];
        })"""
    )
    pairs = []
    for i, (a, ax0, ay0, ax1, ay1) in enumerate(boxes):
        for b, bx0, by0, bx1, by1 in boxes[i + 1 :]:
            if min(ax1, bx1) - max(ax0, bx0) > 20 and min(ay1, by1) - max(ay0, by0) > 20:
                pairs.append((a, b))
    return pairs


def test_adding_and_removing_a_block_leaves_the_others_in_place(page):
    load(page, "first_simulation")  # a preset without saved positions: laid out automatically
    page.wait_for_timeout(500)
    before = flow_positions(page)
    assert len(before) >= 3
    assert overlapping(page) == []

    add(page, "PowerMeter")
    expect(page.locator(".react-flow__node")).to_have_count(len(before) + 1)
    page.wait_for_timeout(300)
    after = flow_positions(page)
    assert {k: after[k] for k in before} == before
    added = next(k for k in after if k not in before)

    page.get_by_test_id("delete").click()  # the block just added is selected
    expect(node(page, added)).to_have_count(0)
    page.wait_for_timeout(300)
    assert flow_positions(page) == before

    # and after a reload: the positions were saved with the context
    page.reload()
    expect(page.locator(".react-flow__node").first).to_be_visible()
    page.wait_for_timeout(500)
    assert flow_positions(page) == before
    assert overlapping(page) == []


def test_removing_a_block_of_a_laid_out_context_keeps_the_others(page):
    before = load(page, "first_simulation")
    page.wait_for_timeout(500)
    positions = flow_positions(page)
    first = before["elements"][0]["id"]
    node(page, first).click()
    page.get_by_test_id("delete").click()
    expect(node(page, first)).to_have_count(0)
    page.wait_for_timeout(300)
    assert flow_positions(page) == {k: v for k, v in positions.items() if k != first}


@pytest.fixture
def editor(page):
    page.goto("/")
    expect(page.get_by_test_id("palette")).to_be_visible()
    page.get_by_test_id("new").click()
    expect(page.locator(".react-flow__node")).to_have_count(0)
    return page


def test_a_port_label_is_its_handle_and_grows_sideways_in_place(editor):
    add(editor, "GaussianLaser")
    add(editor, "PowerMeter")
    source = handle(editor, "gaussianlaser1", "out")
    target = handle(editor, "powermeter1", "in")
    expect(source).to_contain_text("out")
    expect(target).to_contain_text("in")
    box = source.bounding_box()
    assert box["width"] >= 24 and box["height"] >= 16  # easy to catch

    editor.mouse.move(10, 10)
    editor.wait_for_timeout(200)
    still = source.bounding_box()
    source.hover()
    editor.wait_for_timeout(300)
    hovered = source.bounding_box()
    assert hovered["height"] == pytest.approx(still["height"], abs=1)  # it does not grow downwards...
    assert hovered["width"] > still["width"]  # ...but sideways,
    assert hovered["x"] + hovered["width"] == pytest.approx(still["x"] + still["width"], abs=1.5)  # from its edge
    assert centre(hovered)[1] == pytest.approx(centre(still)[1], abs=1)

    # a drag from the label of the output to the label of the input connects them
    start = (still["x"] + 8, centre(still)[1])
    tbox = target.bounding_box()
    drag(editor, start, (tbox["x"] + tbox["width"] - 6, centre(tbox)[1]))
    expect(editor.locator(".react-flow__edge")).to_have_count(1)


def test_the_inspect_button_of_a_block_shows_its_output(editor):
    add(editor, "GaussianLaser")
    node(editor, "gaussianlaser1").get_by_test_id("node-inspect").click()
    inspected = editor.get_by_test_id("inspected")
    expect(inspected).to_be_visible()
    expect(inspected).to_contain_text("gaussianlaser1.out")
    expect(inspected.get_by_test_id("statistics")).to_be_visible(timeout=30_000)


def test_the_categories_of_the_palette_start_collapsed(editor):
    categories = editor.get_by_test_id("palette-category")
    expect(categories.first).to_be_visible()
    expect(editor.get_by_test_id("palette-item")).to_have_count(0)
    categories.first.locator("button").first.click()
    expect(categories.first.get_by_test_id("palette-item").first).to_be_visible()
    editor.get_by_test_id("palette-search").fill("PowerMeter")
    expect(editor.get_by_test_id("palette-item").filter(has_text="PowerMeter")).to_be_visible()


def test_result_cards_hold_their_content(page):
    load(page, "vlt_psf")
    page.get_by_test_id("run").click()
    cards = page.get_by_test_id("results").get_by_test_id("output-card")
    expect(cards).to_have_count(2, timeout=60_000)
    expect(cards.first.get_by_test_id("statistics")).to_be_visible(timeout=30_000)
    bottom = page.get_by_test_id("bottom-panel")
    for size in (280, 200, 150):  # a low bottom panel too
        grip = centre(page.get_by_test_id("resize-bottom").bounding_box())
        drag(page, grip, (grip[0], grip[1] + bottom.bounding_box()["height"] - size))
        assert bottom.bounding_box()["height"] == pytest.approx(size, abs=6)
        spill = page.evaluate(
            """() => [...document.querySelectorAll('[data-testid="output-card"], [data-testid="inspected"]')]
                .map((card) => {
                    const box = card.getBoundingClientRect();
                    const style = getComputedStyle(card);
                    const scrolls = ['auto', 'scroll'].includes(style.overflowY);
                    const out = [...card.querySelectorAll('*')].some((child) => {
                        const r = child.getBoundingClientRect();
                        return r.height > 0 && r.bottom > box.bottom + 1;
                    });
                    return out && !scrolls;
                })
                .some(Boolean)"""
        )
        assert not spill


def test_selecting_a_block_shows_the_hidden_inspector(editor):
    add(editor, "GaussianLaser")
    editor.get_by_test_id("hide-inspector").click()
    expect(editor.get_by_test_id("inspector")).to_have_count(0)
    editor.locator(".react-flow__pane").click(position={"x": 5, "y": 5})
    node(editor, "gaussianlaser1").click()
    expect(editor.get_by_test_id("inspector")).to_be_visible()
    expect(editor.get_by_test_id("element-name")).to_have_value("gaussianlaser1")
