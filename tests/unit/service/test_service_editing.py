"""Editing operations of the service for the visual editor (step P14.5, REQ-WEB-1, REQ-WEB-4, REQ-WEB-6).

Requirements: the positions of the blocks are kept in the context (``ui`` section of the ``.lumos`` file,
ignored by the library); every edit can be undone and redone; elements can be renamed; a selection of elements
becomes a group (its free ports exposed) and a group can be dissolved; elements can be copied and pasted; a
group can be saved as a reusable block (a ``.lumos`` file of kind ``group``, readable by ``lumos.io``) and
inserted elsewhere.

Expected values: structure (identifiers, connections) by construction; physical results by invariance
(grouping, ungrouping or undoing a change does not change what a power meter reads) and by energy conservation
(a power meter behind a laser of power P reads P).
"""

import json

import pytest
from astropy import units as u

import lumos.io
from lumos.core.group import ElementGroup
from lumos.service import Service, ServiceConfig, ServiceError

LASER = "lumos.sources.GaussianLaser"
METER = "lumos.detectors.PowerMeter"
LENS = "lumos.optics.ThinLens"
GAP = "lumos.optics.FreeSpace"


@pytest.fixture
def service(tmp_path):
    service = Service(ServiceConfig(archive_root=str(tmp_path)))
    yield service
    service.shutdown()


@pytest.fixture
def session(service):
    return service.create_session(origin="web").id


def bench(service, session):
    """laser → lens → gap → meter."""
    ctx = service.create_context(session, name="bench").id
    service.add_element(session, ctx, LASER, name="laser", params={"power": "2 mW"})
    service.add_element(session, ctx, LENS, name="lens", params={"focal_length": "200 mm"})
    service.add_element(session, ctx, GAP, name="gap", params={"distance": "100 mm"})
    service.add_element(session, ctx, METER, name="meter")
    for source, target in [("laser.out", "lens.in"), ("lens.out", "gap.in"), ("gap.out", "meter.in")]:
        service.connect(session, ctx, source, target)
    return ctx


def reading(service, session, ctx):
    job = service.wait_job(session, service.run(session, ctx, ["meter.out"]).id, timeout=120)
    assert job.status == "done", job.error
    summary = service.get_result(session, job.id, "meter.out")
    return (summary.values[0] * u.Unit(summary.unit)).to_value(u.mW)


def top(summary):
    return sorted(e.id for e in summary.elements if "." not in e.id)


def links(summary):
    return sorted((c.source, c.target) for c in summary.connections)


# --- Layout ------------------------------------------------------------------------------------------------


def test_positions_are_kept_in_the_context(service, session):
    ctx = bench(service, session)
    summary = service.update_ui(session, ctx, {"positions": {"laser": [10, 20], "meter": [600, 20]}})
    assert summary.ui["positions"]["laser"] == [10, 20]
    service.update_ui(session, ctx, {"collapsed": {"relay": True}})  # other keys are merged
    document = json.loads(service.export_lumos(session, ctx))
    assert document["ui"]["positions"]["meter"] == [600, 20]
    assert document["ui"]["collapsed"] == {"relay": True}


# --- Undo and redo -----------------------------------------------------------------------------------------


def test_every_edit_can_be_undone_and_redone(service, session):
    ctx = bench(service, session)
    service.set_param(session, ctx, "laser.power", "3 mW")
    assert reading(service, session, ctx) == pytest.approx(3.0, rel=1e-3)
    summary = service.undo(session, ctx)
    assert summary.can_redo
    assert reading(service, session, ctx) == pytest.approx(2.0, rel=1e-3)
    summary = service.undo(session, ctx)  # the connection gap → meter
    assert ("gap.out", "meter.in") not in links(summary)
    summary = service.redo(session, ctx)
    assert ("gap.out", "meter.in") in links(summary)
    service.redo(session, ctx)
    assert reading(service, session, ctx) == pytest.approx(3.0, rel=1e-3)
    assert not service.describe_context(session, ctx).can_redo


def test_a_new_edit_forgets_what_could_be_redone(service, session):
    ctx = bench(service, session)
    service.remove_element(session, ctx, "meter")
    service.undo(session, ctx)
    assert service.describe_context(session, ctx).can_redo
    service.set_param(session, ctx, "laser.power", "1 mW")
    assert not service.describe_context(session, ctx).can_redo


def test_undo_without_history_is_explained(service, session):
    ctx = service.create_context(session).id
    assert not service.describe_context(session, ctx).can_undo
    with pytest.raises(ServiceError) as error:
        service.undo(session, ctx)
    assert error.value.hint
    with pytest.raises(ServiceError):
        service.redo(session, ctx)


def test_failed_edits_are_not_recorded(service, session):
    ctx = bench(service, session)
    before = service.describe_context(session, ctx)
    with pytest.raises(ServiceError):
        service.set_param(session, ctx, "laser.power", "3 mm")
    service.undo(session, ctx)  # undoes the last real edit (gap → meter), not the failed one
    assert ("gap.out", "meter.in") not in links(service.describe_context(session, ctx))
    assert ("gap.out", "meter.in") in links(before)


# --- Renaming ----------------------------------------------------------------------------------------------


def test_renaming_an_element_keeps_its_connections_and_position(service, session):
    ctx = bench(service, session)
    service.update_ui(session, ctx, {"positions": {"laser": [0, 0]}})
    summary = service.rename_element(session, ctx, "laser", "source")
    assert "source" in top(summary)
    assert "laser" not in top(summary)
    assert ("source.out", "lens.in") in links(summary)
    assert summary.ui["positions"]["source"] == [0, 0]
    assert reading(service, session, ctx) == pytest.approx(2.0, rel=1e-3)


@pytest.mark.parametrize("name", ["meter", "two words", "a.b", ""])
def test_invalid_or_taken_names_are_refused(service, session, name):
    ctx = bench(service, session)
    with pytest.raises(ServiceError):
        service.rename_element(session, ctx, "laser", name)


# --- Groups ------------------------------------------------------------------------------------------------


def test_a_selection_becomes_a_group_with_its_free_ports_exposed(service, session):
    ctx = bench(service, session)
    before = reading(service, session, ctx)
    summary = service.group_elements(session, ctx, ["lens", "gap"], name="relay")
    assert top(summary) == ["laser", "meter", "relay"]
    assert {"relay.lens", "relay.gap"} <= {e.id for e in summary.elements}
    assert links(summary) == [("laser.out", "relay.in"), ("relay.out", "meter.in")]
    relay = next(e for e in summary.elements if e.id == "relay")
    assert relay.inputs == ["in"]
    assert relay.outputs == ["out"]
    assert reading(service, session, ctx) == pytest.approx(before, rel=1e-9)


def test_ports_of_the_same_name_are_told_apart_by_their_element(service, session):
    ctx = service.create_context(session).id
    service.add_element(session, ctx, LENS, name="lens1")
    service.add_element(session, ctx, LENS, name="lens2")
    relay = next(
        e
        for e in service.group_elements(session, ctx, ["lens1", "lens2"], name="pair").elements
        if e.id == "pair"
    )
    assert sorted(relay.inputs) == ["lens1_in", "lens2_in"]
    assert sorted(relay.outputs) == ["lens1_out", "lens2_out"]


def test_a_group_can_be_dissolved(service, session):
    ctx = bench(service, session)
    original = links(service.describe_context(session, ctx))
    service.group_elements(session, ctx, ["lens", "gap"], name="relay")
    summary = service.ungroup(session, ctx, "relay")
    assert top(summary) == ["gap", "laser", "lens", "meter"]
    assert links(summary) == original
    assert reading(service, session, ctx) == pytest.approx(2.0, rel=1e-3)


def test_grouping_can_be_undone(service, session):
    ctx = bench(service, session)
    service.group_elements(session, ctx, ["lens", "gap"], name="relay")
    assert top(service.undo(session, ctx)) == ["gap", "laser", "lens", "meter"]


@pytest.mark.parametrize(
    ("elements", "name"),
    [([], "g"), (["lens", "gap"], "meter"), (["lens", "nothing"], "g"), (["lens"], "bad name")],
)
def test_invalid_groups_are_refused(service, session, elements, name):
    ctx = bench(service, session)
    with pytest.raises(ServiceError):
        service.group_elements(session, ctx, elements, name=name)
    assert top(service.describe_context(session, ctx)) == ["gap", "laser", "lens", "meter"]


# --- Copy, paste and blocks --------------------------------------------------------------------------------


def test_copied_elements_are_pasted_with_new_names_and_their_inner_connections(service, session):
    ctx = bench(service, session)
    fragment = service.copy_elements(session, ctx, ["lens", "gap"])
    pasted = service.paste_elements(session, ctx, fragment)
    assert pasted == ["lens2", "gap2"]
    summary = service.describe_context(session, ctx)
    assert ("lens2.out", "gap2.in") in links(summary)
    assert not any("lens2" in source and target == "meter.in" for source, target in links(summary))
    lens2 = next(e for e in summary.elements if e.id == "lens2")
    assert u.Quantity(lens2.params["focal_length"]).to_value(u.mm) == pytest.approx(200)
    other = service.create_context(session).id  # into another context: the names are free
    assert service.paste_elements(session, other, fragment) == ["lens", "gap"]


def test_a_group_is_saved_as_a_block_and_inserted_elsewhere(service, session):
    ctx = bench(service, session)
    service.group_elements(session, ctx, ["lens", "gap"], name="relay")
    text = service.export_element(session, ctx, "relay")
    document = json.loads(text)
    assert document["kind"] == "group"
    assert isinstance(lumos.io.from_document(document), ElementGroup)  # the same file works in Python
    other = service.create_context(session).id
    first = service.import_element(session, other, text)
    second = service.import_element(session, other, text)
    assert (first.id, second.id) == ("relay", "relay2")
    summary = service.describe_context(session, other)
    lens = next(e for e in summary.elements if e.id == "relay2.lens")
    assert u.Quantity(lens.params["focal_length"]).to_value(u.mm) == pytest.approx(200)


def test_a_block_must_be_an_element_or_a_group(service, session):
    ctx = bench(service, session)
    with pytest.raises(ServiceError):
        service.import_element(session, ctx, service.export_lumos(session, ctx))


# --- Context -----------------------------------------------------------------------------------------------


def test_the_name_and_description_of_a_context(service, session):
    ctx = bench(service, session)
    summary = service.set_context(session, ctx, name="Relay bench", description="A 4f relay.")
    assert (summary.name, summary.description) == ("Relay bench", "A 4f relay.")
    meta = json.loads(service.export_lumos(session, ctx))["meta"]
    assert meta["name"] == "Relay bench"
    assert meta["description"] == "A 4f relay."
    assert service.undo(session, ctx).name == "bench"


def test_placing_a_new_block_is_part_of_its_addition(service, session):
    ctx = service.create_context(session).id
    service.add_element(session, ctx, LENS, name="lens")
    service.update_ui(session, ctx, {"positions": {"lens": [100, 50]}}, record=False)
    summary = service.undo(session, ctx)  # undoes the addition, not the placement
    assert top(summary) == []
    assert not summary.can_undo


def test_the_summary_describes_the_inside_of_groups(service, session):
    ctx = bench(service, session)
    summary = service.group_elements(session, ctx, ["lens", "gap"], name="relay")
    relay = summary.groups["relay"]
    assert [c.model_dump() for c in relay.connections] == [{"source": "lens.out", "target": "gap.in"}]
    assert relay.exposed == {"in": "lens.in", "out": "gap.out"}
