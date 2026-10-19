"""Revisions and focus of the contexts of a session (step P14.18, review of 19/10/2026).

Requirements: the web interface and an assistant working on the same session (connector link) must see what
the other does. So that the interface can follow:

- every context has a *revision*, given in its summary, which grows at every change of the context (edits,
  interface data, undo and redo, actions, loops) and only then: describing a context does not change it;
- a session has a *focused* context, with the origin of the last change of focus and a counter: a new
  context (created, loaded, imported) takes the focus, and so does a context focused explicitly; closing the
  focused context clears the focus.
"""

import itertools

import pytest

from lumos.service import Service, ServiceConfig, ServiceError
from lumos.service.service import acting_as

LASER = "lumos.sources.GaussianLaser"
METER = "lumos.detectors.PowerMeter"


@pytest.fixture
def service(tmp_path):
    service = Service(ServiceConfig(archive_root=str(tmp_path)))
    yield service
    service.shutdown()


@pytest.fixture
def session(service):
    return service.create_session(origin="web").id


def test_the_revision_grows_at_every_change_and_only_then(service, session):
    summary = service.create_context(session, name="bench")
    cid = summary.id
    revisions = [summary.revision]
    laser = service.add_element(session, cid, LASER)
    revisions.append(service.describe_context(session, cid).revision)
    meter = service.add_element(session, cid, METER)
    service.connect(session, cid, f"{laser.id}.out", f"{meter.id}.in")
    revisions.append(service.describe_context(session, cid).revision)
    service.set_param(session, cid, f"{laser.id}.power", "2 mW")
    revisions.append(service.describe_context(session, cid).revision)
    service.update_ui(session, cid, {"positions": {laser.id: [0, 0]}}, record=False)
    revisions.append(service.describe_context(session, cid).revision)
    service.undo(session, cid)
    revisions.append(service.describe_context(session, cid).revision)
    service.redo(session, cid)
    revisions.append(service.describe_context(session, cid).revision)
    assert all(b > a for a, b in itertools.pairwise(revisions)), revisions

    # reading does not change it
    same = service.describe_context(session, cid).revision
    service.list_contexts(session)
    service.export_lumos(session, cid)
    service.validate(session, cid)
    assert service.describe_context(session, cid).revision == same


def test_a_failed_edit_does_not_count_as_a_change(service, session):
    cid = service.create_context(session).id
    before = service.describe_context(session, cid).revision
    with pytest.raises(ServiceError):
        service.add_element(session, cid, "lumos.optics.NoSuchElement")
    assert service.describe_context(session, cid).revision == before


def test_a_new_context_takes_the_focus_with_its_origin(service, session):
    assert service.get_focus(session).context is None
    first = service.create_context(session, name="first").id
    focus = service.get_focus(session)
    assert (focus.context, focus.origin) == (first, "web")
    with acting_as("mcp"):
        second = service.load_preset(session, "first_simulation").id
    focus2 = service.get_focus(session)
    assert (focus2.context, focus2.origin) == (second, "mcp")
    assert focus2.count > focus.count


def test_a_context_is_focused_explicitly_and_closing_it_clears_the_focus(service, session):
    first = service.create_context(session).id
    second = service.create_context(session).id
    count = service.get_focus(session).count
    focus = service.focus(session, first)
    assert (focus.context, focus.count) == (first, count + 1)
    assert service.focus(session, first).count == count + 1  # focusing it again is no change
    with pytest.raises(ServiceError):
        service.focus(session, "missing")
    service.close_context(session, second)
    assert service.get_focus(session).context == first
    service.close_context(session, first)
    assert service.get_focus(session).context is None
