"""Tests for the explanation of propagations by the service (step P13.1, REQ-MCP-2 ``explain_sampling``).

``explain_sampling`` must describe the computation that will really happen: when an element forces a method,
that method is reported, with the choice the planner would have made for comparison (ADR-0005).
"""

import pytest

from lumos.service import Service


@pytest.fixture
def service():
    service = Service()
    yield service
    service.shutdown()


def test_a_forced_method_is_reported(service):
    session = service.create_session().id
    ctx = service.load_preset(session, "coronagraph")
    forced = next(e for e in ctx.elements if e.id == "image")
    assert forced.settings["method"] == "fresnel_mft"
    explained = service.explain_sampling(session, ctx.id, "image")
    assert explained["method"] == "fresnel_mft"
    assert "forced" in explained["reason"]
    assert explained["planner_method"]
    assert explained["planner_reason"]


def test_the_planner_choice_is_reported_in_auto_mode(service):
    session = service.create_session().id
    ctx = service.load_preset(session, "propagation_methods")
    far = next(e for e in ctx.elements if e.id == "far")
    assert far.settings["method"] == "auto"
    explained = service.explain_sampling(session, ctx.id, "far")
    assert explained["method"] == explained["planner_method"]
    assert explained["reason"] == explained["planner_reason"]
