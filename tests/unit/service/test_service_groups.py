"""Tests for the creation of element groups by the service layer (step P13.2, REQ-MCP-2 ``create_group``).

A group is described in one call (its elements, internal connections and exposed ports), so that an assistant
builds it without handling Python objects; it then behaves as one element of the context (REQ-CTX-7).
The expected reading is physical: a power meter behind a 2 mW laser reads 2 mW.
"""

import pytest

from lumos.service import Service, ServiceError


@pytest.fixture
def service():
    service = Service()
    yield service
    service.shutdown()


def test_a_group_is_created_and_used_as_one_element(service):
    session = service.create_session().id
    ctx = service.create_context(session).id
    group = service.create_group(
        session,
        ctx,
        "source",
        elements=[{"type": "lumos.sources.GaussianLaser", "name": "laser", "params": {"power": "2 mW"}}],
        expose={"out": "laser.out"},
    )
    assert group.id == "source"
    assert group.type_id == "lumos.group"
    assert group.outputs == ["out"]
    service.add_element(session, ctx, "lumos.detectors.PowerMeter", name="meter")
    service.connect(session, ctx, "source.out", "meter.in")
    service.set_param(session, ctx, "source.laser.power", "4 mW")
    job = service.wait_job(session, service.run(session, ctx, ["meter.out"]).id, timeout=60)
    assert service.get_result(session, job.id, "meter.out").values[0] == pytest.approx(4e-3, rel=1e-3)


def test_internal_connections_and_errors(service):
    session = service.create_session().id
    ctx = service.create_context(session).id
    group = service.create_group(
        session,
        ctx,
        "bench",
        elements=[
            {"type": "lumos.sources.GaussianLaser", "name": "laser", "params": {"power": "1 mW"}},
            {"type": "lumos.detectors.PowerMeter", "name": "meter"},
        ],
        connections=[["laser.out", "meter.in"]],
        expose={"reading": "meter.out"},
    )
    assert group.outputs == ["reading"]
    with pytest.raises(ServiceError) as error:
        service.create_group(session, ctx, "broken", elements=[], expose={"out": "nothing.out"})
    assert error.value.hint
    assert "broken" not in {e.id for e in service.describe_context(session, ctx).elements}
