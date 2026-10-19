"""Values of an element that cannot run are problems of the validation (step P14.19).

Requirement: an element may report values that would make a run fail (``Element.problems()``, e.g. a transfer
matrix that amplifies light); the validation of the service lists them as errors of this element, before any
run, so that the editor shows them on its block.
"""

import pytest

from lumos.service import Service, ServiceConfig


@pytest.fixture
def service(tmp_path):
    service = Service(ServiceConfig(archive_root=str(tmp_path)))
    yield service
    service.shutdown()


def test_the_problems_of_an_element_are_errors_of_the_validation(service):
    session = service.create_session(origin="web").id
    ctx = service.create_context(session).id
    service.add_element(
        session,
        ctx,
        "lumos.photonics.TransferMatrix",
        name="mmi",
        params={"amplitudes": "[[1.5, 0], [0, 1]]", "phases": "[[0, 0], [0, 0]] rad"},
    )
    issues = [i for i in service.validate_context(session, ctx) if i.element == "mmi"]
    assert any(i.severity == "error" and "amplif" in i.message for i in issues)
    service.set_param(session, ctx, "mmi.amplitudes", "[[0.5, 0], [0, 1]]")
    issues = [i for i in service.validate_context(session, ctx) if i.element == "mmi"]
    assert not any("amplif" in i.message for i in issues)
