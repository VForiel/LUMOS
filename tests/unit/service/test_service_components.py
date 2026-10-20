"""Components made in the editor, through the service and the web API (step P15.15, ADR-0025).

Requirements (docs/specs/pistes_evolution.md §2): the form view, on every instance, checks a component as it
is typed and saves it, after which it is a type of the palette of the session and can be added like any
element; a composite component is added as a group; the code view (a Python element class) runs only where
scripts are enabled, never on the public instance.
"""

import copy

import pytest

from lumos.service import Service, ServiceConfig, ServiceError

ATTENUATOR = {
    "id": "tester.Attenuator",
    "version": 1,
    "title": "Attenuator",
    "description": "A neutral density.",
    "license": "MIT",
    "authors": ["tester"],
    "template": "gain_phase",
    "params": {"transmission": {"unit": "", "default": 0.5, "bounds": [0, 1], "doc": "power transmission"}},
    "body": {"gain": "$transmission", "phase": "0 rad"},
    "checks": [{"kind": "power", "wavelength": "1 um", "power": "2 mW", "expect": "1 mW", "rtol": 1e-9}],
}

RELAY = {
    "id": "tester.Relay",
    "version": 1,
    "title": "Relay",
    "description": "A thin lens.",
    "license": "MIT",
    "authors": ["tester"],
    "template": "composite",
    "params": {},
    "body": {
        "group": {
            "type": "lumos.group",
            "name": "relay",
            "children": [
                {
                    "id": "lens",
                    "type": "lumos.optics.ThinLens",
                    "type_version": 1,
                    "name": "lens",
                    "params": {"focal_length": {"value": 50.0, "unit": "mm"}},
                    "settings": {},
                    "state": {},
                }
            ],
            "connections": [],
            "exposed_ports": {"in": "lens.in", "out": "lens.out"},
        }
    },
    "checks": [],
}

CODE = '''
from astropy import units as u

from lumos.core.element import Element
from lumos.core.param import Param
from lumos.core.port import Input, Output
from lumos.core.registry import register
from lumos.core.signal import OpticalSignal


@register("plugins.tester.Halver")
class Halver(Element):
    """Halve the power."""

    in_ = Input(OpticalSignal)
    out = Output(OpticalSignal)

    @staticmethod
    def transfer(values, inputs):
        return {"out": inputs["in"] * 0.5**0.5}
'''


def document(spec):
    return {"lumos_format": "1.3", "kind": "component", "component": copy.deepcopy(spec)}


@pytest.fixture
def service(tmp_path):
    service = Service(ServiceConfig(archive_root=str(tmp_path)))
    yield service
    service.shutdown()


def test_a_component_is_checked_saved_listed_and_used(service):
    session = service.create_session().id
    wrong = copy.deepcopy(ATTENUATOR)
    wrong.pop("license")
    assert any("license" in p for p in service.check_component(document(wrong)))
    with pytest.raises(ServiceError, match="license"):
        service.save_component(session, document(wrong))
    info = service.save_component(session, document(ATTENUATOR))
    assert info.type_id == "components.tester.Attenuator"
    assert [c.id for c in service.list_components(session)] == ["tester.Attenuator"]
    assert service.get_component(session, "tester.Attenuator")["component"]["template"] == "gain_phase"
    types = [t.type_id for t in service.list_element_types(session=session, page_size=10_000).items]
    assert "components.tester.Attenuator" in types
    ctx = service.create_context(session).id
    service.add_element(session, ctx, "components.tester.Attenuator", name="nd", params={"transmission": 0.2})
    other = service.create_session().id
    assert service.list_components(other) == []  # a component belongs to the session that made it
    assert "components.tester.Attenuator" not in [
        t.type_id for t in service.list_element_types(session=other, page_size=10_000).items
    ]


def test_a_composite_component_is_added_as_a_group(service):
    session = service.create_session().id
    service.save_component(session, document(RELAY))
    ctx = service.create_context(session).id
    service.add_component(session, ctx, "tester.Relay", name="relay")
    summary = service.describe_context(session, ctx)
    assert "relay" in summary.groups


def test_code_components_need_scripts(tmp_path):
    closed = Service(ServiceConfig(archive_root=str(tmp_path), scripts_enabled=False))
    session = closed.create_session().id
    with pytest.raises(ServiceError, match="script"):
        closed.define_code_component(session, CODE)
    closed.shutdown()
    opened = Service(ServiceConfig(archive_root=str(tmp_path), scripts_enabled=True))
    session = opened.create_session().id
    assert opened.define_code_component(session, CODE) == ["plugins.tester.Halver"]
    assert "plugins.tester.Halver" in [
        t.type_id for t in opened.list_element_types(session=session, page_size=10_000).items
    ]
    opened.shutdown()


def test_the_web_api(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from lumos.web import WebConfig, create_app

    config = WebConfig(service=ServiceConfig(archive_root=str(tmp_path)))
    with TestClient(create_app(config), base_url="http://localhost") as client:
        client.get("/api/info")
        spec = copy.deepcopy(ATTENUATOR)
        spec["id"] = "webtester.Attenuator"
        problems = client.post("/api/components/check", json=document(spec)).json()
        assert problems == []
        saved = client.put("/api/components", json=document(spec))
        assert saved.status_code == 200, saved.text
        assert [c["id"] for c in client.get("/api/components").json()] == ["webtester.Attenuator"]
        palette = client.get("/api/palette").json()
        assert any(t["type_id"] == "components.webtester.Attenuator" for t in palette["types"])
