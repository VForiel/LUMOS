"""Element packages installed by the administrator are elements of every instance (step P15.16, ADR-0026).

Requirements (docs/specs/pistes_evolution.md §2): components with code are shared as Python packages
(``lumos-component-…``) declaring the entry point ``lumos.elements``; once installed where LUMOS runs, their
element types are in the palette, also on the public instance (installed by its administrator, not by
visitors). The service loads them when it starts.
"""

from importlib import metadata

from lumos.core.element import Element
from lumos.core.port import Input, Output
from lumos.core.registry import registry
from lumos.core.signal import OpticalSignal
from lumos.service import Service, ServiceConfig


class FakeEntryPoint:
    def __init__(self, name, load):
        self.name = name
        self.group = "lumos.elements"
        self.value = f"lumos_component_{name}"
        self._load = load

    def load(self):
        return self._load()


def test_the_service_loads_the_installed_element_packages(tmp_path, monkeypatch):
    def load():
        if "plugins.installed.Passthrough" in registry:
            return

        @registry.register("plugins.installed.Passthrough")
        class Passthrough(Element):
            """Let the light through."""

            in_ = Input(OpticalSignal)
            out = Output(OpticalSignal)

            @staticmethod
            def transfer(values, inputs):
                return {"out": inputs["in"]}

    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda **kwargs: (
            [FakeEntryPoint("installed", load)] if kwargs.get("group") == "lumos.elements" else []
        ),
    )
    service = Service(ServiceConfig(archive_root=str(tmp_path), scripts_enabled=False))
    session = service.create_session().id
    types = [t.type_id for t in service.list_element_types(session=session, page_size=10_000).items]
    service.shutdown()
    registry.unregister("plugins.installed.Passthrough")
    assert "plugins.installed.Passthrough" in types
