"""Tests for lumos.core.registry (step P1.5).

Requirements: §10 of the specs (stable type identifiers + type_version, no arbitrary import), REQ-HW-8 and
REQ-CAT-7 (plugins declared through entry points), NF-7 (allow-list registry).
"""

import logging
import re
from importlib import metadata

import pytest
from astropy import units as u

from lumos.core import registry as registry_module
from lumos.core.element import Element
from lumos.core.param import Param
from lumos.core.registry import Registry, UnknownTypeError


class Lens(Element):
    focal_length = Param(u.m, default=1 * u.m)


class Mirror(Element):
    pass


@pytest.fixture
def reg():
    return Registry()


# --- Registration -----------------------------------------------------------------------


def test_register_and_lookup(reg):
    reg.register("mylab.optics.Lens", version=2)(Lens)
    assert reg.get("mylab.optics.Lens") is Lens
    assert reg.type_id(Lens) == "mylab.optics.Lens"
    assert reg.version("mylab.optics.Lens") == 2
    assert "mylab.optics.Lens" in reg
    assert list(reg) == ["mylab.optics.Lens"]
    assert len(reg) == 1


def test_register_returns_the_class_and_defaults_to_version_1(reg):
    decorated = reg.register("mylab.optics.Lens")(Lens)
    assert decorated is Lens
    assert reg.version("mylab.optics.Lens") == 1


@pytest.mark.parametrize(
    "type_id",
    [
        "Lens",
        "mylab",
        "mylab.",
        ".Lens",
        "mylab..Lens",
        "MyLab.optics.Lens",
        "mylab.optics.Le ns",
        "mylab.1optics.L",
    ],
)
def test_invalid_type_ids_are_rejected(reg, type_id):
    with pytest.raises(ValueError):
        reg.register(type_id)(Lens)


@pytest.mark.parametrize("version", [0, -1, 1.5, True, "1"])
def test_invalid_versions_are_rejected(reg, version):
    with pytest.raises((ValueError, TypeError)):
        reg.register("mylab.optics.Lens", version=version)(Lens)


def test_only_elements_can_be_registered(reg):
    with pytest.raises(TypeError):
        reg.register("mylab.misc.Thing")(object)


def test_duplicate_identifier_for_another_class_is_rejected(reg):
    reg.register("mylab.optics.Lens")(Lens)
    with pytest.raises(ValueError, match=re.escape("mylab.optics.Lens")):
        reg.register("mylab.optics.Lens")(Mirror)


def test_class_registered_under_two_identifiers_is_rejected(reg):
    reg.register("mylab.optics.Lens")(Lens)
    with pytest.raises(ValueError):
        reg.register("mylab.optics.OtherLens")(Lens)


def test_registering_the_same_class_twice_is_idempotent(reg):
    reg.register("mylab.optics.Lens")(Lens)
    reg.register("mylab.optics.Lens")(Lens)
    assert len(reg) == 1


def test_lumos_namespace_is_reserved_for_lumos_modules(reg):
    with pytest.raises(ValueError, match="lumos"):
        reg.register("lumos.optics.Lens")(Lens)


def test_lumos_namespace_is_allowed_for_classes_of_lumos(reg):
    class BuiltinLens(Element):
        pass

    BuiltinLens.__module__ = "lumos.elements.optics"
    reg.register("lumos.optics.BuiltinLens")(BuiltinLens)
    assert reg.get("lumos.optics.BuiltinLens") is BuiltinLens


# --- Lookup errors -------------------------------------------------------------------------


def test_unknown_type_raises_with_suggestion(reg):
    reg.register("mylab.optics.Lens")(Lens)
    with pytest.raises(UnknownTypeError, match=re.escape("mylab.optics.Lens")):
        reg.get("mylab.optics.Lense")
    assert issubclass(UnknownTypeError, LookupError)


def test_type_id_of_unregistered_class_raises(reg):
    with pytest.raises(UnknownTypeError):
        reg.type_id(Mirror)


# --- Global registry -------------------------------------------------------------------------


def test_global_registry_and_shortcut():
    assert isinstance(registry_module.registry, Registry)
    assert registry_module.register.__self__ is registry_module.registry


# --- Plugins -----------------------------------------------------------------------------------


class FakeEntryPoint:
    def __init__(self, name, load):
        self.name = name
        self.group = "lumos.elements"
        self.value = f"fake_plugin:{name}"
        self._load = load

    def load(self):
        return self._load()


def test_load_plugins_imports_entry_points(reg, monkeypatch):
    loaded = []

    def load_good():
        loaded.append("good")
        reg.register("goodlab.optics.Lens")(Lens)

    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda **kwargs: (
            [FakeEntryPoint("good", load_good)] if kwargs.get("group") == "lumos.elements" else []
        ),
    )
    names = reg.load_plugins()
    assert names == ["good"]
    assert loaded == ["good"]
    assert reg.get("goodlab.optics.Lens") is Lens


def test_broken_plugin_is_logged_and_others_still_load(reg, monkeypatch, caplog):
    def load_broken():
        raise ImportError("missing dependency")

    def load_good():
        reg.register("goodlab.optics.Lens")(Lens)

    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda **kwargs: [FakeEntryPoint("broken", load_broken), FakeEntryPoint("good", load_good)],
    )
    with caplog.at_level(logging.WARNING, logger="lumos"):
        names = reg.load_plugins()
    assert names == ["good"]
    assert "broken" in caplog.text
    assert "missing dependency" in caplog.text
