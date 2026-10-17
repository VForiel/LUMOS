"""Tests for driver plugins and the driver guide (step P12.9, REQ-HW-8, ADR-0014).

Requirements: drivers of other packages are found through the ``lumos.drivers`` entry points; a guide
"Writing a driver" explains how, and checks that the interface covers the SDKs of the first vendor (Thorlabs
Kinesis for motion, TSI for cameras, TLPM for power meters).
"""

import logging
from pathlib import Path

import pytest

import lumos.hardware.drivers as drivers
from lumos.core.driver import Driver, Reply
from lumos.hardware.drivers.lumoslabs import LinearStageDriver

DOCS = Path(__file__).resolve().parents[3] / "docs" / "hardware"


class VendorStage(Driver):
    def _connect(self):
        pass

    def _disconnect(self):
        pass

    def _send(self, command):
        return Reply()

    def _query(self, name, *arguments):
        return None

    def _stop(self):
        pass


class Entry:
    def __init__(self, name, target):
        self.name = name
        self.value = f"vendor.drivers:{name}"
        self.target = target

    def load(self):
        if isinstance(self.target, Exception):
            raise self.target
        return self.target


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(drivers, "DRIVERS", dict(drivers.DRIVERS))
    monkeypatch.setattr(drivers, "_plugins_loaded", False)
    return drivers


def test_builtin_drivers_are_registered(registry):
    assert registry.driver_for("LL-LS100") is LinearStageDriver
    assert len([m for m in registry.DRIVERS if m.startswith("LL-")]) == 9


def test_registration_rules(registry):
    registry.register_driver("VENDOR-1", VendorStage)
    assert registry.driver_for("VENDOR-1") is VendorStage
    registry.register_driver("VENDOR-1", VendorStage)  # idempotent
    with pytest.raises(ValueError, match="already"):
        registry.register_driver("LL-LS100", VendorStage)
    with pytest.raises(TypeError):
        registry.register_driver("VENDOR-2", object)


def test_plugins_from_entry_points(registry, monkeypatch, caplog):
    entries = [Entry("KDC101", VendorStage), Entry("BROKEN", ImportError("missing SDK"))]
    monkeypatch.setattr(registry, "entry_points", lambda group: entries if group == "lumos.drivers" else [])
    with caplog.at_level(logging.ERROR):
        assert registry.load_plugins() == ["KDC101"]
    assert "BROKEN" in caplog.text or "vendor.drivers:BROKEN" in caplog.text
    assert registry.driver_for("KDC101") is VendorStage


def test_unknown_models_load_the_plugins_first(registry, monkeypatch):
    monkeypatch.setattr(registry, "entry_points", lambda group: [Entry("PM100D", VendorStage)])
    assert registry.driver_for("PM100D") is VendorStage
    with pytest.raises(LookupError, match="NOPE"):
        registry.driver_for("NOPE")


def test_the_driver_guide_covers_the_interface_and_the_thorlabs_sdks():
    text = (DOCS / "writing_a_driver.md").read_text(encoding="utf-8")
    for name in ("_connect", "_disconnect", "_send", "_query", "_stop", "_safe_state", "limits", "dangerous"):
        assert f"`{name}" in text, name
    for sdk in ("Kinesis", "TSI", "TLPM"):
        assert sdk in text
    assert "lumos.drivers" in text
