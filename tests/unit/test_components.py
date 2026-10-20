"""Declarative components (step P15.14, ADR-0025; docs/specs/pistes_evolution.md §2).

Requirements: a component is a document of kind "component" (no code); templates transfer_matrix,
gain_phase, spectral_transmission and composite; parameters with units, bounds and descriptions; checks run
by ``lumos component check``; contexts using components embed their definitions.

Independent references:

- an attenuator of power gain g and phase φ multiplies the field by √g e^{iφ}: the power by g;
- a tabulated transmission is interpolated linearly between its points (a 0.4 → 0.8 ramp gives 0.6 at the
  middle wavelength) and blocks the light outside its table;
- the ideal 3-dB coupler (1/√2)[[1, i], [i, 1]] sends half of the light of one input to each output
  (Pozar, *Microwave Engineering*, §7.5).
"""

import copy
import json
import subprocess
import sys

import numpy as np
import pytest
from astropy import units as u

from lumos import Context
from lumos.components import (
    ComponentError,
    check_component,
    component_from_document,
    create,
    register_component,
)
from lumos.core.element import Element
from lumos.core.registry import Registry
from lumos.fields import Bus, Grid, Wavecube, Wavefront

ATTENUATOR = {
    "id": "acme.Attenuator",
    "version": 1,
    "title": "Attenuator",
    "description": "A neutral density with an adjustable power transmission and a phase.",
    "license": "MIT",
    "authors": ["ACME optics"],
    "template": "gain_phase",
    "params": {
        "transmission": {"unit": "", "default": 0.5, "bounds": [0, 1], "doc": "power transmission"},
        "phase": {"unit": "rad", "default": 0.3, "bounds": [None, None], "doc": "phase added to the field"},
    },
    "body": {"gain": "$transmission", "phase": "$phase"},
    "checks": [{"kind": "power", "wavelength": "1 um", "power": "2 mW", "expect": "1 mW", "rtol": 1e-9}],
}

FILTER = {
    "id": "acme.Ramp",
    "version": 1,
    "title": "Ramp filter",
    "description": "A filter whose transmission grows from 0.4 at 1 um to 0.8 at 2 um.",
    "license": "CC-BY-4.0",
    "authors": ["ACME optics"],
    "template": "spectral_transmission",
    "params": {},
    "body": {"wavelengths": "[1, 2] um", "transmission": [0.4, 0.8]},
    "checks": [{"kind": "power", "wavelength": "1.5 um", "power": "1 W", "expect": "0.6 W", "rtol": 1e-9}],
}

COUPLER = {
    "id": "acme.Coupler",
    "version": 1,
    "title": "3-dB coupler",
    "description": "An ideal 3-dB coupler given by its matrix at 1550 nm.",
    "license": "MIT",
    "authors": ["ACME photonics"],
    "template": "transfer_matrix",
    "params": {},
    "body": {
        "amplitudes": [[0.7071067811865476, 0.7071067811865476], [0.7071067811865476, 0.7071067811865476]],
        "phases": "[[0, 1.5707963267948966], [1.5707963267948966, 0]] rad",
        "wavelength": "1550 nm",
        "dispersion": "achromatic",
    },
    "checks": [{"kind": "power", "wavelength": "1550 nm", "power": "1 mW", "expect": "2 mW", "rtol": 1e-9}],
}

COMPOSITE = {
    "id": "acme.Relay",
    "version": 1,
    "title": "Relay",
    "description": "A thin lens followed by its focal distance.",
    "license": "MIT",
    "authors": ["ACME optics"],
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
                    "params": {"focal_length": {"value": 100.0, "unit": "mm"}},
                    "settings": {},
                    "state": {},
                },
            ],
            "connections": [],
            "exposed_ports": {"in": "lens.in", "out": "lens.out"},
        }
    },
    "checks": [],
}


def document(spec):
    return {"lumos_format": "1.3", "kind": "component", "component": copy.deepcopy(spec)}


def beam(power=2e-3, lam=1e-6):
    return Wavefront.plane(Grid(16, dx=0.1 * u.mm), lam * u.m, power=power * u.W)


@pytest.fixture
def reg():
    return Registry()


def test_a_gain_and_phase_component(reg):
    cls = register_component(ATTENUATOR, registry=reg)
    assert reg.get("components.acme.Attenuator") is cls
    element = cls(name="nd")
    assert element.transmission == 0.5
    element.transmission = 0.25
    incoming = beam()
    out = element.apply(**{"in": incoming})["out"]
    assert out.power.to_value(u.W) == pytest.approx(0.25 * 2e-3, rel=1e-12)
    ratio = np.asarray(out.field)[8, 8] / np.asarray(incoming.field)[8, 8]
    assert np.angle(ratio) == pytest.approx(0.3, abs=1e-12)
    with pytest.raises(ValueError):
        element.transmission = 1.5  # outside the bounds


def test_a_tabulated_transmission(reg):
    cls = register_component(FILTER, registry=reg)
    grid = Grid(8, dx=0.1 * u.mm)
    cube = Wavecube.plane(grid, [1.0, 1.5, 2.0, 2.5] * u.um, power=1 * u.W)
    out = cls().apply(**{"in": cube})["out"]
    np.testing.assert_allclose(
        out.power.to_value(u.W) / cube.power.to_value(u.W), [0.4, 0.6, 0.8, 0.0], atol=1e-12
    )
    lanes = Bus.guided(np.ones((1, 2)), [1.5] * u.um)
    out = cls().apply(**{"in": lanes})["out"]
    np.testing.assert_allclose(np.abs(np.asarray(out.modal.amplitudes)) ** 2, 0.6, rtol=1e-12)


def test_a_transfer_matrix_component(reg):
    cls = register_component(COUPLER, registry=reg)
    bus = Bus.guided(np.array([[1.0, 0.0]]), [1550] * u.nm)
    out = cls().apply(**{"in": bus})["out"]
    np.testing.assert_allclose(np.abs(np.asarray(out.modal.amplitudes)) ** 2, [[0.5, 0.5]], atol=1e-12)


def test_a_composite_component_is_placed_as_a_group(reg):
    register_component(COMPOSITE, registry=reg)
    group = create("acme.Relay", name="relay", registry=reg)
    assert type(group).__name__ == "ElementGroup"
    assert set(group.instance_ports()) == {"in", "out"}


def test_documents_and_contexts_carry_their_components():
    import lumos.io

    cls = register_component(ATTENUATOR)
    ctx = Context()
    laser = ctx.add(lumos.elements.sources.GaussianLaser(name="laser", n_pixels=32, pixel_scale=0.25 * u.mm))
    nd = ctx.add(cls(name="nd", transmission=0.2))
    laser >> nd
    saved = lumos.io.to_document(ctx)
    assert saved["components"]["acme.Attenuator"]["template"] == "gain_phase"
    assert lumos.io.from_document(json.loads(json.dumps(saved)))["nd"].transmission == 0.2
    assert isinstance(Element.from_lumos(nd.to_lumos()), cls)
    spec = component_from_document(document(ATTENUATOR))
    assert spec.to_json() == ATTENUATOR


def test_a_valid_component_has_no_problem():
    for spec in (ATTENUATOR, FILTER, COUPLER, COMPOSITE):
        assert check_component(document(spec)) == [], spec["id"]


def broken(change):
    spec = copy.deepcopy(ATTENUATOR)
    change(spec)
    return document(spec)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda s: s.pop("license"), "license"),
        (lambda s: s.update(description=""), "description"),
        (lambda s: s.update(id="Bad Id"), "id"),
        (lambda s: s["params"]["transmission"].update(default=2.0), "bounds"),
        (lambda s: s["params"]["phase"].update(unit="mm"), "phase"),
        (lambda s: s["params"]["phase"].pop("doc"), "description"),
        (lambda s: s.update(checks=[]), "check"),
        (lambda s: s["checks"][0].update(expect="1.5 mW"), "expect"),
        (lambda s: s["body"].update(gain="$missing"), "missing"),
        (lambda s: s["body"].update(gain=3.0), "amplif"),
        (lambda s: s.update(template="python"), "template"),
        (lambda s: s.update(code="import os"), "code"),
    ],
)
def test_problems_are_reported(change, message):
    problems = check_component(broken(change))
    assert any(message in problem for problem in problems), problems


def test_an_invalid_component_is_not_registered(reg):
    spec = copy.deepcopy(ATTENUATOR)
    spec["template"] = "python"
    with pytest.raises(ComponentError):
        register_component(spec, registry=reg)


def test_another_definition_under_the_same_identifier_is_refused(reg):
    register_component(ATTENUATOR, registry=reg)
    register_component(copy.deepcopy(ATTENUATOR), registry=reg)  # the same definition: nothing to do
    other = copy.deepcopy(ATTENUATOR)
    other["body"]["phase"] = 0.0
    with pytest.raises(ComponentError, match="already"):
        register_component(other, registry=reg)


def test_the_command_line_checks_a_file(tmp_path):
    good, bad = tmp_path / "good.lumos", tmp_path / "bad.lumos"
    good.write_text(json.dumps(document(ATTENUATOR)), encoding="utf-8")
    bad.write_text(json.dumps(broken(lambda s: s.pop("license"))), encoding="utf-8")
    ok = subprocess.run(
        [sys.executable, "-m", "lumos.cli", "component", "check", str(good)], capture_output=True, text=True
    )
    assert ok.returncode == 0, ok.stdout + ok.stderr
    failed = subprocess.run(
        [sys.executable, "-m", "lumos.cli", "component", "check", str(bad)], capture_output=True, text=True
    )
    assert failed.returncode == 1
    assert "license" in failed.stdout + failed.stderr
