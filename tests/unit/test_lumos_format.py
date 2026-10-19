"""Tests for the .lumos format of single elements (step P1.6).

Requirements: REQ-EL-12 (import/export of an element), §10 of the specs (envelope, quantities as
{"value", "unit"}, stable type identifiers + type_version, migrations, no arbitrary import), REQ-EL-3 (free
parameters and priors are part of the description), NF-7 (safe loading).
"""

import builtins
import importlib
import json
import math
import re
from datetime import UTC, datetime

import numpy as np
import pytest
from astropy import units as u

import lumos
from lumos.core.element import Element
from lumos.core.param import Param, Setting, State
from lumos.core.priors import LogUniform, Normal, Uniform
from lumos.core.registry import Registry, UnknownTypeError
from lumos.io import FormatVersionError, LumosFormatError, from_document, load, save, to_document


class Lens(Element):
    focal_length = Param(u.m, default=1 * u.m, bounds=(0 * u.m, None), inclusive=(False, True))
    aberrations = Param(u.nm, default=np.zeros(3) * u.nm, shape=(3,))
    flux = Param(u.W, default=1 * u.W)
    method = Setting(str, default="auto", choices=("auto", "mft"))
    position = State(u.mm, default=0 * u.mm)


class Mirror(Element):
    pass


@pytest.fixture
def reg():
    registry = Registry()
    registry.register("testlab.optics.Lens", version=1)(Lens)
    return registry


def json_round_trip(document):
    return json.loads(json.dumps(document, allow_nan=False))


# --- Writing ---------------------------------------------------------------------------------


def test_document_envelope(reg):
    document = to_document(Lens(name="L1"), registry=reg)
    assert document["lumos_format"] == "1.2"  # ADR-0020 (feedbacks), ADR-0022 (free targets)
    assert document["kind"] == "element"
    assert document["lumos_version"] == lumos.__version__
    created = datetime.fromisoformat(document["created_at"])
    assert created.tzinfo is not None
    assert created.utcoffset() == UTC.utcoffset(None)
    element = document["element"]
    assert element["type"] == "testlab.optics.Lens"
    assert element["type_version"] == 1
    assert element["name"] == "L1"


def test_physical_values_keep_user_units(reg):
    document = to_document(Lens(focal_length=100 * u.mm), registry=reg)
    assert document["element"]["params"]["focal_length"] == {"value": 100.0, "unit": "mm"}
    assert document["element"]["settings"] == {"method": "auto"}
    assert document["element"]["state"]["position"] == {"value": 0.0, "unit": "mm"}


def test_output_is_strict_json_even_with_nan(reg):
    json.dumps(to_document(Lens(flux=math.nan * u.W), registry=reg), allow_nan=False)


def test_unregistered_element_cannot_be_written(reg):
    with pytest.raises(UnknownTypeError):
        to_document(Mirror(), registry=reg)


# --- Round trips -------------------------------------------------------------------------------


def assert_same_element(a, b):
    assert type(a) is type(b)
    assert a.name == b.name
    for name in type(a).declared():
        x, y = getattr(a, name), getattr(b, name)
        if isinstance(x, u.Quantity):
            assert x.unit == y.unit
            np.testing.assert_array_equal(x.value, y.value)
        else:
            assert x == y


def test_round_trip_through_json(reg):
    lens = Lens(
        name="L1",
        focal_length=250 * u.mm,
        aberrations=[10, -5, 2.5] * u.nm,
        flux=math.nan * u.W,
        method="mft",
        position=-3 * u.mm,
    )
    assert_same_element(from_document(json_round_trip(to_document(lens, registry=reg)), registry=reg), lens)


def test_round_trip_keeps_free_parameters_and_priors(reg):
    lens = Lens()
    lens.set_free("focal_length", prior=Normal(1 * u.m, 1 * u.cm))
    lens.set_free("flux", prior=LogUniform(1 * u.mW, 1 * u.W))
    lens.set_free("position", prior=Uniform(-1 * u.mm, 1 * u.mm))
    lens.set_free("aberrations")  # free without prior
    loaded = from_document(json_round_trip(to_document(lens, registry=reg)), registry=reg)
    assert loaded.prior("focal_length") == Normal(1 * u.m, 1 * u.cm)
    assert loaded.prior("flux") == LogUniform(1 * u.mW, 1 * u.W)
    assert loaded.prior("position") == Uniform(-1 * u.mm, 1 * u.mm)
    assert loaded.is_free("aberrations")
    assert loaded.prior("aberrations") is None


def test_round_trip_through_a_file(reg, tmp_path):
    lens = Lens(name="Lentille à 100 mm", focal_length=100 * u.mm)
    path = tmp_path / "lens.lumos"
    save(lens, path, registry=reg)
    text = path.read_text(encoding="utf-8")
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-standard JSON constant {c}"))
    assert_same_element(load(path, registry=reg), lens)


def test_element_shortcuts_use_the_global_registry():
    class Probe(Element):
        gain = Param(default=2.0)

    lumos.core.registry.registry.register("testlab.format.Probe")(Probe)
    probe = Probe(gain=3.0)
    assert_same_element(Probe.from_lumos(json_round_trip(probe.to_lumos())), probe)
    with pytest.raises(TypeError):
        Lens.from_lumos(probe.to_lumos())


# --- Reading invalid documents ---------------------------------------------------------------------


def valid_document(reg):
    return json_round_trip(to_document(Lens(), registry=reg))


@pytest.mark.parametrize("key", ["lumos_format", "kind", "element"])
def test_missing_envelope_keys(reg, key):
    document = valid_document(reg)
    del document[key]
    with pytest.raises(LumosFormatError):
        from_document(document, registry=reg)


@pytest.mark.parametrize("version", ["2.0", "1.3", "one"])  # 1.2 is read since ADR-0022
def test_unsupported_format_versions(reg, version):
    document = valid_document(reg)
    document["lumos_format"] = version
    with pytest.raises(FormatVersionError):
        from_document(document, registry=reg)
    assert issubclass(FormatVersionError, LumosFormatError)
    assert issubclass(LumosFormatError, ValueError)


def test_unknown_kind(reg):
    document = valid_document(reg)
    document["kind"] = "spaceship"
    with pytest.raises(LumosFormatError):
        from_document(document, registry=reg)


def test_unknown_type(reg):
    document = valid_document(reg)
    document["element"]["type"] = "testlab.optics.Telescope"
    with pytest.raises(UnknownTypeError):
        from_document(document, registry=reg)


def test_loading_never_imports_modules(reg, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("loading a document must not import modules")

    document = valid_document(reg)
    document["element"]["type"] = "os.system"
    # Imports are forbidden only during the call, so that pytest can import what it needs to report.
    with monkeypatch.context() as patch:
        patch.setattr(importlib, "import_module", forbidden)
        patch.setattr(builtins, "__import__", forbidden)
        try:
            from_document(document, registry=reg)
            outcome = "loaded"
        except UnknownTypeError:
            outcome = "unknown type"
        except AssertionError as error:
            outcome = str(error)
    assert outcome == "unknown type"


def test_unknown_parameter_is_reported(reg):
    document = valid_document(reg)
    document["element"]["params"]["focal_lenght"] = {"value": 1.0, "unit": "m"}
    with pytest.raises(LumosFormatError, match="focal_lenght"):
        from_document(document, registry=reg)


def test_incompatible_unit_is_reported(reg):
    document = valid_document(reg)
    document["element"]["params"]["focal_length"] = {"value": 1.0, "unit": "s"}
    with pytest.raises(LumosFormatError, match="focal_length"):
        from_document(document, registry=reg)


def test_out_of_bounds_value_is_reported(reg):
    document = valid_document(reg)
    document["element"]["params"]["focal_length"] = {"value": -1.0, "unit": "m"}
    with pytest.raises(LumosFormatError, match="focal_length"):
        from_document(document, registry=reg)


def test_invalid_prior_is_reported(reg):
    document = valid_document(reg)
    document["element"]["free"] = {
        "focal_length": {"prior": {"kind": "uniform", "low": {"value": 1, "unit": "s"}}}
    }
    with pytest.raises(LumosFormatError):
        from_document(document, registry=reg)


# --- Type versions and migrations -------------------------------------------------------------------


def test_newer_type_version_is_refused(reg):
    document = valid_document(reg)
    document["element"]["type_version"] = 2
    with pytest.raises(FormatVersionError, match=re.escape("testlab.optics.Lens")):
        from_document(document, registry=reg)


def test_migrations_are_applied_in_order():
    class Lens3(Element):
        focal = Param(u.m, default=1 * u.m)
        label = Setting(str, default="")

    reg = Registry()
    reg.register("testlab.optics.Lens3", version=3)(Lens3)
    calls = []

    @reg.add_migration("testlab.optics.Lens3", from_version=1)
    def rename(body):
        calls.append(1)
        body["params"]["focal"] = body["params"].pop("focal_length")
        return body

    @reg.add_migration("testlab.optics.Lens3", from_version=2)
    def add_label(body):
        calls.append(2)
        body["settings"]["label"] = "migrated"
        return body

    old = {
        "lumos_format": "1.0",
        "kind": "element",
        "lumos_version": "0.0.1",
        "created_at": "2026-01-01T00:00:00Z",
        "element": {
            "type": "testlab.optics.Lens3",
            "type_version": 1,
            "name": None,
            "params": {"focal_length": {"value": 2.0, "unit": "m"}},
            "settings": {},
            "state": {},
        },
    }
    lens = from_document(old, registry=reg)
    assert calls == [1, 2]
    assert lens.focal == 2 * u.m
    assert lens.label == "migrated"


def test_missing_migration_is_refused():
    class Lens2(Element):
        focal = Param(u.m, default=1 * u.m)

    reg = Registry()
    reg.register("testlab.optics.Lens2", version=2)(Lens2)
    document = {
        "lumos_format": "1.0",
        "kind": "element",
        "lumos_version": "0.0.1",
        "created_at": "2026-01-01T00:00:00Z",
        "element": {
            "type": "testlab.optics.Lens2",
            "type_version": 1,
            "name": None,
            "params": {},
            "settings": {},
            "state": {},
        },
    }
    with pytest.raises(FormatVersionError):
        from_document(document, registry=reg)


@pytest.mark.parametrize("from_version", [0, 1, 2])
def test_migration_versions_must_be_older_than_current(from_version):
    reg = Registry()
    reg.register("testlab.optics.Lens", version=1)(Lens)
    with pytest.raises(ValueError):
        reg.add_migration("testlab.optics.Lens", from_version=from_version)(lambda body: body)
