"""Tests for ``.lumos`` documents of kind ``"context"`` (step P5.8).

Requirements: REQ-CTX-8 (import and export of a whole context), specs §10 (envelope, ``kind`` among
``context | element | group``, quantities as ``{"value", "unit"}``, arrays inline below 64 kB and in a
``.npz`` sidecar above, stable type identifiers, migrations, published JSON Schema, optional and ignored
``ui`` section), ADR-0003.

References: the expected document layout is the example of §10 of the specifications; round trips are
checked by comparing the documents (without the creation date) and the computed results.
"""

import json

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

from lumos import Context
from lumos.core.element import Element
from lumos.core.group import ElementGroup
from lumos.core.param import Param, State
from lumos.core.port import Input, Output
from lumos.core.registry import Registry
from lumos.core.signal import Signal
from lumos.errors import FormatVersionError, LumosFormatError
from lumos.io import document_schema, load, save, to_document


class Number(Signal):
    def __init__(self, value):
        self.value = value


class Constant(Element):
    level = Param(default=1.0)
    out = Output(Number)

    @staticmethod
    def transfer(values, inputs):
        return {"out": Number(values.level)}


class Gain(Element):
    gain = Param(default=2.0)
    position = State(unit=u.mm, default=0 * u.mm)
    in_ = Input(Number)
    out = Output(Number)

    @staticmethod
    def transfer(values, inputs):
        return {"out": Number(values.gain * inputs["in"].value)}


class Mask(Element):
    mask = Param(unit=u.dimensionless_unscaled, shape=None, default=np.ones((2, 2)))
    in_ = Input(Number)
    out = Output(Number)

    @staticmethod
    def transfer(values, inputs):
        return {"out": Number(float(np.sum(values.mask)) * inputs["in"].value)}


@pytest.fixture
def registry():
    reg = Registry()
    reg.register("tests.Constant")(Constant)
    reg.register("tests.Gain")(Gain)
    reg.register("tests.Mask")(Mask)
    return reg


def build():
    ctx = Context(
        mode="operation", twin=False, seed=123, start=Time("2026-10-16T08:00:00", scale="utc"), name="demo"
    )
    ctx.clock.advance(12.5)
    ctx.observation.band = (3.5 * u.um, 4.1 * u.um)
    ctx.observation.n_wavelengths = 16
    ctx.observation.extra = {"site": "Paranal"}
    source = ctx.add(Constant(name="star", level=2.0))
    amp = ElementGroup(name="amp")
    g1, g2 = amp.add(Gain(name="g1", gain=3.0, position=1.5 * u.mm), Gain(name="g2"))
    g1 >> g2
    amp.expose("in", g1.in_)
    amp.expose("out", g2.out)
    source >> amp
    ctx.simulate(source)
    ctx.ui = {"positions": {"star": [0, 0], "amp": [200, 0]}}
    return ctx


def without_date(document):
    return {k: v for k, v in document.items() if k != "created_at"}


# --- Documents -----------------------------------------------------------------------------------


def test_context_document_layout(registry):
    document = to_document(build(), registry=registry)
    assert document["kind"] == "context"
    assert document["meta"]["name"] == "demo"
    settings = document["context"]
    assert settings["mode"] == "operation"
    assert settings["twin"] is False
    assert settings["seed"] == 123
    assert settings["clock"]["start"].startswith("2026-10-16T08:00:00")
    assert settings["clock"]["elapsed"] == {"value": 12.5, "unit": "s"}
    assert settings["observation"]["band"]["min"] == {"value": 3.5, "unit": "um"}
    assert settings["observation"]["n"] == 16
    assert settings["simulate"] == ["star"]
    assert [e["id"] for e in document["elements"]] == ["star", "amp"]
    assert document["elements"][1]["type"] == "lumos.group"
    assert document["connections"] == [{"from": "star.out", "to": "amp.in"}]
    assert document["ui"]["positions"]["amp"] == [200, 0]


def test_context_round_trip(tmp_path, registry):
    ctx = build()
    path = tmp_path / "demo.lumos"
    save(ctx, path, registry=registry)
    copy = load(path, registry=registry)
    assert isinstance(copy, Context)
    assert without_date(to_document(copy, registry=registry)) == without_date(
        to_document(ctx, registry=registry)
    )
    assert copy.mode == "operation"
    assert copy.mode_of(copy["star"]) == "simulation"
    assert copy.elapsed == 12.5 * u.s
    assert copy["amp.g1"].position == 1.5 * u.mm
    copy.mode = ctx.mode = "simulation"
    assert copy.run(["amp.out"])["amp.out"].value == ctx.run(["amp.out"])["amp.out"].value == 12.0


def test_random_streams_survive_the_round_trip(tmp_path, registry):
    ctx = build()
    path = tmp_path / "demo.lumos"
    save(ctx, path, registry=registry)
    copy = load(path, registry=registry)
    np.testing.assert_array_equal(copy.rng("star").random(4), ctx.rng("star").random(4))


def test_elements_and_groups_still_save_alone(tmp_path, registry):
    path = tmp_path / "gain.lumos"
    save(Gain(name="g", gain=5.0), path, registry=registry)
    assert load(path, registry=registry).gain == 5.0


# --- Sidecar for large arrays ---------------------------------------------------------------------


def test_large_arrays_go_to_a_sidecar(tmp_path, registry):
    ctx = Context()
    big = np.random.default_rng(0).random((128, 128))  # 128 kB of float64
    ctx.add(Constant(name="c")) >> Mask(name="big", mask=big)
    ctx.add(Mask(name="small", mask=np.ones((3, 3))))
    path = tmp_path / "masks.lumos"
    save(ctx, path, registry=registry)
    text = path.read_text(encoding="utf-8")
    assert '"$ref": "masks.npz#' in text
    assert (tmp_path / "masks.npz").exists()
    data = json.loads(text)
    small = next(e for e in data["elements"] if e["id"] == "small")
    assert "data" in small["params"]["mask"]["value"]  # inline
    copy = load(path, registry=registry)
    np.testing.assert_array_equal(copy["big"].mask.value, big)
    np.testing.assert_array_equal(copy["small"].mask.value, np.ones((3, 3)))


def test_missing_sidecar_is_reported(tmp_path, registry):
    ctx = Context()
    ctx.add(Mask(name="big", mask=np.zeros((128, 128))))
    path = tmp_path / "m.lumos"
    save(ctx, path, registry=registry)
    (tmp_path / "m.npz").unlink()
    with pytest.raises(LumosFormatError, match="sidecar"):
        load(path, registry=registry)


def test_sidecar_references_cannot_escape_the_directory(tmp_path, registry):
    ctx = Context()
    ctx.add(Mask(name="big", mask=np.zeros((128, 128))))
    path = tmp_path / "m.lumos"
    save(ctx, path, registry=registry)
    text = path.read_text(encoding="utf-8").replace('"$ref": "m.npz#', '"$ref": "../m.npz#')
    path.write_text(text, encoding="utf-8")
    with pytest.raises(LumosFormatError):
        load(path, registry=registry)


# --- Schema and versions ---------------------------------------------------------------------


def test_documents_follow_the_published_schema(registry):
    import jsonschema

    schema = document_schema()
    for document in (to_document(build(), registry=registry), to_document(Gain(), registry=registry)):
        jsonschema.validate(document, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"lumos_format": "1.0", "kind": "nonsense"}, schema)


def test_the_schema_is_shipped_with_the_package():
    from importlib import resources

    shipped = json.loads(
        resources.files("lumos.io").joinpath("schemas/lumos-1.2.schema.json").read_text("utf-8")
    )
    assert shipped == document_schema()


def test_newer_format_versions_are_refused(tmp_path, registry):
    path = tmp_path / "future.lumos"
    document = to_document(build(), registry=registry)
    document["lumos_format"] = "2.0"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(FormatVersionError):
        load(path, registry=registry)


def test_unknown_context_settings_are_refused(registry):
    from lumos.io import from_document

    document = to_document(build(), registry=registry)
    document["context"]["warp_drive"] = True
    with pytest.raises(LumosFormatError, match="warp_drive"):
        from_document(document, registry=registry)
