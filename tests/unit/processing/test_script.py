"""Tests for script blocks (step P11.5, REQ-PIPE-6).

Requirement: a user function wrapped as an element with declared ports. Security (NF-7): a ``.lumos`` file
names the script, it never holds code; loading needs the script to be defined (imported) first.

The expected values are computed by hand from the functions defined here.
"""

import json

import numpy as np
import pytest
from astropy import units as u

import lumos.io
from lumos import Context
from lumos.core.element import Element
from lumos.core.port import Output, PortCompatibilityError
from lumos.core.registry import UnknownTypeError, register
from lumos.fields import DataProduct, Wavefront
from lumos.processing.script import script


@script(name="testlab.scripts.Scale", inputs={"in": DataProduct}, outputs={"out": DataProduct})
def scale(frame, *, factor: float = 2.0, offset: u.Quantity = 0 * u.adu):
    """Multiply a frame by ``factor`` and add ``offset``."""
    return DataProduct(frame.data * factor + offset)


@script(
    name="testlab.scripts.Split",
    inputs={"in": DataProduct},
    outputs={"positive": DataProduct, "negative": DataProduct},
)
def split(frame):
    """Separate the positive and negative parts."""
    data = frame.data
    return {
        "positive": DataProduct(np.maximum(data, 0 * data.unit)),
        "negative": DataProduct(np.minimum(data, 0 * data.unit)),
    }


@register("testlab.scripts.Frames")
class Frames(Element):
    out = Output(DataProduct)

    @staticmethod
    def transfer(values, inputs):
        return {"out": DataProduct(np.array([-1.0, 2.0, 3.0]) * u.adu)}


def test_a_script_is_an_element_with_its_ports_and_parameters():
    block = scale(name="scale", factor=3.0, offset=1 * u.adu)
    assert isinstance(block, Element)
    assert set(block.instance_ports()) == {"in", "out"}
    assert block.factor == 3.0
    assert block.offset == 1 * u.adu
    assert "Multiply a frame" in type(block).__doc__
    with pytest.raises(u.UnitsError):
        scale(offset=1 * u.kg)


def test_scripts_run_in_a_context():
    ctx = Context()
    frames = ctx.add(Frames(name="frames"))
    frames >> ctx.add(scale(name="scale", factor=3.0, offset=1 * u.adu)) >> ctx.add(split(name="split"))
    result = ctx.run(["split.positive", "split.negative"])
    np.testing.assert_allclose(result["split.positive"].data.to_value(u.adu), [0, 7, 10])
    np.testing.assert_allclose(result["split.negative"].data.to_value(u.adu), [-2, 0, 0])


def test_declared_port_types_are_checked():
    ctx = Context()
    frames = ctx.add(Frames(name="frames"))
    block = ctx.add(scale(name="scale"))

    @script(name="testlab.scripts.Field", inputs={"in": Wavefront}, outputs={"out": Wavefront})
    def identity(field):
        return field

    with pytest.raises(PortCompatibilityError):
        ctx.connect(frames.port("out"), ctx.add(identity(name="identity")).port("in"))
    assert block.port("in")


def test_wrong_outputs_are_refused():
    @script(name="testlab.scripts.Broken", inputs={"in": DataProduct}, outputs={"out": DataProduct})
    def broken(frame):
        return 42

    with pytest.raises(TypeError, match="out"):
        broken().apply(**{"in": DataProduct(np.ones(2) * u.adu)})


def test_scripts_are_saved_by_name_and_need_their_definition(tmp_path):
    ctx = Context()
    ctx.add(Frames(name="frames")) >> ctx.add(scale(name="scale", factor=5.0))
    path = tmp_path / "ctx.lumos"
    lumos.io.save(ctx, path)
    text = path.read_text(encoding="utf-8")
    assert "testlab.scripts.Scale" in text
    assert "def scale" not in text  # no code in the file
    loaded = lumos.io.load(path)
    assert loaded["scale"].factor == 5.0
    document = json.loads(text)
    for entry in document["elements"]:
        if entry["id"] == "scale":
            entry["type"] = "testlab.scripts.Undefined"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(UnknownTypeError):
        lumos.io.load(path)


def test_redefining_a_script_replaces_it():
    @script(name="testlab.scripts.Again", inputs={"in": DataProduct}, outputs={"out": DataProduct})
    def again(frame):
        return frame

    @script(name="testlab.scripts.Again", inputs={"in": DataProduct}, outputs={"out": DataProduct})
    def again(frame):  # noqa: F811 - same function defined again (a notebook cell run twice)
        return DataProduct(frame.data * 2)

    out = again().apply(**{"in": DataProduct(np.ones(2) * u.adu)})["out"]
    np.testing.assert_allclose(out.data.value, 2.0)

    with pytest.raises(ValueError, match="already"):

        @script(name="testlab.scripts.Again", inputs={"in": DataProduct}, outputs={"out": DataProduct})
        def other(frame):
            return frame
