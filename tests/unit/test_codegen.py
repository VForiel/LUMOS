"""Tests for the generation of an equivalent Python script (step P5.9).

Requirements: REQ-CTX-8 (generation of a Python script equivalent to a context). The plan of phase 5 asks
that the ``.lumos`` round trip and the generated script give identical results.

Reference: the generated script is executed and the context it builds is compared with the original one
(same document, without the creation date, and same computed outputs).
"""

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

from lumos import Context
from lumos.core.element import Element
from lumos.core.group import ElementGroup
from lumos.core.param import Param, Setting, State
from lumos.core.port import Input, Output
from lumos.core.priors import Normal
from lumos.core.registry import Registry
from lumos.core.signal import Signal
from lumos.io import to_document
from lumos.io.codegen import to_python


class Number(Signal):
    def __init__(self, value):
        self.value = value


class Constant(Element):
    level = Param(unit=u.W, default=1 * u.W)
    label = Setting(str, default="source")
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
    mask = Param(unit=u.dimensionless_unscaled, shape=None, default=np.ones(2))
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
        mode="operation", twin=False, seed=7, start=Time("2026-10-16T08:00:00", scale="utc"), name="demo"
    )
    ctx.clock.advance(3.0)
    ctx.observation.band = (1 * u.um, 2 * u.um)
    ctx.observation.n_wavelengths = 5
    ctx.observation.rule = "gauss"
    source = ctx.add(Constant(name="star", level=3 * u.mW, label="Vega"))
    source.set_free("level", True, Normal(3 * u.mW, 0.1 * u.mW))
    amp = ElementGroup(name="amp")
    g1, g2 = amp.add(Gain(name="g1", gain=3.0, position=2 * u.mm), Gain(name="g2"))
    g1 >> g2
    amp.expose("in", g1.in_)
    amp.expose("out", g2.out)
    source >> amp >> Mask(name="mask", mask=np.arange(6.0).reshape(2, 3))
    ctx.simulate(source)
    ctx.ui = {"positions": {"star": [0, 0]}}
    return ctx


def execute(script):
    namespace = {}
    exec(compile(script, "<generated>", "exec"), namespace)
    return namespace["ctx"]


def without_date(document):
    return {k: v for k, v in document.items() if k != "created_at"}


def test_generated_script_rebuilds_the_same_context(registry):
    ctx = build()
    copy = execute(to_python(ctx))
    assert isinstance(copy, Context)
    assert without_date(to_document(copy, registry=registry)) == without_date(
        to_document(ctx, registry=registry)
    )


def test_generated_script_gives_the_same_results():
    ctx = build()
    copy = execute(to_python(ctx))
    for c in (ctx, copy):
        c.mode = "simulation"
    assert copy.run(["mask.out"])["mask.out"].value == ctx.run(["mask.out"])["mask.out"].value
    np.testing.assert_array_equal(copy.rng("star").random(3), ctx.rng("star").random(3))


def test_generated_script_is_readable_python():
    script = to_python(build())
    compile(script, "<generated>", "exec")
    assert "from lumos import Context" in script
    assert "ctx = Context(" in script
    assert "ctx.connect(" in script
    assert "Constant(" in script
    assert script == to_python(build())  # deterministic


def test_context_method_and_empty_context():
    script = Context(seed=1).to_python()
    copy = execute(script)
    assert len(copy) == 0
    assert copy.seed == 1
