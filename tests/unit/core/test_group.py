"""Tests for lumos.core.group.ElementGroup (step P5.4).

Requirements: REQ-CTX-7 (groups with exposed ports, usable as elements, nested freely, serialised),
REQ-EL-12 (individual export of a group in ``.lumos``, ``kind: "group"``, specs §10).

References: expected values are written by hand for affine transfers on scalar signals.
"""

from collections import Counter

import pytest

from lumos import Context
from lumos.core.element import Element
from lumos.core.graph import GraphError
from lumos.core.group import ElementGroup
from lumos.core.param import Param
from lumos.core.port import Input, Output, PortCompatibilityError
from lumos.core.registry import Registry
from lumos.core.signal import DataSignal, Signal
from lumos.io import from_document

CALLS: Counter = Counter()


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
    in_ = Input(Number)
    out = Output(Number)

    @staticmethod
    def transfer(values, inputs):
        CALLS["gain"] += 1
        return {"out": Number(values.gain * inputs["in"].value)}


class Sum(Element):
    a = Input(Number)
    b = Input(Number)
    out = Output(Number)

    @staticmethod
    def transfer(values, inputs):
        return {"out": Number(inputs["a"].value + inputs["b"].value)}


class Data(Element):
    in_ = Input(DataSignal)


@pytest.fixture(autouse=True)
def reset():
    CALLS.clear()


@pytest.fixture
def registry():
    reg = Registry()
    reg.register("tests.Constant")(Constant)
    reg.register("tests.Gain")(Gain)
    reg.register("tests.Sum")(Sum)
    return reg


def amplifier(name="amp", g1=2.0, g2=3.0):
    group = ElementGroup(name=name)
    first, second = group.add(Gain(name="g1", gain=g1), Gain(name="g2", gain=g2))
    first >> second
    group.expose("in", first.in_)
    group.expose("out", second.out)
    return group


# --- Exposed ports ---------------------------------------------------------------------------------


def test_exposed_ports_are_the_group_ports():
    group = amplifier()
    ports = group.instance_ports()
    assert list(ports) == ["in", "out"]
    assert ports["in"].direction == "in"
    assert ports["out"].direction == "out"
    assert ports["in"].signal_type is Number


def test_group_runs_like_an_element_in_a_context():
    ctx = Context()
    source = ctx.add(Constant(level=5.0))
    group = amplifier()
    source >> group
    assert ctx.run([group.port("out")])[group.port("out")].value == 30.0  # 3 · 2 · 5
    assert ctx.run(["amp.out"])["amp.out"].value == 30.0


def test_group_ports_connect_to_other_elements():
    ctx = Context()
    source = ctx.add(Constant(level=1.0))
    group = amplifier()
    final = Gain(name="final", gain=10.0)
    source >> group >> final
    assert ctx.run([final.out])[final.out].value == 60.0


def test_exposing_checks_ports_and_names():
    group = amplifier()
    outsider = Gain()
    with pytest.raises(GraphError, match="not in the group"):
        group.expose("x", outsider.in_)
    with pytest.raises(GraphError, match="already"):
        group.expose("out", group["g1"].out)
    with pytest.raises(GraphError):
        group.expose("bad name!", group["g1"].out)


def test_exposed_port_types_are_checked_outside():
    ctx = Context()
    group = ctx.add(amplifier())
    sink = Data()
    with pytest.raises(PortCompatibilityError):
        ctx.connect(group.port("out"), sink.in_)


def test_internal_access_by_path():
    group = amplifier()
    assert group["g2"].gain == 3.0
    ctx = Context()
    ctx.add(group)
    assert ctx["amp.g1"].gain == 2.0


# --- Caching and invalidation ----------------------------------------------------------------


def test_child_parameter_changes_invalidate_the_outer_results():
    ctx = Context()
    source = ctx.add(Constant(level=1.0))
    group = amplifier()
    source >> group
    assert ctx.run(["amp.out"])["amp.out"].value == 6.0
    CALLS.clear()
    ctx.run(["amp.out"])
    assert not CALLS  # cached
    ctx["amp.g2"].gain = 4.0
    assert ctx.run(["amp.out"])["amp.out"].value == 8.0


def test_new_inputs_are_propagated_inside():
    ctx = Context()
    source = ctx.add(Constant(level=1.0))
    group = amplifier()
    source >> group
    ctx.run(["amp.out"])
    source.level = 2.0
    assert ctx.run(["amp.out"])["amp.out"].value == 12.0


# --- Nesting ---------------------------------------------------------------------------------


def test_groups_nest():
    outer = ElementGroup(name="outer")
    inner = outer.add(amplifier(name="inner"))
    tail = outer.add(Gain(name="tail", gain=0.5))
    inner >> tail
    outer.expose("in", inner.port("in"))
    outer.expose("out", tail.out)
    ctx = Context()
    source = ctx.add(Constant(level=2.0))
    source >> outer
    assert ctx.run(["outer.out"])["outer.out"].value == 6.0  # 0.5 · 3 · 2 · 2
    assert ctx["outer.inner.g1"].gain == 2.0
    ctx["outer.inner.g1"].gain = 1.0
    assert ctx.run(["outer.out"])["outer.out"].value == 3.0


def test_group_with_two_inputs():
    group = ElementGroup(name="mix")
    total = group.add(Sum(name="sum"))
    group.expose("left", total.a)
    group.expose("right", total.b)
    group.expose("out", total.out)
    ctx = Context()
    a, b = ctx.add(Constant(name="a", level=1.0), Constant(name="b", level=2.0))
    ctx.add(group)
    ctx.connect(a.out, group.port("left"))
    ctx.connect(b.out, group.port("right"))
    assert ctx.run(["mix.out"])["mix.out"].value == 3.0


# --- Serialisation ---------------------------------------------------------------------------


def test_group_document_round_trip(registry):
    group = amplifier(g1=1.5, g2=4.0)
    document = group.to_lumos(registry=registry)
    assert document["kind"] == "group"
    entry = document["group"]
    assert entry["type"] == "lumos.group"
    assert [child["id"] for child in entry["children"]] == ["g1", "g2"]
    assert entry["connections"] == [{"from": "g1.out", "to": "g2.in"}]
    assert entry["exposed_ports"] == {"in": "g1.in", "out": "g2.out"}

    copy = from_document(document, registry=registry)
    assert isinstance(copy, ElementGroup)
    assert copy["g1"].gain == 1.5
    ctx = Context()
    source = ctx.add(Constant(level=2.0))
    source >> copy
    assert ctx.run(["amp.out"])["amp.out"].value == 12.0


def test_nested_group_round_trip(registry):
    outer = ElementGroup(name="outer")
    inner = outer.add(amplifier(name="inner"))
    outer.expose("in", inner.port("in"))
    outer.expose("out", inner.port("out"))
    copy = from_document(outer.to_lumos(registry=registry), registry=registry)
    assert isinstance(copy["inner"], ElementGroup)
    assert copy["inner.g2"].gain == 3.0
    assert list(copy.instance_ports()) == ["in", "out"]
