"""Feedback loops: time iterations driven by the context (step P14.14, REQ-CTX-2, ADR-0020).

Requirements:

- a feedback connects an output port to a field (``"element.field"``) and is applied between two iterations,
  with the mode ``"set"`` (the field takes the value) or ``"add"`` (the value is added); the graph of
  signals stays acyclic;
- ``ctx.loop(steps, period=..., outputs=...)`` computes the outputs at ``t_k = t_0 + k·period``, applies the
  feedbacks, then advances the clock; it returns one step per iteration (index, time, outputs) and keeps the
  fed fields and the clock, unless ``restore=True``;
- feedbacks are checked (ports, fields, modes), listed, removed, saved in ``.lumos`` documents (format 1.1)
  and written by ``to_python``; a source element may refuse a target (``check_feedback``).

Expected values are computed by hand: with a source giving ``x + 1`` from a state ``x`` fed back with
``"set"``, the state counts the iterations; with ``"add"`` and a constant source ``c``, it grows by ``c`` per
iteration (an integrator).
"""

import json

import pytest
from astropy import units as u

import lumos.io
from lumos import Context
from lumos.core.element import Element
from lumos.core.param import Param, State
from lumos.core.port import Input, Output
from lumos.core.registry import Registry
from lumos.core.signal import Signal
from lumos.fields import DataProduct


class Number(Signal):
    def __init__(self, value):
        self.value = value


class Counter(Element):
    """Outputs its state plus one (DataProduct in metres) and the elapsed time."""

    time_dependent = True
    x = State(unit=u.m, default=0 * u.m)
    out = Output(DataProduct)
    clock = Output(Number)

    @staticmethod
    def transfer(values, inputs):
        return {"out": DataProduct((values.x + 1) * u.m), "clock": Number(values.time)}


class Constant(Element):
    level = Param(unit=u.mm, default=2 * u.mm)
    out = Output(DataProduct)

    @staticmethod
    def transfer(values, inputs):
        return {"out": DataProduct(values.level * u.m)}


class Picky(Element):
    """Accepts feedbacks to fields named ``x`` only."""

    out = Output(DataProduct)

    @staticmethod
    def transfer(values, inputs):
        return {"out": DataProduct(1 * u.m)}

    def check_feedback(self, target, field):
        if field != "x":
            raise ValueError(f"{self.name} only drives fields named x, not {field}")


class Sink(Element):
    in_ = Input(DataProduct)
    out = Output(DataProduct)

    @staticmethod
    def transfer(values, inputs):
        return {"out": inputs["in"]}


def counter_context():
    ctx = Context(seed=1)
    ctx.add(Counter(name="counter"))
    return ctx


def test_a_set_feedback_counts_the_iterations():
    ctx = counter_context()
    ctx.feedback("counter.out", "counter.x")
    steps = ctx.loop(4, period=2 * u.ms, outputs=["counter.out", "counter.clock"])
    assert [s.step for s in steps] == [0, 1, 2, 3]
    assert [s.time.to_value(u.ms) for s in steps] == pytest.approx([0, 2, 4, 6])
    assert [float(s.outputs["counter.out"].data.to_value(u.m)) for s in steps] == [1, 2, 3, 4]
    assert [s.outputs["counter.clock"].value for s in steps] == pytest.approx([0, 2e-3, 4e-3, 6e-3])
    assert ctx["counter"].x == 4 * u.m  # kept: the loop has run
    assert ctx.elapsed.to_value(u.ms) == pytest.approx(8)


def test_an_add_feedback_integrates():
    ctx = counter_context()
    ctx.add(Constant(name="source"))
    ctx.feedback("source.out", "counter.x", mode="add")
    ctx.loop(5, period=1 * u.s, outputs=["counter.out"])
    assert ctx["counter"].x.to_value(u.mm) == pytest.approx(10)


def test_restore_puts_back_the_fields_and_the_clock():
    ctx = counter_context()
    ctx.feedback("counter.out", "counter.x")
    ctx.loop(3, period=1 * u.s, outputs=["counter.out"], restore=True)
    assert ctx["counter"].x == 0 * u.m
    assert ctx.elapsed.to_value(u.s) == 0


def test_feedbacks_are_listed_checked_and_removed():
    ctx = counter_context()
    ctx.add(Picky(name="picky"))
    ctx.add(Sink(name="sink"))
    ctx.feedback("counter.out", "counter.x", mode="add")
    assert [(f.source, f.target, f.mode) for f in ctx.feedbacks] == [("counter.out", "counter.x", "add")]
    with pytest.raises(ValueError, match="mode"):
        ctx.feedback("counter.out", "counter.x", mode="multiply")
    with pytest.raises(KeyError):
        ctx.feedback("nothing.out", "counter.x")
    with pytest.raises(KeyError):
        ctx.feedback("counter.nothing", "counter.x")
    with pytest.raises(AttributeError):
        ctx.feedback("counter.out", "counter.y")
    ctx.add(Constant(name="source"))
    with pytest.raises(ValueError, match="only drives"):
        ctx.feedback("picky.out", "source.level")
    with pytest.raises(ValueError, match="already"):
        ctx.feedback("counter.out", "counter.x")
    ctx.remove_feedback("counter.out", "counter.x")
    assert ctx.feedbacks == []
    with pytest.raises(KeyError):
        ctx.remove_feedback("counter.out", "counter.x")


def test_a_value_of_another_dimension_stops_the_loop_with_a_clear_error():
    ctx = counter_context()
    ctx.add(Constant(name="source"))

    class Timer(Element):
        t = State(unit=u.s, default=0 * u.s)
        out = Output(DataProduct)

        @staticmethod
        def transfer(values, inputs):
            return {"out": DataProduct(values.t * u.s)}

    ctx.add(Timer(name="timer"))
    ctx.feedback("source.out", "timer.t")
    with pytest.raises(u.UnitsError, match=r"source\.out.*timer\.t"):
        ctx.loop(2, period=1 * u.s, outputs=["timer.out"])
    with pytest.raises(ValueError, match="steps"):
        ctx.loop(0, period=1 * u.s, outputs=["timer.out"])
    with pytest.raises(u.UnitsError):
        ctx.loop(1, period=1 * u.m, outputs=["timer.out"])


def test_feedbacks_are_saved_loaded_and_written_in_python(tmp_path):
    registry = Registry()
    registry.register("tests.Counter")(Counter)
    registry.register("tests.Constant")(Constant)
    ctx = counter_context()
    ctx.add(Constant(name="source"))
    ctx.feedback("source.out", "counter.x", mode="add")
    document = lumos.io.to_document(ctx, registry=registry)
    assert document["lumos_format"] == "1.2"  # 1.1 added the feedbacks, 1.2 the free targets
    assert document["feedbacks"] == [{"from": "source.out", "to": "counter.x", "mode": "add"}]

    path = tmp_path / "loop.lumos"
    lumos.io.save(ctx, path, registry=registry)
    again = lumos.io.load(path, registry=registry)
    assert [(f.source, f.target, f.mode) for f in again.feedbacks] == [("source.out", "counter.x", "add")]

    older = json.loads(json.dumps(document))
    older["lumos_format"] = "1.0"
    del older["feedbacks"]
    assert lumos.io.from_document(older, registry=registry).feedbacks == []  # 1.0 documents are still read


def test_python_scripts_rebuild_the_feedbacks():
    from lumos.elements.atmosphere import AOController, DeformableMirror

    ctx = Context(seed=1)
    ctx.add(DeformableMirror(name="dm"))
    ctx.add(AOController(name="ctrl"))
    ctx.feedback("ctrl.commands", "dm.commands", mode="add")
    assert 'ctx.feedback("ctrl.commands", "dm.commands", mode="add")' in ctx.to_python()
