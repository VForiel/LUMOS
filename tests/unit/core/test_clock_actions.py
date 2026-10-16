"""Tests for the clock of the context and timed actions (step P5.5).

Requirements: REQ-CTX-5 (start date as ``astropy.time.Time`` and elapsed time, advanced by actions or by
``ctx.wait``; ``with ctx.parallel():`` advances by the longest duration; time-dependent elements read the
time), REQ-EL-5 (actions update the virtual state and advance the clock in simulation), REQ-EL-6 (fixed
durations or durations computed from the element properties).

References: durations are computed by hand from the kinematics of the test stage, ``|Δx| / v + t_settle``.
"""

from collections import Counter

import pytest
from astropy import units as u
from astropy.time import Time

from lumos import Context
from lumos.core.action import action
from lumos.core.element import Element
from lumos.core.param import Param, State
from lumos.core.port import Output
from lumos.core.signal import Signal

CALLS: Counter = Counter()
START = Time("2026-10-16T00:00:00", scale="utc")


class Number(Signal):
    def __init__(self, value):
        self.value = value


class Stage(Element):
    velocity = Param(unit=u.mm / u.s, default=2 * u.mm / u.s)
    settle = Param(unit=u.s, default=0.5 * u.s)
    position = State(unit=u.mm, default=0 * u.mm)

    @action(duration=lambda self, target: abs(target - self.position) / self.velocity + self.settle)
    def move(self, target):
        self.position = target
        return self.position


class Shutter(Element):
    is_open = State(unit=u.dimensionless_unscaled, default=0.0)

    @action(duration=20 * u.ms)
    def open(self):
        self.is_open = 1.0

    @action()
    def reset(self):
        self.is_open = 0.0


class Clocked(Element):
    time_dependent = True
    rate = Param(unit=u.Hz, default=1 * u.Hz)
    out = Output(Number)

    @staticmethod
    def transfer(values, inputs):
        CALLS["clocked"] += 1
        return {"out": Number(values.rate * values.time)}


class Static(Element):
    out = Output(Number)

    @staticmethod
    def transfer(values, inputs):
        CALLS["static"] += 1
        return {"out": Number(1.0)}


@pytest.fixture(autouse=True)
def reset():
    CALLS.clear()


def elapsed_s(ctx):
    return ctx.elapsed.to_value(u.s)


# --- Clock -------------------------------------------------------------------------------------


def test_clock_starts_at_zero_elapsed_time():
    ctx = Context(start=START)
    assert elapsed_s(ctx) == 0.0
    assert ctx.time == START
    assert isinstance(ctx.time, Time)


def test_wait_advances_the_clock():
    ctx = Context(start=START)
    ctx.wait(10 * u.s)
    ctx.wait(2 * u.min)
    assert elapsed_s(ctx) == pytest.approx(130.0)
    assert (ctx.time - START).to_value(u.s) == pytest.approx(130.0)


def test_wait_validates_the_duration():
    ctx = Context()
    with pytest.raises(u.UnitsError):
        ctx.wait(1 * u.m)
    with pytest.raises(ValueError):
        ctx.wait(-1 * u.s)
    with pytest.raises(TypeError):
        ctx.wait(3.0)


# --- Actions -------------------------------------------------------------------------------------


def test_actions_advance_the_clock_by_their_computed_duration():
    ctx = Context(start=START)
    stage = ctx.add(Stage())
    stage.move(10 * u.mm)  # 10 / 2 + 0.5 = 5.5 s
    assert elapsed_s(ctx) == pytest.approx(5.5)
    stage.move(4 * u.mm)  # 6 / 2 + 0.5 = 3.5 s
    assert elapsed_s(ctx) == pytest.approx(9.0)
    assert stage.position.to_value(u.mm) == pytest.approx(4.0)


def test_fixed_and_zero_durations():
    ctx = Context()
    shutter = ctx.add(Shutter())
    shutter.open()
    assert elapsed_s(ctx) == pytest.approx(0.02)
    shutter.reset()
    assert elapsed_s(ctx) == pytest.approx(0.02)
    assert shutter.is_open.value == 0.0


def test_action_return_values_are_kept():
    ctx = Context()
    stage = ctx.add(Stage())
    assert stage.move(3 * u.mm) == 3 * u.mm


def test_parallel_advances_by_the_longest_duration():
    ctx = Context()
    stage, shutter = ctx.add(Stage(), Shutter())
    with ctx.parallel():
        stage.move(10 * u.mm)  # 5.5 s
        shutter.open()  # 0.02 s
    assert elapsed_s(ctx) == pytest.approx(5.5)
    assert stage.position.to_value(u.mm) == pytest.approx(10.0)
    assert shutter.is_open.value == 1.0


def test_waits_inside_parallel_blocks_count_as_durations():
    ctx = Context()
    shutter = ctx.add(Shutter())
    with ctx.parallel():
        shutter.open()
        ctx.wait(1 * u.s)
    assert elapsed_s(ctx) == pytest.approx(1.0)


def test_nested_parallel_blocks():
    ctx = Context()
    a, b = ctx.add(Stage(name="a"), Stage(name="b", velocity=1 * u.mm / u.s))
    with ctx.parallel():
        a.move(4 * u.mm)  # 2.5 s
        with ctx.parallel():
            b.move(1 * u.mm)  # 1.5 s
            ctx.wait(3 * u.s)
    assert elapsed_s(ctx) == pytest.approx(3.0)


def test_actions_without_a_context_only_update_the_state():
    stage = Stage()
    stage.move(5 * u.mm)
    assert stage.position.to_value(u.mm) == pytest.approx(5.0)


def test_invalid_durations_are_refused():
    class Bad(Element):
        @action(duration=lambda self: -1 * u.s)
        def negative(self):
            pass

        @action(duration=lambda self: 2 * u.m)
        def length(self):
            pass

    ctx = Context()
    bad = ctx.add(Bad())
    with pytest.raises(ValueError):
        bad.negative()
    with pytest.raises(u.UnitsError):
        bad.length()
    assert elapsed_s(ctx) == 0.0


def test_actions_are_listed():
    assert set(Stage.actions()) == {"move"}
    assert set(Shutter.actions()) == {"open", "reset"}


# --- Time-dependent elements ------------------------------------------------------------------


def test_time_dependent_elements_read_the_elapsed_time():
    ctx = Context()
    clocked = ctx.add(Clocked(rate=2 * u.Hz))
    assert ctx.run([clocked.out])[clocked.out].value == 0.0
    ctx.wait(3 * u.s)
    assert ctx.run([clocked.out])[clocked.out].value == pytest.approx(6.0)


def test_only_time_dependent_elements_are_recomputed_when_time_passes():
    ctx = Context()
    clocked, static = ctx.add(Clocked(), Static())
    ctx.run([clocked.out, static.out])
    CALLS.clear()
    ctx.wait(1 * u.s)
    ctx.run([clocked.out, static.out])
    assert Counter({"clocked": 1}) == CALLS
