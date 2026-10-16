"""Tests for the structured journal of a context (step P5.7).

Requirements: REQ-CTX-9 (structured journal through the standard ``logging`` module: commands, states,
numerical choices, physical warnings, durations, and the origin of each action: Python, web or MCP),
REQ-HW-7 groundwork (the list of commands of a script is the same in simulation and in operation).

References: expected entries are written by hand from the scripted actions.
"""

import json
import logging

import numpy as np
import pytest
from astropy import units as u

from lumos import Context
from lumos.core.action import action
from lumos.core.driver import Command, Driver, Reply
from lumos.core.element import Element
from lumos.core.journal import JsonFormatter
from lumos.core.param import Param, State
from lumos.core.port import Output
from lumos.fields import Grid, Wavefront


class Stage(Element):
    velocity = Param(unit=u.mm / u.s, default=2 * u.mm / u.s)
    position = State(unit=u.mm, default=0 * u.mm)

    @action(duration=lambda self, target: abs(target - self.position) / self.velocity)
    def move(self, target):
        self.position = target


class Beam(Element):
    out = Output(Wavefront)

    @staticmethod
    def transfer(values, inputs):
        from lumos.propagation import propagate

        grid = Grid(32, dx=2e-6 * u.m)
        xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
        wf = Wavefront(np.exp(-(xx**2 + yy**2) / (10e-6) ** 2), grid, 1 * u.um)
        return {"out": propagate(wf, 0.1 * u.mm)}


class EchoDriver(Driver):
    def _connect(self):
        pass

    def _disconnect(self):
        pass

    def _send(self, command):
        return Reply(duration=0.5 * u.s)

    def _query(self, name):
        return None

    def _stop(self):
        pass


def script(ctx, stage):
    stage.move(4 * u.mm)
    ctx.wait(1 * u.s)
    stage.move(1 * u.mm)


def test_actions_are_journaled():
    ctx = Context()
    stage = ctx.add(Stage(name="stage"))
    stage.move(4 * u.mm)
    (entry,) = ctx.journal.entries("action")
    assert entry.element == "stage"
    assert entry.action == "move"
    assert entry.args == "4.0 mm"
    assert entry.mode == "simulation"
    assert entry.origin == "python"
    assert entry.predicted == pytest.approx(2.0)
    assert entry.measured is None
    assert entry.elapsed == 0.0  # the clock value when the action started


def test_waits_are_journaled_with_their_duration():
    ctx = Context()
    stage = ctx.add(Stage(name="stage"))
    script(ctx, stage)
    kinds = [e.kind for e in ctx.journal]
    assert kinds == ["action", "wait", "action"]
    assert ctx.journal.entries("wait")[0].predicted == pytest.approx(1.0)
    assert ctx.journal.entries("action")[1].elapsed == pytest.approx(3.0)


def test_origin_is_recorded():
    ctx = Context()
    stage = ctx.add(Stage(name="stage"))
    with ctx.as_origin("mcp"):
        stage.move(1 * u.mm)
    with ctx.as_origin("web"):
        stage.move(2 * u.mm)
    assert [e.origin for e in ctx.journal.entries("action")] == ["mcp", "web"]
    with pytest.raises(ValueError, match="origin"), ctx.as_origin("telepathy"):
        pass


def test_operation_entries_have_the_measured_duration_and_driver():
    ctx = Context(mode="operation")
    stage = ctx.add(Stage(name="stage"))
    stage.driver = EchoDriver()
    stage.driver.connect()
    stage.move(4 * u.mm)
    (entry,) = ctx.journal.entries("action")
    assert entry.mode == "operation"
    assert entry.predicted == pytest.approx(2.0)
    assert entry.measured == pytest.approx(0.5)
    assert entry.driver == "EchoDriver"


def test_the_same_script_gives_the_same_commands_in_both_modes():
    sim = Context()
    script(sim, sim.add(Stage(name="stage")))
    op = Context(mode="operation")
    stage = op.add(Stage(name="stage"))
    stage.driver = EchoDriver()
    stage.driver.connect()
    script(op, stage)
    assert sim.journal.commands() == op.journal.commands()
    assert sim.journal.commands()[0] == Command("stage", "move", (4 * u.mm,), {})


def test_log_records_carry_the_structured_entry(caplog):
    ctx = Context()
    stage = ctx.add(Stage(name="stage"))
    with caplog.at_level(logging.INFO, logger="lumos.context"):
        stage.move(2 * u.mm)
    record = next(r for r in caplog.records if hasattr(r, "lumos"))
    assert record.lumos["action"] == "move"
    assert record.lumos["origin"] == "python"


def test_json_formatter_writes_one_json_object_per_record(caplog):
    ctx = Context()
    stage = ctx.add(Stage(name="stage"))
    with caplog.at_level(logging.INFO, logger="lumos.context"):
        stage.move(2 * u.mm)
    record = next(r for r in caplog.records if hasattr(r, "lumos"))
    data = json.loads(JsonFormatter().format(record))
    assert data["logger"] == "lumos.context"
    assert data["level"] == "INFO"
    assert data["lumos"]["element"] == "stage"


def test_numerical_choices_made_during_a_run_are_journaled():
    ctx = Context()
    beam = ctx.add(Beam(name="beam"))
    ctx.run([beam.out])
    logs = ctx.journal.entries("log")
    assert any(e.logger == "lumos.propagation" and "planner" in e.message for e in logs)


def test_journal_is_json_serialisable():
    ctx = Context()
    stage = ctx.add(Stage(name="stage"))
    script(ctx, stage)
    data = json.loads(json.dumps(ctx.journal.to_json()))
    assert [d["kind"] for d in data] == ["action", "wait", "action"]
