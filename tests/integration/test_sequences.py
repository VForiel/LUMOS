"""Tests for recorded sequences of actions (step P12.8, REQ-HW-7, ADR-0014).

Requirements: record and replay sequences of actions, in simulation then in operation; test of the plan: a
same script gives an identical journal of commands in simulation and in operation. The sequences are saved as
JSON (no pickle).
"""

import json

import pytest
from astropy import units as u

from lumos import Context
from lumos.core.sequence import Sequence
from lumos.elements.lab import FilterWheel, LinearStage, Shutter
from lumos.hardware.drivers.lumoslabs import FilterWheelDriver, LinearStageDriver, ShutterDriver
from lumos.hardware.emulators import EmulatorServer, create_emulator


def bench(mode="simulation"):
    ctx = Context(mode=mode)
    ctx.add(LinearStage(name="stage", velocity=50 * u.mm / u.s, acceleration=500 * u.mm / u.s**2))
    ctx.add(Shutter(name="shutter"))
    ctx.add(FilterWheel(name="wheel"))
    return ctx


def script(ctx):
    ctx["stage"].home()
    ctx["stage"].move(10 * u.mm)
    ctx["shutter"].open()
    ctx["wheel"].select(3)
    ctx["stage"].move(2.5 * u.mm)
    ctx["shutter"].close()


def commands(ctx):
    return [entry.command for entry in ctx.journal.entries("action")]


@pytest.fixture
def devices():
    servers = {
        name: EmulatorServer(create_emulator(model, latency=0.0)).start()
        for name, model in (("stage", "LL-LS100"), ("shutter", "LL-SH1"), ("wheel", "LL-FW6"))
    }
    yield servers
    for server in servers.values():
        server.stop()


def operate(ctx, servers):
    for name, cls in (("stage", LinearStageDriver), ("shutter", ShutterDriver), ("wheel", FilterWheelDriver)):
        ctx[name].driver = cls(servers[name].url, poll=0.005)
        ctx[name].driver.connect()


def test_recording_in_simulation():
    ctx = bench()
    with ctx.record() as sequence:
        script(ctx)
    assert isinstance(sequence, Sequence)
    assert [str(step.command) for step in sequence] == [
        "stage.home()",
        "stage.move(10.0 mm)",
        "shutter.open()",
        "wheel.select(3)",
        "stage.move(2.5 mm)",
        "shutter.close()",
    ]
    times = [step.time.to_value(u.s) for step in sequence]
    assert times[0] == 0
    assert times == sorted(times)
    assert times[1] == pytest.approx(0.1)  # after the homing of 0.1 s


def test_sequences_are_saved_as_json(tmp_path):
    ctx = bench()
    with ctx.record() as sequence:
        script(ctx)
    path = tmp_path / "sequence.json"
    sequence.save(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["kind"] == "sequence"
    loaded = Sequence.load(path)
    assert [s.command for s in loaded] == [s.command for s in sequence]
    assert loaded[1].command.args[0] == 10 * u.mm


def test_same_script_same_commands_in_simulation_and_operation(devices):
    simulated = bench()
    script(simulated)
    operated = bench("operation")
    operate(operated, devices)
    script(operated)
    assert commands(operated) == commands(simulated)
    assert devices["stage"].emulator.position == pytest.approx(2.5)
    assert devices["wheel"].emulator.slot == 3


def test_a_sequence_recorded_in_simulation_replays_in_operation(devices):
    simulated = bench()
    with simulated.record() as sequence:
        script(simulated)
    operated = bench("operation")
    operate(operated, devices)
    operated.replay(sequence)
    assert commands(operated) == [step.command for step in sequence]
    assert devices["shutter"].emulator.state == "CLOSED"
    assert operated["stage"].position.to_value(u.mm) == pytest.approx(2.5)


def test_replay_needs_the_elements_of_the_sequence():
    ctx = bench()
    with ctx.record() as sequence:
        script(ctx)
    other = Context()
    other.add(LinearStage(name="stage"))
    with pytest.raises(KeyError, match="shutter"):
        other.replay(sequence)
