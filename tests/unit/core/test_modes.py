"""Tests for operating modes, drivers and the digital twin skeleton (step P5.6).

Requirements: REQ-EL-4 (simulation and operation modes on the context, per-element override), REQ-EL-5 (in
simulation no real command is sent: the command is logged, the virtual state updated and the clock advanced;
in operation the command goes to the driver, the state is read back and the real duration recorded),
REQ-EL-7 (digital twin: in operation each command is also applied to the simulated model, on by default),
REQ-HW-1 (``Driver`` interface: connect, disconnect, send, query, status, stop; ``SimulatedDriver`` logs the
commands and returns the virtual state).

References: durations are written by hand; the fake driver below records every call made to its transport.
"""

import logging

import pytest
from astropy import units as u

from lumos import Context
from lumos.core.action import action
from lumos.core.driver import Command, Driver, DriverError, Reply, SimulatedDriver
from lumos.core.element import Element
from lumos.core.param import Param, State


class Stage(Element):
    velocity = Param(unit=u.mm / u.s, default=2 * u.mm / u.s)
    position = State(unit=u.mm, default=0 * u.mm)
    body_calls = 0

    @action(duration=lambda self, target: abs(target - self.position) / self.velocity)
    def move(self, target):
        type(self).body_calls += 1
        self.position = target


class RecordingDriver(Driver):
    """Driver whose transport records the commands and replies with a measured duration and state."""

    def __init__(self, duration=1.25 * u.s, position=None):
        super().__init__()
        self.sent = []
        self.duration = duration
        self.position = position

    def _connect(self):
        pass

    def _disconnect(self):
        pass

    def _send(self, command):
        self.sent.append(command)
        state = {} if self.position is None else {"position": self.position}
        return Reply(duration=self.duration, state=state)

    def _query(self, name):
        return self.position

    def _stop(self):
        self.sent.append("STOP")


@pytest.fixture(autouse=True)
def reset_counter():
    Stage.body_calls = 0


# --- Simulation ------------------------------------------------------------------------------------


def test_simulation_never_calls_the_transport(caplog):
    ctx = Context()
    stage = ctx.add(Stage(name="stage"))
    driver = RecordingDriver()
    stage.driver = driver
    with caplog.at_level(logging.INFO, logger="lumos.context"):
        stage.move(4 * u.mm)
    assert driver.sent == []
    assert stage.position == 4 * u.mm
    assert ctx.elapsed.to_value(u.s) == pytest.approx(2.0)
    assert any("stage.move" in record.getMessage() for record in caplog.records)


# --- Operation -------------------------------------------------------------------------------------


def test_operation_sends_the_command_and_records_the_real_duration():
    ctx = Context(mode="operation")
    stage = ctx.add(Stage(name="stage"))
    stage.driver = RecordingDriver(duration=1.25 * u.s)
    stage.driver.connect()
    stage.move(4 * u.mm)  # predicted 2 s, measured 1.25 s
    (command,) = stage.driver.sent
    assert isinstance(command, Command)
    assert command.element == "stage"
    assert command.action == "move"
    assert command.args == (4 * u.mm,)
    assert ctx.elapsed.to_value(u.s) == pytest.approx(1.25)


def test_operation_with_twin_also_updates_the_simulated_model():
    ctx = Context(mode="operation", twin=True)
    stage = ctx.add(Stage(name="stage"))
    stage.driver = RecordingDriver(position=3.9 * u.mm)
    stage.driver.connect()
    stage.move(4 * u.mm)
    assert Stage.body_calls == 1
    assert stage.position == 4 * u.mm  # the twin keeps the simulated state
    assert ctx.twin_state(stage)["position"] == 3.9 * u.mm  # the measured state, side by side


def test_operation_without_twin_reads_the_state_back():
    ctx = Context(mode="operation", twin=False)
    stage = ctx.add(Stage(name="stage"))
    stage.driver = RecordingDriver(position=3.9 * u.mm)
    stage.driver.connect()
    stage.move(4 * u.mm)
    assert Stage.body_calls == 0
    assert stage.position == 3.9 * u.mm


def test_operation_needs_a_connected_driver():
    ctx = Context(mode="operation")
    stage = ctx.add(Stage(name="stage"))
    with pytest.raises(DriverError, match="driver"):
        stage.move(1 * u.mm)
    stage.driver = RecordingDriver()
    with pytest.raises(DriverError, match="connected"):
        stage.move(1 * u.mm)
    assert ctx.elapsed.to_value(u.s) == 0.0


def test_elements_can_be_forced_to_simulation():
    ctx = Context(mode="operation")
    stage = ctx.add(Stage(name="stage"))
    ctx.simulate(stage)
    assert ctx.mode_of(stage) == "simulation"
    stage.move(2 * u.mm)
    assert stage.position == 2 * u.mm
    assert ctx.elapsed.to_value(u.s) == pytest.approx(1.0)
    ctx.simulate(stage, False)
    assert ctx.mode_of(stage) == "operation"


def test_driver_assignment_is_checked():
    stage = Stage()
    with pytest.raises(TypeError):
        stage.driver = "COM3"
    stage.driver = None
    assert stage.driver is None


# --- Drivers ---------------------------------------------------------------------------------------


def test_driver_lifecycle_and_stop():
    driver = RecordingDriver()
    assert not driver.connected
    driver.connect()
    assert driver.connected
    assert driver.status()["connected"] is True
    driver.stop()
    assert driver.sent == ["STOP"]
    driver.disconnect()
    assert not driver.connected
    with pytest.raises(DriverError):
        driver.send(Command("stage", "move", (1 * u.mm,), {}))


def test_simulated_driver_logs_commands_and_returns_the_virtual_state():
    stage = Stage(position=2 * u.mm)
    driver = SimulatedDriver(stage)
    driver.connect()
    reply = driver.send(Command("stage", "move", (5 * u.mm,), {}))
    assert reply.duration is None
    assert driver.commands == [Command("stage", "move", (5 * u.mm,), {})]
    assert driver.query("position") == 2 * u.mm
    assert driver.status()["commands"] == 1


def test_commands_are_json_serialisable():
    command = Command("stage", "move", (5 * u.mm,), {"speed": 1 * u.mm / u.s})
    data = command.to_json()
    assert data["element"] == "stage"
    assert Command.from_json(data) == command
