"""Tests for the safety mechanisms of drivers (step P12.1, REQ-HW-1, REQ-HW-4, ADR-0014).

Requirements: lifecycle, errors, timeouts, recovery, emergency stop, software limits, dry run, confirmations,
safe state on interruption. Tests of the plan: limits hold even for a command sent directly to the driver; the
emergency stop interrupts a motion in progress; recovery after an injected disconnection.

The equipment is a small in-memory stage written here: its behaviour (a move of d at speed v lasts d/v) is the
reference.
"""

import threading
import time
from typing import ClassVar

import pytest
from astropy import units as u

from lumos import Context
from lumos.core.action import action
from lumos.core.driver import (
    ActionRefused,
    Command,
    ConnectionLost,
    Driver,
    DriverError,
    DriverStopped,
    DriverTimeout,
    LimitError,
    Reply,
)
from lumos.core.element import Element
from lumos.core.param import Param, State


class Stage(Element):
    velocity = Param(unit=u.mm / u.s, default=10 * u.mm / u.s)
    position = State(unit=u.mm, default=0 * u.mm)

    @action(duration=lambda self, target: abs(target - self.position) / self.velocity)
    def move(self, target):
        self.position = target

    @action()
    def laser_on(self):
        pass


class MemoryStage(Driver):
    """A stage moving at 10 mm/s, interrupted by stop(); optional faults."""

    signatures: ClassVar = {"move": ("target",)}
    limits: ClassVar = {"move.target": (0 * u.mm, 50 * u.mm)}
    dangerous: ClassVar = frozenset({"laser_on"})

    def __init__(self, *, losses=0, timeouts=0, speed=10.0, **options):
        super().__init__(**options)
        self.position = 0.0
        self.speed = speed
        self.losses = losses
        self.timeouts = timeouts
        self.sent = []
        self.connections = 0
        self.safe = 0

    def _connect(self):
        self.connections += 1

    def _disconnect(self):
        pass

    def _send(self, command):
        if self.losses:
            self.losses -= 1
            raise ConnectionLost("cable unplugged")
        if self.timeouts:
            self.timeouts -= 1
            raise DriverTimeout("no answer within the timeout")
        self.sent.append(command)
        if command.action == "move":
            target = command.args[0].to_value(u.mm)
            start, began = self.position, time.perf_counter()
            duration = abs(target - start) / self.speed
            finished = self.wait(duration)
            fraction = 1.0 if finished else min((time.perf_counter() - began) / duration, 1.0)
            self.position = start + (target - start) * fraction
            if not finished:
                raise DriverStopped(f"move interrupted at {self.position:.2f} mm")
            return Reply(state={"position": self.position * u.mm})
        return Reply()

    def _query(self, name):
        return self.position * u.mm

    def _stop(self):
        self.sent.append("STOP")

    def _safe_state(self):
        self.safe += 1


def connected(**options):
    driver = MemoryStage(**options)
    driver.connect()
    return driver


# --- Lifecycle --------------------------------------------------------------------------------------------


def test_lifecycle_states():
    driver = MemoryStage()
    assert driver.state == "disconnected"
    driver.connect()
    assert driver.state == "ready"
    driver.speed = 1e6
    driver.send(Command("stage", "move", (1 * u.mm,)))
    assert driver.state == "ready"
    driver.disconnect()
    assert driver.state == "disconnected"


# --- Limits, dry run, confirmations -----------------------------------------------------------------------


def test_limits_hold_for_commands_sent_directly():
    driver = connected(speed=1e6)
    with pytest.raises(LimitError, match=r"move\.target"):
        driver.send(Command("stage", "move", (60 * u.mm,)))
    with pytest.raises(LimitError):
        driver.send(Command("stage", "move", (), {"target": -1 * u.mm}))
    with pytest.raises(LimitError):
        driver.send(Command("stage", "move", (5.1 * u.cm,)))  # units are converted
    driver.send(Command("stage", "move", (5 * u.cm,)))
    assert driver.position == pytest.approx(50.0)
    assert len(driver.sent) == 1


def test_limits_hold_in_operation_mode():
    ctx = Context(mode="operation")
    stage = ctx.add(Stage(name="stage"))
    stage.driver = connected(speed=1e6)
    with pytest.raises(LimitError):
        stage.move(80 * u.mm)
    assert stage.position == 0 * u.mm
    stage.move(20 * u.mm)
    assert stage.position == 20 * u.mm


def test_dry_run_validates_without_sending(caplog):
    driver = connected(dry_run=True)
    reply = driver.send(Command("stage", "move", (10 * u.mm,)))
    assert driver.sent == []
    assert reply.value == "dry-run"
    assert "dry-run" in caplog.text
    with pytest.raises(LimitError):
        driver.send(Command("stage", "move", (100 * u.mm,)))


def test_dangerous_actions_need_a_confirmation():
    driver = connected()
    with pytest.raises(ActionRefused, match="confirmation"):
        driver.send(Command("stage", "laser_on"))
    asked = []
    driver.confirm = lambda command: asked.append(command) or False
    with pytest.raises(ActionRefused):
        driver.send(Command("stage", "laser_on"))
    driver.confirm = lambda command: True
    driver.send(Command("stage", "laser_on"))
    assert asked == [Command("stage", "laser_on")]
    assert driver.sent[-1] == Command("stage", "laser_on")


# --- Errors, timeouts, recovery ---------------------------------------------------------------------------


def test_recovery_after_a_disconnection(caplog):
    driver = connected(losses=1, speed=1e6)
    driver.send(Command("stage", "move", (3 * u.mm,)))
    assert driver.position == pytest.approx(3.0)
    assert driver.connections == 2
    assert "reconnect" in caplog.text


def test_persistent_disconnection_fails_after_the_retries():
    driver = connected(losses=10, retries=2)
    with pytest.raises(DriverError, match="2 retries"):
        driver.send(Command("stage", "move", (3 * u.mm,)))
    assert driver.state == "error"


def test_timeouts_are_reported():
    driver = connected(timeouts=1)
    with pytest.raises(DriverTimeout):
        driver.send(Command("stage", "move", (3 * u.mm,)))


# --- Emergency stop and safe state ------------------------------------------------------------------------


def test_emergency_stop_interrupts_a_motion_in_progress():
    ctx = Context(mode="operation")
    stage = ctx.add(Stage(name="stage"))
    stage.driver = driver = connected()  # 10 mm/s: a move of 40 mm lasts 4 s
    errors = []

    def move():
        try:
            stage.move(40 * u.mm)
        except DriverStopped as error:
            errors.append(error)

    thread = threading.Thread(target=move)
    began = time.perf_counter()
    thread.start()
    time.sleep(0.3)
    ctx.stop_all()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert time.perf_counter() - began < 1.5
    assert errors
    assert 0 < driver.position < 40
    assert "STOP" in driver.sent
    assert driver.safe == 1
    assert driver.state == "stopped"
    with pytest.raises(DriverError, match="reset"):
        driver.send(Command("stage", "move", (1 * u.mm,)))
    driver.reset()
    driver.speed = 1e6
    driver.send(Command("stage", "move", (1 * u.mm,)))


def test_operation_guard_brings_the_equipment_to_a_safe_state():
    ctx = Context(mode="operation")
    stage = ctx.add(Stage(name="stage"))
    stage.driver = driver = connected(speed=1e6)

    def interrupted():
        with ctx.operation_guard():
            stage.move(5 * u.mm)
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        interrupted()
    assert driver.safe == 1
    assert "STOP" in driver.sent
