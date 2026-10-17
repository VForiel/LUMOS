"""Tests for the LumosLabs drivers and lab elements (step P12.5, REQ-HW-1 to REQ-HW-5, ADR-0014).

Requirements: one driver per LumosLabs device, over a real transport (here TCP to the emulators); lab
elements whose actions go to the drivers in operation mode; detectors returning ``DataProduct`` measurements.
Tests of the plan: limits respected; the emergency stop interrupts a motion; recovery after an injected
disconnection; the measured duration of an action is consistent with the predicted one.

References: the behaviours of the emulators (``docs/hardware/lumoslabs_protocol.md``), e.g. a move of d at v
and a lasts d/v + v/a.
"""

import threading
import time

import numpy as np
import pytest
from astropy import units as u

from lumos import Context
from lumos.core.driver import ActionRefused, DriverError, DriverStopped, LimitError
from lumos.elements.detectors import Camera, PowerMeter
from lumos.elements.lab import FilterWheel, LaserSource, LinearStage, PhaseController, Piezo, Shutter
from lumos.hardware.drivers.lumoslabs import (
    CameraDriver,
    FilterWheelDriver,
    LaserDriver,
    LinearStageDriver,
    PhaseControllerDriver,
    PiezoDriver,
    PowerMeterDriver,
    ShutterDriver,
)
from lumos.hardware.emulators import EmulatorServer, create_emulator


@pytest.fixture
def served():
    servers = []

    def serve(model, **options):
        emulator = create_emulator(model, latency=0.0, **options)
        server = EmulatorServer(emulator).start()
        servers.append(server)
        return emulator, server.url

    yield serve
    for server in servers:
        server.stop()


def operated(element, driver):
    ctx = Context(mode="operation", twin=False)
    ctx.add(element)
    element.driver = driver
    driver.connect()
    return ctx


def test_stage_moves_with_the_predicted_duration(served):
    emulator, url = served("LL-LS100")
    stage = LinearStage(name="stage", velocity=50 * u.mm / u.s, acceleration=500 * u.mm / u.s**2)
    ctx = operated(stage, LinearStageDriver(url, poll=0.005))
    stage.home()
    stage.move(20 * u.mm)
    assert stage.position.to_value(u.mm) == pytest.approx(20, abs=1e-3)
    entry = ctx.journal.entries("action")[-1]
    predicted = 20 / 50 + 50 / 500
    # the move starts from the encoder reading after homing (noise of 10 nm, i.e. 2e-7 s at 50 mm/s)
    assert entry.predicted == pytest.approx(predicted, abs=1e-6)
    assert entry.measured == pytest.approx(predicted, abs=0.06)  # polling period and latency
    assert emulator.velocity == pytest.approx(50)  # the driver sent the velocity of the element


def test_stage_limits_come_from_the_device(served):
    _, url = served("LL-LS100")
    stage = LinearStage(name="stage")
    driver = LinearStageDriver(url)
    operated(stage, driver)
    assert driver.limits["move.target"][1] == 100 * u.mm
    stage.home()
    with pytest.raises(LimitError):
        stage.move(150 * u.mm)


def test_emergency_stop_over_tcp(served):
    emulator, url = served("LL-LS100")
    stage = LinearStage(name="stage", velocity=10 * u.mm / u.s, acceleration=1000 * u.mm / u.s**2)
    ctx = operated(stage, LinearStageDriver(url, poll=0.01))
    stage.home()
    errors = []

    def move():
        try:
            stage.move(40 * u.mm)
        except DriverStopped as error:
            errors.append(error)

    thread = threading.Thread(target=move)
    thread.start()
    time.sleep(0.3)
    ctx.stop_all()
    thread.join(timeout=2)
    assert errors
    assert emulator.stopped
    assert 1 < emulator.position < 10
    stage.driver.reset()  # sends *RST
    assert not emulator.stopped


def test_recovery_after_an_injected_disconnection(served, caplog):
    emulator, url = served("LL-LS100")
    stage = LinearStage(name="stage", velocity=50 * u.mm / u.s, acceleration=500 * u.mm / u.s**2)
    operated(stage, LinearStageDriver(url, poll=0.005))
    stage.home()
    emulator.inject("DISCONNECT")
    stage.move(3 * u.mm)
    assert stage.position.to_value(u.mm) == pytest.approx(3, abs=1e-3)
    assert "reconnect" in caplog.text


def test_the_driver_checks_the_model(served):
    _, url = served("LL-SH1")
    with pytest.raises(DriverError, match="LL-LS100"):
        LinearStageDriver(url).connect()


def test_shutter_wheel_and_piezo(served):
    _, shutter_url = served("LL-SH1")
    shutter = Shutter(name="shutter")
    operated(shutter, ShutterDriver(shutter_url, poll=0.002))
    shutter.open()
    assert shutter.driver.query("state") == "OPEN"
    shutter.close()
    assert shutter.driver.query("state") == "CLOSED"

    wheel_emulator, wheel_url = served("LL-FW6")
    wheel = FilterWheel(name="wheel")
    operated(wheel, FilterWheelDriver(wheel_url, poll=0.01))
    wheel.select(6)
    assert wheel_emulator.slot == 6

    piezo_emulator, piezo_url = served("LL-PZ3")
    piezo = Piezo(name="piezo")
    operated(piezo, PiezoDriver(piezo_url))
    piezo.move("Y", 12 * u.um)
    assert piezo_emulator.targets["Y"] == pytest.approx(12)
    with pytest.raises(LimitError):
        piezo.move("Y", 40 * u.um)


def test_laser_emission_needs_a_confirmation_and_stops_in_the_safe_state(served):
    emulator, url = served("LL-LD")
    laser = LaserSource(name="laser")
    driver = LaserDriver(url)
    ctx = operated(laser, driver)
    laser.set_power(3 * u.mW)
    with pytest.raises(ActionRefused):
        laser.emit(True)
    driver.confirm = lambda command: True
    laser.emit(True)
    assert emulator.emitting
    ctx.stop_all()
    assert not emulator.emitting


def test_detectors_return_data_products(served):
    meter_emulator, meter_url = served("LL-PM", seed=3)
    driver = PowerMeterDriver(meter_url)
    driver.connect()
    driver.feed(2e-6 * u.W)
    reading = driver.acquire()
    assert reading.data.unit == u.W
    assert reading.data[0].to_value(u.W) == pytest.approx(1.02 * 2e-6 + 1e-9, rel=0.01)

    camera_emulator, camera_url = served("LL-CAM", seed=4)
    camera = CameraDriver(camera_url, poll=0.005)
    camera.connect()
    camera.feed(np.full((8, 10), 400.0) * u.electron / u.s)
    camera.configure(exposure=0.05 * u.s)
    frame = camera.acquire()
    assert frame.shape == (8, 10)
    assert frame.data.unit == u.adu
    normal = frame.data.value[camera_emulator.hot_pixels == 0]
    assert normal.mean() == pytest.approx(100 + (400 + 5) * 0.05 / 2, rel=0.05)


def test_phase_controller(served):
    emulator, url = served("LL-PCU")
    pcu = PhaseController(name="pcu")
    operated(pcu, PhaseControllerDriver(url, poll=0.002))
    pcu.set_voltage(3, 5 * u.V)
    assert emulator.voltages[2] == pytest.approx(5)
    assert pcu.driver.query("phase", 3).to_value(u.rad) == pytest.approx(np.pi)
    with pytest.raises(LimitError):
        pcu.set_voltage(3, 11 * u.V)


def test_lab_elements_simulate_their_actions():
    ctx = Context()
    stage = ctx.add(LinearStage(name="stage", velocity=10 * u.mm / u.s, acceleration=100 * u.mm / u.s**2))
    stage.home()
    stage.move(30 * u.mm)
    assert stage.position == 30 * u.mm
    assert ctx.clock.elapsed.to_value(u.s) == pytest.approx(0.1 + 30 / 10 + 10 / 100)
    shutter = ctx.add(Shutter(name="shutter"))
    shutter.open()
    assert shutter.is_open == 1
    detectors = (Camera, PowerMeter)
    assert all(getattr(cls, "lumoslabs_model", None) for cls in detectors)
