"""Tests for the digital twin (step P12.7, REQ-EL-7, REQ-HW-5, REQ-HW-6, ADR-0014).

Requirements: in operation, commands go to the equipment and to the simulated model; real and simulated
outputs are produced side by side with their residuals; LL-CAM and LL-PM measure what the twin sends them,
with their own noise and defects, which creates a controlled gap between real and simulated outputs;
actuators drive parameters of the optical model (``ctx.link``).

References: the documented models of the emulators (``docs/hardware/lumoslabs_protocol.md``): the power meter
reads (1 + 2 %) P + 1 nW; the camera counts bias + [Poisson((S + D) t) + read noise] / gain with a bias of
100 adu, D = 5 e⁻/s, 3 e⁻ of read noise and 2 e⁻/adu.
"""

import numpy as np
import pytest
from astropy import units as u

from lumos import Context
from lumos.elements.detectors import Camera, PowerMeter
from lumos.elements.lab import LaserSource, LinearStage
from lumos.elements.optics import FreeSpace, ThinLens
from lumos.fields import DataProduct, Grid
from lumos.hardware.drivers.lumoslabs import CameraDriver, LaserDriver, LinearStageDriver, PowerMeterDriver
from lumos.hardware.emulators import EmulatorServer, create_emulator


@pytest.fixture
def emulators():
    started = {}

    def serve(model, **options):
        emulator = create_emulator(model, latency=0.0, **options)
        server = EmulatorServer(emulator).start()
        started[model] = (emulator, server)
        return emulator, server.url

    yield serve
    for _, server in started.values():
        server.stop()


def connect(element, driver):
    element.driver = driver
    driver.connect()
    return driver


def test_power_meter_twin_gives_real_simulated_and_residual(emulators):
    _, laser_url = emulators("LL-LD")
    _, meter_url = emulators("LL-PM", seed=5)
    ctx = Context(mode="operation")
    laser = ctx.add(LaserSource(name="laser", power=2 * u.mW, n_pixels=64, pixel_scale=60 * u.um))
    meter = ctx.add(PowerMeter(name="meter"))
    laser >> meter
    connect(laser, LaserDriver(laser_url, confirm=lambda command: True))
    connect(meter, PowerMeterDriver(meter_url))
    laser.emit(True)
    result = ctx.run(["meter.out"])
    simulated = result.twin["meter.out"].data.to_value(u.W).sum()
    real = result["meter.out"].data.to_value(u.W).sum()
    assert simulated == pytest.approx(2e-3, rel=1e-3)  # the whole beam reaches the meter
    assert real == pytest.approx(1.02 * simulated + 1e-9, rel=1e-4)
    residual = result.residuals["meter.out"].data.to_value(u.W).sum()
    assert residual == pytest.approx(0.02 * simulated + 1e-9, rel=0.01)
    assert result["meter.out"].meta["device"] == "LL-PM"


def test_camera_twin_differs_by_the_defects_of_the_device(emulators):
    camera_emulator, camera_url = emulators("LL-CAM", seed=6)
    ctx = Context(mode="operation")
    laser = ctx.add(
        LaserSource(name="laser", power=1e-9 * u.mW, waist=0.2 * u.mm, n_pixels=32, pixel_scale=20 * u.um)
    )
    camera = ctx.add(
        Camera(
            name="camera",
            exposure=0.05 * u.s,
            gain=2 * u.electron / u.adu,
            bias=100 * u.adu,
            dark_current=5 * u.electron / u.s,
            read_noise=3 * u.electron,
            quantum_efficiency=[1.0],
            noise=False,
        )
    )
    laser >> camera
    ctx.simulate(laser)  # no laser device: the source stays simulated
    laser.emit(True)
    connect(camera, CameraDriver(camera_url, poll=0.005))
    result = ctx.run(["camera.out"])
    real, simulated = result["camera.out"], result.twin["camera.out"]
    assert real.shape == simulated.shape == (32, 32)
    normal = camera_emulator.hot_pixels == 0
    residual = result.residuals["camera.out"].data.to_value(u.adu)
    # same nominal parameters: only noise and hot pixels differ
    assert abs(residual[normal].mean()) < 0.5
    assert residual[~normal].mean() == pytest.approx(500 * 0.05 / 2, rel=0.2)
    # a fault injected in the device shows in the residuals
    camera_emulator.inject("OFFSET", 40)
    shifted = ctx.run(["camera.out"]).residuals["camera.out"].data.to_value(u.adu)
    assert shifted[normal].mean() == pytest.approx(40, abs=0.5)


def test_commands_go_to_the_equipment_and_to_the_model(emulators):
    stage_emulator, url = emulators("LL-LS100")
    ctx = Context(mode="operation")
    stage = ctx.add(LinearStage(name="stage", velocity=50 * u.mm / u.s, acceleration=500 * u.mm / u.s**2))
    connect(stage, LinearStageDriver(url, poll=0.005))
    stage.home()
    stage.move(12 * u.mm)
    assert stage.position == 12 * u.mm  # the simulated model
    assert stage_emulator.position == pytest.approx(12)  # the equipment
    assert ctx.twin_state(stage)["position"].to_value(u.mm) == pytest.approx(12, abs=1e-3)


def test_actuators_drive_the_optical_model():
    ctx = Context()
    stage = ctx.add(LinearStage(name="stage", velocity=50 * u.mm / u.s, acceleration=500 * u.mm / u.s**2))
    laser = ctx.add(LaserSource(name="laser", waist=1 * u.mm, n_pixels=128, pixel_scale=40 * u.um))
    lens = ctx.add(ThinLens(name="lens", focal_length=200 * u.mm))
    gap = ctx.add(FreeSpace(name="gap", distance=180 * u.mm, output_grid=Grid(64, dx=4 * u.um)))
    laser >> lens >> gap
    ctx.link("stage.position", "gap.distance", offset=180 * u.mm)
    laser.emit(True)
    stage.home()
    stage.move(20 * u.mm)
    assert gap.distance == 200 * u.mm
    focused = np.abs(np.asarray(ctx.run(["gap.out"])["gap.out"].field)) ** 2
    stage.move(5 * u.mm)
    assert gap.distance == 185 * u.mm
    defocused = np.abs(np.asarray(ctx.run(["gap.out"])["gap.out"].field)) ** 2
    assert focused.max() > 2 * defocused.max()


def test_without_twin_only_the_measurements_are_returned(emulators):
    _, meter_url = emulators("LL-PM", seed=7)
    ctx = Context(mode="operation", twin=False)
    laser = ctx.add(LaserSource(name="laser", n_pixels=32, pixel_scale=100 * u.um))
    meter = ctx.add(PowerMeter(name="meter"))
    laser >> meter
    ctx.simulate(laser)
    connect(meter, PowerMeterDriver(meter_url))
    result = ctx.run(["meter.out"])
    assert isinstance(result["meter.out"], DataProduct)
    assert result["meter.out"].meta["device"] == "LL-PM"
    assert dict(result.twin) == {}
