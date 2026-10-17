"""Tests for the LumosLabs emulators (step P12.4, REQ-HW-3, ADR-0014).

Requirements: emulators of LL-LS100, LL-RS360, LL-PZ3, LL-SH1, LL-FW6, LL-LD, LL-PM, LL-CAM and LL-PCU
(§5.8.1) speaking the protocol of ``docs/hardware/lumoslabs_protocol.md``, with latencies, noise, limits and
fault injection, started by ``lumos emulators start``.

References (the behaviours documented for each device):

- a trapezoidal move of d at velocity v and acceleration a (d >= v²/a) lasts d/v + v/a and passes through the
  middle of its course at half of that time (symmetric profile);
- the laser warms up as P(t) = P_set (1 - exp(-t/τ)) with τ = 2 s;
- the power meter reads (1 + g) P + offset with g = 2 % and an offset of 1 nW, plus noise;
- the camera counts bias + [Poisson((S + D) t) + read noise] / gain;
- the phase of heater i is κ (V_i² + c (V_{i-1}² + V_{i+1}²)), κ = π/25 rad/V², c = 5 %.

A fake clock makes the timings deterministic.
"""

import math
import time

import numpy as np
import pytest
from astropy import units as u

from lumos.core.driver import ConnectionLost
from lumos.hardware import protocol
from lumos.hardware.emulators import EMULATORS, EmulatorServer, create_emulator
from lumos.hardware.transport import TCPTransport


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def device(model, **options):
    clock = Clock()
    return create_emulator(model, clock=clock, latency=0.0, **options), clock


def ask(emulator, line):
    return protocol.parse_reply(emulator.handle(line))


def test_every_model_identifies_itself():
    assert set(EMULATORS) == {
        "LL-LS100",
        "LL-RS360",
        "LL-PZ3",
        "LL-SH1",
        "LL-FW6",
        "LL-LD",
        "LL-PM",
        "LL-CAM",
        "LL-PCU",
    }
    for model in EMULATORS:
        emulator, _ = device(model, serial="SN7")
        identity = protocol.parse_identity(emulator.handle("*IDN?")[3:])
        assert (identity.vendor, identity.model, identity.serial) == ("LUMOSLABS", model, "SN7")
        assert ask(emulator, "STATUS?") == ["READY"]
        with pytest.raises(protocol.DeviceError) as error:
            ask(emulator, "FLY 3")
        assert error.value.code == 100


# --- Stages -----------------------------------------------------------------------------------------------


def test_linear_stage_motion_profile():
    stage, clock = device("LL-LS100")
    with pytest.raises(protocol.DeviceError) as error:
        ask(stage, "MOVE 10[mm]")
    assert error.value.code == 301  # not homed
    ask(stage, "HOME")
    clock.now += 1.0
    assert ask(stage, "BUSY?") == [0]
    assert ask(stage, "LIMITS? POS") == [0 * u.mm, 100 * u.mm]
    ask(stage, "VEL 25[mm/s]")
    ask(stage, "ACC 50[mm/s2]")
    start = clock.now
    ask(stage, "MOVE 50[mm]")
    duration = 50 / 25 + 25 / 50
    clock.now = start + duration / 2
    assert ask(stage, "BUSY?") == [1]
    assert ask(stage, "POS?")[0].to_value(u.mm) == pytest.approx(25, abs=1e-3)
    clock.now = start + duration - 0.01
    assert ask(stage, "BUSY?") == [1]
    clock.now = start + duration + 1e-6
    assert ask(stage, "BUSY?") == [0]
    assert ask(stage, "POS?")[0].to_value(u.mm) == pytest.approx(50, abs=1e-3)
    with pytest.raises(protocol.DeviceLimitError):
        ask(stage, "MOVE 150[mm]")
    with pytest.raises(protocol.DeviceError) as error:
        ask(stage, "MOVE 3[s]")
    assert error.value.code == 102


def test_stop_latches_and_reset_clears():
    stage, clock = device("LL-LS100")
    ask(stage, "HOME")
    clock.now += 1
    ask(stage, "VEL 10[mm/s]")
    ask(stage, "ACC 1000[mm/s2]")
    ask(stage, "MOVE 40[mm]")
    clock.now += 1.0
    ask(stage, "STOP")
    position = ask(stage, "POS?")[0].to_value(u.mm)
    assert 5 < position < 15
    clock.now += 10
    assert ask(stage, "POS?")[0].to_value(u.mm) == pytest.approx(position, abs=1e-3)
    assert ask(stage, "STATUS?") == ["STOPPED"]
    with pytest.raises(protocol.DeviceStopped):
        ask(stage, "MOVE 1[mm]")
    ask(stage, "*RST")
    assert ask(stage, "STATUS?") == ["READY"]


def test_rotation_stage_works_in_degrees():
    stage, clock = device("LL-RS360")
    ask(stage, "HOME")
    clock.now += 1
    ask(stage, "MOVE 90[deg]")
    clock.now += 100
    assert ask(stage, "POS?")[0].to_value(u.deg) == pytest.approx(90, abs=1e-3)
    assert ask(stage, "LIMITS? POS") == [-360 * u.deg, 360 * u.deg]


def test_piezo_hysteresis():
    piezo, clock = device("LL-PZ3")
    ask(piezo, "MOVE X 5[um]")
    ask(piezo, "MOVE X 10[um]")  # approached from below
    clock.now += 1
    below = ask(piezo, "POS? X")[0].to_value(u.um)
    ask(piezo, "MOVE X 20[um]")
    ask(piezo, "MOVE X 10[um]")  # approached from above
    clock.now += 1
    above = ask(piezo, "POS? X")[0].to_value(u.um)
    assert below == pytest.approx(9.7, abs=0.05)
    assert above == pytest.approx(10.3, abs=0.05)
    with pytest.raises(protocol.DeviceLimitError):
        ask(piezo, "MOVE Z 31[um]")


# --- Shutter, filter wheel --------------------------------------------------------------------------------


def test_shutter_timings():
    shutter, clock = device("LL-SH1")
    assert ask(shutter, "STATE?") == ["CLOSED"]
    ask(shutter, "OPEN")
    clock.now += 0.007
    assert ask(shutter, "STATE?") == ["MOVING"]
    clock.now += 0.002
    assert ask(shutter, "STATE?") == ["OPEN"]


@pytest.mark.parametrize(("target", "steps"), [(4, 3), (6, 1), (2, 1)])
def test_filter_wheel_takes_the_shortest_way(target, steps):
    wheel, clock = device("LL-FW6")
    assert ask(wheel, "SLOT?") == [1]
    ask(wheel, f"SLOT {target}")
    clock.now += 0.4 * steps - 0.01
    assert ask(wheel, "BUSY?") == [1]
    clock.now += 0.02
    assert ask(wheel, "BUSY?") == [0]
    assert ask(wheel, "SLOT?") == [target]


# --- Laser, power meter, camera, phase controller ---------------------------------------------------------


def test_laser_warm_up():
    laser, clock = device("LL-LD")
    assert ask(laser, "WL?") == [635 * u.nm]
    ask(laser, "POWER 5[mW]")
    assert ask(laser, "POWER?")[0].to_value(u.mW) == 0  # not emitting
    ask(laser, "EMIT ON")
    clock.now += 2.0
    assert ask(laser, "POWER?")[0].to_value(u.mW) == pytest.approx(5 * (1 - math.exp(-1)), rel=0.01)
    clock.now += 20
    assert ask(laser, "POWER?")[0].to_value(u.mW) == pytest.approx(5, rel=0.01)
    with pytest.raises(protocol.DeviceLimitError):
        ask(laser, "POWER 20[mW]")
    ask(laser, "EMIT OFF")
    assert ask(laser, "EMIT?") == ["OFF"]


def test_power_meter_measures_what_the_twin_sends():
    meter, _ = device("LL-PM", seed=1)
    ask(meter, "SIM:POWER 1e-06[W]")
    ask(meter, "INT 0.1[s]")
    readings = np.array([ask(meter, "READ?")[0].to_value(u.W) for _ in range(400)])
    expected = 1.02 * 1e-6 + 1e-9
    sigma = 2e-10 / math.sqrt(0.1)
    assert readings.mean() == pytest.approx(expected, abs=5 * sigma / math.sqrt(400))
    assert readings.std() == pytest.approx(sigma, rel=0.15)
    ask(meter, "RANGE 5e-07[W]")
    assert ask(meter, "READ?")[0].to_value(u.W) == pytest.approx(5e-7)  # saturated


def test_camera_exposes_the_image_of_the_twin():
    camera, clock = device("LL-CAM", seed=2)
    image = np.full((16, 16), 1000.0, dtype=np.float32)  # e-/s
    camera.handle("SIM:IMAGE " + " ".join(protocol.encode_array(image)))
    ask(camera, "EXP 0.5[s]")
    assert ask(camera, "GAIN?") == [2.0]
    ask(camera, "ACQ")
    assert ask(camera, "BUSY?") == [1]
    clock.now += 0.5
    assert ask(camera, "BUSY?") == [0]
    frame = protocol.decode_array(*ask(camera, "FRAME?"))
    assert frame.dtype == np.uint16
    assert frame.shape == (16, 16)
    normal = frame[camera.hot_pixels == 0]
    expected = 100 + (1000 + 5) * 0.5 / 2  # bias + (signal + dark) t / gain
    assert normal.mean() == pytest.approx(expected, rel=0.01)
    ask(camera, "ROI 2 3 8 4")
    ask(camera, "ACQ")
    clock.now += 0.5
    assert protocol.decode_array(*ask(camera, "FRAME?")).shape == (4, 8)


def test_phase_controller_crosstalk():
    pcu, clock = device("LL-PCU")
    assert ask(pcu, "CHAN?") == [8]
    ask(pcu, "VOLT 3 5[V]")
    clock.now += 0.01
    kappa = math.pi / 25
    assert ask(pcu, "PHASE? 3")[0].to_value(u.rad) == pytest.approx(kappa * 25)
    ask(pcu, "VOLT 4 2[V]")
    clock.now += 0.01
    assert ask(pcu, "PHASE? 3")[0].to_value(u.rad) == pytest.approx(kappa * (25 + 0.05 * 4))
    assert ask(pcu, "PHASE? 2")[0].to_value(u.rad) == pytest.approx(kappa * 0.05 * 25)
    with pytest.raises(protocol.DeviceLimitError):
        ask(pcu, "VOLT 1 12[V]")


# --- Faults -----------------------------------------------------------------------------------------------


def test_fault_injection():
    stage, clock = device("LL-LS100")
    ask(stage, "HOME")
    clock.now += 1
    ask(stage, "SIM:FAULT STUCK")
    ask(stage, "MOVE 10[mm]")
    clock.now += 100
    assert ask(stage, "BUSY?") == [1]
    ask(stage, "SIM:FAULT CLEAR")
    ask(stage, "SIM:FAULT OFFSET 0.5")
    clock.now += 100
    assert ask(stage, "POS?")[0].to_value(u.mm) == pytest.approx(10.5, abs=1e-3)
    ask(stage, "SIM:FAULT CLEAR")
    ask(stage, "SIM:FAULT DISCONNECT")
    with pytest.raises(ConnectionLost):
        stage.handle("POS?")
    assert ask(stage, "POS?")[0].to_value(u.mm) == pytest.approx(10, abs=1e-3)  # one-shot


def test_delay_fault_uses_real_time():
    stage = create_emulator("LL-LS100", latency=0.0)
    stage.inject("DELAY", 0.2)
    began = time.perf_counter()
    stage.handle("*IDN?")
    assert time.perf_counter() - began >= 0.19


# --- Server and command line ------------------------------------------------------------------------------


def test_tcp_server():
    stage = create_emulator("LL-LS100", latency=0.0)
    with EmulatorServer(stage, port=0) as server:
        transport = TCPTransport(*server.address, timeout=1.0)
        transport.open()
        assert transport.query("*IDN?").startswith("OK LUMOSLABS,LL-LS100")
        stage.inject("DISCONNECT")
        with pytest.raises(ConnectionLost):
            transport.query("POS?")
        transport.open()  # the server accepts a new connection
        assert transport.query("STATUS?") == "OK READY"
        transport.close()


def test_command_line_starts_the_emulators(capsys):
    from lumos.cli import main

    assert main(["emulators", "list"]) == 0
    assert "LL-PCU" in capsys.readouterr().out
    assert (
        main(["emulators", "start", "--port", "0", "--duration", "0.3", "--devices", "LL-SH1", "LL-PM"]) == 0
    )
    out = capsys.readouterr().out
    assert "LL-SH1" in out
    assert "tcp://127.0.0.1:" in out
