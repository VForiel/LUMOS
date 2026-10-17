"""Tests for the discovery of equipment (step P12.6, REQ-HW-2, ADR-0014).

Requirements: enumerate the connected devices (TCP, serial, VISA) and associate them with the elements of a
context. References: emulators started here with known models and serial numbers; a TCP server that is not a
LumosLabs device; a closed port; pyserial's ``loop://`` port, which echoes the question instead of answering.
"""

import socket
import socketserver
import threading

import pytest
from astropy import units as u

from lumos import Context
from lumos.core.driver import DriverError
from lumos.elements.detectors import Camera
from lumos.elements.lab import LinearStage, Shutter
from lumos.hardware.discovery import DeviceInfo, associate, discover
from lumos.hardware.drivers.lumoslabs import CameraDriver, LinearStageDriver, ShutterDriver
from lumos.hardware.emulators import EmulatorServer, create_emulator


class Other(socketserver.StreamRequestHandler):
    def handle(self):
        for _ in self.rfile:
            self.wfile.write(b"HELLO FROM ANOTHER DEVICE\n")


@pytest.fixture
def bench():
    servers = [
        EmulatorServer(create_emulator(model, serial=serial, latency=0.0)).start()
        for model, serial in (
            ("LL-LS100", "SN0001"),
            ("LL-LS100", "SN0002"),
            ("LL-SH1", "SN0003"),
            ("LL-CAM", "SN0004"),
        )
    ]
    other = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Other)
    other.daemon_threads = True
    threading.Thread(target=other.serve_forever, daemon=True).start()
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    closed = probe.getsockname()[1]
    probe.close()
    urls = [s.url for s in servers] + [
        f"tcp://127.0.0.1:{other.server_address[1]}",
        f"tcp://127.0.0.1:{closed}",
    ]
    yield urls
    for server in servers:
        server.stop()
    other.shutdown()
    other.server_close()


def test_discovery_finds_the_lumoslabs_devices(bench):
    devices = discover(tcp=bench, timeout=0.5)
    assert [(d.model, d.serial) for d in devices] == [
        ("LL-LS100", "SN0001"),
        ("LL-LS100", "SN0002"),
        ("LL-SH1", "SN0003"),
        ("LL-CAM", "SN0004"),
    ]
    assert all(isinstance(d, DeviceInfo) and d.vendor == "LUMOSLABS" for d in devices)
    assert devices[0].url == bench[0]


def test_serial_ports_that_do_not_answer_are_skipped():
    pytest.importorskip("serial")
    assert discover(serial=["loop://"], timeout=0.2) == []


def test_association_attaches_and_connects_drivers(bench):
    devices = discover(tcp=bench, timeout=0.5)
    ctx = Context(mode="operation")
    stage_a = ctx.add(LinearStage(name="stage_a"))
    stage_b = ctx.add(LinearStage(name="stage_b"))
    shutter = ctx.add(Shutter(name="shutter"))
    camera = ctx.add(Camera(name="camera"))
    found = associate(ctx, devices, assignments={"stage_a": "SN0002", "stage_b": "SN0001"})
    assert found["stage_a"].serial == "SN0002"
    assert isinstance(stage_a.driver, LinearStageDriver)
    assert isinstance(shutter.driver, ShutterDriver)
    assert isinstance(camera.driver, CameraDriver)
    assert all(d.connected for d in ctx.drivers().values())
    assert stage_b.driver.status()["serial"] == "SN0001"
    stage_a.home()
    stage_a.move(1 * u.mm)
    for driver in ctx.drivers().values():
        driver.disconnect()


def test_ambiguous_association_is_refused(bench):
    devices = discover(tcp=bench, timeout=0.5)
    ctx = Context(mode="operation")
    ctx.add(LinearStage(name="stage_a"))
    ctx.add(LinearStage(name="stage_b"))
    with pytest.raises(DriverError, match="SN0001"):
        associate(ctx, devices, connect=False)


def test_missing_devices_are_reported(bench):
    devices = discover(tcp=bench[2:3], timeout=0.5)  # the shutter only
    ctx = Context(mode="operation")
    ctx.add(LinearStage(name="stage"))
    with pytest.raises(DriverError, match="LL-LS100"):
        associate(ctx, devices, connect=False)
