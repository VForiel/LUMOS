"""Tests for transports (step P12.2, REQ-HW-2, REQ-HW-3, ADR-0014).

Requirements: serial (including virtual), TCP and VISA transports behind one testable interface, exchanging
``\\n``-terminated ASCII lines, with timeouts and loss of connection reported as driver errors.

References: a TCP server written here with the standard library (it upper-cases each line), pyserial's
``loop://`` (a loop-back port that returns what is written) and ``socket://`` URLs, and a fake VISA resource
manager that records the calls (pyvisa's ``open_resource``/``write``/``read`` interface).
"""

import socket
import socketserver
import threading

import pytest

from lumos.core.driver import ConnectionLost, DriverTimeout
from lumos.hardware.transport import (
    PipeTransport,
    SerialTransport,
    TCPTransport,
    VisaTransport,
    open_transport,
)


class Upper(socketserver.StreamRequestHandler):
    def handle(self):
        for raw in self.rfile:
            line = raw.decode().strip()
            if line == "SILENT":
                continue
            if line == "QUIT":
                return
            self.wfile.write((line.upper() + "\n").encode())


@pytest.fixture
def server():
    tcp = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Upper)
    tcp.daemon_threads = True
    thread = threading.Thread(target=tcp.serve_forever, daemon=True)
    thread.start()
    yield tcp.server_address
    tcp.shutdown()
    tcp.server_close()


def test_tcp_transport(server):
    host, port = server
    transport = TCPTransport(host, port, timeout=1.0)
    transport.open()
    assert transport.is_open
    assert transport.query("hello") == "HELLO"
    assert transport.query("two words") == "TWO WORDS"
    with pytest.raises(DriverTimeout):
        transport.query("SILENT", timeout=0.2)
    transport.write_line("QUIT")
    with pytest.raises(ConnectionLost):
        transport.read_line(timeout=1.0)
    transport.close()
    assert not transport.is_open


def test_tcp_connection_refused():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    with pytest.raises(ConnectionLost):
        TCPTransport("127.0.0.1", port, timeout=0.5).open()


def test_virtual_serial_loop():
    pytest.importorskip("serial")
    transport = SerialTransport("loop://", timeout=0.2)
    transport.open()
    assert transport.query("PING") == "PING"  # the loop-back returns the line written
    with pytest.raises(DriverTimeout):
        transport.read_line(timeout=0.1)
    transport.close()


def test_serial_over_a_socket(server):
    pytest.importorskip("serial")
    host, port = server
    transport = SerialTransport(f"socket://{host}:{port}", timeout=1.0)
    transport.open()
    assert transport.query("abc") == "ABC"
    transport.close()


class FakeResource:
    def __init__(self, log):
        self.log = log
        self.timeout = None
        self.read_termination = None
        self.write_termination = None

    def write(self, text):
        self.log.append(("write", text))

    def read(self):
        if self.log[-1] == ("write", "SILENT"):
            raise TimeoutError("VI_ERROR_TMO")
        return self.log[-1][1].upper()

    def close(self):
        self.log.append(("close",))


class FakeManager:
    def __init__(self):
        self.log = []
        self.opened = []

    def open_resource(self, name):
        self.opened.append(name)
        return FakeResource(self.log)


def test_visa_transport_with_an_injected_resource_manager():
    manager = FakeManager()
    transport = VisaTransport("TCPIP0::192.0.2.1::5025::SOCKET", resource_manager=manager, timeout=0.5)
    transport.open()
    assert manager.opened == ["TCPIP0::192.0.2.1::5025::SOCKET"]
    assert transport.query("*idn?") == "*IDN?"
    with pytest.raises(DriverTimeout):
        transport.query("SILENT")
    transport.close()
    assert manager.log[-1] == ("close",)


def test_pipe_transport_calls_its_handler():
    transport = PipeTransport(lambda line: f"OK {line[::-1]}")
    transport.open()
    assert transport.query("abc") == "OK cba"
    transport.close()
    with pytest.raises(ConnectionLost):
        transport.query("abc")


def test_open_transport_from_urls(server):
    host, port = server
    tcp = open_transport(f"tcp://{host}:{port}", timeout=1.0)
    assert isinstance(tcp, TCPTransport)
    assert tcp.url == f"tcp://{host}:{port}"
    pytest.importorskip("serial")
    assert isinstance(open_transport("serial://loop://"), SerialTransport)
    assert isinstance(open_transport("serial://COM7?baudrate=9600"), SerialTransport)
    assert open_transport("serial://COM7?baudrate=9600").baudrate == 9600
    assert isinstance(open_transport("visa://GPIB0::12::INSTR"), VisaTransport)
    with pytest.raises(ValueError, match="scheme"):
        open_transport("ftp://example.org")
