r"""Transports of ASCII lines to equipment: TCP, serial, VISA and in-process (step P12.2, ADR-0014).

A :class:`Transport` exchanges lines of text terminated by ``\n`` (the terminator is added on writing and
removed on reading). Errors are reported as driver errors, so that drivers handle them uniformly:

- :class:`~lumos.core.driver.DriverTimeout` when no line arrives within the timeout;
- :class:`~lumos.core.driver.ConnectionLost` when the connection cannot be opened or is closed by the other
  end (a driver then reconnects and retries).

Transports
----------
- :class:`TCPTransport`: a TCP socket (``tcp://host:port``), used by the LumosLabs emulators;
- :class:`SerialTransport`: a serial port through pyserial (extra ``[hardware]``), any URL of
  ``serial.serial_for_url``: ``COM3``, ``/dev/ttyUSB0``, the virtual ``loop://`` and ``socket://host:port``;
- :class:`VisaTransport`: a VISA resource through pyvisa (``GPIB0::12::INSTR``, ``TCPIP0::...::SOCKET``...);
  the resource manager can be injected for tests;
- :class:`PipeTransport`: an in-process function answering each line (an emulator in the same process).

:func:`open_transport` creates a transport from a URL: ``tcp://``, ``serial://`` (followed by the port or the
pyserial URL, options as ``?baudrate=9600``), ``visa://`` and ``pipe://`` is not a URL (use
:class:`PipeTransport`).
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs

from lumos.core.driver import ConnectionLost, DriverTimeout

__all__ = ["PipeTransport", "SerialTransport", "TCPTransport", "Transport", "VisaTransport", "open_transport"]

logger = logging.getLogger("lumos.hardware")

TERMINATOR = "\n"


class Transport(ABC):
    """Exchange of ASCII lines with an equipment.

    Parameters
    ----------
    timeout
        Default time to wait for a line, in seconds.
    """

    def __init__(self, *, timeout: float = 5.0) -> None:
        self.timeout = float(timeout)
        self._lock = threading.RLock()

    @property
    @abstractmethod
    def url(self) -> str:
        """URL of the transport."""

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """Whether the transport is open."""

    @abstractmethod
    def open(self) -> None:
        """Open the connection (raises :class:`ConnectionLost` if impossible)."""

    @abstractmethod
    def close(self) -> None:
        """Close the connection."""

    @abstractmethod
    def write_line(self, line: str) -> None:
        """Send ``line`` followed by the terminator."""

    @abstractmethod
    def read_line(self, timeout: float | None = None) -> str:
        """Return the next line, without its terminator."""

    def query(self, line: str, timeout: float | None = None) -> str:
        """Send ``line`` and return the line answered (atomic with respect to other threads)."""
        with self._lock:
            self.write_line(line)
            return self.read_line(timeout)

    def __repr__(self) -> str:
        """Show the URL."""
        return f"{type(self).__name__}({self.url!r})"


# --- TCP --------------------------------------------------------------------------------------------------


class TCPTransport(Transport):
    """Lines over a TCP connection."""

    def __init__(self, host: str, port: int, *, timeout: float = 5.0) -> None:
        super().__init__(timeout=timeout)
        self.host, self.port = host, int(port)
        self._socket: socket.socket | None = None
        self._buffer = b""

    @property
    def url(self) -> str:
        """``tcp://host:port``."""
        return f"tcp://{self.host}:{self.port}"

    @property
    def is_open(self) -> bool:
        """Whether the socket is open."""
        return self._socket is not None

    def open(self) -> None:
        """Connect to the server."""
        try:
            self._socket = socket.create_connection((self.host, self.port), timeout=self.timeout)
        except OSError as error:
            raise ConnectionLost(f"cannot connect to {self.url}: {error}") from error
        self._buffer = b""

    def close(self) -> None:
        """Close the socket."""
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None

    def _require(self) -> socket.socket:
        if self._socket is None:
            raise ConnectionLost(f"{self.url} is not open")
        return self._socket

    def write_line(self, line: str) -> None:
        """Send a line."""
        sock = self._require()
        try:
            sock.sendall((line + TERMINATOR).encode("ascii"))
        except OSError as error:
            self.close()
            raise ConnectionLost(f"{self.url}: {error}") from error

    def read_line(self, timeout: float | None = None) -> str:
        """Receive a line."""
        sock = self._require()
        deadline = time.monotonic() + (self.timeout if timeout is None else float(timeout))
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DriverTimeout(f"{self.url}: no answer within {timeout or self.timeout:g} s")
            sock.settimeout(remaining)
            try:
                chunk = sock.recv(65536)
            except TimeoutError as error:
                raise DriverTimeout(f"{self.url}: no answer within {timeout or self.timeout:g} s") from error
            except OSError as error:
                self.close()
                raise ConnectionLost(f"{self.url}: {error}") from error
            if not chunk:
                self.close()
                raise ConnectionLost(f"{self.url}: connection closed by the equipment")
            self._buffer += chunk
        line, _, self._buffer = self._buffer.partition(b"\n")
        return line.decode("ascii").rstrip("\r")


# --- Serial -----------------------------------------------------------------------------------------------


class SerialTransport(Transport):
    """Lines over a serial port (pyserial URL: ``COM3``, ``/dev/ttyUSB0``, ``loop://``, ``socket://...``)."""

    def __init__(self, port: str, *, baudrate: int = 115200, timeout: float = 5.0) -> None:
        super().__init__(timeout=timeout)
        self.port = port
        self.baudrate = int(baudrate)
        self._serial: Any = None

    @property
    def url(self) -> str:
        """``serial://port``."""
        return f"serial://{self.port}"

    @property
    def is_open(self) -> bool:
        """Whether the port is open."""
        return self._serial is not None

    def open(self) -> None:
        """Open the port."""
        try:
            import serial
        except ImportError as error:
            raise ImportError(
                "serial transports need pyserial: pip install 'lumos-maxima[hardware]'"
            ) from error
        try:
            self._serial = serial.serial_for_url(self.port, baudrate=self.baudrate, timeout=self.timeout)
        except (serial.SerialException, OSError, ValueError) as error:
            raise ConnectionLost(f"cannot open {self.url}: {error}") from error

    def close(self) -> None:
        """Close the port."""
        if self._serial is not None:
            try:
                self._serial.close()
            finally:
                self._serial = None

    def _require(self) -> Any:
        if self._serial is None:
            raise ConnectionLost(f"{self.url} is not open")
        return self._serial

    def write_line(self, line: str) -> None:
        """Send a line."""
        port = self._require()
        try:
            port.write((line + TERMINATOR).encode("ascii"))
            port.flush()
        except OSError as error:
            self.close()
            raise ConnectionLost(f"{self.url}: {error}") from error

    def read_line(self, timeout: float | None = None) -> str:
        """Receive a line (pyserial returns what arrived before the timeout)."""
        port = self._require()
        port.timeout = self.timeout if timeout is None else float(timeout)
        try:
            raw = port.read_until(b"\n")
        except OSError as error:
            self.close()
            raise ConnectionLost(f"{self.url}: {error}") from error
        if not raw.endswith(b"\n"):
            raise DriverTimeout(f"{self.url}: no complete line within {port.timeout:g} s")
        return raw[:-1].decode("ascii").rstrip("\r")


# --- VISA -------------------------------------------------------------------------------------------------


def _is_timeout(error: BaseException) -> bool:
    return (
        isinstance(error, TimeoutError)
        or "TMO" in str(getattr(error, "error_code", ""))
        or "TMO" in str(error)
    )


class VisaTransport(Transport):
    """Lines with a VISA resource through pyvisa (or an injected resource manager)."""

    def __init__(self, resource: str, *, resource_manager: Any = None, timeout: float = 5.0) -> None:
        super().__init__(timeout=timeout)
        self.resource = resource
        self._manager = resource_manager
        self._instrument: Any = None

    @property
    def url(self) -> str:
        """``visa://resource``."""
        return f"visa://{self.resource}"

    @property
    def is_open(self) -> bool:
        """Whether the resource is open."""
        return self._instrument is not None

    def open(self) -> None:
        """Open the resource."""
        if self._manager is None:
            try:
                import pyvisa
            except ImportError as error:
                raise ImportError(
                    "VISA transports need pyvisa: pip install 'lumos-maxima[hardware]'"
                ) from error
            self._manager = pyvisa.ResourceManager()
        try:
            instrument: Any = self._manager.open_resource(self.resource)
        except Exception as error:
            raise ConnectionLost(f"cannot open {self.url}: {error}") from error
        instrument.timeout = int(self.timeout * 1000)  # milliseconds
        instrument.read_termination = TERMINATOR
        instrument.write_termination = TERMINATOR
        self._instrument = instrument

    def close(self) -> None:
        """Close the resource."""
        if self._instrument is not None:
            try:
                self._instrument.close()
            finally:
                self._instrument = None

    def _require(self) -> Any:
        if self._instrument is None:
            raise ConnectionLost(f"{self.url} is not open")
        return self._instrument

    def write_line(self, line: str) -> None:
        """Send a line."""
        try:
            self._require().write(line)
        except (ConnectionLost, DriverTimeout):
            raise
        except Exception as error:
            raise ConnectionLost(f"{self.url}: {error}") from error

    def read_line(self, timeout: float | None = None) -> str:
        """Receive a line."""
        instrument = self._require()
        if timeout is not None:
            instrument.timeout = int(float(timeout) * 1000)
        try:
            return str(instrument.read()).rstrip("\r\n")
        except Exception as error:
            if _is_timeout(error):
                raise DriverTimeout(f"{self.url}: no answer in time") from error
            raise ConnectionLost(f"{self.url}: {error}") from error
        finally:
            if timeout is not None:
                instrument.timeout = int(self.timeout * 1000)


# --- In-process -------------------------------------------------------------------------------------------


class PipeTransport(Transport):
    """Lines answered by a function of the same process (``handler(line) -> reply``), e.g. an emulator."""

    def __init__(self, handler: Callable[[str], str], *, name: str = "pipe", timeout: float = 5.0) -> None:
        super().__init__(timeout=timeout)
        self.handler = handler
        self.name = name
        self._open = False
        self._replies: list[str] = []

    @property
    def url(self) -> str:
        """``pipe://name``."""
        return f"pipe://{self.name}"

    @property
    def is_open(self) -> bool:
        """Whether the pipe is open."""
        return self._open

    def open(self) -> None:
        """Open the pipe."""
        self._open = True
        self._replies.clear()

    def close(self) -> None:
        """Close the pipe."""
        self._open = False

    def write_line(self, line: str) -> None:
        """Give the line to the handler and keep its answer."""
        if not self._open:
            raise ConnectionLost(f"{self.url} is not open")
        try:
            reply = self.handler(line)
        except ConnectionLost:
            self._open = False
            raise
        if reply is not None:
            self._replies.append(reply)

    def read_line(self, timeout: float | None = None) -> str:
        """Return the next answer."""
        if not self._open:
            raise ConnectionLost(f"{self.url} is not open")
        if not self._replies:
            raise DriverTimeout(f"{self.url}: no answer")
        return self._replies.pop(0)


# --- URLs -------------------------------------------------------------------------------------------------


def open_transport(url: str, *, timeout: float = 5.0) -> Transport:
    """Create (without opening) the transport of ``url``: ``tcp://``, ``serial://`` or ``visa://``."""
    scheme, separator, rest = url.partition("://")
    if not separator:
        raise ValueError(f"{url!r} is not a transport URL (scheme://...)")
    if scheme == "tcp":
        host, _, port = rest.rpartition(":")
        return TCPTransport(host, int(port), timeout=timeout)
    if scheme == "serial":
        target, _, query = rest.partition("?")
        options = {key: values[-1] for key, values in parse_qs(query).items()}
        return SerialTransport(target, baudrate=int(options.get("baudrate", 115200)), timeout=timeout)
    if scheme == "visa":
        return VisaTransport(rest, timeout=timeout)
    raise ValueError(f"unknown transport scheme {scheme!r} in {url!r}: use tcp://, serial:// or visa://")
