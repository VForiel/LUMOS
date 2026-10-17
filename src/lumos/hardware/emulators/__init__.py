"""Emulators of the LumosLabs devices (step P12.4, REQ-HW-3, ADR-0014).

An emulator answers the LumosLabs protocol (``docs/hardware/lumoslabs_protocol.md``) line by line
(:meth:`Emulator.handle`). It runs in the process (through a
:class:`~lumos.hardware.transport.PipeTransport`) or behind a TCP server (:class:`EmulatorServer`,
``lumos emulators start``), so that the drivers go through a real transport.
"""

from __future__ import annotations

import logging
import socketserver
import threading
from typing import Any

from lumos.core.driver import ConnectionLost
from lumos.hardware.emulators.base import Emulator, EmulatorError
from lumos.hardware.emulators.devices import (
    Camera,
    FilterWheel,
    LaserSource,
    LinearStage,
    PhaseController,
    Piezo,
    PowerMeter,
    RotationStage,
    Shutter,
)

__all__ = ["EMULATORS", "Emulator", "EmulatorError", "EmulatorServer", "create_emulator"]

logger = logging.getLogger("lumos.hardware")

#: Emulator classes by model.
EMULATORS: dict[str, type[Emulator]] = {
    cls.model: cls
    for cls in (
        LinearStage,
        RotationStage,
        Piezo,
        Shutter,
        FilterWheel,
        LaserSource,
        PowerMeter,
        Camera,
        PhaseController,
    )
}


def create_emulator(model: str, **options: Any) -> Emulator:
    """Return an emulator of ``model`` (e.g. ``"LL-LS100"``); options as for :class:`Emulator`."""
    try:
        cls = EMULATORS[model.upper()]
    except KeyError:
        raise ValueError(f"unknown LumosLabs model {model!r}; known: {', '.join(EMULATORS)}") from None
    return cls(**options)


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        emulator: Emulator = self.server.emulator  # type: ignore[attr-defined]
        for raw in self.rfile:
            line = raw.decode("ascii", errors="replace").strip()
            if not line:
                continue
            try:
                reply = emulator.handle(line)
            except ConnectionLost:
                logger.info("%s: connection dropped (injected fault)", emulator.model)
                return
            self.wfile.write((reply + "\n").encode("ascii"))
            self.wfile.flush()


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


class EmulatorServer:
    """TCP server of one emulator (use as a context manager, or :meth:`start` and :meth:`stop`)."""

    def __init__(self, emulator: Emulator, *, host: str = "127.0.0.1", port: int = 0) -> None:
        self.emulator = emulator
        self._server = _Server((host, int(port)), _Handler)
        self._server.emulator = emulator  # type: ignore[attr-defined]
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        """``(host, port)`` the server listens on."""
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    @property
    def url(self) -> str:
        """``tcp://host:port``."""
        host, port = self.address
        return f"tcp://{host}:{port}"

    def start(self) -> EmulatorServer:
        """Serve in a background thread."""
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name=self.emulator.model
        )
        self._thread.start()
        logger.info("%s %s emulated on %s", self.emulator.model, self.emulator.serial, self.url)
        return self

    def stop(self) -> None:
        """Stop serving."""
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> EmulatorServer:
        """Start the server."""
        return self.start()

    def __exit__(self, *exc: object) -> None:
        """Stop the server."""
        self.stop()
