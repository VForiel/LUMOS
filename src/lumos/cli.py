"""Command line of LUMOS (``lumos``).

- ``lumos emulators list``: the LumosLabs devices that can be emulated;
- ``lumos emulators start [--devices ...] [--host H] [--port P] [--duration S]``: serve emulators over TCP,
  one port per device from ``P`` (``0``: free ports), until Ctrl+C or for ``S`` seconds.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

__all__ = ["main"]


def _write(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _emulators(arguments: argparse.Namespace) -> int:
    from lumos.hardware.emulators import EMULATORS, EmulatorServer, create_emulator

    if arguments.action == "list":
        for model, cls in EMULATORS.items():
            _write(f"{model:10s} {cls.description}")
        return 0
    servers = []
    models = arguments.devices or list(EMULATORS)
    try:
        for index, model in enumerate(models):
            port = 0 if arguments.port == 0 else arguments.port + index
            emulator = create_emulator(model, serial=f"SN{index + 1:04d}")
            servers.append(EmulatorServer(emulator, host=arguments.host, port=port).start())
        for server in servers:
            _write(f"{server.emulator.model:10s} {server.url:28s} {server.emulator.serial}")
        _write("emulators running; press Ctrl+C to stop" if arguments.duration is None else "")
        if arguments.duration is None:
            while True:
                time.sleep(3600)
        else:
            time.sleep(arguments.duration)
    except KeyboardInterrupt:
        pass
    finally:
        for server in servers:
            server.stop()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lumos", description="LUMOS command line")
    commands = parser.add_subparsers(dest="command", required=True)
    emulators = commands.add_parser("emulators", help="LumosLabs emulators")
    emulators.add_argument("action", choices=("list", "start"))
    emulators.add_argument("--devices", nargs="*", help="models to emulate (default: all)")
    emulators.add_argument("--host", default="127.0.0.1", help="address to listen on (default: local only)")
    emulators.add_argument("--port", type=int, default=5100, help="first port (0: free ports)")
    emulators.add_argument(
        "--duration", type=float, default=None, help="seconds to run (default: until Ctrl+C)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return the exit status."""
    arguments = _parser().parse_args(argv)
    if arguments.command == "emulators":
        return _emulators(arguments)
    return 1  # pragma: no cover - argparse requires a command


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
