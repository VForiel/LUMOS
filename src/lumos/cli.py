"""Command line of LUMOS (``lumos``).

- ``lumos emulators list``: the LumosLabs devices that can be emulated;
- ``lumos emulators start [--devices ...] [--host H] [--port P] [--duration S]``: serve emulators over TCP,
  one port per device from ``P`` (``0``: free ports), until Ctrl+C or for ``S`` seconds;
- ``lumos mcp [--archive-root DIR] [--presets-dir DIR] [--enable-operation]``: serve the MCP server for AI
  assistants over stdio, or over streamable HTTP with ``--transport http [--host H] [--port P] [--path /mcp]``
  (local only unless ``--allow-remote``);
- ``lumos web [--host H] [--port P] [--profile local|public] [--open]``: serve the web interface
  (``http://127.0.0.1:8000`` by default).
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


def _mcp(arguments: argparse.Namespace) -> int:
    import logging

    try:
        from lumos.mcp import create_server
    except ImportError as error:  # pragma: no cover - depends on the installed extras
        sys.stderr.write(f"the MCP server needs the extra 'mcp': pip install 'lumos-maxima[mcp]' ({error})\n")
        return 2
    from lumos.service import ServiceConfig

    local = arguments.host in ("127.0.0.1", "localhost", "::1")
    if arguments.transport == "http" and not local:
        if not arguments.allow_remote:
            sys.stderr.write(
                f"refusing to listen on {arguments.host}: the MCP server has no authentication before the\n"
                "web instance; pass --allow-remote to accept requests from other machines anyway\n"
            )
            return 2
        if arguments.enable_operation:
            sys.stderr.write(
                "the operation of equipment is never allowed to unauthenticated remote clients\n"
            )
            return 2
    # stdout carries the stdio protocol: the logs go to stderr
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    config = ServiceConfig(
        archive_root=arguments.archive_root,
        presets_dir=arguments.presets_dir,
        operation_enabled=arguments.enable_operation,
    )
    server = create_server(config=config)
    if arguments.transport == "stdio":
        server.run("stdio")
    else:
        import uvicorn

        from lumos.mcp import create_http_app

        app = create_http_app(server, path=arguments.path, host=arguments.host)
        uvicorn.run(app, host=arguments.host, port=arguments.port, log_level="warning")
    return 0


def _web(arguments: argparse.Namespace) -> int:
    import logging

    try:
        import uvicorn

        from lumos.web import WebConfig, create_app
    except ImportError as error:  # pragma: no cover - depends on the installed extras
        sys.stderr.write(
            f"the web interface needs the extra 'web': pip install 'lumos-maxima[web]' ({error})\n"
        )
        return 2
    from lumos.service import ServiceConfig

    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = WebConfig(
        profile=arguments.profile,
        host=arguments.host,
        port=arguments.port,
        service=ServiceConfig(archive_root=arguments.archive_root, presets_dir=arguments.presets_dir),
    )
    app = create_app(config)
    url = (
        f"http://{'localhost' if arguments.host in ('127.0.0.1', '::1') else arguments.host}:{arguments.port}"
    )
    sys.stderr.write(f"LUMOS web interface on {url} (press Ctrl+C to stop)\n")
    if arguments.open:
        import threading
        import webbrowser

        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=arguments.host, port=arguments.port, log_level="warning")
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
    mcp = commands.add_parser("mcp", help="MCP server for AI assistants (Claude Code, Claude Desktop...)")
    mcp.add_argument("--archive-root", default=None, help="directory of the archives of the runs")
    mcp.add_argument("--presets-dir", default=None, help="directory of the preset .lumos files")
    mcp.add_argument(
        "--transport", choices=("stdio", "http"), default="stdio", help="stdio (default) or streamable HTTP"
    )
    mcp.add_argument("--host", default="127.0.0.1", help="HTTP: address to listen on (default: 127.0.0.1)")
    mcp.add_argument("--port", type=int, default=8765, help="HTTP: port (default: 8765)")
    mcp.add_argument("--path", default="/mcp", help="HTTP: path of the endpoint (default: /mcp)")
    mcp.add_argument(
        "--allow-remote", action="store_true", help="HTTP: accept a non-local address (no authentication)"
    )
    mcp.add_argument(
        "--enable-operation",
        action="store_true",
        help="allow actions on real equipment, each one confirmed by the user (default: disabled)",
    )
    web = commands.add_parser("web", help="web interface (visual editor)")
    web.add_argument("--host", default="127.0.0.1", help="address to listen on (default: 127.0.0.1)")
    web.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    web.add_argument(
        "--profile", choices=("local", "public"), default="local", help="deployment profile (default: local)"
    )
    web.add_argument("--archive-root", default=None, help="directory of the archives of the runs")
    web.add_argument("--presets-dir", default=None, help="directory of the preset .lumos files")
    web.add_argument("--open", action="store_true", help="open the interface in the default browser")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return the exit status."""
    arguments = _parser().parse_args(argv)
    if arguments.command == "emulators":
        return _emulators(arguments)
    if arguments.command == "mcp":
        return _mcp(arguments)
    if arguments.command == "web":
        return _web(arguments)
    return 1  # pragma: no cover - argparse requires a command


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
