"""Command line of LUMOS (``lumos``).

- ``lumos emulators list``: the LumosLabs devices that can be emulated;
- ``lumos emulators start [--devices ...] [--host H] [--port P] [--duration S]``: serve emulators over TCP,
  one port per device from ``P`` (``0``: free ports), until Ctrl+C or for ``S`` seconds;
- ``lumos mcp [--archive-root DIR] [--presets-dir DIR] [--enable-operation]``: serve the MCP server for AI
  assistants over stdio, or over streamable HTTP with ``--transport http [--host H] [--port P] [--path /mcp]``
  (local only unless ``--allow-remote``);
- ``lumos web [--config FILE] [--host H] [--port P] [--profile local|public] [--open] ...``: serve the web
  interface (``http://127.0.0.1:8000`` by default);
- ``lumos users add|remove|list [NAME] [--role R] [--password] [--ssh-key FILE]``: users of an instance;
- ``lumos login URL --user NAME [--key FILE]``: log in with an SSH key and print a bearer token.
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
        from lumos.web.config import ConfigError, load_config
    except ImportError as error:  # pragma: no cover - depends on the installed extras
        sys.stderr.write(
            f"the web interface needs the extra 'web': pip install 'lumos-maxima[web]' ({error})\n"
        )
        return 2
    from lumos.service import ServiceConfig

    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    options = {
        "profile": arguments.profile,
        "host": arguments.host,
        "port": arguments.port,
        "users_file": arguments.users,
        "tls_certfile": arguments.tls_cert,
        "tls_keyfile": arguments.tls_key,
        "trust_proxy": arguments.trust_proxy or None,
        "listen_all": arguments.listen_all or None,
        "audit_log": arguments.audit_log,
    }
    given = {key: value for key, value in options.items() if value is not None}
    if arguments.config:  # the file wins over the defaults of the command line
        for key, default in (("profile", "local"), ("host", "127.0.0.1"), ("port", 8000)):
            if given.get(key) == default:
                del given[key]
    service_options = {
        "archive_root": arguments.archive_root,
        "presets_dir": arguments.presets_dir,
        "operation_enabled": arguments.enable_operation or None,
    }
    service = {key: value for key, value in service_options.items() if value is not None}
    try:
        if arguments.config:
            config = load_config(arguments.config, **given, service=service)
        else:
            config = WebConfig(**given, service=ServiceConfig(**service))
        app = create_app(config)
    except ConfigError as error:
        sys.stderr.write(f"invalid configuration: {error}\n")
        return 2
    scheme = "https" if config.tls_certfile else "http"
    shown = "localhost" if config.host in ("127.0.0.1", "::1") else config.host
    url = f"{scheme}://{shown}:{config.port}"
    sys.stderr.write(f"LUMOS web interface ({config.profile} profile) on {url} (press Ctrl+C to stop)\n")
    if arguments.open:
        import threading
        import webbrowser

        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="warning",
        ssl_certfile=config.tls_certfile,
        ssl_keyfile=config.tls_keyfile,
        proxy_headers=config.trust_proxy,
        forwarded_allow_ips="*" if config.trust_proxy else None,
    )
    return 0


def _users(arguments: argparse.Namespace) -> int:
    import getpass
    from pathlib import Path

    from lumos.web.auth import ConfigError, User, hash_password, load_users, save_users

    path = Path(arguments.file)
    try:
        users = load_users(path) if path.exists() else {}
    except ConfigError as error:
        sys.stderr.write(f"{error}\n")
        return 2
    if arguments.action == "list":
        for user in sorted(users.values(), key=lambda u: u.name):
            methods = [m for m, ok in (("password", user.password), ("ssh", user.ssh_keys)) if ok]
            _write(f"{user.name:20s} {user.role:10s} {', '.join(methods) or 'no credential'}")
        return 0
    if not arguments.name:
        sys.stderr.write("give the name of the user\n")
        return 2
    if arguments.action == "remove":
        if users.pop(arguments.name, None) is None:
            sys.stderr.write(f"no user {arguments.name} in {path}\n")
            return 1
        save_users(path, users)
        return 0
    user = users.get(arguments.name) or User(name=arguments.name, role=arguments.role or "simulator")
    if arguments.role:
        user.role = arguments.role
    password = None
    if arguments.password_stdin:
        password = sys.stdin.readline().rstrip("\r\n")
    elif arguments.password:
        password = getpass.getpass(f"password of {arguments.name}: ")
        if password != getpass.getpass("again: "):
            sys.stderr.write("the passwords differ\n")
            return 1
    if password is not None:
        if len(password) < 8:
            sys.stderr.write("use a password of at least 8 characters\n")
            return 1
        user.password = hash_password(password)
    for key in arguments.ssh_key or []:
        line = Path(key).expanduser().read_text(encoding="utf-8").strip()
        if line not in user.ssh_keys:
            user.ssh_keys.append(line)
    candidate = {**users, user.name: user}
    save_users(path, candidate)
    try:
        load_users(path)
    except ConfigError as error:  # e.g. not a public key file: restore the previous file
        save_users(path, users)
        sys.stderr.write(f"{error}\n")
        return 2
    _write(f"{user.name}: {user.role} in {path}")
    return 0


def _component(arguments: argparse.Namespace) -> int:
    """Check component documents; exit status 1 if one is invalid."""
    import json
    from pathlib import Path

    from lumos.components import check_component

    failed = False
    for name in arguments.files:
        try:
            document = json.loads(Path(name).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            sys.stderr.write(f"{name}: cannot read it: {error}\n")
            failed = True
            continue
        problems = check_component(document)
        if problems:
            failed = True
            _write(f"{name}: {len(problems)} problem(s)")
            for problem in problems:
                _write(f"  - {problem}")
        else:
            _write(f"{name}: OK")
    return 1 if failed else 0


def _login(arguments: argparse.Namespace) -> int:
    from lumos.web.login import LoginError, login

    try:
        token = login(arguments.url, arguments.user, arguments.key)
    except LoginError as error:
        sys.stderr.write(f"{error}\n")
        return 1
    sys.stderr.write("logged in; send the token as 'Authorization: Bearer <token>'\n")
    _write(token)
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
    web.add_argument("--config", default=None, help="TOML configuration file (see the hosting guide)")
    web.add_argument("--host", default="127.0.0.1", help="address to listen on (default: 127.0.0.1)")
    web.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    web.add_argument(
        "--profile", choices=("local", "public"), default="local", help="deployment profile (default: local)"
    )
    web.add_argument("--users", default=None, help="users file (lumos users add)")
    web.add_argument("--tls-cert", default=None, help="certificate (PEM) to serve HTTPS")
    web.add_argument("--tls-key", default=None, help="private key (PEM) of the certificate")
    web.add_argument("--trust-proxy", action="store_true", help="behind a reverse proxy terminating TLS")
    web.add_argument(
        "--listen-all",
        action="store_true",
        help="in a container published on the local machine only: listen on 0.0.0.0 with the local profile",
    )
    web.add_argument("--audit-log", default=None, help="JSON-lines file of the security events")
    web.add_argument("--archive-root", default=None, help="directory of the archives of the runs")
    web.add_argument("--presets-dir", default=None, help="directory of the preset .lumos files")
    web.add_argument(
        "--enable-operation", action="store_true", help="allow actions on real equipment, each one confirmed"
    )
    web.add_argument("--open", action="store_true", help="open the interface in the default browser")
    users = commands.add_parser("users", help="users of a web instance")
    users.add_argument("action", choices=("add", "remove", "list"))
    users.add_argument("name", nargs="?", help="user name")
    users.add_argument("--file", default="users.toml", help="users file (default: users.toml)")
    users.add_argument("--role", choices=("viewer", "simulator", "operator"), default=None)
    users.add_argument("--password", action="store_true", help="ask for a password")
    users.add_argument(
        "--password-stdin", action="store_true", help="read the password from the standard input"
    )
    users.add_argument("--ssh-key", action="append", help="public key file (id_ed25519.pub); repeatable")
    component = commands.add_parser("component", help="declarative components (ADR-0025)")
    component.add_argument("action", choices=("check",), help="check: list what makes a component invalid")
    component.add_argument("files", nargs="+", help="component documents (.lumos of kind 'component')")
    login = commands.add_parser("login", help="log in to a web instance with an SSH key")
    login.add_argument("url", help="address of the instance, e.g. https://lumos.example.org")
    login.add_argument("--user", required=True, help="user name")
    login.add_argument("--key", default="~/.ssh/id_ed25519", help="private key (default: ~/.ssh/id_ed25519)")
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
    if arguments.command == "users":
        return _users(arguments)
    if arguments.command == "login":
        return _login(arguments)
    if arguments.command == "component":
        return _component(arguments)
    return 1  # pragma: no cover - argparse requires a command


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
