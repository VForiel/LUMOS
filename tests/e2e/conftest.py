"""Browser tests of the web interface (Playwright, step P14.11).

They start ``lumos web`` on a free port with the built interface (``npm run build`` in ``web/``) and drive
Chromium. Run them with ``pytest -m e2e tests/e2e`` after ``playwright install chromium``.
"""

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright")

STATIC = Path(__file__).resolve().parents[2] / "src" / "lumos" / "web" / "static" / "index.html"


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def start_server(tmp_path, *arguments):
    """Start ``lumos web`` and return ``(process, base_url)`` once it answers."""
    import httpx

    if not STATIC.is_file():
        pytest.skip("the web interface is not built (npm ci && npm run build in web/)")
    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "lumos.cli",
            "web",
            "--port",
            str(port),
            "--archive-root",
            str(tmp_path / "archives"),
            *arguments,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    url = f"http://localhost:{port}"
    deadline = time.monotonic() + 60
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"lumos web stopped: {process.stderr.read() if process.stderr else ''}")
        try:
            httpx.get(f"{url}/api/info", timeout=2)
            return process, url
        except httpx.TransportError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.3)


def stop_server(process):
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    if process.stderr is not None:
        process.stderr.close()


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    process, url = start_server(tmp_path_factory.mktemp("web"))
    yield url
    stop_server(process)


@pytest.fixture(scope="session")
def base_url(server):
    return server


@pytest.fixture
def browser_context_args(browser_context_args):
    return {**browser_context_args, "locale": "en-GB", "viewport": {"width": 1440, "height": 900}}


@pytest.fixture
def errors(page):
    """Errors written to the console of the page (they fail the tests that check them)."""
    found = []
    page.on("console", lambda message: found.append(message.text) if message.type == "error" else None)
    page.on("pageerror", lambda error: found.append(str(error)))
    return found


@pytest.fixture
def public_server(tmp_path):
    """An instance with the public profile (anonymous simulators, quotas, no operation)."""
    process, url = start_server(tmp_path, "--profile", "public", "--host", "127.0.0.1")
    yield url
    stop_server(process)


@pytest.fixture
def operation_server(tmp_path):
    """A local instance where the operation of equipment is enabled."""
    process, url = start_server(tmp_path, "--enable-operation")
    yield url
    stop_server(process)


@pytest.fixture
def stage_emulator():
    """A LumosLabs LL-LS100 linear stage emulated over TCP."""
    from lumos.hardware.emulators import EmulatorServer, create_emulator

    server = EmulatorServer(create_emulator("LL-LS100", latency=0.0)).start()
    yield server
    server.stop()


@pytest.fixture
def users_server(tmp_path):
    """A local instance with a users file: simon (simulator, password "simulator password")."""
    from argon2 import PasswordHasher

    users = tmp_path / "users.toml"
    users.write_text(
        f'[users.simon]\nrole = "simulator"\npassword = "{PasswordHasher().hash("simulator password")}"\n',
        encoding="utf-8",
    )
    process, url = start_server(tmp_path, "--users", str(users))
    yield url
    stop_server(process)
