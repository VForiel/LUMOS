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
