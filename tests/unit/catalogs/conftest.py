"""Replay of recorded catalogue responses: the catalogue tests never reach the network.

The fixtures of ``tests/fixtures/catalogs`` were recorded from the real services by
``tools/record_catalog_fixtures.py`` (plain ``requests`` calls, independent of LUMOS). A request of LUMOS is
answered with the fixture of the same URL whose ``match`` strings all appear in its parameters or body; any
other request fails the test.
"""

import base64
import json
from pathlib import Path

import pytest

from lumos.catalogs.http import Response, settings, use_transport

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "catalogs"


class ReplayTransport:
    """Answer requests with recorded responses."""

    def __init__(self) -> None:
        self.fixtures = [
            json.loads(path.read_text(encoding="utf-8")) for path in sorted(FIXTURES.glob("*.json"))
        ]
        self.calls: list[dict] = []

    def request(self, method, url, *, params=None, data=None, timeout, verify):
        assert verify is True
        assert timeout > 0
        text = json.dumps({"params": params or {}, "data": data or {}}, sort_keys=True)
        self.calls.append({"method": method, "url": url, "params": params, "data": data})
        for fixture in self.fixtures:
            if fixture["url"].rstrip("/") != url.rstrip("/"):
                continue
            if all(token in text for token in fixture["match"]):
                content = base64.b64decode(fixture["content"])
                return Response(fixture["status"], content, fixture["content_type"])
        raise AssertionError(
            f"unexpected catalogue request (no recorded response): {method} {url} {text[:300]}"
        )


@pytest.fixture
def replay(tmp_path, monkeypatch):
    """Install the replay transport and an empty cache."""
    monkeypatch.setattr(settings, "cache_dir", tmp_path / "catalog-cache")
    monkeypatch.setattr(settings, "offline", False)
    transport = ReplayTransport()
    with use_transport(transport):
        yield transport
