"""Tests for the network layer of the catalogues (step P9.1, REQ-CAT-5, NF-7).

Requirements: user cache (``platformdirs``) with a configurable lifetime, offline mode, TLS always verified,
a timeout on every request; providers are pluggable (``Provider``). No test reaches the network: requests
go to a fake transport that records what it is asked.
"""

import json
import time

import pytest

from lumos.catalogs import http
from lumos.catalogs.http import (
    CatalogSettings,
    OfflineError,
    Response,
    fetch,
    settings,
    use_transport,
)
from lumos.catalogs.provider import Provider, providers, register_provider


class FakeTransport:
    """Answer every request with a fixed body and record the calls."""

    def __init__(self, body=b'{"ok": true}', status=200, content_type="application/json"):
        self.calls = []
        self.body, self.status, self.content_type = body, status, content_type

    def request(self, method, url, *, params=None, data=None, timeout, verify):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "params": params,
                "data": data,
                "timeout": timeout,
                "verify": verify,
            }
        )
        return Response(self.status, self.body, self.content_type)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_dir", tmp_path / "cache")
    monkeypatch.setattr(settings, "offline", False)
    monkeypatch.setattr(settings, "ttl", 7 * 86400.0)
    monkeypatch.setattr(settings, "timeout", 30.0)


def test_fetch_uses_the_transport_with_tls_and_a_timeout():
    transport = FakeTransport()
    with use_transport(transport):
        response = fetch("https://example.org/tap", params={"q": "1"})
    assert response.json() == {"ok": True}
    assert transport.calls[0]["verify"] is True
    assert transport.calls[0]["timeout"] == pytest.approx(30.0)
    assert response.from_cache is False


def test_per_request_timeout():
    transport = FakeTransport()
    with use_transport(transport):
        fetch("https://example.org/tap", timeout=5.0)
    assert transport.calls[0]["timeout"] == pytest.approx(5.0)


def test_timeouts_must_be_positive():
    with use_transport(FakeTransport()), pytest.raises(ValueError, match="timeout"):
        fetch("https://example.org/tap", timeout=0)


def test_tls_verification_cannot_be_disabled():
    with pytest.raises(ValueError, match="TLS"):
        fetch("https://example.org/tap", verify=False)


def test_only_https_is_accepted():
    with use_transport(FakeTransport()), pytest.raises(ValueError, match="https"):
        fetch("http://example.org/tap")


def test_responses_are_cached():
    transport = FakeTransport()
    with use_transport(transport):
        first = fetch("https://example.org/tap", params={"q": "1"})
        second = fetch("https://example.org/tap", params={"q": "1"})
        fetch("https://example.org/tap", params={"q": "2"})
    assert len(transport.calls) == 2
    assert second.from_cache is True
    assert second.content == first.content


def test_cache_lives_in_the_configured_directory(tmp_path):
    with use_transport(FakeTransport()):
        fetch("https://example.org/tap")
    files = list((tmp_path / "cache").glob("*.json"))
    assert len(files) == 1
    entry = json.loads(files[0].read_text(encoding="utf-8"))
    assert entry["url"] == "https://example.org/tap"


def test_expired_entries_are_fetched_again(monkeypatch):
    transport = FakeTransport()
    with use_transport(transport):
        fetch("https://example.org/tap")
        monkeypatch.setattr(settings, "ttl", 0.0)
        time.sleep(0.01)
        fetch("https://example.org/tap")
    assert len(transport.calls) == 2


def test_cache_can_be_bypassed():
    transport = FakeTransport()
    with use_transport(transport):
        fetch("https://example.org/tap")
        fetch("https://example.org/tap", cache=False)
    assert len(transport.calls) == 2


def test_offline_mode_reads_the_cache_only(monkeypatch):
    transport = FakeTransport()
    with use_transport(transport):
        fetch("https://example.org/tap", params={"q": "cached"})
        monkeypatch.setattr(settings, "offline", True)
        assert fetch("https://example.org/tap", params={"q": "cached"}).from_cache
        with pytest.raises(OfflineError):
            fetch("https://example.org/tap", params={"q": "new"})
    assert len(transport.calls) == 1


def test_offline_mode_ignores_expiry(monkeypatch):
    transport = FakeTransport()
    with use_transport(transport):
        fetch("https://example.org/tap")
        monkeypatch.setattr(settings, "ttl", 0.0)
        monkeypatch.setattr(settings, "offline", True)
        assert fetch("https://example.org/tap").from_cache


def test_http_errors_are_raised_and_not_cached():
    transport = FakeTransport(body=b"down", status=503, content_type="text/plain")
    with use_transport(transport), pytest.raises(http.CatalogError, match="503"):
        fetch("https://example.org/tap")
    with use_transport(transport), pytest.raises(http.CatalogError):
        fetch("https://example.org/tap")
    assert len(transport.calls) == 2


def test_binary_bodies_round_trip_through_the_cache():
    body = bytes(range(256))
    transport = FakeTransport(body=body, content_type="application/fits")
    with use_transport(transport):
        fetch("https://example.org/file.fits")
        cached = fetch("https://example.org/file.fits")
    assert cached.content == body


def test_settings_defaults():
    defaults = CatalogSettings()
    assert defaults.ttl > 0
    assert defaults.timeout > 0
    assert defaults.offline is False
    assert "lumos" in str(defaults.cache_dir).lower()


def test_providers_are_pluggable():
    @register_provider
    class Dummy(Provider):
        name = "dummy-test"
        description = "a test provider"

    assert providers()["dummy-test"] is Dummy
    with pytest.raises(ValueError, match="already"):

        @register_provider
        class Other(Provider):
            name = "dummy-test"


def test_no_network_without_a_transport_in_tests(monkeypatch):
    """The default transport is the real one; tests always install a fake one."""
    assert http.default_transport().__class__.__name__ == "RequestsTransport"


def test_settings_from_the_environment(monkeypatch, tmp_path):
    """Notebooks and CI select a cache and the offline mode with environment variables (step P9.9)."""
    monkeypatch.setenv("LUMOS_CATALOGS_CACHE", str(tmp_path / "shared"))
    monkeypatch.setenv("LUMOS_CATALOGS_OFFLINE", "1")
    configured = CatalogSettings()
    assert configured.cache_dir == tmp_path / "shared"
    assert configured.offline is True
    monkeypatch.setenv("LUMOS_CATALOGS_OFFLINE", "0")
    assert CatalogSettings().offline is False
