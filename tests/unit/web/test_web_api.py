"""Tests for the HTTP backend of the web interface (step P14.1, REQ-SVC-1, REQ-WEB-5, REQ-WEB-6, REQ-WEB-9a).

Requirements: the backend is a thin FastAPI adapter of the service layer (ADR-0015): every operation of the
service is reachable through a JSON API, with the same errors (message, hint, code) mapped to HTTP statuses;
each browser has its own service session (origin ``web``) behind an HttpOnly cookie; the progress of runs is
pushed over a WebSocket; the server listens on ``localhost`` by default and, in the local profile, refuses
requests addressed to another host name (protection against DNS rebinding) and WebSockets opened from
another origin.

Expected values are physical: a power meter behind a laser of power P reads P (energy conservation; the
Gaussian beam truncated by the grid loses less than 1e-4 of its power).
"""

import json
import socket
import subprocess
import sys
import time

import pytest
from astropy import units as u

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from lumos.cli import _parser
from lumos.service import Service, ServiceConfig
from lumos.web import WebConfig, create_app

LASER = "lumos.sources.GaussianLaser"
METER = "lumos.detectors.PowerMeter"


@pytest.fixture
def app(tmp_path):
    config = WebConfig(service=ServiceConfig(archive_root=str(tmp_path / "archives")))
    return create_app(config)


@pytest.fixture
def client(app):
    with TestClient(app, base_url="http://localhost") as client:
        yield client


def bench(client, power="2 mW"):
    response = client.post("/api/contexts", json={"name": "bench"})
    assert response.status_code == 201, response.text
    cid = response.json()["id"]
    response = client.post(
        f"/api/contexts/{cid}/elements", json={"type_id": LASER, "name": "laser", "params": {"power": power}}
    )
    assert response.status_code == 201, response.text
    assert client.post(f"/api/contexts/{cid}/elements", json={"type_id": METER, "name": "meter"}).is_success
    response = client.post(
        f"/api/contexts/{cid}/connections", json={"source": "laser.out", "target": "meter.in"}
    )
    assert response.status_code == 201, response.text
    return cid


def wait_done(client, job, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = client.get(f"/api/jobs/{job}").json()
        if info["status"] in ("done", "failed"):
            return info
        time.sleep(0.05)
    raise TimeoutError(job)


def reading(summary):
    return (summary["values"][0] * u.Unit(summary["unit"])).to(u.mW)


# --- Information and sessions ------------------------------------------------------------------------------


def test_info_describes_the_instance(client):
    info = client.get("/api/info").json()
    assert info["profile"] == "local"
    assert info["operation"] is False
    assert set(info["languages"]) == {"en", "fr"}
    assert info["version"]


def test_each_browser_gets_its_own_session_behind_an_http_only_cookie(app):
    with (
        TestClient(app, base_url="http://localhost") as alice,
        TestClient(app, base_url="http://localhost") as bob,
    ):
        response = alice.post("/api/contexts", json={"name": "alice"})
        cookie = response.headers["set-cookie"].lower()
        assert "lumos_session=" in cookie
        assert "httponly" in cookie
        assert "samesite=strict" in cookie
        cid = response.json()["id"]
        assert alice.get(f"/api/contexts/{cid}").status_code == 200
        assert bob.get(f"/api/contexts/{cid}").status_code == 404
        assert [c["id"] for c in bob.get("/api/contexts").json()] == []


def test_the_sessions_of_the_web_are_journaled_with_the_origin_web(client):
    cid = bench(client)
    job = client.post(f"/api/contexts/{cid}/runs", json={"outputs": ["meter.out"]}).json()["id"]
    wait_done(client, job)
    entries = client.get(f"/api/contexts/{cid}/journal", params={"kind": "run"}).json()
    assert entries
    assert all(entry["origin"] == "web" for entry in entries)


# --- Discovery ---------------------------------------------------------------------------------------------


def test_element_types_and_schemas(client):
    page = client.get("/api/element-types", params={"query": "laser"}).json()
    assert any(item["type_id"] == LASER for item in page["items"])
    schema = client.get(f"/api/element-types/{LASER}/schema").json()
    assert schema["properties"]["params"]["properties"]["power"]["x-lumos-unit"] == "mW"


def test_the_palette_gives_every_type_with_its_schema_category_and_the_signal_hierarchy(client):
    palette = client.get("/api/palette").json()
    types = {item["type_id"]: item for item in palette["types"]}
    total = client.get("/api/element-types", params={"page_size": 1}).json()["total"]
    assert len(types) == total
    assert types[LASER]["category"] == "sources"
    assert types[LASER]["schema"]["x-lumos-ports"]["outputs"][0]["name"] == "out"
    signals = palette["signals"]
    # a Wavefront is an OpticalSignal, which is a Signal (lumos.core.signal, lumos.fields)
    assert "OpticalSignal" in signals["Wavefront"]["parents"]
    assert "Signal" in signals["OpticalSignal"]["parents"]


def test_documentation_search(client):
    hits = client.get("/api/docs/search", params={"q": "coronagraph"}).json()
    assert hits
    assert {"title", "path", "excerpt"} <= set(hits[0])


# --- Contexts, editing, errors -----------------------------------------------------------------------------


def test_build_and_run_a_bench_over_http(client):
    cid = bench(client, power="2 mW")
    assert client.patch(
        f"/api/contexts/{cid}/params", json={"path": "laser.power", "value": "3 mW"}
    ).is_success
    assert client.get(f"/api/contexts/{cid}/validate").json() == []
    response = client.post(f"/api/contexts/{cid}/runs", json={"outputs": ["meter.out"]})
    assert response.status_code == 202
    info = wait_done(client, response.json()["id"])
    assert info["status"] == "done", info
    assert info["progress"] == 1.0
    summary = client.get(f"/api/jobs/{info['id']}/results/meter.out").json()
    assert reading(summary).value == pytest.approx(3.0, rel=1e-4)


def test_describe_remove_and_disconnect(client):
    cid = bench(client)
    summary = client.get(f"/api/contexts/{cid}").json()
    assert {e["id"] for e in summary["elements"]} == {"laser", "meter"}
    assert summary["connections"] == [{"source": "laser.out", "target": "meter.in"}]
    response = client.delete(
        f"/api/contexts/{cid}/connections", params={"source": "laser.out", "target": "meter.in"}
    )
    assert response.status_code == 204
    issues = client.get(f"/api/contexts/{cid}/validate").json()
    assert any("meter.in" in issue["message"] for issue in issues)
    assert client.delete(f"/api/contexts/{cid}/elements/meter").status_code == 204
    assert [e["id"] for e in client.get(f"/api/contexts/{cid}").json()["elements"]] == ["laser"]
    assert client.delete(f"/api/contexts/{cid}").status_code == 204
    assert client.get(f"/api/contexts/{cid}").status_code == 404


def test_observation_band(client):
    cid = client.post("/api/contexts", json={}).json()["id"]
    response = client.put(
        f"/api/contexts/{cid}/observation",
        json={"minimum": "1.5 um", "maximum": "1.7 um", "n_wavelengths": 3},
    )
    assert response.status_code == 200
    assert response.json()["n_wavelengths"] == 3
    band = [u.Quantity(value).to_value(u.um) for value in response.json()["band"]]
    assert band == pytest.approx([1.5, 1.7])


@pytest.mark.parametrize(
    ("request_", "status", "code"),
    [
        (("post", "/api/contexts/{cid}/elements", {"type_id": "lumos.sources.Laser"}), 404, "not_found"),
        (
            ("post", "/api/contexts/{cid}/connections", {"source": "meter.out", "target": "laser.out"}),
            400,
            "invalid",
        ),
        (("patch", "/api/contexts/{cid}/params", {"path": "laser.power", "value": "3 mm"}), 400, "invalid"),
        (("get", "/api/contexts/unknown", None), 404, "not_found"),
    ],
)
def test_errors_of_the_service_keep_their_message_hint_and_code(client, request_, status, code):
    cid = bench(client)
    method, path, body = request_
    response = client.request(method, path.format(cid=cid), json=body)
    assert response.status_code == status, response.text
    error = response.json()
    assert error["code"] == code
    assert error["error"]
    assert error["hint"]


def test_quotas_of_the_service_answer_429(tmp_path):
    config = WebConfig(service=ServiceConfig(max_contexts_per_session=1))
    with TestClient(create_app(config), base_url="http://localhost") as client:
        assert client.post("/api/contexts", json={}).status_code == 201
        response = client.post("/api/contexts", json={})
        assert response.status_code == 429
        assert response.json()["code"] == "quota"


def test_malformed_requests_are_explained(client):
    response = client.post("/api/contexts/x/elements", json={"name": "no type"})
    assert response.status_code == 400
    assert response.json()["code"] == "invalid"
    assert "type_id" in response.json()["error"]


# --- Presets, import and export ----------------------------------------------------------------------------


def test_presets_are_listed_and_loaded(client):
    names = [p["name"] for p in client.get("/api/presets").json()]
    assert "coronagraph" in names
    response = client.post("/api/presets/coronagraph/load")
    assert response.status_code == 201
    ids = {e["id"] for e in response.json()["elements"]}
    assert {"star", "telescope", "coronagraph"} <= ids


def test_export_and_import_lumos_and_export_python(client):
    cid = bench(client)
    response = client.get(f"/api/contexts/{cid}/export.lumos")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    assert response.headers["content-disposition"].endswith('.lumos"')
    document = response.text
    assert json.loads(document)["kind"] == "context"
    imported = client.post(
        "/api/contexts/import", content=document, headers={"content-type": "application/json"}
    )
    assert imported.status_code == 201
    assert {e["id"] for e in imported.json()["elements"]} == {"laser", "meter"}
    script = client.get(f"/api/contexts/{cid}/export.py")
    assert script.status_code == 200
    assert "Context(" in script.text
    assert script.headers["content-disposition"].endswith('.py"')


def test_invalid_lumos_documents_are_refused(client):
    response = client.post(
        "/api/contexts/import", content="{not json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 400
    assert response.json()["hint"]


def test_parity_the_web_exports_the_same_document_as_the_service(client, tmp_path):
    cid = bench(client, power="2 mW")
    web = json.loads(client.get(f"/api/contexts/{cid}/export.lumos").text)
    service = Service(ServiceConfig(archive_root=str(tmp_path)))
    session = service.create_session(origin="python").id
    ctx = service.create_context(session, name="bench").id
    service.add_element(session, ctx, LASER, name="laser", params={"power": "2 mW"})
    service.add_element(session, ctx, METER, name="meter")
    service.connect(session, ctx, "laser.out", "meter.in")
    direct = json.loads(service.export_lumos(session, ctx))
    service.shutdown()
    for document in (web, direct):
        document.pop("created_at")
    assert web == direct


# --- Inspection --------------------------------------------------------------------------------------------


def test_inspect_any_port_with_a_summary_and_a_png_figure(client):
    cid = bench(client, power="2 mW")
    summary = client.get(f"/api/contexts/{cid}/ports/laser.out").json()
    assert summary["kind"] == "Wavefront"
    assert u.Quantity(summary["details"]["power"]).to_value(u.mW) == pytest.approx(2.0, rel=1e-4)
    assert "figure_png" not in summary
    figure = client.get(f"/api/contexts/{cid}/ports/laser.out/figure.png")
    assert figure.status_code == 200
    assert figure.headers["content-type"] == "image/png"
    assert figure.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_archives_are_listed(client):
    assert client.get("/api/archives").json() == []


# --- WebSocket ---------------------------------------------------------------------------------------------


def test_the_progress_of_runs_is_pushed_over_the_websocket(client):
    cid = bench(client)
    with client.websocket_connect("/api/ws", headers={"origin": "http://localhost"}) as socket_:
        job = client.post(f"/api/contexts/{cid}/runs", json={"outputs": ["meter.out"]}).json()["id"]
        statuses = []
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            message = socket_.receive_json()
            if message["type"] == "job" and message["job"]["id"] == job:
                statuses.append(message["job"]["status"])
                if message["job"]["status"] in ("done", "failed"):
                    assert message["job"]["progress"] == 1.0
                    break
        assert statuses[-1] == "done"


def test_the_logs_of_runs_are_pushed_over_the_websocket(client):
    cid = bench(client)
    with client.websocket_connect("/api/ws", headers={"origin": "http://localhost"}) as socket_:
        client.post(f"/api/contexts/{cid}/runs", json={"outputs": ["meter.out"]})
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            message = socket_.receive_json()
            if message["type"] == "journal":
                assert message["context"] == cid
                assert message["entry"]["kind"] == "run"
                break
        else:
            pytest.fail("no journal entry received")


# --- Local binding -----------------------------------------------------------------------------------------


def test_the_command_line_listens_on_localhost_by_default():
    arguments = _parser().parse_args(["web"])
    assert arguments.host == "127.0.0.1"
    assert arguments.profile == "local"


def test_requests_for_another_host_name_are_refused_in_the_local_profile(app):
    with TestClient(app, base_url="http://attacker.example") as client:
        assert client.get("/api/info").status_code == 400


def test_websockets_from_another_origin_are_refused(client):
    from starlette.websockets import WebSocketDisconnect

    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect("/api/ws", headers={"origin": "http://attacker.example"}) as socket_,
    ):
        socket_.receive_json()


def test_the_root_serves_the_interface(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_lumos_web_serves_the_api_on_localhost(tmp_path):
    import httpx

    port = free_port()
    process = subprocess.Popen(
        [sys.executable, "-m", "lumos.cli", "web", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 60
        while True:
            if process.poll() is not None:
                pytest.fail(f"the server stopped: {process.stderr.read() if process.stderr else ''}")
            try:
                response = httpx.get(f"http://127.0.0.1:{port}/api/info", timeout=2)
                break
            except httpx.TransportError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.3)
        assert response.json()["profile"] == "local"
    finally:
        process.terminate()
        process.wait(timeout=30)
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()
