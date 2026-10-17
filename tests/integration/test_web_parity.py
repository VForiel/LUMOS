"""Parity of the Python API, the web interface and the MCP server (step P14.11, objective O4, criterion v0.7).

Requirements: the same bench built with the Python library, through the HTTP API of the web interface and by
an assistant through the MCP server of the web instance gives the same ``.lumos`` document and result.

Expected value: a power meter behind a 2 mW laser, a lens and a free-space gap reads 2 mW (lossless optics;
the Gaussian beam truncated by the grid loses less than 1e-3 of its power).
"""

import json

import pytest
from astropy import units as u

pytest.importorskip("fastapi")
pytest.importorskip("mcp")

from fastapi.testclient import TestClient

import lumos.io
from lumos import Context
from lumos.elements.detectors import PowerMeter
from lumos.elements.optics import FreeSpace, ThinLens
from lumos.elements.sources import GaussianLaser
from lumos.web import create_app

ELEMENTS = [
    ("lumos.sources.GaussianLaser", "laser", {"power": "2 mW"}),
    ("lumos.optics.ThinLens", "lens", {"focal_length": "200 mm"}),
    ("lumos.optics.FreeSpace", "gap", {"distance": "100 mm"}),
    ("lumos.detectors.PowerMeter", "meter", {}),
]
LINKS = [("laser.out", "lens.in"), ("lens.out", "gap.in"), ("gap.out", "meter.in")]


class Mcp:
    """Raw JSON-RPC over the streamable HTTP transport of MCP (specification 2025-06-18)."""

    def __init__(self, client):
        self.client, self.session, self.next_id = client, None, 0

    def post(self, message):
        headers = {"accept": "application/json, text/event-stream", "content-type": "application/json"}
        if self.session:
            headers.update({"mcp-session-id": self.session, "mcp-protocol-version": "2025-06-18"})
        return self.client.post("/mcp", content=json.dumps(message), headers=headers)

    def request(self, method, params):
        self.next_id += 1
        response = self.post({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params})
        self.session = response.headers.get("mcp-session-id", self.session)
        for line in response.text.splitlines():
            if line.startswith("data:") and json.loads(line[5:]).get("id") == self.next_id:
                return json.loads(line[5:])
        return response.json()

    def start(self):
        info = {"name": "parity", "version": "1"}
        self.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": info})
        self.post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def call(self, name, arguments):
        return self.request("tools/call", {"name": name, "arguments": arguments})["result"]


def comparable(document):
    document = json.loads(json.dumps(document))
    document.pop("created_at", None)
    document["meta"].pop("name", None)
    document["context"].pop("seed")
    document["context"]["clock"].pop("start")
    document["ui"] = {}
    return document


def python_bench():
    ctx = Context(name="parity")
    laser = ctx.add(GaussianLaser(name="laser", power=2 * u.mW))
    lens = ctx.add(ThinLens(name="lens", focal_length=200 * u.mm))
    gap = ctx.add(FreeSpace(name="gap", distance=100 * u.mm))
    meter = ctx.add(PowerMeter(name="meter"))
    ctx.connect(laser.port("out"), lens.port("in"))
    ctx.connect(lens.port("out"), gap.port("in"))
    ctx.connect(gap.port("out"), meter.port("in"))
    reading = ctx.run([meter.port("out")])[meter.port("out")]
    return comparable(lumos.io.to_document(ctx)), u.Quantity(reading.data).to_value(u.mW)


def web_bench(client):
    cid = client.post("/api/contexts", json={"name": "parity"}).json()["id"]
    for type_id, name, params in ELEMENTS:
        assert client.post(
            f"/api/contexts/{cid}/elements", json={"type_id": type_id, "name": name, "params": params}
        ).is_success
    for source, target in LINKS:
        assert client.post(
            f"/api/contexts/{cid}/connections", json={"source": source, "target": target}
        ).is_success
    reading = client.get(f"/api/contexts/{cid}/ports/meter.out").json()
    document = json.loads(client.get(f"/api/contexts/{cid}/export.lumos").text)
    return comparable(document), (reading["values"][0] * u.Unit(reading["unit"])).to_value(u.mW)


def mcp_bench(client):
    mcp = Mcp(client)
    mcp.start()
    created = mcp.call("create_context", {"name": "parity"})
    cid = created["structuredContent"]["id"]
    for type_id, name, params in ELEMENTS:
        assert not mcp.call(
            "add_element", {"type_id": type_id, "name": name, "params": params, "context": cid}
        ).get("isError")
    for source, target in LINKS:
        assert not mcp.call("connect", {"source": source, "target": target, "context": cid}).get("isError")
    reading = mcp.call("inspect_port", {"port": "meter.out", "context": cid, "figure": False})[
        "structuredContent"
    ]
    document = json.loads(client.get(f"/api/contexts/{cid}/export.lumos").text)
    return comparable(document), (reading["values"][0] * u.Unit(reading["unit"])).to_value(u.mW)


def test_python_web_and_mcp_build_the_same_bench_and_read_the_same_power():
    python_document, python_reading = python_bench()
    with TestClient(create_app(), base_url="http://localhost") as client:
        client.get("/api/info")
        web_document, web_reading = web_bench(client)
        mcp_document, mcp_reading = mcp_bench(client)
    assert web_document == python_document
    assert mcp_document == python_document
    assert python_reading == pytest.approx(2.0, rel=1e-3)
    assert web_reading == pytest.approx(python_reading, rel=1e-12)
    assert mcp_reading == pytest.approx(python_reading, rel=1e-12)
