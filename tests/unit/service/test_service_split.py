"""Splitting a bus port in the editor (step P15.10, ADR-0023; docs/specs/pistes_evolution.md §1, step 4).

Requirements: "Éclater le port" inserts a ``Mux`` before a bus input, or a ``Demux`` after a bus output, of as
many lanes as the port draws, **attached** to the block (``ui.attached``: it moves and is deleted with it).
The document holds explicit ``Demux`` / ``Mux`` elements, so Python and the MCP server see exactly what is in
the file. Splitting an input already wired to a bus keeps the system unchanged: the wire goes through a
``Demux`` and the lanes are wired straight to the ``Mux``, ready for components on single lanes. A port of
free width (a waveguide acts on any number of lanes) cannot be split.
"""

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from lumos.service import Service, ServiceError
from lumos.web import WebConfig, create_app


@pytest.fixture
def setup():
    service = Service()
    session = service.create_session().id
    ctx = service.create_context(session).id
    return service, session, ctx


def elements(service, session, ctx):
    return {e.id: e for e in service.describe_context(session, ctx).elements}


def connections(service, session, ctx):
    return {(c.source, c.target) for c in service.describe_context(session, ctx).connections}


def test_splitting_an_input_attaches_a_mux(setup):
    service, session, ctx = setup
    service.add_element(session, ctx, "lumos.photonics.MMI", name="mmi", params={"n_ports": 4})
    summary = service.split_port(session, ctx, "mmi.in")
    found = elements(service, session, ctx)
    assert found["mmi_in"].type_id == "lumos.buses.Mux"
    assert found["mmi_in"].settings["n_lanes"] == 4
    assert ("mmi_in.out", "mmi.in") in connections(service, session, ctx)
    assert summary.ui["attached"]["mmi_in"] == {"to": "mmi", "port": "in", "side": "in"}


def test_splitting_an_output_attaches_a_demux(setup):
    service, session, ctx = setup
    service.add_element(session, ctx, "lumos.photonics.MMI", name="mmi", params={"n_ports": 3})
    summary = service.split_port(session, ctx, "mmi.out")
    found = elements(service, session, ctx)
    assert found["mmi_out"].type_id == "lumos.buses.Demux"
    assert found["mmi_out"].outputs == ["out0", "out1", "out2"]
    assert ("mmi.out", "mmi_out.in") in connections(service, session, ctx)
    assert summary.ui["attached"]["mmi_out"]["side"] == "out"


def test_splitting_a_wired_input_keeps_the_results(setup):
    service, session, ctx = setup
    service.set_observation(session, ctx, minimum="3.5 um", maximum="4 um", n_wavelengths=2)
    service.add_element(session, ctx, "lumos.sources.PointSource", name="star")
    service.add_element(
        session,
        ctx,
        "lumos.telescopes.IdealInjectionArray",
        name="array",
        params={"frame": "sky", "positions": "[[0, 0], [20, 0], [0, 30], [25, 25]] m"},
    )
    service.add_element(session, ctx, "lumos.photonics.MMI", name="mmi", params={"n_ports": 4})
    service.add_element(session, ctx, "lumos.detectors.PowerMeter", name="meter")
    service.connect(session, ctx, "star.out", "array.in")
    service.connect(session, ctx, "array.out", "mmi.in")
    service.connect(session, ctx, "mmi.out", "meter.in")

    def reading():
        job = service.run(session, ctx, outputs=["meter.out"])
        service.wait_job(session, job.id, timeout=120)
        return np.array(service.get_result(session, job.id, "meter.out").values)

    before = reading()
    service.split_port(session, ctx, "mmi.in")
    wires = connections(service, session, ctx)
    assert ("array.out", "mmi_in_lanes.in") in wires
    assert {(f"mmi_in_lanes.out{k}", f"mmi_in.in{k}") for k in range(4)} <= wires
    assert ("array.out", "mmi.in") not in wires
    np.testing.assert_allclose(reading(), before, rtol=1e-12)


def test_attached_elements_follow_their_block(setup):
    service, session, ctx = setup
    service.add_element(session, ctx, "lumos.photonics.MMI", name="mmi", params={"n_ports": 2})
    service.split_port(session, ctx, "mmi.in")
    service.split_port(session, ctx, "mmi.out")
    summary = service.rename_element(session, ctx, "mmi", "coupler")
    assert {k: v["to"] for k, v in summary.ui["attached"].items()} == {
        "mmi_in": "coupler",
        "mmi_out": "coupler",
    }
    service.remove_element(session, ctx, "mmi_out")  # removing the split alone keeps the block
    assert "mmi_out" not in service.describe_context(session, ctx).ui["attached"]
    service.remove_element(session, ctx, "coupler")
    assert set(elements(service, session, ctx)) == set()
    assert service.describe_context(session, ctx).ui["attached"] == {}


def test_a_port_of_free_width_cannot_be_split(setup):
    service, session, ctx = setup
    service.add_element(session, ctx, "lumos.photonics.Waveguide", name="guide")
    with pytest.raises(ServiceError, match="lanes"):
        service.split_port(session, ctx, "guide.in")
    service.add_element(session, ctx, "lumos.sources.GaussianLaser", name="laser")
    with pytest.raises(ServiceError, match="bus"):
        service.split_port(session, ctx, "laser.out")


def test_the_web_api_splits_a_port(tmp_path):
    with TestClient(create_app(WebConfig(archive_root=str(tmp_path))), base_url="http://localhost") as client:
        client.get("/api/info")
        cid = client.post("/api/contexts", json={"name": "split"}).json()["id"]
        client.post(f"/api/contexts/{cid}/elements", json={"type_id": "lumos.photonics.MMI", "name": "mmi"})
        response = client.post(f"/api/contexts/{cid}/split", json={"port": "mmi.in"})
        assert response.status_code == 200, response.text
        assert response.json()["ui"]["attached"]["mmi_in"]["to"] == "mmi"
