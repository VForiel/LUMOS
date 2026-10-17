"""Routes of the editing operations of the web interface (step P14.5, REQ-WEB-1, REQ-WEB-4, REQ-WEB-6).

Requirements: every editing operation of the service (layout, undo and redo, renaming, groups, copy and paste,
blocks, name of the context) is reachable over HTTP, with the summary of the context telling whether undo and
redo are possible; a viewer cannot edit.
"""

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from lumos.web import WebConfig, create_app

LENS = "lumos.optics.ThinLens"
GAP = "lumos.optics.FreeSpace"


@pytest.fixture
def client():
    with TestClient(create_app(), base_url="http://localhost") as client:
        yield client


def chain(client):
    cid = client.post("/api/contexts", json={"name": "chain"}).json()["id"]
    for type_id, name in [(LENS, "lens"), (GAP, "gap")]:
        client.post(f"/api/contexts/{cid}/elements", json={"type_id": type_id, "name": name})
    client.post(f"/api/contexts/{cid}/connections", json={"source": "lens.out", "target": "gap.in"})
    return cid


def ids(summary):
    return sorted(e["id"] for e in summary["elements"] if "." not in e["id"])


def test_layout_undo_and_redo(client):
    cid = chain(client)
    summary = client.put(f"/api/contexts/{cid}/ui", json={"positions": {"lens": [5, 6]}}).json()
    assert summary["ui"]["positions"]["lens"] == [5, 6]
    assert summary["can_undo"] is True
    assert client.post(f"/api/contexts/{cid}/undo").json()["can_redo"] is True
    assert client.post(f"/api/contexts/{cid}/redo").json()["ui"]["positions"]["lens"] == [5, 6]


def test_rename_group_ungroup(client):
    cid = chain(client)
    summary = client.post(f"/api/contexts/{cid}/elements/lens/rename", json={"name": "focus"}).json()
    assert ids(summary) == ["focus", "gap"]
    summary = client.post(
        f"/api/contexts/{cid}/groups", json={"elements": ["focus", "gap"], "name": "relay"}
    ).json()
    assert ids(summary) == ["relay"]
    summary = client.post(f"/api/contexts/{cid}/groups/relay/ungroup").json()
    assert ids(summary) == ["focus", "gap"]


def test_copy_paste_and_blocks(client):
    cid = chain(client)
    fragment = client.post(f"/api/contexts/{cid}/copy", json={"elements": ["lens", "gap"]}).json()
    assert client.post(f"/api/contexts/{cid}/paste", json={"fragment": fragment}).json() == {
        "ids": ["lens2", "gap2"]
    }
    client.post(f"/api/contexts/{cid}/groups", json={"elements": ["lens", "gap"], "name": "relay"})
    block = client.get(f"/api/contexts/{cid}/elements/relay/export.lumos")
    assert block.headers["content-disposition"].endswith('relay.lumos"')
    assert json.loads(block.text)["kind"] == "group"
    inserted = client.post(
        f"/api/contexts/{cid}/blocks", content=block.text, headers={"content-type": "application/json"}
    )
    assert inserted.status_code == 201
    assert inserted.json()["id"] == "relay2"


def test_name_and_description(client):
    cid = chain(client)
    summary = client.patch(f"/api/contexts/{cid}", json={"name": "Relay", "description": "4f"}).json()
    assert (summary["name"], summary["description"]) == ("Relay", "4f")


def test_a_viewer_cannot_edit(tmp_path):
    config = WebConfig(profile="public", anonymous_role="viewer")
    with TestClient(create_app(config), base_url="http://testserver") as client:
        cid = client.post("/api/presets/first_simulation/load").json()["id"]
        element = client.get(f"/api/contexts/{cid}").json()["elements"][0]["id"]
        refused = [
            client.put(f"/api/contexts/{cid}/ui", json={"positions": {}}),
            client.post(f"/api/contexts/{cid}/undo"),
            client.post(f"/api/contexts/{cid}/elements/{element}/rename", json={"name": "x"}),
            client.post(f"/api/contexts/{cid}/groups", json={"elements": [element], "name": "g"}),
            client.post(f"/api/contexts/{cid}/paste", json={"fragment": {"elements": [], "connections": []}}),
            client.patch(f"/api/contexts/{cid}", json={"name": "x"}),
        ]
        assert [r.status_code for r in refused] == [403] * len(refused)
        assert client.post(f"/api/contexts/{cid}/copy", json={"elements": [element]}).status_code == 200


def test_the_palette_gives_the_bounds_in_the_unit_of_each_parameter(client):
    types = {item["type_id"]: item for item in client.get("/api/palette").json()["types"]}
    power = types["lumos.sources.GaussianLaser"]["schema"]["properties"]["params"]["properties"]["power"]
    assert power["x-lumos-unit"] == "mW"
    assert power["x-lumos-bounds"][0]["unit"] == "W"  # as declared by the element
    assert power["x-lumos-bounds-value"] == [0.0, None]  # in mW, for the forms
    for item in types.values():
        for field in item["schema"]["properties"]["params"]["properties"].values():
            assert len(field["x-lumos-bounds-value"]) == 2
