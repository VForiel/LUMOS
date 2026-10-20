"""The community catalogue (step P15.16, ADR-0026; docs/specs/pistes_evolution.md §2 "Catalogue").

Requirements: a ``community/`` folder of the repository holds the contributions (declarative components,
shared presets); an index lists one entry per contribution (identifier, title, author, version, license,
minimum ``.lumos`` format); every contribution passes ``lumos component check`` (its checks carry independent
reference values) and declares its license; the index is rebuilt from the files and must match them; the
catalogue is shipped with the package, listed by the service, and a declarative component is imported, a
preset opened, in one call; nothing outside the index can be read through it.
"""

import json
import tomllib
from pathlib import Path

import pytest

import lumos.io
from lumos.catalogue import build_index, check_catalogue
from lumos.components import check_component
from lumos.io.lumos_format import FORMAT_VERSION
from lumos.service import Service, ServiceConfig, ServiceError

ROOT = Path(__file__).resolve().parents[2]
COMMUNITY = ROOT / "community"


def components():
    return sorted((COMMUNITY / "components").rglob("*.lumos"))


def presets():
    return sorted((COMMUNITY / "presets").rglob("*.lumos"))


def version(text):
    return tuple(int(part) for part in str(text).split("."))


def test_the_catalogue_has_components_and_presets():
    assert len(components()) >= 3
    assert len(presets()) >= 1
    assert (COMMUNITY / "README.md").is_file()


@pytest.mark.parametrize("path", components(), ids=lambda p: p.stem)
def test_every_component_passes_its_checks(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    assert check_component(document) == []
    component = document["component"]
    author, name = component["id"].split(".")
    assert path.parent.name == author  # components/<author>/<Name>.lumos
    assert path.stem == name
    assert version(document["lumos_format"]) <= version(FORMAT_VERSION)


@pytest.mark.parametrize("path", presets(), ids=lambda p: p.stem)
def test_every_preset_opens_and_declares_its_license(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    meta = document["meta"]
    assert meta.get("license")
    assert meta.get("authors")
    assert meta.get("description")
    ctx = lumos.io.load(path)
    assert len(ctx.elements) >= 2


def test_the_index_matches_the_files():
    assert check_catalogue(COMMUNITY) == []
    committed = json.loads((COMMUNITY / "index.json").read_text(encoding="utf-8"))
    assert committed == build_index(COMMUNITY)
    entries = committed["entries"]
    assert {entry["kind"] for entry in entries} == {"component", "preset"}
    for entry in entries:
        for key in ("kind", "id", "title", "description", "authors", "license", "lumos_format", "path"):
            assert entry[key], (entry["id"], key)


def test_a_contribution_without_license_is_refused(tmp_path):
    source = components()[0]
    document = json.loads(source.read_text(encoding="utf-8"))
    document["component"].pop("license")
    target = tmp_path / "components" / source.parent.name / source.name
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(document), encoding="utf-8")
    problems = check_catalogue(tmp_path)
    assert any("license" in problem for problem in problems)


def test_the_catalogue_is_shipped_in_the_wheel():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    included = config["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert included.get("community") == "lumos/community"


@pytest.fixture
def service(tmp_path):
    service = Service(ServiceConfig(archive_root=str(tmp_path)))
    yield service
    service.shutdown()


def test_the_service_lists_imports_and_opens(service):
    entries = service.list_catalogue()
    kinds = {(entry.kind, entry.id) for entry in entries}
    assert ("component", "lumos.NeutralDensity") in kinds
    preset = next(entry for entry in entries if entry.kind == "preset")
    session = service.create_session().id
    info = service.use_catalogue_component(session, "lumos.NeutralDensity")
    assert info.type_id == "components.lumos.NeutralDensity"
    assert "lumos.NeutralDensity" in [c.id for c in service.list_components(session)]
    summary = service.open_catalogue_preset(session, preset.id)
    assert len(summary.elements) >= 2
    assert service.list_catalogue(query="coupler")[0].id == "lumos.Coupler3dB"


@pytest.mark.parametrize("identifier", ["lumos.Unknown", "../../pyproject", "lumos/../../../README"])
def test_nothing_outside_the_index_is_read(service, identifier):
    session = service.create_session().id
    with pytest.raises(ServiceError):
        service.use_catalogue_component(session, identifier)
    with pytest.raises(ServiceError):
        service.open_catalogue_preset(session, identifier)


def test_the_web_api(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from lumos.web import WebConfig, create_app

    config = WebConfig(service=ServiceConfig(archive_root=str(tmp_path)))
    with TestClient(create_app(config), base_url="http://localhost") as client:
        client.get("/api/info")
        entries = client.get("/api/catalogue").json()
        assert any(entry["id"] == "lumos.NeutralDensity" for entry in entries)
        used = client.post("/api/catalogue/component/lumos.NeutralDensity")
        assert used.status_code == 200, used.text
        assert any(
            t["type_id"] == "components.lumos.NeutralDensity"
            for t in client.get("/api/palette").json()["types"]
        )
        preset = next(entry for entry in entries if entry["kind"] == "preset")
        opened = client.post(f"/api/catalogue/preset/{preset['id']}")
        assert opened.status_code == 201, opened.text
        assert client.post("/api/catalogue/preset/nothing").status_code == 404
