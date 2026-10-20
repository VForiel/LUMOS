"""The community catalogue: shared components and presets of the ``community/`` folder (ADR-0026).

Layout of a catalogue directory::

    components/<author>/<Name>.lumos   documents of kind "component", identifier <author>.<Name>
    presets/<author>/<name>.lumos      contexts whose meta gives name, description, authors and license
    index.json                         one entry per contribution, made by :func:`build_index`

:func:`check_catalogue` lists what refuses a contribution; the CI of the repository runs it on every change of
``community/``. The catalogue is shipped with the package (``lumos/community``) and read by the service.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lumos

__all__ = ["INDEX", "build_index", "catalogue_dir", "check_catalogue", "read_index"]

#: File name of the index of a catalogue.
INDEX = "index.json"


def catalogue_dir() -> Path:
    """Directory of the catalogue shipped with LUMOS (the ``community/`` folder of the repository)."""
    packaged = Path(lumos.__file__).resolve().parent / "community"
    if packaged.is_dir():
        return packaged
    return Path(lumos.__file__).resolve().parents[2] / "community"


def _version(text: Any) -> tuple[int, ...]:
    return tuple(int(part) for part in str(text).split("."))


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _component_entry(root: Path, path: Path) -> dict[str, Any]:
    document = _read(path)
    component = document.get("component", {})
    return {
        "kind": "component",
        "id": str(component.get("id", "")),
        "title": str(component.get("title", "")),
        "description": str(component.get("description", "")),
        "authors": list(component.get("authors", [])),
        "version": component.get("version"),
        "license": str(component.get("license", "")),
        "template": str(component.get("template", "")),
        "lumos_format": str(document.get("lumos_format", "")),
        "path": path.relative_to(root).as_posix(),
    }


def _preset_entry(root: Path, path: Path) -> dict[str, Any]:
    document = _read(path)
    meta = document.get("meta", {})
    return {
        "kind": "preset",
        "id": f"{path.parent.name}/{path.stem}",
        "title": str(meta.get("name", "")),
        "description": str(meta.get("description", "")),
        "authors": list(meta.get("authors", [])),
        "version": meta.get("version"),
        "license": str(meta.get("license", "")),
        "template": "",
        "lumos_format": str(document.get("lumos_format", "")),
        "path": path.relative_to(root).as_posix(),
    }


def build_index(root: str | Path) -> dict[str, Any]:
    """Return the index of the catalogue in ``root``, made from its files (components first)."""
    root = Path(root)
    entries = [_component_entry(root, p) for p in sorted((root / "components").glob("*/*.lumos"))]
    entries += [_preset_entry(root, p) for p in sorted((root / "presets").glob("*/*.lumos"))]
    return {"catalogue": "lumos-community", "entries": entries}


def read_index(root: str | Path) -> list[dict[str, Any]]:
    """Return the entries of the committed index of the catalogue in ``root`` (empty when there is none)."""
    path = Path(root) / INDEX
    if not path.is_file():
        return []
    return list(_read(path).get("entries", []))


def check_catalogue(root: str | Path) -> list[str]:
    """Return what refuses contributions of the catalogue in ``root`` (empty when all are accepted).

    A component must pass :func:`lumos.components.check_component` and lie at the place of its identifier; a
    preset must declare its name, description, authors and license, and load; no file may need a format newer
    than this version of LUMOS; the committed index must equal the one built from the files.
    """
    from lumos.components import check_component
    from lumos.io.lumos_format import FORMAT_VERSION, from_document

    root = Path(root)
    problems: list[str] = []
    index = build_index(root)
    for entry in index["entries"]:
        where = entry["path"]
        try:
            document = _read(root / where)
        except (OSError, ValueError) as error:
            problems.append(f"{where}: cannot be read ({error})")
            continue
        try:
            if _version(entry["lumos_format"]) > _version(FORMAT_VERSION):
                problems.append(f"{where}: format {entry['lumos_format']} is newer than {FORMAT_VERSION}")
        except ValueError:
            problems.append(f"{where}: give the lumos_format of the file")
        if entry["kind"] == "component":
            problems += [f"{where}: {problem}" for problem in check_component(document)]
            expected = f"{Path(where).parent.name}.{Path(where).stem}"
            if entry["id"] != expected:
                problems.append(f"{where}: the identifier {entry['id']!r} must be {expected!r} (its place)")
        else:
            for key in ("name", "description", "authors", "license"):
                if not document.get("meta", {}).get(key):
                    problems.append(f"{where}: give the {key} of the preset in its meta")
            try:
                from_document(document)
            except Exception as error:  # any failure to load refuses the contribution
                problems.append(f"{where}: does not load ({error})")
    committed = root / INDEX
    if not committed.is_file() or _read(committed) != index:
        problems.append(f"{INDEX} is not up to date: run python tools/build_catalogue.py")
    return problems
