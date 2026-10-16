"""The ``.lumos`` document format (ADR-0003).

A ``.lumos`` file is a UTF-8 strict-JSON document. This module handles documents of kind ``"element"``,
which describe a single element, and ``"group"``, which describe an element group::

    {
      "lumos_format": "1.0",
      "kind": "element",
      "lumos_version": "0.1.0",
      "created_at": "2026-10-16T09:00:00Z",
      "element": {
        "type": "lumos.optics.ThinLens", "type_version": 1, "name": "L1",
        "params": {"focal_length": {"value": 100.0, "unit": "mm"}},
        "settings": {}, "state": {},
        "free": {"focal_length": {"prior": {"kind": "normal", "mean": ..., "std": ...}}}
      }
    }

A group entry has the type ``"lumos.group"`` and lists its ``children`` (element or group entries, each with
an ``id``), its internal ``connections`` (``{"from": "id.port", "to": "id.port"}``) and its
``exposed_ports`` (``{"name": "id.port"}``).

Physical values keep the user's unit (see :func:`lumos.units.quantity_to_json`). The element class is found
in the :class:`~lumos.core.registry.Registry` from its ``type`` identifier; nothing is ever imported while
reading a document. Older type versions are upgraded with the registered migrations.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
from astropy import units as u

import lumos
from lumos.core import registry as _registry_module
from lumos.core.element import Element
from lumos.core.group import GROUP_TYPE, ElementGroup
from lumos.core.param import Param, Setting
from lumos.core.priors import prior_from_json, prior_to_json
from lumos.core.registry import Registry
from lumos.errors import FormatVersionError, LumosFormatError
from lumos.units import quantity_from_json, quantity_to_json

__all__ = ["FORMAT_VERSION", "entry_of", "from_document", "from_entry", "load", "save", "to_document"]

FORMAT_VERSION = "1.0"
_SUPPORTED = (1, 0)
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)$")
_BODY_KEYS = ("name", "params", "settings", "state")


def _registry(registry: Registry | None) -> Registry:
    return _registry_module.registry if registry is None else registry


def to_document(element: Element, *, registry: Registry | None = None) -> dict[str, Any]:
    """Return the ``.lumos`` document (a JSON-compatible dictionary) describing an element or a group."""
    reg = _registry(registry)
    kind = "group" if isinstance(element, ElementGroup) else "element"
    return {
        "lumos_format": FORMAT_VERSION,
        "kind": kind,
        "lumos_version": lumos.__version__,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        kind: entry_of(element, reg),
    }


def entry_of(element: Element, registry: Registry | None = None) -> dict[str, Any]:
    """Return the document entry of an element or a group (without the envelope)."""
    reg = _registry(registry)
    if isinstance(element, ElementGroup):
        graph = element.graph
        children = []
        for child_id, child in graph.elements.items():
            children.append({"id": child_id, **entry_of(child, reg)})
        connections = [
            {
                "from": f"{graph.id_of(c.source.owner)}.{c.source.name}",
                "to": f"{graph.id_of(c.target.owner)}.{c.target.name}",
            }
            for c in graph.connections
        ]
        exposed = {name: f"{graph.id_of(ref.owner)}.{ref.name}" for name, ref in element.exposed().items()}
        return {
            "type": GROUP_TYPE,
            "name": element.name,
            "children": children,
            "connections": connections,
            "exposed_ports": exposed,
        }
    type_id = reg.type_id(type(element))
    groups: dict[str, dict[str, Any]] = {"param": {}, "setting": {}, "state": {}}
    free: dict[str, Any] = {}
    for name, field in type(element).declared().items():
        value = getattr(element, name)
        groups[field.kind][name] = value if isinstance(field, Setting) else quantity_to_json(value)
        if isinstance(field, Param) and element.is_free(name):
            prior = element.prior(name)
            free[name] = {"prior": None if prior is None else prior_to_json(prior)}
    return {
        "type": type_id,
        "type_version": reg.version(type_id),
        "name": element.name,
        "params": groups["param"],
        "settings": groups["setting"],
        "state": groups["state"],
        "free": free,
    }


def _check_format_version(version: Any) -> None:
    match = _VERSION_RE.match(version) if isinstance(version, str) else None
    if match is None:
        raise FormatVersionError(f"invalid lumos_format {version!r}: expected 'major.minor'")
    major, minor = int(match[1]), int(match[2])
    if major != _SUPPORTED[0] or minor > _SUPPORTED[1]:
        raise FormatVersionError(
            f"lumos_format {version} is not supported by LUMOS {lumos.__version__} "
            f"(reads format {FORMAT_VERSION}); upgrade LUMOS to read this document"
        )


def from_document(document: Mapping[str, Any], *, registry: Registry | None = None) -> Element:
    """Create the element described by a ``.lumos`` document.

    Raises
    ------
    LumosFormatError
        If the document is malformed or describes invalid values.
    FormatVersionError
        If the format or type version cannot be read by this version of LUMOS.
    lumos.core.registry.UnknownTypeError
        If the element type is not registered.
    """
    reg = _registry(registry)
    if not isinstance(document, Mapping):
        raise LumosFormatError("a .lumos document must be a JSON object")
    missing = [key for key in ("lumos_format", "kind") if key not in document]
    if missing:
        raise LumosFormatError(f"missing key(s) in the document: {missing}")
    _check_format_version(document["lumos_format"])
    kind = document["kind"]
    if kind not in ("element", "group"):
        raise LumosFormatError(f"unsupported document kind {kind!r}")
    if kind not in document:
        raise LumosFormatError(f"missing key {kind!r} in the document")
    return from_entry(document[kind], registry=reg)


def from_entry(entry: Any, *, registry: Registry | None = None) -> Element:
    """Create the element or group described by a document entry."""
    reg = _registry(registry)
    if isinstance(entry, Mapping) and entry.get("type") == GROUP_TYPE:
        return _group_from_entry(entry, reg)
    if not isinstance(entry, Mapping) or "type" not in entry or "type_version" not in entry:
        raise LumosFormatError("the element entry needs 'type' and 'type_version'")
    type_id = entry["type"]
    cls = reg.get(type_id)
    version = entry["type_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise LumosFormatError(f"{type_id}: invalid type_version {version!r}")
    body = {key: entry[key] for key in _BODY_KEYS if key in entry}
    if version != reg.version(type_id):
        body = reg.migrate(type_id, body, version)

    try:
        jsonschema.validate(body, cls.schema())
    except jsonschema.ValidationError as error:
        location = "/".join(str(part) for part in error.absolute_path) or "element"
        raise LumosFormatError(f"{type_id}: {error.message} (at {location})") from error

    values: dict[str, Any] = dict(body["settings"])
    try:
        for group in ("params", "state"):
            for name, encoded in body[group].items():
                values[name] = quantity_from_json(encoded)
        element = cls(name=body.get("name"), **values)
        _apply_free(element, entry.get("free"))
    except (TypeError, ValueError, u.UnitsError) as error:
        raise LumosFormatError(f"{type_id}: {error}") from error
    return element


def _group_from_entry(entry: Mapping[str, Any], reg: Registry) -> Element:
    unknown = set(entry) - {"type", "name", "children", "connections", "exposed_ports", "id"}
    if unknown:
        raise LumosFormatError(f"unknown key(s) in a group entry: {sorted(unknown)}")
    group = ElementGroup(name=entry.get("name"))
    for child in entry.get("children", []):
        if not isinstance(child, Mapping) or not isinstance(child.get("id"), str):
            raise LumosFormatError("every child of a group needs a string 'id'")
        body = {key: value for key, value in child.items() if key != "id"}
        group.add(from_entry(body, registry=reg), id=child["id"])
    try:
        for connection in entry.get("connections", []):
            group.connect(_port(group, connection["from"]), _port(group, connection["to"]))
        for name, path in entry.get("exposed_ports", {}).items():
            group.expose(name, _port(group, path))
    except (KeyError, TypeError, ValueError) as error:
        raise LumosFormatError(f"invalid group {entry.get('name')!r}: {error}") from error
    return group


def _port(group: Any, path: Any) -> Any:
    if not isinstance(path, str) or "." not in path:
        raise LumosFormatError(f"invalid port reference {path!r}: expected 'id.port'")
    owner, _, name = path.rpartition(".")
    return group[owner].port(name)


def _apply_free(element: Element, free: Any) -> None:
    if free is None:
        return  # keep the defaults declared by the class
    if not isinstance(free, Mapping):
        raise ValueError("'free' must be an object")
    physical = [name for name, field in type(element).declared().items() if isinstance(field, Param)]
    unknown = sorted(set(free) - set(physical))
    if unknown:
        raise ValueError(f"unknown free parameter(s) {unknown}")
    for name in physical:
        if name not in free:
            element.set_free(name, False)
            continue
        spec = free[name]
        if not isinstance(spec, Mapping) or set(spec) != {"prior"}:
            raise ValueError(f"free parameter {name!r} must be described as {{'prior': ...}}")
        prior = None if spec["prior"] is None else prior_from_json(spec["prior"])
        element.set_free(name, True, prior)


def _reject_constant(constant: str) -> Any:
    raise LumosFormatError(f"non-standard JSON constant {constant!r} (use the strings 'nan', 'inf', '-inf')")


def save(element: Element, path: str | Path, *, registry: Registry | None = None) -> None:
    """Write ``element`` to a ``.lumos`` file (UTF-8, indented strict JSON)."""
    text = json.dumps(to_document(element, registry=registry), indent=2, ensure_ascii=False, allow_nan=False)
    Path(path).write_text(text + "\n", encoding="utf-8")


def load(path: str | Path, *, registry: Registry | None = None) -> Element:
    """Read an element from a ``.lumos`` file."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise LumosFormatError(f"{path}: invalid JSON ({error})") from error
    return from_document(document, registry=registry)
