"""Edits of ``.lumos`` context documents used by the visual editor (step P14.5, REQ-WEB-1, REQ-WEB-4).

Renaming, grouping, ungrouping, copying and pasting change identifiers in several places of a document
(element entries, connections, exposed ports of groups, the outputs listed in ``meta``, the elements forced to
simulation, the positions of the ``ui`` section). These pure functions transform the document; the service
then rebuilds the context from it, so that the result is always a valid context (or nothing changes).
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Mapping
from typing import Any

from lumos.service.errors import ServiceError

__all__ = ["NAME", "fragment_of", "group", "paste", "rename", "ungroup", "unique_name"]

#: Valid identifiers of elements: a letter, then letters, digits or underscores.
NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

Document = dict[str, Any]


def check_name(name: str) -> None:
    """Refuse names that cannot be element identifiers."""
    if not NAME.match(name or ""):
        raise ServiceError(
            f"{name!r} is not a valid name",
            hint="start with a letter, then use letters, digits or '_' (no space, no dot)",
        )


def unique_name(base: str, taken: Iterable[str]) -> str:
    """Return ``base`` if free, else its stem (without final digits) and the smallest free number from 2."""
    used = set(taken)
    if base not in used:
        return base
    stem = base.rstrip("0123456789") or base
    number = 2
    while f"{stem}{number}" in used:
        number += 1
    return f"{stem}{number}"


def _split(reference: str) -> tuple[str, str]:
    owner, _, port = reference.rpartition(".")
    return owner, port


def _top_ids(document: Mapping[str, Any]) -> list[str]:
    return [entry["id"] for entry in document["elements"]]


def _rename_references(document: Document, mapping: Mapping[str, str]) -> None:
    """Rename top-level elements in the connections, ``meta.outputs``, ``simulate`` and ``ui.positions``."""

    def renamed(reference: str) -> str:
        owner, port = _split(reference)
        head, dot, rest = owner.partition(".")
        return f"{mapping.get(head, head)}{dot}{rest}.{port}" if owner else reference

    for connection in document.get("connections", []):
        connection["from"] = renamed(connection["from"])
        connection["to"] = renamed(connection["to"])
    meta = document.setdefault("meta", {})
    if "outputs" in meta:
        meta["outputs"] = [renamed(output) for output in meta["outputs"]]
    settings = document.get("context", {})
    settings["simulate"] = [
        mapping.get(identifier, identifier) for identifier in settings.get("simulate", [])
    ]
    positions = document.setdefault("ui", {}).get("positions")
    if positions:
        updated = {}
        for key, value in positions.items():
            head, dot, rest = key.partition(".")
            updated[f"{mapping.get(head, head)}{dot}{rest}"] = value
        document["ui"]["positions"] = updated


def rename(document: Document, element: str, name: str) -> Document:
    """Rename the element ``element`` (a top-level identifier or a path ``group.child``)."""
    check_name(name)
    result = copy.deepcopy(document)
    owner, _, child = element.rpartition(".")
    if not owner:
        ids = _top_ids(result)
        if element not in ids:
            raise ServiceError(
                f"no element {element!r}", hint=f"elements: {', '.join(ids)}", code="not_found"
            )
        if name in ids and name != element:
            raise ServiceError(f"the name {name!r} is taken", hint="choose another name")
        for entry in result["elements"]:
            if entry["id"] == element:
                entry["id"] = entry["name"] = name
        _rename_references(result, {element: name})
        return result
    group_entry = _find_group(result, owner)
    ids = [entry["id"] for entry in group_entry["children"]]
    if child not in ids:
        raise ServiceError(
            f"no element {element!r}", hint=f"elements of {owner}: {', '.join(ids)}", code="not_found"
        )
    if name in ids and name != child:
        raise ServiceError(f"the name {name!r} is taken in {owner}", hint="choose another name")
    for entry in group_entry["children"]:
        if entry["id"] == child:
            entry["id"] = entry["name"] = name

    def inner(reference: str) -> str:
        head, port = _split(reference)
        return f"{name}.{port}" if head == child else reference

    for connection in group_entry.get("connections", []):
        connection["from"], connection["to"] = inner(connection["from"]), inner(connection["to"])
    group_entry["exposed_ports"] = {k: inner(v) for k, v in group_entry.get("exposed_ports", {}).items()}
    positions = result.setdefault("ui", {}).get("positions") or {}
    if element in positions:
        positions[f"{owner}.{name}"] = positions.pop(element)
    return result


def _find_group(document: Mapping[str, Any], path: str) -> Document:
    entries: list[Any] = document["elements"]
    found: Document | None = None
    for part in path.split("."):
        found = next((entry for entry in entries if entry["id"] == part), None)
        if found is None or "children" not in found:
            raise ServiceError(f"no group {path!r}", hint="give the path of a group", code="not_found")
        entries = found["children"]
    assert found is not None
    return found


def _free_ports(
    entries: list[Document], connections: list[Mapping[str, str]], ports: Mapping[str, Any]
) -> list:
    """``(direction, element, port)`` of the ports of ``entries`` not connected between them."""
    inside_targets = {c["to"] for c in connections}
    inside_sources = {c["from"] for c in connections}
    free = []
    for entry in entries:
        inputs, outputs = ports[entry["id"]]
        free += [("in", entry["id"], p) for p in inputs if f"{entry['id']}.{p}" not in inside_targets]
        free += [("out", entry["id"], p) for p in outputs if f"{entry['id']}.{p}" not in inside_sources]
    return free


def group(document: Document, elements: list[str], name: str, ports: Mapping[str, Any]) -> Document:
    """Move the top-level ``elements`` into a new group ``name`` exposing their free ports.

    ``ports`` maps each element to its ``(inputs, outputs)`` port names. Exposed ports keep the name of the
    inner port when it is unique in its direction, else ``<element>_<port>``.
    """
    check_name(name)
    result = copy.deepcopy(document)
    ids = _top_ids(result)
    chosen = list(dict.fromkeys(elements))
    if not chosen:
        raise ServiceError("no element to group", hint="select at least one element")
    unknown = [e for e in chosen if e not in ids]
    if unknown:
        raise ServiceError(
            f"cannot group {', '.join(unknown)}",
            hint="group elements of the context itself (not elements inside a group)",
            code="not_found",
        )
    if name in ids and name not in chosen:
        raise ServiceError(f"the name {name!r} is taken", hint="choose another name for the group")
    members = set(chosen)
    entries = [entry for entry in result["elements"] if entry["id"] in members]
    owner = lambda reference: _split(reference)[0].partition(".")[0]  # noqa: E731
    inside = [c for c in result["connections"] if owner(c["from"]) in members and owner(c["to"]) in members]
    free = _free_ports(entries, inside, ports)
    counts: dict[tuple[str, str], int] = {}
    for direction, _, port in free:
        counts[(direction, port)] = counts.get((direction, port), 0) + 1
    exposed: dict[str, str] = {}
    by_inner: dict[str, str] = {}
    for direction, element, port in free:
        label = port if counts[(direction, port)] == 1 else f"{element}_{port}"
        exposed[label] = f"{element}.{port}"
        by_inner[f"{element}.{port}"] = f"{name}.{label}"
    outside = []
    for connection in result["connections"]:
        if connection in inside:
            continue
        outside.append(
            {
                "from": by_inner.get(connection["from"], connection["from"]),
                "to": by_inner.get(connection["to"], connection["to"]),
            }
        )
    first = min(ids.index(e) for e in chosen)
    group_entry = {
        "id": name,
        "type": "lumos.group",
        "name": name,
        "children": entries,
        "connections": inside,
        "exposed_ports": exposed,
    }
    remaining = [entry for entry in result["elements"] if entry["id"] not in members]
    remaining.insert(min(first, len(remaining)), group_entry)
    result["elements"] = remaining
    result["connections"] = outside
    meta = result.setdefault("meta", {})
    if "outputs" in meta:
        meta["outputs"] = [by_inner.get(output, output) for output in meta["outputs"]]
    settings = result.get("context", {})
    settings["simulate"] = [i for i in settings.get("simulate", []) if i not in members]
    positions = result.setdefault("ui", {}).setdefault("positions", {})
    placed = [positions[e] for e in chosen if e in positions]
    for element in chosen:
        if element in positions:
            positions[f"{name}.{element}"] = positions.pop(element)
    if placed:
        positions[name] = [
            round(sum(p[0] for p in placed) / len(placed)),
            round(sum(p[1] for p in placed) / len(placed)),
        ]
    return result


def ungroup(document: Document, name: str) -> Document:
    """Dissolve the top-level group ``name``: its children join the context, its connections are rewired."""
    result = copy.deepcopy(document)
    group_entry = next((e for e in result["elements"] if e["id"] == name), None)
    if group_entry is None or "children" not in group_entry:
        raise ServiceError(
            f"no group {name!r}", hint="give the name of a group of the context", code="not_found"
        )
    others = [e["id"] for e in result["elements"] if e["id"] != name]
    mapping: dict[str, str] = {}
    for child in group_entry["children"]:
        new = unique_name(child["id"], others + list(mapping.values()))
        mapping[child["id"]] = new

    def promoted(reference: str) -> str:
        head, port = _split(reference)
        return f"{mapping.get(head, head)}.{port}"

    exposed = {
        f"{name}.{label}": promoted(inner) for label, inner in group_entry.get("exposed_ports", {}).items()
    }
    children = []
    for child in group_entry["children"]:
        child = dict(child)
        child["id"] = mapping[child["id"]]
        child["name"] = child["id"]
        children.append(child)
    connections: list[dict[str, Any]] = [
        {"from": promoted(c["from"]), "to": promoted(c["to"])} for c in group_entry.get("connections", [])
    ]
    for connection in result["connections"]:
        connections.append(
            {
                "from": exposed.get(connection["from"], connection["from"]),
                "to": exposed.get(connection["to"], connection["to"]),
            }
        )
    index = [e["id"] for e in result["elements"]].index(name)
    result["elements"][index : index + 1] = children
    result["connections"] = connections
    meta = result.setdefault("meta", {})
    if "outputs" in meta:
        meta["outputs"] = [exposed.get(output, output) for output in meta["outputs"]]
    positions = result.setdefault("ui", {}).setdefault("positions", {})
    centre = positions.pop(name, None)
    for offset, child in enumerate(group_entry["children"]):
        key = f"{name}.{child['id']}"
        if key in positions:
            positions[mapping[child["id"]]] = positions.pop(key)
        elif centre is not None:
            positions[mapping[child["id"]]] = [centre[0] + 290 * offset, centre[1]]
    return result


def fragment_of(document: Mapping[str, Any], elements: list[str]) -> Document:
    """Return the entries of the top-level ``elements``, the connections between them and their positions."""
    ids = _top_ids(document)
    unknown = [e for e in elements if e not in ids]
    if unknown:
        raise ServiceError(
            f"no element {', '.join(unknown)}", hint=f"elements: {', '.join(ids)}", code="not_found"
        )
    members = set(elements)
    owner = lambda reference: _split(reference)[0].partition(".")[0]  # noqa: E731
    positions = (document.get("ui") or {}).get("positions") or {}
    return {
        "kind": "fragment",
        "elements": [copy.deepcopy(e) for e in document["elements"] if e["id"] in members],
        "connections": [
            dict(c)
            for c in document.get("connections", [])
            if owner(c["from"]) in members and owner(c["to"]) in members
        ],
        "positions": {k: v for k, v in positions.items() if k.partition(".")[0] in members},
    }


def paste(
    document: Document, fragment: Mapping[str, Any], *, offset: tuple[int, int] = (40, 40)
) -> tuple[Document, list[str]]:
    """Add the elements of ``fragment`` with free names; return the document and the new identifiers."""
    if not isinstance(fragment, Mapping) or not isinstance(fragment.get("elements"), list):
        raise ServiceError("not a fragment of elements", hint="copy elements first (copy_elements)")
    result = copy.deepcopy(document)
    taken = _top_ids(result)
    mapping: dict[str, str] = {}
    for entry in fragment["elements"]:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), str):
            raise ServiceError("invalid element in the fragment", hint="copy elements first (copy_elements)")
        new = unique_name(entry["id"], taken + list(mapping.values()))
        mapping[entry["id"]] = new
    for entry in fragment["elements"]:
        item = copy.deepcopy(dict(entry))
        item["id"] = item["name"] = mapping[entry["id"]]
        result["elements"].append(item)

    def moved(reference: str) -> str:
        owner, port = _split(reference)
        head, dot, rest = owner.partition(".")
        return f"{mapping.get(head, head)}{dot}{rest}.{port}"

    for connection in fragment.get("connections", []):
        result["connections"].append({"from": moved(connection["from"]), "to": moved(connection["to"])})
    positions = result.setdefault("ui", {}).setdefault("positions", {})
    for key, (x, y) in (fragment.get("positions") or {}).items():
        head, dot, rest = key.partition(".")
        if head in mapping:
            positions[f"{mapping[head]}{dot}{rest}"] = [x + offset[0], y + offset[1]]
    return result, [mapping[e["id"]] for e in fragment["elements"]]
