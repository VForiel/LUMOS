"""JSON Schema of ``.lumos`` documents, format 1.x (specs §10, ADR-0003).

The schema describes the envelope and the structure of element, group and context documents. The values of
the parameters of each element type are described by :meth:`lumos.core.element.Element.schema` and checked
when the element is loaded. The same schema is shipped as ``lumos/io/schemas/lumos-1.3.schema.json`` (format
1.3: feedbacks of contexts since 1.1, ADR-0020; free settings and components since 1.2, ADR-0022; settings
``channels`` designating the lanes of a bus since 1.3, ADR-0023); the schemas of the formats 1.0 to 1.2 are
kept next to it.
"""

from __future__ import annotations

from typing import Any

__all__ = ["SCHEMA_ID", "document_schema"]

SCHEMA_ID = "https://lumos-maxima.readthedocs.io/en/latest/lumos-1.3.schema.json"


def document_schema() -> dict[str, Any]:
    """Return the JSON Schema (draft 2020-12) of ``.lumos`` documents."""
    port_path = {"type": "string", "pattern": r"^[A-Za-z_][A-Za-z0-9_\-]*(\.[A-Za-z_][A-Za-z0-9_\-]*)+$"}
    identifier = {"type": "string", "pattern": r"^[A-Za-z_][A-Za-z0-9_\-]*$"}
    quantity = {
        "type": "object",
        "properties": {"value": {}, "unit": {"type": "string"}},
        "required": ["value", "unit"],
        "additionalProperties": False,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "title": "LUMOS document",
        "description": "A .lumos file: an element, an element group or a whole context.",
        "type": "object",
        "required": ["lumos_format", "kind"],
        "properties": {
            "lumos_format": {"type": "string", "pattern": r"^1\.[0-9]+$"},
            "kind": {"enum": ["element", "group", "context", "procedure", "component"]},
            "lumos_version": {"type": "string"},
            "created_at": {"type": "string"},
            "meta": {"type": "object"},
            "element": {"$ref": "#/$defs/element"},
            "group": {"$ref": "#/$defs/group"},
            "context": {"$ref": "#/$defs/settings"},
            "elements": {"type": "array", "items": {"$ref": "#/$defs/child"}},
            "connections": {"type": "array", "items": {"$ref": "#/$defs/connection"}},
            "feedbacks": {"type": "array", "items": {"$ref": "#/$defs/feedback"}},
            "procedures": {"type": "object", "additionalProperties": {"$ref": "#/$defs/procedure"}},
            "procedure": {"$ref": "#/$defs/procedure"},
            "component": {"$ref": "#/$defs/component"},
            "components": {"type": "object", "additionalProperties": {"$ref": "#/$defs/component"}},
            "ui": {"type": "object"},
        },
        "additionalProperties": False,
        "allOf": [
            {"if": {"properties": {"kind": {"const": "element"}}}, "then": {"required": ["element"]}},
            {"if": {"properties": {"kind": {"const": "group"}}}, "then": {"required": ["group"]}},
            {"if": {"properties": {"kind": {"const": "procedure"}}}, "then": {"required": ["procedure"]}},
            {"if": {"properties": {"kind": {"const": "component"}}}, "then": {"required": ["component"]}},
            {
                "if": {"properties": {"kind": {"const": "context"}}},
                "then": {"required": ["context", "elements", "connections"]},
            },
        ],
        "$defs": {
            "component": {
                "description": "A declarative component (ADR-0025): a template filled with data, no code.",
                "type": "object",
                "required": ["id", "version", "template", "body"],
                "properties": {
                    "id": {"type": "string", "pattern": r"^[a-z][a-z0-9_]*\.[A-Z][A-Za-z0-9_]*$"},
                    "version": {"type": "integer", "minimum": 1},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "license": {"type": "string"},
                    "authors": {"type": "array", "items": {"type": "string"}},
                    "template": {
                        "enum": ["transfer_matrix", "gain_phase", "spectral_transmission", "composite"]
                    },
                    "params": {"type": "object"},
                    "body": {"type": "object"},
                    "checks": {"type": "array", "items": {"type": "object"}},
                },
                "additionalProperties": False,
            },
            "procedure": {
                "description": "A procedure: steps run in order (ADR-0024), each naming its kind in 'do'.",
                "type": "object",
                "required": ["name", "steps"],
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "steps": {"type": "array", "items": {"$ref": "#/$defs/step"}},
                },
                "additionalProperties": False,
            },
            "step": {
                "type": "object",
                "required": ["do"],
                "properties": {
                    "do": {
                        "enum": [
                            "action",
                            "measure",
                            "repeat",
                            "for_each",
                            "process",
                            "set",
                            "use",
                            "wait",
                            "if",
                            "save",
                            "show",
                        ]
                    },
                    "steps": {"type": "array", "items": {"$ref": "#/$defs/step"}},
                    "then": {"type": "array", "items": {"$ref": "#/$defs/step"}},
                    "else": {"type": "array", "items": {"$ref": "#/$defs/step"}},
                },
            },
            "quantity": quantity,
            "element": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "type_version": {"type": "integer", "minimum": 1},
                    "name": {"type": ["string", "null"]},
                    "params": {"type": "object", "additionalProperties": {"$ref": "#/$defs/quantity"}},
                    "settings": {"type": "object"},
                    "state": {"type": "object", "additionalProperties": {"$ref": "#/$defs/quantity"}},
                    "free": {"type": "object"},
                    "driver": {"type": "object"},
                },
                "required": ["type", "type_version"],
            },
            "group": {
                "type": "object",
                "properties": {
                    "type": {"const": "lumos.group"},
                    "group_type": {"type": "string", "description": "registered specialised group class"},
                    "name": {"type": ["string", "null"]},
                    "children": {"type": "array", "items": {"$ref": "#/$defs/child"}},
                    "connections": {"type": "array", "items": {"$ref": "#/$defs/connection"}},
                    "exposed_ports": {"type": "object", "additionalProperties": port_path},
                },
                "required": ["type", "children"],
            },
            "child": {
                "type": "object",
                "properties": {"id": identifier},
                "required": ["id", "type"],
                "anyOf": [{"$ref": "#/$defs/element"}, {"$ref": "#/$defs/group"}],
            },
            "connection": {
                "type": "object",
                "properties": {"from": port_path, "to": port_path},
                "required": ["from", "to"],
                "additionalProperties": False,
            },
            "feedback": {
                "type": "object",
                "properties": {"from": port_path, "to": port_path, "mode": {"enum": ["set", "add"]}},
                "required": ["from", "to"],
                "additionalProperties": False,
            },
            "settings": {
                "type": "object",
                "properties": {
                    "mode": {"enum": ["simulation", "operation"]},
                    "twin": {"type": "boolean"},
                    "seed": {"type": "integer"},
                    "backend": {"type": "object", "properties": {"preferred": {"type": "string"}}},
                    "clock": {
                        "type": "object",
                        "properties": {"start": {"type": "string"}, "elapsed": {"$ref": "#/$defs/quantity"}},
                        "required": ["start", "elapsed"],
                    },
                    "observation": {
                        "type": "object",
                        "properties": {
                            "band": {
                                "oneOf": [
                                    {"type": "null"},
                                    {
                                        "type": "object",
                                        "properties": {
                                            "min": {"$ref": "#/$defs/quantity"},
                                            "max": {"$ref": "#/$defs/quantity"},
                                        },
                                        "required": ["min", "max"],
                                    },
                                ]
                            },
                            "n": {"type": "integer", "minimum": 1},
                            "rule": {"enum": ["midpoint", "gauss"]},
                            "extra": {"type": "object"},
                        },
                    },
                    "simulate": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["mode", "twin", "seed", "clock"],
            },
        },
    }
