"""Review of the public API before its freeze for 1.x (step P15.1, ADR-0021).

Requirements:

- every element type has a docstring, a stable identifier ``lumos.<family>.<ClassName>`` and a description of
  every field (shown by the web interface, the JSON schemas and the MCP server);
- public identifiers (names, fields, parameters) use the American spelling of the scientific Python
  libraries (NumPy, astropy, matplotlib: ``normalize``, ``center``, ``color``); the prose is free;
- a field renamed by the review keeps reading the documents written before (type version and migration);
- every public subpackage lists its names in ``__all__``, every listed name exists, and the API reference
  documents every public subpackage.
"""

import importlib
import inspect
import pathlib
import pkgutil
import re

import pytest

import lumos
import lumos.elements
import lumos.processing
import lumos.scenes
import lumos.symbolic.source
from lumos.core.registry import registry
from lumos.io import from_document

PUBLIC = [
    "backend",
    "catalogs",
    "core",
    "elements",
    "fields",
    "hardware",
    "io",
    "plotting",
    "processing",
    "propagation",
    "results",
    "scenes",
    "service",
    "symbolic",
]
BRITISH = re.compile(
    r"(normalis|centre|colour|realis|regularis|polaris|serialis|vectoris|analys)", re.IGNORECASE
)
REFERENCE = pathlib.Path(__file__).resolve().parents[2] / "docs" / "api" / "index.md"


def test_every_element_type_is_documented_with_a_stable_identifier():
    for type_id in registry:
        cls = registry.get(type_id)
        family, name = type_id.split(".")[1:]
        assert type_id == f"lumos.{family}.{name}", type_id
        assert name == cls.__name__, type_id
        assert (cls.__doc__ or "").strip(), f"{type_id} has no docstring"
        missing = [field for field, declared in cls.declared().items() if not getattr(declared, "doc", "")]
        assert not missing, f"{type_id}: fields without a description {missing}"


def public_identifiers():
    for module_info in pkgutil.walk_packages(lumos.__path__, "lumos."):
        if "._" in module_info.name or module_info.name.startswith("lumos.web"):
            continue
        module = importlib.import_module(module_info.name)
        for name, obj in vars(module).items():
            if name.startswith("_") or getattr(obj, "__module__", None) != module.__name__:
                continue
            yield f"{module.__name__}.{name}", name
            if inspect.isfunction(obj):
                for parameter in inspect.signature(obj).parameters:
                    yield f"{module.__name__}.{name}({parameter})", parameter
            if inspect.isclass(obj):
                for attribute in vars(obj):
                    if not attribute.startswith("_"):
                        yield f"{module.__name__}.{name}.{attribute}", attribute


def test_public_identifiers_use_the_american_spelling():
    found = sorted({where for where, identifier in public_identifiers() if BRITISH.search(identifier)})
    assert found == []


@pytest.mark.parametrize(
    ("type_id", "old", "new"),
    [
        ("lumos.processing.FlatField", "normalise", "normalize"),
        ("lumos.processing.KernelObservables", "normalise", "normalize"),
    ],
)
def test_renamed_fields_read_the_documents_written_before(type_id, old, new):
    assert registry.version(type_id) >= 2
    document = {
        "lumos_format": "1.0",
        "kind": "element",
        "element": {
            "type": type_id,
            "type_version": 1,
            "name": "x",
            "params": {},
            "settings": {old: False},
            "state": {},
        },
    }
    element = from_document(document)
    assert getattr(element, new) is False


def test_public_subpackages_list_their_names():
    for name in PUBLIC:
        module = importlib.import_module(f"lumos.{name}")
        listed = getattr(module, "__all__", None)
        assert listed, f"lumos.{name} has no __all__"
        for item in listed:
            assert hasattr(module, item), f"lumos.{name}.__all__ lists {item}, which does not exist"


def test_the_api_reference_documents_every_public_subpackage():
    text = REFERENCE.read_text(encoding="utf-8")
    missing = [name for name in PUBLIC if not re.search(rf"automodule:: lumos\.{name}\b", text)]
    assert missing == []
