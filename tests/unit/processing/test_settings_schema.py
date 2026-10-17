"""The JSON Schema of every registered element type is a JSON document (step P11.1, REQ-WEB-2).

Settings holding a quantity (``QuantitySetting``, e.g. a centre in pixels) must give their default in their
JSON form (``{"value", "unit"}``), as in a ``.lumos`` file: the web interface and the MCP server send
these schemas as JSON.
"""

import json

import lumos.elements
import lumos.processing
import lumos.symbolic.source  # noqa: F401 - registers the symbolic source
from lumos.core.registry import registry


def test_every_schema_is_json():
    for type_id in registry:
        json.dumps(registry.get(type_id).schema())


def test_quantity_settings_give_their_default_as_value_and_unit():
    schema = registry.get("lumos.processing.Crop").schema()
    default = schema["properties"]["settings"]["properties"]["x"]["default"]
    assert default == {"value": 0.0, "unit": "pix"}
