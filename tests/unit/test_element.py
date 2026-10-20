"""Tests for lumos.core.element (step P1.4).

Requirements: REQ-EL-1 (declarative parameters and ports, schema), REQ-EL-8 and REQ-EL-10 (pure numerical
transfer functions), REQ-EL-11 (describe, rich Jupyter representation, JSON Schema), and principle 7 of the
specs (interfaces derived from the schema). JSON Schema validity is checked against the official
draft 2020-12 meta-schema with the independent ``jsonschema`` library.
"""

import jsonschema
import numpy as np
import pytest
from astropy import units as u

from lumos.core.element import Element, Values
from lumos.core.param import Param, Setting, State
from lumos.core.port import Input, Output
from lumos.core.signal import DataSignal, OpticalSignal


class Field(OpticalSignal):
    """Minimal optical signal used by the tests."""

    def __init__(self, amplitude):
        self.amplitude = amplitude


class Attenuator(Element):
    """Attenuate a field.

    The output amplitude is the input amplitude times the square root of the transmission.
    """

    transmission = Param(default=0.5, bounds=(0, 1), doc="Power transmission")
    thickness = Param(u.mm, default=1 * u.mm, doc="Glass thickness")
    label = Setting(str, default="ND")
    temperature = State(u.K, default=293 * u.K)
    in_ = Input(Field)
    out = Output(Field)

    @staticmethod
    def transfer(values, inputs):
        return {"out": Field(inputs["in"].amplitude * np.sqrt(values.transmission))}


class Lens(Element):
    focal_length = Param(u.m, bounds=(0 * u.m, None), inclusive=(False, True))
    modes = Setting(int, default=1, choices=(1, 2, 4))


# --- Construction --------------------------------------------------------------------


def test_name_and_parameters():
    attenuator = Attenuator(name="ND1", transmission=0.25, thickness=2 * u.mm)
    assert attenuator.name == "ND1"
    assert attenuator.transmission == 0.25 * u.dimensionless_unscaled
    assert attenuator.thickness == 2 * u.mm


def test_name_is_optional():
    assert Attenuator().name is None


def test_name_must_be_a_string():
    with pytest.raises(TypeError):
        Attenuator(name=3)


def test_unknown_attribute_assignment_is_rejected_with_suggestion():
    attenuator = Attenuator()
    with pytest.raises(AttributeError, match="transmission"):
        attenuator.transmision = 0.1


def test_private_attributes_are_allowed():
    attenuator = Attenuator()
    attenuator._cache = 1
    assert attenuator._cache == 1


def test_field_name_reserved_by_element_api_is_rejected():
    with pytest.raises(TypeError, match="name"):

        class Bad(Element):
            name = Param(u.m, default=1 * u.m)


def test_port_and_field_with_same_name_are_rejected():
    with pytest.raises(TypeError, match="out"):

        class Bad(Element):
            out = Param(u.m, default=1 * u.m)
            out_ = Output(Field)


def test_two_ports_with_same_name_are_rejected():
    with pytest.raises(TypeError, match="in"):

        class Bad(Element):
            in_ = Input(Field)
            other = Input(Field, name="in")


def test_ports_are_listed():
    ports = Attenuator.ports()
    assert list(ports) == ["in", "out"]
    assert ports["in"].direction == "in"


# --- Values snapshot ---------------------------------------------------------------------


def test_values_are_an_immutable_si_snapshot():
    attenuator = Attenuator(thickness=2 * u.mm)
    values = attenuator.values()
    assert isinstance(values, Values)
    assert values.thickness == pytest.approx(0.002, rel=1e-15)
    assert values["label"] == "ND"
    assert set(values) == {"transmission", "thickness", "label", "temperature"}
    with pytest.raises(AttributeError):
        values.thickness = 1.0  # type: ignore[misc]
    attenuator.thickness = 5 * u.mm
    assert values.thickness == pytest.approx(0.002, rel=1e-15)  # the snapshot did not change


# --- Pure transfer ---------------------------------------------------------------------------


def test_transfer_must_be_a_staticmethod():
    with pytest.raises(TypeError, match="staticmethod"):

        class Impure(Element):
            out = Output(Field)

            def transfer(self, values, inputs):
                return {}


def test_apply_runs_the_pure_transfer():
    attenuator = Attenuator(transmission=0.25)
    outputs = attenuator.apply(**{"in": Field(2.0)})
    assert outputs["out"].amplitude == pytest.approx(1.0)  # 2 * sqrt(0.25)


def test_apply_rejects_unknown_or_missing_inputs():
    attenuator = Attenuator()
    with pytest.raises(TypeError, match="in"):
        attenuator.apply()
    with pytest.raises(TypeError, match="extra"):
        attenuator.apply(**{"in": Field(1.0), "extra": Field(1.0)})


def test_apply_checks_input_signal_types():
    class Data(DataSignal):
        pass

    with pytest.raises(TypeError, match="in"):
        Attenuator().apply(**{"in": Data()})


def test_apply_checks_outputs_of_the_transfer():
    class Broken(Element):
        out = Output(Field)

        @staticmethod
        def transfer(values, inputs):
            return {"wrong": Field(1.0)}

    with pytest.raises(TypeError, match="out"):
        Broken().apply()


def test_default_transfer_is_not_implemented():
    with pytest.raises(NotImplementedError):
        Lens(focal_length=1 * u.m).apply()


# --- Introspection -----------------------------------------------------------------------------


def test_describe_lists_name_fields_and_ports():
    text = Attenuator(name="ND1", thickness=2 * u.mm).describe()
    for expected in [
        "Attenuator",
        "ND1",
        "transmission",
        "thickness",
        "2.0 mm",
        "label",
        "temperature",
        "in",
        "out",
    ]:
        assert expected in text


def test_html_representation_is_a_table():
    html = Attenuator(name="<b>")._repr_html_()
    assert "<table" in html
    assert "thickness" in html
    assert "<b>" not in html  # user content is escaped


# --- JSON Schema ----------------------------------------------------------------------------------


def valid_body(**params):
    return {
        "name": "L1",
        "params": {"focal_length": {"value": 100.0, "unit": "mm"}, **params},
        "settings": {"modes": 2},
        "state": {},
    }


def test_schema_is_a_valid_draft_2020_12_schema():
    jsonschema.Draft202012Validator.check_schema(Lens.schema())
    jsonschema.Draft202012Validator.check_schema(Attenuator.schema())


def test_schema_accepts_a_valid_element_body():
    jsonschema.validate(valid_body(), Lens.schema())


def test_schema_rejects_unknown_param():
    body = valid_body(focal_lenght={"value": 1.0, "unit": "m"})
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(body, Lens.schema())


def test_schema_rejects_missing_required_param():
    body = valid_body()
    del body["params"]["focal_length"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(body, Lens.schema())


@pytest.mark.parametrize("modes", [3, "2", 2.5])
def test_schema_rejects_invalid_setting(modes):
    body = valid_body()
    body["settings"]["modes"] = modes
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(body, Lens.schema())


def test_schema_documents_units_bounds_defaults_and_ports():
    schema = Attenuator.schema()
    assert schema["title"] == "Attenuator"
    assert schema["description"].startswith("Attenuate a field.")
    thickness = schema["properties"]["params"]["properties"]["thickness"]
    assert u.Unit(thickness["x-lumos-unit"]) == u.mm
    assert thickness["description"] == "Glass thickness"
    assert thickness["default"] == {"value": 1.0, "unit": "mm"}
    transmission = schema["properties"]["params"]["properties"]["transmission"]
    assert transmission["x-lumos-bounds"] == [
        {"value": 0.0, "unit": ""},
        {"value": 1.0, "unit": ""},
    ]
    inputs = schema["x-lumos-ports"]["inputs"]
    assert inputs == [{"name": "in", "signal": "Field", "multiple": False, "bus": False, "doc": ""}]
    assert schema["x-lumos-ports"]["outputs"][0]["name"] == "out"
