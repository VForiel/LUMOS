"""Tests for bus ports and the mapping of one-lane elements over buses (step P15.6, ADR-0023).

Specification: docs/specs/pistes_evolution.md §1.2 — "a one-lane component applied to a bus acts on every
lane, as N identical copies"; an input declared with ``bus=True`` receives the bus as it is.

Consequences checked here:

- mapping gives, lane by lane, exactly what the element gives on that lane alone, and keeps the names and
  positions of the bus;
- several bus inputs must have the same number of lanes, and other inputs are shared by every lane;
- a stochastic element mapped over a bus draws an independent stream per lane (N copies have N streams);
- an element with a data output cannot be mapped (its outputs would not be lanes);
- the ``bus`` flag of ports is published in the schema of the element.
"""

from typing import Any

import numpy as np
import pytest
from astropy import units as u

from lumos.core.element import Element, Values, generator
from lumos.core.param import Param
from lumos.core.port import Input, Output
from lumos.core.signal import OpticalSignal
from lumos.fields import Bus, DataProduct, ModalField

LAM = [1.55] * u.um


def lane(value: complex) -> ModalField:
    return ModalField(np.array([[value]]), LAM)


class Gain(Element):
    factor = Param(default=2.0)
    in_ = Input(ModalField)
    out = Output(ModalField)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        return {"out": inputs["in"] * values.factor}


class Sum(Element):
    left = Input(ModalField)
    right = Input(ModalField)
    out = Output(ModalField)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        left, right = inputs["left"], inputs["right"]
        return {"out": ModalField(np.asarray(left.amplitudes) + np.asarray(right.amplitudes), LAM)}


class Noisy(Element):
    stochastic = True
    in_ = Input(ModalField)
    out = Output(ModalField)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        return {"out": inputs["in"] * complex(generator(values).normal())}


class Meter(Element):
    in_ = Input(ModalField)
    out = Output(DataProduct)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        return {"out": DataProduct(inputs["in"].total_power)}


class Counter(Element):
    in_ = Input(OpticalSignal, bus=True)
    out = Output(DataProduct)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        signal = inputs["in"]
        n = len(signal) if isinstance(signal, Bus) else 0
        return {"out": DataProduct(np.array([float(n)]) * u.dimensionless_unscaled)}


def amplitudes(bus: Bus) -> list[complex]:
    return [complex(np.asarray(lane_.amplitudes)[0, 0]) for lane_ in bus]


def test_a_one_lane_element_acts_on_every_lane_of_a_bus():
    bus = Bus([lane(1), lane(2j), lane(-3)], names=["a", "b", "c"], positions=[[0, 0], [1, 0], [2, 0]] * u.m)
    out = Gain(factor=3.0).apply(**{"in": bus})["out"]
    assert isinstance(out, Bus)
    assert amplitudes(out) == [3, 6j, -9]
    assert out.names == ("a", "b", "c")
    np.testing.assert_allclose(out.positions.to_value(u.m), [[0, 0], [1, 0], [2, 0]])


def test_a_single_lane_is_not_turned_into_a_bus():
    out = Gain().apply(**{"in": lane(1)})["out"]
    assert isinstance(out, ModalField)


def test_bus_inputs_are_mapped_together_and_other_inputs_shared():
    left = Bus([lane(1), lane(2)])
    out = Sum().apply(left=left, right=Bus([lane(10), lane(20)]))["out"]
    assert amplitudes(out) == [11, 22]
    shared = Sum().apply(left=left, right=lane(100))["out"]
    assert amplitudes(shared) == [101, 102]
    with pytest.raises(ValueError, match="lanes"):
        Sum().apply(left=left, right=Bus([lane(1), lane(2), lane(3)]))


def test_a_stochastic_element_draws_an_independent_stream_per_lane():
    out = Noisy().apply(**{"in": Bus([lane(1), lane(1), lane(1)])})["out"]
    draws = amplitudes(out)
    assert len(set(draws)) == 3


def test_an_element_with_a_data_output_cannot_be_mapped():
    with pytest.raises(TypeError, match="bus"):
        Meter().apply(**{"in": Bus([lane(1), lane(2)])})


def test_a_bus_input_receives_the_bus_as_it_is():
    reading = Counter().apply(**{"in": Bus([lane(1), lane(2)])})["out"]
    assert float(reading.data.value[0]) == 2
    single = Counter().apply(**{"in": lane(1)})["out"]
    assert float(single.data.value[0]) == 0


def test_the_bus_flag_is_in_the_schema():
    inputs = Counter.schema()["x-lumos-ports"]["inputs"]
    assert inputs[0]["bus"] is True
    assert Gain.schema()["x-lumos-ports"]["outputs"][0]["bus"] is False


def test_mapping_works_through_a_context():
    from lumos import Context
    from lumos.core.element import Element as _Element

    class Source(_Element):
        out = Output(ModalField, bus=True)

        @staticmethod
        def transfer(values: Values, inputs: Any) -> dict[str, Any]:
            return {"out": Bus([lane(1), lane(2)])}

    ctx = Context()
    source = ctx.add(Source(name="source"))
    gain = ctx.add(Gain(name="gain", factor=5.0))
    ctx.connect(source.port("out"), gain.port("in"))
    assert amplitudes(ctx.run(["gain.out"])["gain.out"]) == [5, 10]
