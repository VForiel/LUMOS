"""Tests for pull execution and caching of a graph (step P5.2).

Requirements: REQ-CTX-4 (``run(outputs)`` computes only the required sub-graph, in topological order, with a
cache per port and downstream invalidation whenever a parameter or a state changes), REQ-EL-1 (changing a
parameter invalidates the downstream cache).

References: the expected values are written by hand (affine transfers on scalar signals) and the numbers of
transfer calls are counted for each element.
"""

from collections import Counter

import pytest
from astropy import units as u

from lumos.core.element import Element
from lumos.core.graph import Graph, GraphError
from lumos.core.param import Param, State
from lumos.core.port import Input, Output
from lumos.core.signal import Signal

CALLS: Counter = Counter()


class Number(Signal):
    def __init__(self, value):
        self.value = value


class Constant(Element):
    level = Param(default=1.0)
    out = Output(Number)

    @staticmethod
    def transfer(values, inputs):
        CALLS["constant"] += 1
        return {"out": Number(values.level)}


class Gain(Element):
    gain = Param(default=2.0)
    offset = State(default=0.0)
    in_ = Input(Number)
    out = Output(Number)

    @staticmethod
    def transfer(values, inputs):
        CALLS["gain"] += 1
        return {"out": Number(values.gain * inputs["in"].value + values.offset)}


class Sum(Element):
    in_ = Input(Number, multiple=True)
    out = Output(Number)

    @staticmethod
    def transfer(values, inputs):
        CALLS["sum"] += 1
        return {"out": Number(sum(s.value for s in inputs["in"]))}


class Splitter(Element):
    in_ = Input(Number)
    a = Output(Number)
    b = Output(Number)

    @staticmethod
    def transfer(values, inputs):
        CALLS["splitter"] += 1
        return {"a": Number(inputs["in"].value), "b": Number(-inputs["in"].value)}


class LengthSource(Element):
    length = Param(unit=u.m, default=1 * u.mm)
    out = Output(Number)

    @staticmethod
    def transfer(values, inputs):
        CALLS["length"] += 1
        return {"out": Number(values.length)}


@pytest.fixture(autouse=True)
def reset_counters():
    CALLS.clear()


def chain():
    g = Graph()
    c, g1, g2 = Constant(name="c", level=3.0), Gain(name="g1"), Gain(name="g2", gain=10.0)
    for e in (c, g1, g2):
        g.add(e)
    g.connect(c.out, g1.in_)
    g.connect(g1.out, g2.in_)
    return g, c, g1, g2


def test_run_returns_the_requested_outputs():
    g, c, g1, g2 = chain()
    result = g.run([g2.out])
    assert result[g2.out].value == 60.0  # 10 · (2 · 3)
    assert result["g2.out"].value == 60.0


def test_only_the_required_subgraph_is_computed():
    g, c, g1, g2 = chain()
    other = Gain(name="unused")
    g.add(other)
    g.connect(c.out, other.in_)
    g.run([g1.out])
    assert Counter({"constant": 1, "gain": 1}) == CALLS


def test_results_are_cached():
    g, c, g1, g2 = chain()
    g.run([g2.out])
    g.run([g2.out])
    g.run([g1.out])
    assert Counter({"constant": 1, "gain": 2}) == CALLS


def test_changing_a_parameter_recomputes_only_downstream():
    g, c, g1, g2 = chain()
    g.run([g2.out])
    CALLS.clear()
    g1.gain = 5.0
    assert g.run([g2.out])[g2.out].value == 150.0
    assert Counter({"gain": 2}) == CALLS  # g1 and g2, not the constant


def test_changing_the_last_element_recomputes_only_it():
    g, c, g1, g2 = chain()
    g.run([g2.out])
    CALLS.clear()
    g2.gain = 1.0
    assert g.run([g2.out])[g2.out].value == 6.0
    assert Counter({"gain": 1}) == CALLS


def test_changing_a_state_invalidates_too():
    g, c, g1, g2 = chain()
    g.run([g2.out])
    CALLS.clear()
    g1.offset = 1.0
    assert g.run([g2.out])[g2.out].value == 70.0
    assert Counter({"gain": 2}) == CALLS


def test_unit_parameters_reach_the_transfer_in_si():
    g = Graph()
    s = LengthSource()
    g.add(s)
    assert g.run([s.out])[s.out].value == pytest.approx(1e-3)
    s.length = 2 * u.cm
    assert g.run([s.out])[s.out].value == pytest.approx(2e-2)


def test_multiple_inputs_receive_a_list_in_connection_order():
    g = Graph()
    a, b, total = Constant(name="a", level=1.0), Constant(name="b", level=10.0), Sum()
    for e in (a, b, total):
        g.add(e)
    g.connect(a.out, total.in_)
    g.connect(b.out, total.in_)
    assert g.run([total.out])[total.out].value == 11.0


def test_several_outputs_of_one_element_are_computed_once():
    g = Graph()
    c, s = Constant(level=4.0), Splitter()
    g.add(c)
    g.add(s)
    g.connect(c.out, s.in_)
    result = g.run([s.a, s.b])
    assert (result[s.a].value, result[s.b].value) == (4.0, -4.0)
    assert CALLS["splitter"] == 1


def test_requesting_an_element_returns_all_its_outputs():
    g = Graph()
    c, s = Constant(level=4.0), Splitter(name="split")
    g.add(c)
    g.add(s)
    g.connect(c.out, s.in_)
    result = g.run([s])
    assert set(result) == {s.a, s.b}


def test_unconnected_inputs_are_reported():
    g = Graph()
    gain = Gain(name="lonely")
    g.add(gain)
    with pytest.raises(GraphError, match=r"lonely.*in"):
        g.run([gain.out])


def test_connecting_invalidates_downstream():
    g, c, g1, g2 = chain()
    g.run([g2.out])
    CALLS.clear()
    g.disconnect(c.out, g1.in_)
    other = Constant(name="other", level=1.0)
    g.add(other)
    g.connect(other.out, g1.in_)
    assert g.run([g2.out])[g2.out].value == 20.0
    assert Counter({"constant": 1, "gain": 2}) == CALLS


def test_removed_elements_stop_notifying_the_graph():
    g, c, g1, g2 = chain()
    g.run([g2.out])
    g.remove(g2)
    CALLS.clear()
    g2.gain = 3.0  # no error, no effect on the graph
    g.run([g1.out])
    assert Counter() == CALLS


def test_invalidate_clears_the_cache():
    g, c, g1, g2 = chain()
    g.run([g2.out])
    CALLS.clear()
    g.invalidate()
    g.run([g2.out])
    assert Counter({"constant": 1, "gain": 2}) == CALLS
