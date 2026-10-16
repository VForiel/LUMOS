"""Tests for lumos.core.graph.Graph (step P5.1).

Requirements: REQ-CTX-1 (container of the elements), REQ-CTX-2 (directed **acyclic** graph of signals),
REQ-SIG-11 (typed ports: incompatible connections are refused), REQ-CTX-4 (only the required sub-graph is
computed, in topological order).

References: the expected orders and sub-graphs are written by hand for small graphs; a topological order is
valid when every connection goes from an earlier to a later element (Cormen et al., *Introduction to
Algorithms*, §22.4), which is checked on random acyclic graphs generated with Hypothesis.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from lumos.core.element import Element
from lumos.core.graph import CycleError, Graph, GraphError
from lumos.core.port import Input, Output, PortCompatibilityError
from lumos.core.signal import DataSignal, OpticalSignal, Signal


class Source(Element):
    out = Output(OpticalSignal)


class Optic(Element):
    in_ = Input(OpticalSignal)
    out = Output(OpticalSignal)


class Combiner(Element):
    in_ = Input(OpticalSignal, multiple=True)
    out = Output(OpticalSignal)


class Detector(Element):
    in_ = Input(OpticalSignal)
    out = Output(DataSignal)


class Anything(Element):
    in_ = Input(Signal)
    out = Output(Signal)


def chain():
    g = Graph()
    s, o, d = Source(name="src"), Optic(name="lens"), Detector(name="cam")
    for e in (s, o, d):
        g.add(e)
    g.connect(s.out, o.in_)
    g.connect(o.out, d.in_)
    return g, s, o, d


# --- Elements and identifiers --------------------------------------------------------------------


def test_add_uses_the_name_as_identifier():
    g = Graph()
    lens = Optic(name="lens")
    assert g.add(lens) == "lens"
    assert g["lens"] is lens
    assert "lens" in g
    assert len(g) == 1


def test_unnamed_elements_get_unique_identifiers():
    g = Graph()
    ids = [g.add(Optic()) for _ in range(3)]
    assert len(set(ids)) == 3
    assert all(i.startswith("optic") for i in ids)


def test_explicit_identifier_and_duplicates():
    g = Graph()
    g.add(Optic(), id="l1")
    with pytest.raises(GraphError, match="l1"):
        g.add(Optic(), id="l1")
    with pytest.raises(GraphError):
        g.add(Optic(), id="not valid!")


def test_an_element_belongs_to_one_graph_once():
    g = Graph()
    lens = Optic()
    g.add(lens)
    with pytest.raises(GraphError):
        g.add(lens)
    assert g.id_of(lens) in g


def test_remove_deletes_connections():
    g, _s, o, _d = chain()
    g.remove(o)
    assert "lens" not in g
    assert g.connections == []


# --- Connections ---------------------------------------------------------------------------------


def test_connections_are_recorded():
    g, s, o, d = chain()
    assert [(c.source, c.target) for c in g.connections] == [(s.out, o.in_), (o.out, d.in_)]
    assert g.upstream(d) == [o]
    assert g.downstream(s) == [o]


def test_incompatible_signal_types_are_refused():
    g = Graph()
    d1, d2 = Detector(), Detector()
    g.add(d1)
    g.add(d2)
    with pytest.raises(PortCompatibilityError):
        g.connect(d1.out, d2.in_)  # DataSignal into an OpticalSignal input


def test_subtype_signals_are_accepted():
    g = Graph()
    s, a = Source(), Anything()
    g.add(s)
    g.add(a)
    g.connect(s.out, a.in_)


def test_direction_is_checked():
    g = Graph()
    o1, o2 = Optic(), Optic()
    g.add(o1)
    g.add(o2)
    with pytest.raises(PortCompatibilityError):
        g.connect(o1.in_, o2.out)


def test_single_inputs_take_one_connection():
    g = Graph()
    s1, s2, o = Source(), Source(), Optic()
    for e in (s1, s2, o):
        g.add(e)
    g.connect(s1.out, o.in_)
    with pytest.raises(GraphError, match="already connected"):
        g.connect(s2.out, o.in_)


def test_multiple_inputs_keep_the_connection_order():
    g = Graph()
    s1, s2, c = Source(name="a"), Source(name="b"), Combiner()
    for e in (s1, s2, c):
        g.add(e)
    g.connect(s2.out, c.in_)
    g.connect(s1.out, c.in_)
    assert [conn.source.owner for conn in g.incoming(c.in_)] == [s2, s1]


def test_ports_of_foreign_elements_are_refused():
    g = Graph()
    s, o = Source(), Optic()
    g.add(s)
    with pytest.raises(GraphError, match="not in the graph"):
        g.connect(s.out, o.in_)


def test_cycles_are_refused():
    g = Graph()
    a, b, c = Optic(name="a"), Optic(name="b"), Optic(name="c")
    for e in (a, b, c):
        g.add(e)
    g.connect(a.out, b.in_)
    g.connect(b.out, c.in_)
    with pytest.raises(CycleError, match="a"):
        g.connect(c.out, a.in_)
    assert len(g.connections) == 2  # the graph is unchanged


def test_self_loops_are_refused():
    g = Graph()
    a = Optic()
    g.add(a)
    with pytest.raises(CycleError):
        g.connect(a.out, a.in_)


def test_disconnect():
    g, s, o, _d = chain()
    g.disconnect(s.out, o.in_)
    assert g.upstream(o) == []
    with pytest.raises(GraphError):
        g.disconnect(s.out, o.in_)


# --- Topological order and required sub-graph -------------------------------------------------


def test_topological_order_of_a_chain():
    g, s, o, d = chain()
    assert g.topological_order() == [s, o, d]


def test_required_subgraph_contains_only_upstream_elements():
    g = Graph()
    s, o1, o2, d1, d2 = Source(name="s"), Optic(name="o1"), Optic(name="o2"), Detector(), Detector()
    for e in (s, o1, o2, d1, d2):
        g.add(e)
    g.connect(s.out, o1.in_)
    g.connect(s.out, o2.in_)
    g.connect(o1.out, d1.in_)
    g.connect(o2.out, d2.in_)
    assert g.required([d1.out]) == [s, o1, d1]
    assert g.required([o2]) == [s, o2]
    assert set(g.required([d1, d2])) == {s, o1, o2, d1, d2}


@given(st.data())
def test_random_dags_have_valid_topological_orders(data):
    n = data.draw(st.integers(2, 12))
    g = Graph()
    nodes = [Combiner(name=f"n{i}") for i in range(n)]
    for node in nodes:
        g.add(node)
    edges = data.draw(
        st.sets(st.tuples(st.integers(0, n - 1), st.integers(0, n - 1)).filter(lambda e: e[0] < e[1]))
    )
    for i, j in sorted(edges):
        g.connect(nodes[i].out, nodes[j].in_)
    order = g.topological_order()
    position = {id(e): k for k, e in enumerate(order)}
    assert len(order) == n
    for conn in g.connections:
        assert position[id(conn.source.owner)] < position[id(conn.target.owner)]


@given(st.data())
def test_closing_any_path_into_a_loop_is_refused(data):
    n = data.draw(st.integers(2, 8))
    g = Graph()
    nodes = [Combiner(name=f"n{i}") for i in range(n)]
    for node in nodes:
        g.add(node)
    for i in range(n - 1):
        g.connect(nodes[i].out, nodes[i + 1].in_)
    j = data.draw(st.integers(0, n - 2))
    with pytest.raises(CycleError):
        g.connect(nodes[n - 1].out, nodes[j].in_)
