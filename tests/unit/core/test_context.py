"""Tests for lumos.Context (step P5.3).

Requirements: REQ-CTX-1 (the context holds and coordinates the elements), REQ-CTX-3 (``ctx.add``,
``ctx.connect``, ``a >> b`` when ports are unique, access by name), REQ-CTX-6 (global observation
settings: spectral sampling, random seeds), REQ-EL-4 (simulation or operation mode, set on the context),
REQ-EL-7 (digital twin enabled by default), REQ-HPC-1 (backend chosen per context).

References: expected values are written by hand; random streams are compared with NumPy generators built
from the same seed sequence, ``numpy.random.SeedSequence(seed, spawn_key=...)`` (NumPy documentation,
"Parallel random number generation").
"""

import numpy as np
import pytest
from astropy import units as u

from lumos import Context, backend
from lumos.core.element import Element
from lumos.core.graph import GraphError
from lumos.core.param import Param
from lumos.core.port import Input, Output
from lumos.core.signal import Signal

SEEN_BACKENDS: list[str] = []


class Number(Signal):
    def __init__(self, value):
        self.value = value


class Source(Element):
    level = Param(default=1.0)
    out = Output(Number)

    @staticmethod
    def transfer(values, inputs):
        SEEN_BACKENDS.append(backend.current().name)
        return {"out": Number(values.level)}


class Double(Element):
    in_ = Input(Number)
    out = Output(Number)

    @staticmethod
    def transfer(values, inputs):
        return {"out": Number(2 * inputs["in"].value)}


class TwoOutputs(Element):
    in_ = Input(Number)
    a = Output(Number)
    b = Output(Number)

    @staticmethod
    def transfer(values, inputs):
        return {"a": inputs["in"], "b": inputs["in"]}


# --- Settings --------------------------------------------------------------------------------


def test_defaults():
    ctx = Context()
    assert ctx.mode == "simulation"
    assert ctx.twin is True
    assert ctx.backend == "auto"
    assert len(ctx) == 0


def test_mode_is_validated():
    with pytest.raises(ValueError, match="mode"):
        Context(mode="dream")
    ctx = Context()
    ctx.mode = "operation"
    assert ctx.mode == "operation"
    with pytest.raises(ValueError):
        ctx.mode = "fast"


def test_backend_is_validated():
    with pytest.raises(ValueError):
        Context(backend="fortran")


def test_run_uses_the_context_backend():
    SEEN_BACKENDS.clear()
    ctx = Context(backend="numpy")
    s = ctx.add(Source())
    ctx.run([s.out])
    assert SEEN_BACKENDS == ["numpy"]


# --- Elements, names and connections -----------------------------------------------------------


def test_add_returns_the_element_and_names_reach_it():
    ctx = Context()
    lens = ctx.add(Double(name="lens"))
    assert ctx["lens"] is lens
    assert "lens" in ctx
    assert ctx.id_of(lens) == "lens"


def test_add_several_elements():
    ctx = Context()
    a, b = ctx.add(Source(name="a"), Double(name="b"))
    assert list(ctx.elements) == ["a", "b"]
    assert a.context is ctx
    assert b.context is ctx


def test_an_element_belongs_to_one_context():
    s = Source()
    Context().add(s)
    with pytest.raises(GraphError, match="context"):
        Context().add(s)


def test_remove_detaches_the_element():
    ctx = Context()
    s = ctx.add(Source(name="s"))
    ctx.remove(s)
    assert "s" not in ctx
    assert s.context is None


def test_connect_and_run():
    ctx = Context()
    s, d = ctx.add(Source(level=3.0), Double(name="d"))
    ctx.connect(s.out, d.in_)
    assert ctx.run([d.out])[d.out].value == 6.0
    assert ctx.run(["d.out"])["d.out"].value == 6.0


def test_rshift_connects_unique_ports_and_adds_the_target():
    ctx = Context()
    s = ctx.add(Source(level=1.5))
    d1, d2 = Double(name="d1"), Double(name="d2")
    last = s >> d1 >> d2
    assert last is d2
    assert d1.context is ctx
    assert ctx.run([d2.out])[d2.out].value == 6.0


def test_rshift_from_a_port_and_to_a_port():
    ctx = Context()
    s, t = ctx.add(Source(level=2.0), TwoOutputs(name="t"))
    s >> t.in_
    d = Double()
    t.b >> d
    assert ctx.run([d.out])[d.out].value == 4.0


def test_rshift_refuses_ambiguous_ports():
    ctx = Context()
    s, t = ctx.add(Source(), TwoOutputs())
    s >> t
    with pytest.raises(GraphError, match="ambiguous"):
        t >> Double()


def test_rshift_needs_a_context():
    with pytest.raises(GraphError, match="context"):
        Source() >> Double()


def test_parameter_changes_invalidate_the_results():
    ctx = Context()
    s = ctx.add(Source(level=1.0))
    d = Double()
    s >> d
    assert ctx.run([d.out])[d.out].value == 2.0
    s.level = 4.0
    assert ctx.run([d.out])[d.out].value == 8.0


# --- Seeds ---------------------------------------------------------------------------------------


def test_random_streams_are_reproducible_and_independent():
    ctx1, ctx2 = Context(seed=42), Context(seed=42)
    a1, a2 = ctx1.add(Source(name="a")), ctx2.add(Source(name="a"))
    b1 = ctx1.add(Source(name="b"))
    x1 = ctx1.rng(a1).normal(size=5)
    x2 = ctx2.rng(a2).normal(size=5)
    np.testing.assert_array_equal(x1, x2)
    assert not np.allclose(ctx1.rng(b1).normal(size=5), x1)
    assert not np.allclose(Context(seed=43).rng("a").normal(size=5), x1)


def test_random_streams_follow_numpy_seed_sequences():
    ctx = Context(seed=7)
    ctx.add(Source(name="star"))
    expected = np.random.default_rng(np.random.SeedSequence(7, spawn_key=ctx.spawn_key("star"))).random(3)
    np.testing.assert_array_equal(ctx.rng("star").random(3), expected)


def test_a_fresh_stream_is_returned_on_each_call():
    ctx = Context(seed=1)
    ctx.add(Source(name="s"))
    np.testing.assert_array_equal(ctx.rng("s").random(2), ctx.rng("s").random(2))


# --- Observation ---------------------------------------------------------------------------------


def test_observation_band_gives_the_spectral_samples():
    ctx = Context()
    ctx.observation.band = (1.5 * u.um, 1.7 * u.um)
    ctx.observation.n_wavelengths = 4
    lam, weights = ctx.observation.wavelengths()
    np.testing.assert_allclose(lam.to_value(u.um), [1.525, 1.575, 1.625, 1.675])
    np.testing.assert_allclose(weights, 0.25)


def test_observation_band_is_validated():
    ctx = Context()
    with pytest.raises((TypeError, ValueError)):
        ctx.observation.band = (2 * u.um, 1 * u.um)
    with pytest.raises(TypeError):
        ctx.observation.band = (1.0, 2.0)
    with pytest.raises(ValueError):
        ctx.observation.wavelengths()  # no band set
