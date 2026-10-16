"""Tests for stochastic elements (step P6.5, needed by the turbulent atmosphere and later by detector noise).

Requirements: REQ-CTX-6 (random seeds of the context), ADR-0002 (pure transfers: the random stream of an
element is passed explicitly in its values).

Reference: the stream given to a stochastic transfer is the NumPy generator of the context stream of the
element, ``numpy.random.SeedSequence(seed, spawn_key=...)`` (NumPy documentation, "Parallel random number
generation").
"""

import numpy as np
import pytest

from lumos import Context
from lumos.core.element import Element, generator
from lumos.core.port import Output
from lumos.core.signal import Signal

CALLS: list[float] = []


class Number(Signal):
    def __init__(self, value):
        self.value = value


class Noise(Element):
    stochastic = True
    out = Output(Number)

    @staticmethod
    def transfer(values, inputs):
        value = float(generator(values).normal())
        CALLS.append(value)
        return {"out": Number(value)}


def test_stochastic_transfer_draws_from_the_context_stream():
    ctx = Context(seed=11)
    noise = ctx.add(Noise(name="noise"))
    value = ctx.run(["noise.out"])["noise.out"].value
    assert value == ctx.rng(noise).normal()


def test_same_seed_same_draws_and_other_seed_other_draws():
    def draw(seed):
        ctx = Context(seed=seed)
        ctx.add(Noise(name="noise"))
        return ctx.run(["noise.out"])["noise.out"].value

    assert draw(3) == draw(3)
    assert draw(3) != draw(4)


def test_changing_the_seed_recomputes_stochastic_elements():
    ctx = Context(seed=1)
    ctx.add(Noise(name="noise"))
    first = ctx.run(["noise.out"])["noise.out"].value
    ctx.seed = 2
    second = ctx.run(["noise.out"])["noise.out"].value
    assert first != second
    assert second == ctx.rng("noise").normal()


def test_outside_a_context_the_stream_is_fixed():
    a = Noise().apply()["out"].value
    b = Noise().apply()["out"].value
    assert a == b


def test_random_key_is_a_reserved_name():
    from lumos.core.param import Param

    with pytest.raises(TypeError, match="random_key"):

        class Bad(Element):
            stochastic = True
            random_key = Param(default=1.0)

    assert np.isfinite(CALLS).all()
