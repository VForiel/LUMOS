"""Noise realizations of sweeps and Monte-Carlo runs (step P14.13, REQ-CTX-6, REQ-CTX-10).

Requirements:

- the random streams of a context depend on its seed and on a *realization* index (0 by default, the streams
  of the earlier versions); each realization gives independent streams, reproducible with the same seed;
- a Monte-Carlo run (``ctx.sample``) draws a new noise realization at every point by default
  (``noise="independent"``), or keeps the noise of the context (``noise="fixed"``, common random numbers);
  without free parameters it repeats the run with independent noise;
- a sweep keeps the noise of the context by default (the points differ by their parameters only) and can draw
  a new realization per point;
- the seed and the realization are restored afterwards, and every point tells its realization.

Reference: NumPy, "Parallel random number generation": streams spawned from ``SeedSequence(seed,
spawn_key=key)`` with distinct keys are independent. The key of an element is the UTF-8 bytes of its path; a
realization ``r > 0`` appends ``0xFF`` (never a byte of UTF-8) and ``r``, so that keys never collide.
"""

import numpy as np
import pytest
from astropy import units as u

from lumos import Context
from lumos.core.element import Element, generator
from lumos.core.param import Param
from lumos.core.port import Output
from lumos.core.priors import Uniform
from lumos.core.signal import Signal


class Number(Signal):
    def __init__(self, value):
        self.value = value


class Noisy(Element):
    """``offset`` plus a standard normal draw."""

    stochastic = True
    offset = Param(default=0.0)
    out = Output(Number)

    @staticmethod
    def transfer(values, inputs):
        return {"out": Number(values.offset + float(generator(values).normal()))}


def context(seed=7):
    ctx = Context(seed=seed)
    ctx.add(Noisy(name="noise"))
    return ctx


def stream(seed, path, realization):
    key = tuple(path.encode("utf-8")) + ((0xFF, realization) if realization else ())
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=key))


def test_realization_zero_keeps_the_streams_of_the_seed():
    ctx = context()
    assert ctx.realization == 0
    assert ctx.run(["noise.out"])["noise.out"].value == stream(7, "noise", 0).normal()


def test_each_realization_has_its_own_reproducible_stream():
    ctx = context()
    ctx.realization = 3
    value = ctx.run(["noise.out"])["noise.out"].value
    assert value == stream(7, "noise", 3).normal()
    assert ctx.rng("noise").normal() == stream(7, "noise", 3).normal()
    ctx.realization = 4
    assert ctx.run(["noise.out"])["noise.out"].value == stream(7, "noise", 4).normal()
    with pytest.raises(ValueError, match="realization"):
        ctx.realization = -1


def test_monte_carlo_draws_independent_noise_by_default():
    ctx = context()
    points = ctx.sample(5, outputs=["noise.out"])  # no free parameter: the noise alone varies
    assert [p.realization for p in points] == [1, 2, 3, 4, 5]
    expected = [stream(7, "noise", r).normal() for r in range(1, 6)]
    assert [p.outputs["noise.out"].value for p in points] == pytest.approx(expected)
    assert ctx.realization == 0
    again = context().sample(5, outputs=["noise.out"])
    assert [p.outputs["noise.out"].value for p in again] == [p.outputs["noise.out"].value for p in points]


def test_monte_carlo_with_fixed_noise_keeps_the_noise_of_the_context():
    ctx = context()
    ctx["noise"].set_free("offset", True, Uniform(0.0, 1.0))
    points = ctx.sample(4, outputs=["noise.out"], noise="fixed")
    noise = stream(7, "noise", 0).normal()
    offsets = [p.params["noise.offset"].value for p in points]
    assert [p.outputs["noise.out"].value for p in points] == pytest.approx([o + noise for o in offsets])
    assert {p.realization for p in points} == {0}
    with pytest.raises(ValueError, match="nothing varies"):
        context().sample(3, outputs=["noise.out"], noise="fixed")
    with pytest.raises(ValueError, match="noise"):
        ctx.sample(3, outputs=["noise.out"], noise="sometimes")


def test_sweeps_keep_the_noise_unless_asked():
    ctx = context()
    fixed = ctx.sweep({"noise.offset": [0.0, 1.0]}, outputs=["noise.out"])
    noise = stream(7, "noise", 0).normal()
    assert [p.outputs["noise.out"].value for p in fixed] == pytest.approx([noise, 1.0 + noise])
    independent = ctx.sweep({"noise.offset": [0.0, 1.0]}, outputs=["noise.out"], noise="independent")
    assert [p.realization for p in independent] == [1, 2]
    expected = [stream(7, "noise", 1).normal(), 1.0 + stream(7, "noise", 2).normal()]
    assert [p.outputs["noise.out"].value for p in independent] == pytest.approx(expected)
    assert ctx.realization == 0


def test_the_mean_of_independent_realizations_converges():
    # standard normal draws: the mean of n independent draws has a standard deviation of 1/sqrt(n)
    n = 400
    points = context(seed=123).sample(n, outputs=["noise.out"])
    values = np.array([p.outputs["noise.out"].value for p in points])
    assert abs(values.mean()) < 4 / np.sqrt(n)
    assert values.std(ddof=1) == pytest.approx(1.0, abs=4 * np.sqrt(2 / (n - 1)))


def test_units_of_sampled_quantities_are_kept():
    ctx = Context(seed=1)

    class Source(Element):
        power = Param(unit=u.W, default=1 * u.W)
        out = Output(Number)

        @staticmethod
        def transfer(values, inputs):
            return {"out": Number(values.power)}

    source = ctx.add(Source(name="src"))
    source.set_free("power", True, Uniform(1 * u.mW, 2 * u.mW))
    points = ctx.sample(3, outputs=["src.out"])
    assert all(1e-3 <= p.outputs["src.out"].value <= 2e-3 for p in points)
    assert all(p.params["src.power"].unit.is_equivalent(u.W) for p in points)
