"""Tests for parameter sweeps and Monte-Carlo sampling (step P5.10).

Requirements: REQ-CTX-10 (parametric sweeps and Monte-Carlo runs, ``ctx.sweep``), REQ-EL-3 (free
parameters with priors).

References: expected outputs are computed by hand (affine transfers); Monte-Carlo draws are compared with
NumPy generators fed with the same seed sequence (``uniform``, ``normal`` and the exponential of a uniform
draw in log space for the log-uniform prior, NumPy random ``Generator`` documentation).
"""

import itertools

import numpy as np
import pytest
from astropy import units as u

from lumos import Context
from lumos.core.element import Element
from lumos.core.group import ElementGroup
from lumos.core.param import Param
from lumos.core.port import Input, Output
from lumos.core.priors import LogUniform, Normal, Uniform
from lumos.core.signal import Signal


class Number(Signal):
    def __init__(self, value):
        self.value = value


class Source(Element):
    power = Param(unit=u.W, default=1 * u.W)
    out = Output(Number)

    @staticmethod
    def transfer(values, inputs):
        return {"out": Number(values.power)}


class Gain(Element):
    gain = Param(default=2.0)
    in_ = Input(Number)
    out = Output(Number)

    @staticmethod
    def transfer(values, inputs):
        return {"out": Number(values.gain * inputs["in"].value)}


def system():
    ctx = Context(seed=11)
    source = ctx.add(Source(name="src"))
    group = ElementGroup(name="grp")
    gain = group.add(Gain(name="g"))
    group.expose("in", gain.in_)
    group.expose("out", gain.out)
    source >> group
    return ctx, source, gain


def test_single_parameter_sweep():
    ctx, source, _ = system()
    powers = [1, 2, 5] * u.mW
    results = ctx.sweep({"src.power": powers}, outputs=["grp.out"])
    assert len(results) == 3
    for point, power in zip(results, powers, strict=True):
        assert point.params["src.power"] == power
        assert point.outputs["grp.out"].value == pytest.approx(2 * power.to_value(u.W))
    assert source.power == 1 * u.W  # restored


def test_grid_sweep_is_the_cartesian_product():
    ctx, _, gain = system()
    powers = [1, 2] * u.W
    gains = [1.0, 3.0, 10.0]
    results = ctx.sweep({"src.power": powers, "grp.g.gain": gains}, outputs=["grp.out"])
    expected = [p.value * g for p, g in itertools.product(powers, gains)]
    assert [r.outputs["grp.out"].value for r in results] == pytest.approx(expected)
    assert gain.gain == 2.0


def test_zip_sweep_pairs_the_values():
    ctx, *_ = system()
    results = ctx.sweep(
        {"src.power": [1, 2] * u.W, "grp.g.gain": [3.0, 4.0]}, outputs=["grp.out"], mode="zip"
    )
    assert [r.outputs["grp.out"].value for r in results] == pytest.approx([3.0, 8.0])
    with pytest.raises(ValueError, match="length"):
        ctx.sweep({"src.power": [1, 2] * u.W, "grp.g.gain": [3.0]}, outputs=["grp.out"], mode="zip")


def test_sweep_paths_are_checked():
    ctx, *_ = system()
    with pytest.raises(KeyError):
        ctx.sweep({"nothing.power": [1] * u.W}, outputs=["grp.out"])
    with pytest.raises(AttributeError):
        ctx.sweep({"src.colour": [1]}, outputs=["grp.out"])
    with pytest.raises(ValueError, match="mode"):
        ctx.sweep({"src.power": [1] * u.W}, outputs=["grp.out"], mode="random")


def test_parameters_are_restored_after_an_error():
    ctx, source, _ = system()
    with pytest.raises(u.UnitsError):
        ctx.sweep({"src.power": [1 * u.W, 2 * u.s]}, outputs=["grp.out"])
    assert source.power == 1 * u.W


# --- Priors and Monte-Carlo ---------------------------------------------------------------------


def test_prior_samples_follow_numpy_generators():
    rng_a, rng_b = np.random.default_rng(5), np.random.default_rng(5)
    np.testing.assert_allclose(
        Uniform(1 * u.W, 3 * u.W).sample(rng_a, 4).to_value(u.W), rng_b.uniform(1, 3, 4)
    )
    np.testing.assert_allclose(
        Normal(2 * u.mm, 0.1 * u.mm).sample(rng_a, 3).to_value(u.mm), rng_b.normal(2, 0.1, 3)
    )
    np.testing.assert_allclose(
        LogUniform(1e-3, 1e3).sample(rng_a, 5).value, np.exp(rng_b.uniform(np.log(1e-3), np.log(1e3), 5))
    )


def test_monte_carlo_draws_the_free_parameters_from_their_priors():
    ctx, source, gain = system()
    source.set_free("power", True, Uniform(1 * u.W, 2 * u.W))
    gain.set_free("gain", True, Normal(3.0, 0.5))
    results = ctx.sample(4, outputs=["grp.out"])
    rng = np.random.default_rng(np.random.SeedSequence(11, spawn_key=ctx.spawn_key("sample")))
    powers = rng.uniform(1, 2, 4)
    gains = rng.normal(3.0, 0.5, 4)
    assert [r.params["src.power"].to_value(u.W) for r in results] == pytest.approx(list(powers))
    assert [r.params["grp.g.gain"].value for r in results] == pytest.approx(list(gains))
    assert [r.outputs["grp.out"].value for r in results] == pytest.approx(list(powers * gains))
    assert source.power == 1 * u.W


def test_monte_carlo_needs_priors():
    ctx, source, _ = system()
    source.set_free("power", True)
    with pytest.raises(ValueError, match="prior"):
        ctx.sample(2, outputs=["grp.out"])
