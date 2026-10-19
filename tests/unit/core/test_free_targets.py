"""Free settings and components of array parameters (step P14.17, ADR-0022; REQ-EL-3, REQ-CTX-10).

Requirements (review of 19/10/2026: "many parameters cannot be used in the Monte-Carlo runs"):

- a numeric setting (``int`` or ``float``, without choices) can be free with a dimensionless prior; boolean,
  text and choice settings cannot;
- a component ``"field[i]"`` or ``"field[i, j]"`` of an array parameter can be free with a prior in the unit
  of the parameter; an index outside the array is refused;
- sweeps and Monte-Carlo runs vary these targets; an integer setting takes the nearest integer of the value;
  setting a component changes this component only; every target gets its value back afterwards;
- ``.lumos`` documents (format 1.2) keep the free targets; documents of format 1.1 are still read.

Expected values: by construction (affine toy elements, outputs computed by hand); the integers drawn from a
uniform prior on [2.5, 6.5] are 3, 4, 5 and 6 with probability 1/4 each (rounding to the nearest integer);
sample means within 4 standard errors of the prior mean.
"""

import numpy as np
import pytest
from astropy import units as u

import lumos.io
from lumos import Context
from lumos.core.element import Element
from lumos.core.param import Param, Setting, parse_target
from lumos.core.port import Input, Output
from lumos.core.priors import Normal, Uniform
from lumos.core.registry import register, registry
from lumos.core.signal import Signal
from lumos.io.lumos_format import FORMAT_VERSION


class Number(Signal):
    def __init__(self, value):
        self.value = value


@register("tests.free_targets.Bank")
class Bank(Element):
    """Sum of an array of powers."""

    powers = Param(unit=u.W, shape=None, default=[1.0, 2.0, 3.0] * u.W, doc="powers summed")
    weights = Param(shape=None, default=[[1.0, 0.0], [0.0, 1.0]], doc="a matrix (2-D array)")
    out = Output(Number)

    @staticmethod
    def transfer(values, inputs):
        return {"out": Number(float(np.sum(values.powers)))}


@register("tests.free_targets.Stages")
class Stages(Element):
    """Multiplies by ``factor`` once per stage."""

    stages = Setting(int, default=1, doc="number of stages")
    factor = Setting(float, default=2.0, doc="gain of a stage")
    flag = Setting(bool, default=False, doc="a boolean setting")
    mode = Setting(str, default="a", choices=["a", "b"], doc="a choice")
    in_ = Input(Number)
    out = Output(Number)

    @staticmethod
    def transfer(values, inputs):
        return {"out": Number(inputs["in"].value * values.factor**values.stages)}


def system():
    ctx = Context(seed=3)
    bank = ctx.add(Bank(name="bank"))
    stages = ctx.add(Stages(name="stages"))
    bank >> stages
    return ctx, bank, stages


def test_targets_are_named_by_a_field_and_indices():
    assert parse_target("powers") == ("powers", None)
    assert parse_target("powers[2]") == ("powers", (2,))
    assert parse_target("weights[1, 0]") == ("weights", (1, 0))
    assert parse_target("weights[1,0]") == ("weights", (1, 0))
    with pytest.raises(ValueError):
        parse_target("powers[x]")


def test_a_numeric_setting_is_free_with_a_dimensionless_prior():
    _, _, stages = system()
    stages.set_free("stages", True, Uniform(2.5, 6.5))
    stages.set_free("factor", True, Normal(2.0, 0.1))
    assert stages.is_free("stages")
    assert stages.prior("factor") == Normal(2.0, 0.1)
    assert stages.free_names() == ["stages", "factor"]
    stages.set_target("stages", 3.7 * u.one)
    assert stages.stages == 4
    assert isinstance(stages.stages, int)
    with pytest.raises(ValueError):
        stages.set_free("factor", True, Normal(1 * u.m, 1 * u.cm))  # a setting has no unit


@pytest.mark.parametrize("name", ["flag", "mode"])
def test_boolean_and_choice_settings_cannot_be_free(name):
    _, _, stages = system()
    with pytest.raises(TypeError):
        stages.set_free(name, True)


def test_a_component_of_an_array_parameter_is_free_in_its_unit():
    _, bank, _ = system()
    bank.set_free("powers[1]", True, Normal(2 * u.W, 0.1 * u.W))
    bank.set_free("weights[0, 1]", True, Uniform(0.0, 0.5))
    assert bank.free_names() == ["powers[1]", "weights[0, 1]"]
    assert bank.get_target("powers[1]") == 2 * u.W
    bank.set_target("powers[1]", 500 * u.mW)
    assert u.allclose(bank.powers, [1.0, 0.5, 3.0] * u.W)
    bank.set_target("weights[0, 1]", 0.25)
    assert np.allclose(np.asarray(bank.weights), [[1.0, 0.25], [0.0, 1.0]])
    with pytest.raises(IndexError):
        bank.set_free("powers[3]", True, Normal(1 * u.W, 1 * u.mW))
    with pytest.raises(IndexError):
        bank.set_free("weights[2]", True, Uniform(0.0, 1.0))  # one index for a 2-D array
    with pytest.raises(ValueError):
        bank.set_free("powers[0]", True, Normal(1 * u.m, 1 * u.cm))


def test_a_sweep_varies_components_and_integer_settings():
    ctx, bank, stages = system()
    points = ctx.sweep(
        {"bank.powers[0]": [0, 10] * u.W, "stages.stages": [0.6, 2.2]}, outputs=["stages.out"], mode="zip"
    )
    # stages rounded to 1 and 2; sums 0 + 2 + 3 and 10 + 2 + 3
    assert [p.outputs["stages.out"].value for p in points] == pytest.approx([5 * 2, 15 * 4])
    assert u.allclose(bank.powers, [1.0, 2.0, 3.0] * u.W)  # restored
    assert stages.stages == 1


def test_a_monte_carlo_run_draws_settings_and_components():
    ctx, bank, stages = system()
    stages.set_free("stages", True, Uniform(2.5, 6.5))
    bank.set_free("powers[2]", True, Normal(3 * u.W, 0.5 * u.W))
    assert list(ctx.free_parameters()) == ["bank.powers[2]", "stages.stages"]
    n = 400
    points = ctx.sample(n, outputs=["stages.out"])
    drawn = np.array([p.params["stages.stages"] for p in points], dtype=float)
    assert set(np.unique(drawn)) == {3.0, 4.0, 5.0, 6.0}
    for value in (3, 4, 5, 6):  # 1/4 each, within 4 standard deviations of a binomial count
        assert abs(np.sum(drawn == value) - n / 4) < 4 * np.sqrt(n * 0.25 * 0.75)
    third = np.array([p.params["bank.powers[2]"].to_value(u.W) for p in points])
    assert abs(third.mean() - 3.0) < 4 * 0.5 / np.sqrt(n)
    for point in points[:10]:
        expected = (1.0 + 2.0 + point.params["bank.powers[2]"].to_value(u.W)) * 2.0 ** int(
            point.params["stages.stages"]
        )
        assert point.outputs["stages.out"].value == pytest.approx(expected)
    assert stages.stages == 1
    assert u.allclose(bank.powers, [1.0, 2.0, 3.0] * u.W)


def test_documents_keep_the_free_targets(tmp_path):
    ctx, bank, stages = system()
    stages.set_free("factor", True, Normal(2.0, 0.1))
    bank.set_free("powers[1]", True, Normal(2 * u.W, 0.1 * u.W))
    document = lumos.io.to_document(ctx)
    assert FORMAT_VERSION == "1.2"
    assert document["lumos_format"] == "1.2"
    loaded = lumos.io.from_document(document)
    assert list(loaded.free_parameters()) == ["bank.powers[1]", "stages.factor"]
    assert loaded.free_parameters()["bank.powers[1]"] == Normal(2 * u.W, 0.1 * u.W)

    old = lumos.io.to_document(system()[0])
    old["lumos_format"] = "1.1"
    assert len(lumos.io.from_document(old)) == 2  # 1.1 documents are still read


def teardown_module():
    for type_id in ("tests.free_targets.Bank", "tests.free_targets.Stages"):
        registry.unregister(type_id)
