"""Tests for lumos.jit and the JAX pytree form of Values (step P2.5).

Requirements: REQ-HPC-3 (costly functions compiled with the best technique: JAX jit for vectorisable and
differentiable kernels), REQ-PROP-6 / REQ-EL-10 (pure transfer functions differentiable with JAX), ADR-0002
(Values snapshots must be JAX pytrees). Gradients are compared with analytical derivatives.
"""

import importlib.util
import subprocess
import sys

import numpy as np
import pytest
from astropy import units as u

import lumos
from lumos import backend
from lumos.core.element import Element, Values
from lumos.core.param import Param, Setting
from lumos.core.port import Input, Output
from lumos.core.signal import OpticalSignal

HAS_JAX = importlib.util.find_spec("jax") is not None
requires_jax = pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")


@pytest.fixture
def numpy_backend():
    previous = lumos.config.backend
    lumos.config.backend = "numpy"
    yield
    lumos.config.backend = previous


def test_jit_is_exposed_at_package_level():
    assert lumos.jit is backend.jit


# --- NumPy path ------------------------------------------------------------------------------


def test_numpy_inputs_run_the_plain_function(numpy_backend):
    @lumos.jit
    def double(x):
        return 2 * x

    result = double(np.arange(3.0))
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [0.0, 2.0, 4.0])


def test_numpy_path_does_not_import_jax():
    code = (
        "import sys, numpy as np, lumos\n"
        "lumos.config.backend = 'numpy'\n"
        "f = lumos.jit(lambda x: x + 1)\n"
        "f(np.ones(2))\n"
        "print('jax' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


# --- JAX path --------------------------------------------------------------------------------------


@requires_jax
def test_jax_inputs_are_compiled_once_per_shape(numpy_backend):
    traces = []

    @lumos.jit
    def square(x):
        traces.append(1)  # executed only while JAX traces the function
        return x * x

    x = backend.convert(np.arange(4.0), "jax")
    first = square(x)
    second = square(x + 1)
    assert backend.backend_of(first) == "jax"
    np.testing.assert_array_equal(backend.to_numpy(second), (np.arange(4.0) + 1) ** 2)
    assert len(traces) == 1


@requires_jax
def test_current_jax_backend_compiles_python_inputs():
    traces = []

    @lumos.jit
    def add(x, y):
        traces.append(1)
        return x + y

    with backend.use("jax"):
        result = add(1.0, 2.0)
        add(3.0, 4.0)
    assert backend.backend_of(result) == "jax"
    assert float(backend.to_numpy(result)) == 3.0
    assert len(traces) == 1


@requires_jax
def test_static_arguments(numpy_backend):
    traces = []

    @lumos.jit(static_argnames=("n",))
    def tile(x, n):
        traces.append(n)
        xp = backend.get("jax").xp
        return xp.concat([x] * n)

    x = backend.convert(np.ones(2), "jax")
    assert tile(x, n=2).shape == (4,)
    assert tile(x, n=3).shape == (6,)
    assert tile(x, n=3).shape == (6,)
    assert traces == [2, 3]


# --- Values as a pytree ------------------------------------------------------------------------------


class Gain(Element):
    a = Param(u.m, default=3 * u.m)
    b = Param(default=2.0)
    label = Setting(str, default="x")
    modes = Setting(int, default=2)


@requires_jax
def test_values_leaves_are_the_numerical_values_only():
    import jax

    values = Gain().values()
    leaves = jax.tree_util.tree_leaves(values)
    assert sorted(float(leaf) for leaf in leaves) == [2.0, 3.0]


@requires_jax
def test_tree_map_keeps_settings_and_structure():
    import jax

    doubled = jax.tree_util.tree_map(lambda leaf: leaf * 2, Gain().values())
    assert isinstance(doubled, Values)
    assert doubled.a == 6.0
    assert doubled.b == 4.0
    assert doubled.label == "x"
    assert doubled.modes == 2


@requires_jax
def test_gradient_with_respect_to_values_matches_analytic_derivatives():
    import jax

    def f(values):
        return values.a**2 * values.b

    gradient = jax.grad(f)(Gain(a=3 * u.m, b=2.0).values())
    assert float(gradient.a) == pytest.approx(2 * 3.0 * 2.0)  # ∂f/∂a = 2ab
    assert float(gradient.b) == pytest.approx(3.0**2)  # ∂f/∂b = a²


class Field(OpticalSignal):
    def __init__(self, amplitude):
        self.amplitude = amplitude


class Attenuator(Element):
    transmission = Param(default=0.25, bounds=(0, 1))
    in_ = Input(Field)
    out = Output(Field)

    @staticmethod
    def transfer(values, inputs):
        return {"out": inputs["in"] * values.transmission**0.5}


@requires_jax
def test_transfer_functions_can_be_compiled_and_differentiated():
    import jax

    attenuator = Attenuator(transmission=0.25)
    amplitude = backend.convert(np.array([2.0, 4.0]), "jax")
    compiled = lumos.jit(Attenuator.transfer)
    out = compiled(attenuator.values(), {"in": amplitude})["out"]
    np.testing.assert_allclose(backend.to_numpy(out), [1.0, 2.0])

    def power(values):
        return (Attenuator.transfer(values, {"in": amplitude})["out"] ** 2).sum()

    # P = T * sum(A²) so dP/dT = sum(A²) = 4 + 16
    assert float(jax.grad(power)(attenuator.values()).transmission) == pytest.approx(20.0)
