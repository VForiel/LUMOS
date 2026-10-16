"""Benchmarks of the propagation methods on the available backends (step P4.11, REQ-HPC-6).

Not run by default: ``pytest -m benchmark tests/benchmarks/test_bench_propagation.py``. GPU backends are
measured when a GPU is present (CuPy, or JAX on a GPU device); results are reported in
``docs/learn/performance.md``. GPU backends are asynchronous: every run waits for its result.
"""

import importlib.util

import numpy as np
import pytest
from astropy import units as u

import lumos
from lumos import backend
from lumos.fields import Grid, Wavefront
from lumos.propagation import clear_cache, propagate

pytestmark = pytest.mark.benchmark

BACKENDS = ["numpy"] + (["jax"] if importlib.util.find_spec("jax") else [])
if importlib.util.find_spec("cupy"):
    BACKENDS.append("cupy")

CASES = {
    "asm": (0.5 * u.mm, None),
    "fresnel_tf": (0.5 * u.mm, None),
    "fresnel_mft": (5 * u.mm, Grid(256, dx=4e-6 * u.m)),
    "fraunhofer": (None, Grid(256, dx=2e-4 * u.rad)),
}


@pytest.fixture(params=["double", "single"])
def precision(request):
    previous = lumos.config.precision
    lumos.config.precision = request.param
    clear_cache()
    yield request.param
    lumos.config.precision = previous
    clear_cache()


def gaussian(n):
    grid = Grid(n, dx=2e-6 * u.m)
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    return Wavefront(np.exp(-(xx**2 + yy**2) / (n * 2e-6 / 8) ** 2), grid, 1 * u.um)


def wait(field):
    """Block until an asynchronous GPU computation of ``field`` is finished."""
    if hasattr(field, "block_until_ready"):
        field.block_until_ready()  # JAX
    elif backend.backend_of(field) == "cupy":
        import cupy

        cupy.cuda.get_current_stream().synchronize()
    return field


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize("method", sorted(CASES))
@pytest.mark.parametrize("n", [256, 1024])
def test_propagation(benchmark, precision, name, method, n):
    distance, out_grid = CASES[method]
    wf = gaussian(n).to_backend(name)

    def run():
        return wait(propagate(wf, distance, output_grid=out_grid, method=method).field)

    run()  # warm-up: kernel cache and JAX compilation of the FFTs
    benchmark(run)
    assert backend.backend_of(run()) == name


@pytest.mark.skipif(not importlib.util.find_spec("jax"), reason="JAX is not installed")
@pytest.mark.parametrize("method", sorted(CASES))
@pytest.mark.parametrize("n", [256, 1024])
def test_compiled_jax_propagation(benchmark, precision, method, n):
    import jax

    distance, out_grid = CASES[method]
    wf = gaussian(n).to_backend("jax")
    compiled = jax.jit(lambda w: propagate(w, distance, output_grid=out_grid, method=method).field)

    def run():
        return wait(compiled(wf))

    run()  # warm-up: compilation
    benchmark(run)
