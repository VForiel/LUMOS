"""Benchmarks of the propagation methods on the available backends (step P4.11, REQ-HPC-6).

Not run by default: ``pytest -m benchmark tests/benchmarks/test_bench_propagation.py``. GPU backends are
measured when a GPU is present (CuPy, or JAX on a GPU device); results are reported in
``docs/learn/performance.md``.
"""

import importlib.util

import numpy as np
import pytest
from astropy import units as u

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


def gaussian(n):
    grid = Grid(n, dx=2e-6 * u.m)
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    return Wavefront(np.exp(-(xx**2 + yy**2) / (n * 2e-6 / 8) ** 2), grid, 1 * u.um)


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize("method", sorted(CASES))
@pytest.mark.parametrize("n", [256, 1024])
def test_propagation(benchmark, name, method, n):
    distance, out_grid = CASES[method]
    wf = gaussian(n).to_backend(name)

    def run():
        out = propagate(wf, distance, output_grid=out_grid, method=method)
        field = out.field
        if hasattr(field, "block_until_ready"):
            field.block_until_ready()  # JAX is asynchronous: wait for the result
        return out

    clear_cache()
    run()  # warm-up: kernel cache and JAX compilation of the FFTs
    benchmark(run)
    assert backend.backend_of(run().field) == name
