"""Benchmarks of the centred 2-D FFT on the available backends (REQ-HPC-6).

Not run by default: ``pytest -m benchmark tests/benchmarks``.
"""

import importlib.util

import numpy as np
import pytest

from lumos import backend
from lumos.backend import fft

pytestmark = pytest.mark.benchmark

BACKENDS = ["numpy"] + (["jax"] if importlib.util.find_spec("jax") else [])


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize("n", [256, 1024, 2048])
def test_fft2c(benchmark, name, n):
    rng = np.random.default_rng(0)
    data = backend.convert(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)), name)

    def run():
        result = fft.fft2c(data)
        if name == "jax":
            result.block_until_ready()  # JAX is asynchronous: wait for the result
        return result

    run()  # warm-up (compilation for JAX)
    benchmark(run)
