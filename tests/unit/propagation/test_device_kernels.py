"""Cached propagation kernels stay on the device (step P4.7, REQ-HPC-6).

A cached kernel (transfer function, MFT matrix, chirp) is converted to the array type and dtype of the
backend once: copying a 2048 × 2048 complex128 kernel (64 MiB) from the host to a GPU at every propagation
costs about 10 ms over PCIe 3, as much as the propagation itself. Arrays created while JAX traces a
compiled function are tracers and must not be kept.
"""

import importlib.util

import numpy as np
import pytest
from astropy import units as u

from lumos import backend
from lumos.backend.memory import dtypes
from lumos.fields import Grid, Wavefront
from lumos.propagation import clear_cache, propagate
from lumos.propagation._common import KERNEL_CACHE

HAS_JAX = importlib.util.find_spec("jax") is not None
HAS_CUPY = importlib.util.find_spec("cupy") is not None
GRID = Grid(32, dx=2e-6 * u.m)


def build():
    return np.exp(1j * np.arange(16.0)).reshape(4, 4)


def kernel_twice(name, precision):
    clear_cache()
    xp = backend.get(name).xp
    dtype = dtypes(precision, backend=name)[1]
    first = KERNEL_CACHE.get(("probe",), build, xp, dtype)
    second = KERNEL_CACHE.get(("probe",), build, xp, dtype)
    clear_cache()
    return first, second


@pytest.mark.parametrize("precision", ["double", "single"])
def test_numpy_kernels_are_converted_once(precision):
    first, second = kernel_twice("numpy", precision)
    assert second is first
    assert first.dtype == dtypes(precision, backend="numpy")[1]


@pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")
def test_jax_kernels_are_converted_once():
    first, second = kernel_twice("jax", "double")
    assert second is first


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUPY, reason="CuPy is not installed")
def test_cupy_kernels_are_converted_once():
    first, second = kernel_twice("cupy", "double")
    assert second is first


@pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")
def test_kernels_built_during_a_jax_trace_are_not_kept():
    import jax

    clear_cache()
    rng = np.random.default_rng(0)
    wf = Wavefront(rng.standard_normal(GRID.shape) + 0j, GRID, 1 * u.um).to_backend("jax")
    compiled = jax.jit(lambda w: propagate(w, 0.1 * u.mm, method="asm").field)
    traced = compiled(wf)
    eager = propagate(wf, 0.1 * u.mm, method="asm").field  # reuses the cached kernel outside the trace
    np.testing.assert_allclose(np.asarray(eager), np.asarray(traced), rtol=1e-12, atol=1e-15)
    clear_cache()
