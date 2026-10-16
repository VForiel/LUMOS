"""Tests for precision settings, memory estimates and chunked computation (step P2.4).

Requirements: REQ-HPC-4 (memory estimate, automatic chunking over wavelengths, modes and fields when memory
is insufficient), REQ-HPC-5 (configurable precision, double by default).
Reference values come from the definition of the dtypes (itemsize) and from monolithic computations.
"""

import importlib.util
import math

import numpy as np
import psutil
import pytest
from hypothesis import given
from hypothesis import strategies as st

import lumos
from lumos import backend
from lumos.backend import fft
from lumos.backend.memory import available_memory, chunk_slices, dtypes, estimate_bytes, map_chunked

HAS_JAX = importlib.util.find_spec("jax") is not None
BACKENDS = [
    "numpy",
    pytest.param("jax", marks=pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")),
]


@pytest.fixture
def restore_precision():
    previous = lumos.config.precision
    yield
    lumos.config.precision = previous


# --- Precision -------------------------------------------------------------------------------


def test_default_precision_is_double():
    assert lumos.config.precision == "double"


def test_precision_is_validated(restore_precision):
    with pytest.raises(ValueError):
        lumos.config.precision = "half"


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize(
    ("precision", "real", "complex_"),
    [("double", np.float64, np.complex128), ("single", np.float32, np.complex64)],
)
def test_dtypes_follow_the_precision(name, precision, real, complex_, restore_precision):
    lumos.config.precision = precision
    real_dtype, complex_dtype = dtypes(backend=name)
    xp = backend.get(name).xp
    assert backend.to_numpy(xp.zeros(1, dtype=real_dtype)).dtype == real
    assert backend.to_numpy(xp.zeros(1, dtype=complex_dtype)).dtype == complex_


def test_explicit_precision_overrides_the_setting(restore_precision):
    lumos.config.precision = "double"
    real_dtype, complex_dtype = dtypes("single", backend="numpy")
    assert np.dtype(real_dtype) == np.float32
    assert np.dtype(complex_dtype) == np.complex64


# --- Memory estimates ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "dtype", "expected"),
    [
        ((2048, 2048), np.complex128, 2048 * 2048 * 16),
        ((32, 512, 512), np.complex64, 32 * 512 * 512 * 8),
        ((), np.float64, 8),
    ],
)
def test_estimate_bytes(shape, dtype, expected):
    assert estimate_bytes(shape, dtype) == expected


def test_available_memory_on_cpu_is_bounded_by_the_ram():
    memory = available_memory("numpy")
    assert 0 < memory <= psutil.virtual_memory().total


# --- Chunking ----------------------------------------------------------------------------------------


@given(
    n=st.integers(min_value=0, max_value=500),
    item_bytes=st.integers(min_value=1, max_value=10_000),
    budget=st.integers(min_value=1, max_value=100_000),
)
def test_chunk_slices_cover_everything_in_order_within_budget(n, item_bytes, budget):
    slices = chunk_slices(n, item_bytes, budget)
    covered = [i for s in slices for i in range(n)[s]]
    assert covered == list(range(n))
    limit = max(1, budget // item_bytes)  # at least one item per chunk, even over budget
    assert all(len(range(n)[s]) <= limit for s in slices)
    assert len(slices) == math.ceil(n / limit)


@pytest.mark.parametrize("name", BACKENDS)
def test_map_chunked_equals_the_monolithic_computation(name):
    rng = np.random.default_rng(1)
    cube = backend.convert(rng.standard_normal((10, 16, 16)) + 1j * rng.standard_normal((10, 16, 16)), name)
    calls = []

    def transform(block):
        calls.append(block.shape[0])
        return fft.fft2c(block)

    one_slice = 16 * 16 * 16  # bytes of one wavelength of complex128
    chunked = map_chunked(transform, cube, axis=0, budget_bytes=3 * one_slice)
    assert calls == [3, 3, 3, 1]
    assert backend.backend_of(chunked) == name
    np.testing.assert_allclose(
        backend.to_numpy(chunked), backend.to_numpy(fft.fft2c(cube)), rtol=0, atol=1e-12
    )


def test_map_chunked_along_another_axis():
    data = np.arange(24.0).reshape(2, 12)
    result = map_chunked(lambda block: block * 2, data, axis=1, budget_bytes=5 * 2 * 8)
    np.testing.assert_array_equal(result, data * 2)


def test_map_chunked_without_budget_fits_available_memory():
    data = np.ones((4, 8, 8))
    np.testing.assert_array_equal(map_chunked(lambda block: block + 1, data), data + 1)


def test_map_chunked_rejects_functions_changing_the_chunked_axis():
    with pytest.raises(ValueError):
        # Budget of 32 bytes: chunks of two rows of 2 x 8 bytes, which block[:1] reduces to one row.
        map_chunked(lambda block: block[:1], np.ones((4, 2)), axis=0, budget_bytes=32)
