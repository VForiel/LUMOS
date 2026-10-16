"""Tests for lumos.backend: array namespaces and conversions (step P2.1).

Requirements: REQ-HPC-1 (one code for NumPy, JAX, CuPy, PyTorch through the Array API; LUMOS data are
backend arrays), REQ-HPC-5 (complex128 by default, so conversions must not lose precision), NF-9 (fast
import: heavy backends are imported lazily). NumPy is the reference implementation.
"""

import importlib.util
import subprocess
import sys
import threading

import numpy as np
import pytest

from lumos import backend
from lumos.backend import Backend, BackendUnavailableError

HAS_JAX = importlib.util.find_spec("jax") is not None
requires_jax = pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")


@pytest.fixture
def numpy_default():
    """Make NumPy the default backend for the duration of a test."""
    previous = backend.get_default()
    backend.set_default("numpy")
    yield
    backend.set_default(previous)


# --- Discovery ---------------------------------------------------------------------------


def test_known_backends():
    assert backend.known() == ("numpy", "jax", "cupy", "torch")


def test_numpy_is_always_available():
    assert backend.available()[0] == "numpy"


def test_available_backends_match_installed_packages():
    for name in ("jax", "cupy", "torch"):
        installed = importlib.util.find_spec(name) is not None
        assert (name in backend.available()) == installed


def test_unknown_backend_name():
    with pytest.raises(ValueError, match="tensorflow"):
        backend.get("tensorflow")


@pytest.mark.parametrize("name", ["cupy", "torch"])
def test_known_but_missing_backend(name):
    if importlib.util.find_spec(name) is not None:
        pytest.skip(f"{name} is installed")
    with pytest.raises(BackendUnavailableError, match=name):
        backend.get(name)
    assert issubclass(BackendUnavailableError, ImportError)


def test_importing_lumos_backend_does_not_import_heavy_libraries():
    code = "import sys, lumos.backend; print(sorted(m for m in ('jax', 'cupy', 'torch') if m in sys.modules))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


# --- Backend objects -----------------------------------------------------------------------


def test_numpy_backend():
    numpy_backend = backend.get("numpy")
    assert isinstance(numpy_backend, Backend)
    assert numpy_backend.name == "numpy"
    array = numpy_backend.asarray([1.0, 2.0])
    assert isinstance(array, np.ndarray)
    assert numpy_backend.is_array(array)
    np.testing.assert_array_equal(numpy_backend.xp.sum(array), 3.0)


@requires_jax
def test_jax_backend_namespace_follows_the_array_api():
    jax_backend = backend.get("jax")
    xp = jax_backend.xp
    array = jax_backend.asarray(np.arange(6.0).reshape(2, 3))
    assert jax_backend.is_array(array)
    assert not jax_backend.is_array(np.zeros(2))
    np.testing.assert_allclose(backend.to_numpy(xp.sum(array, axis=0)), [3.0, 5.0, 7.0])


@requires_jax
@pytest.mark.parametrize("dtype", [np.float64, np.complex128, np.float32, np.complex64])
def test_jax_conversion_keeps_dtype_and_values(dtype):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 3)).astype(dtype)
    if np.iscomplexobj(data):
        data = data + 1j * rng.standard_normal((4, 3)).astype(dtype)
    converted = backend.convert(data, "jax")
    assert backend.backend_of(converted) == "jax"
    back = backend.to_numpy(converted)
    assert back.dtype == dtype
    np.testing.assert_array_equal(back, data)  # exact: no precision lost on the way


def test_to_numpy_accepts_numpy_arrays_and_python_values():
    data = np.ones(3)
    assert backend.to_numpy(data) is data
    np.testing.assert_array_equal(backend.to_numpy([1, 2]), np.array([1, 2]))


def test_backend_of_numpy_array_and_non_arrays():
    assert backend.backend_of(np.zeros(2)) == "numpy"
    with pytest.raises(TypeError):
        backend.backend_of([1, 2])


# --- Current backend ----------------------------------------------------------------------------


@requires_jax
def test_use_switches_and_restores(numpy_default):
    assert backend.current().name == "numpy"
    with backend.use("jax") as active:
        assert active.name == "jax"
        assert backend.current().name == "jax"
        with backend.use("numpy"):
            assert backend.current().name == "numpy"
        assert backend.current().name == "jax"
    assert backend.current().name == "numpy"


def test_use_restores_after_an_exception(numpy_default):
    with pytest.raises(RuntimeError), backend.use("numpy"):
        raise RuntimeError
    assert backend.current().name == "numpy"


@requires_jax
def test_use_is_local_to_the_current_thread(numpy_default):
    seen = []
    with backend.use("jax"):
        thread = threading.Thread(target=lambda: seen.append(backend.current().name))
        thread.start()
        thread.join()
    assert seen == ["numpy"]


def test_set_default_validates_the_name():
    with pytest.raises(ValueError):
        backend.set_default("tensorflow")
