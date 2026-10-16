"""Tests for lumos.backend.fft (step P2.2).

Requirements: REQ-HPC-1 (same code for every backend), REQ-HPC-2 (number of FFT threads), REQ-HPC-5
(single precision stays single precision), §9 of the specs (centre pixel at index N // 2).

References: numpy.fft (independent implementation), Parseval's theorem with orthonormal normalisation, and
the property that the Fourier transform of a real even function is real.
"""

import importlib.util

import numpy as np
import pytest
import scipy.fft

import lumos
from lumos import backend
from lumos.backend import fft

HAS_JAX = importlib.util.find_spec("jax") is not None
BACKENDS = [
    "numpy",
    pytest.param("jax", marks=pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")),
]


def random_field(shape, dtype=np.complex128, seed=0):
    rng = np.random.default_rng(seed)
    real_dtype = np.float32 if dtype in (np.complex64, np.float32) else np.float64
    data = rng.standard_normal(shape).astype(real_dtype)
    if np.dtype(dtype).kind == "c":
        data = (data + 1j * rng.standard_normal(shape).astype(real_dtype)).astype(dtype)
    return data


def on(name, array):
    return backend.convert(array, name)


# --- Agreement with numpy.fft -------------------------------------------------------------


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
@pytest.mark.parametrize("shape", [(8, 8), (7, 12), (5, 6, 9)])
def test_fft2_matches_numpy(name, norm, shape):
    data = random_field(shape)
    result = fft.fft2(on(name, data), norm=norm)
    assert backend.backend_of(result) == name
    np.testing.assert_allclose(backend.to_numpy(result), np.fft.fft2(data, norm=norm), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", BACKENDS)
def test_ifft2_matches_numpy(name):
    data = random_field((6, 10))
    np.testing.assert_allclose(
        backend.to_numpy(fft.ifft2(on(name, data))), np.fft.ifft2(data), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("name", BACKENDS)
def test_custom_axes(name):
    cube = random_field((4, 6, 3))
    result = backend.to_numpy(fft.fft2(on(name, cube), axes=(0, 1)))
    np.testing.assert_allclose(result, np.fft.fft2(cube, axes=(0, 1)), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize("n", [8, 9])
def test_shifts_match_numpy(name, n):
    data = random_field((n, n + 1))
    np.testing.assert_array_equal(
        backend.to_numpy(fft.fftshift(on(name, data))), np.fft.fftshift(data, axes=(-2, -1))
    )
    np.testing.assert_array_equal(
        backend.to_numpy(fft.ifftshift(on(name, data))), np.fft.ifftshift(data, axes=(-2, -1))
    )


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize("n", [8, 9])
def test_fftfreq_matches_numpy(name, n):
    with backend.use(name):
        frequencies = fft.fftfreq(n, d=0.5)
    assert backend.backend_of(frequencies) == name
    np.testing.assert_allclose(backend.to_numpy(frequencies), np.fft.fftfreq(n, d=0.5), rtol=1e-15)


# --- Mathematical properties ------------------------------------------------------------------------


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize("shape", [(16, 16), (15, 15), (16, 15)])
def test_centred_transform_of_real_even_function_is_real(name, shape):
    # Gaussian centred on pixel N // 2 (convention of §9): real and even about the centre pixel.
    ny, nx = shape
    y = np.arange(ny) - ny // 2
    x = np.arange(nx) - nx // 2
    gaussian = np.exp(-(y[:, None] ** 2 + x[None, :] ** 2) / (2 * 2.0**2)).astype(np.complex128)
    result = backend.to_numpy(fft.fft2c(on(name, gaussian)))
    assert np.max(np.abs(result.imag)) < 1e-12 * np.max(np.abs(result.real))
    assert np.argmax(np.abs(result)) == np.ravel_multi_index(
        (ny // 2, nx // 2), shape
    )  # zero frequency at centre


@pytest.mark.parametrize("name", BACKENDS)
def test_centred_delta_has_flat_spectrum(name):
    delta = np.zeros((9, 8), dtype=np.complex128)
    delta[9 // 2, 8 // 2] = 1.0
    np.testing.assert_allclose(backend.to_numpy(fft.fft2c(on(name, delta))), np.ones((9, 8)), atol=1e-15)


@pytest.mark.parametrize("name", BACKENDS)
def test_parseval_with_orthonormal_normalisation(name):
    data = random_field((12, 10))
    spectrum = backend.to_numpy(fft.fft2c(on(name, data), norm="ortho"))
    assert np.sum(np.abs(spectrum) ** 2) == pytest.approx(np.sum(np.abs(data) ** 2), rel=1e-12)


@pytest.mark.parametrize("name", BACKENDS)
def test_centred_round_trip(name):
    data = random_field((7, 10))
    np.testing.assert_allclose(backend.to_numpy(fft.ifft2c(fft.fft2c(on(name, data)))), data, atol=1e-13)


# --- Precision ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (np.complex64, np.complex64),
        (np.complex128, np.complex128),
        (np.float32, np.complex64),
        (np.float64, np.complex128),
    ],
)
def test_precision_is_preserved(name, dtype, expected):
    result = fft.fft2(on(name, random_field((8, 8), dtype=dtype)))
    assert backend.to_numpy(result).dtype == expected


# --- Threads ------------------------------------------------------------------------------------------------


def test_fft_workers_setting_is_validated():
    previous = lumos.config.fft_workers
    try:
        lumos.config.fft_workers = 2
        assert lumos.config.fft_workers == 2
        lumos.config.fft_workers = -1  # all cores
        with pytest.raises(ValueError):
            lumos.config.fft_workers = 0
        with pytest.raises(TypeError):
            lumos.config.fft_workers = 1.5
    finally:
        lumos.config.fft_workers = previous


def test_numpy_transforms_use_the_configured_workers(monkeypatch):
    calls = []
    original = scipy.fft.fft2

    def spy(*args, **kwargs):
        calls.append(kwargs.get("workers"))
        return original(*args, **kwargs)

    monkeypatch.setattr(scipy.fft, "fft2", spy)
    previous = lumos.config.fft_workers
    try:
        lumos.config.fft_workers = 3
        fft.fft2(np.ones((4, 4), dtype=complex))
    finally:
        lumos.config.fft_workers = previous
    assert calls == [3]


def test_unknown_config_attribute_is_rejected():
    with pytest.raises(AttributeError):
        lumos.config.fft_worker = 2  # type: ignore[attr-defined]
