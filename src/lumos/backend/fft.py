"""Two-dimensional FFTs on any backend.

Every function works on the backend of its input array: JAX arrays are transformed by JAX, CuPy arrays on
the GPU, and NumPy arrays by :mod:`scipy.fft`, which keeps single precision and uses
``lumos.config.fft_workers`` CPU threads.

The *centred* transforms :func:`fft2c` and :func:`ifft2c` assume that the origin of both the input and the
output grids is the pixel of index ``N // 2`` along each axis (convention of LUMOS, §9 of the specs). They
are the natural transforms for optics, where fields are centred on the optical axis.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import scipy.fft

from lumos._config import config
from lumos.backend._core import backend_of, get

__all__ = ["fft2", "fft2c", "fftfreq", "fftshift", "ifft2", "ifft2c", "ifftshift"]

Norm = Literal["backward", "ortho", "forward"]
Axes = tuple[int, int]


def _namespace(array: Any) -> tuple[str, Any]:
    name = backend_of(array)
    return name, get(name).xp


def fft2(array: Any, *, axes: Axes = (-2, -1), norm: Norm = "backward") -> Any:
    """Return the discrete Fourier transform of ``array`` over ``axes`` (origin at index 0)."""
    name, xp = _namespace(array)
    if name == "numpy":
        return scipy.fft.fft2(array, axes=axes, norm=norm, workers=config.fft_workers)
    return xp.fft.fftn(array, axes=axes, norm=norm)


def ifft2(array: Any, *, axes: Axes = (-2, -1), norm: Norm = "backward") -> Any:
    """Return the inverse discrete Fourier transform of ``array`` over ``axes`` (origin at index 0)."""
    name, xp = _namespace(array)
    if name == "numpy":
        return scipy.fft.ifft2(array, axes=axes, norm=norm, workers=config.fft_workers)
    return xp.fft.ifftn(array, axes=axes, norm=norm)


def fftshift(array: Any, *, axes: Axes = (-2, -1)) -> Any:
    """Move the zero-frequency sample from index 0 to index ``N // 2`` along ``axes``."""
    name, xp = _namespace(array)
    if name == "numpy":
        return np.fft.fftshift(array, axes=axes)
    return xp.fft.fftshift(array, axes=axes)


def ifftshift(array: Any, *, axes: Axes = (-2, -1)) -> Any:
    """Inverse of :func:`fftshift`: move the sample at index ``N // 2`` to index 0 along ``axes``."""
    name, xp = _namespace(array)
    if name == "numpy":
        return np.fft.ifftshift(array, axes=axes)
    return xp.fft.ifftshift(array, axes=axes)


def fftfreq(n: int, d: float = 1.0, *, backend: str | None = None) -> Any:
    """Return the sample frequencies of an ``n``-point FFT with sample spacing ``d``.

    The array is created with ``backend`` (the current backend by default), in FFT order (zero first).
    """
    return get(backend).xp.fft.fftfreq(n, d=d)


def fft2c(array: Any, *, axes: Axes = (-2, -1), norm: Norm = "backward") -> Any:
    """Return the centred Fourier transform of ``array`` (origins of both grids at index ``N // 2``)."""
    return fftshift(fft2(ifftshift(array, axes=axes), axes=axes, norm=norm), axes=axes)


def ifft2c(array: Any, *, axes: Axes = (-2, -1), norm: Norm = "backward") -> Any:
    """Return the centred inverse Fourier transform of ``array`` (inverse of :func:`fft2c`)."""
    return fftshift(ifft2(ifftshift(array, axes=axes), axes=axes, norm=norm), axes=axes)
