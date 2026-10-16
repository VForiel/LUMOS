"""Numerical precision, memory estimates and chunked computation (REQ-HPC-4, REQ-HPC-5).

Large computations (a wavecube of hundreds of wavelengths, thousands of incoherent modes...) may not fit in
memory at once. Most of them are independent along one axis, so :func:`map_chunked` cuts the input along
that axis into chunks that fit a memory budget, applies the computation to each chunk and concatenates the
results, which are identical to those of a single computation.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import psutil

from lumos._config import config
from lumos.backend._core import backend_of, get

__all__ = ["available_memory", "chunk_slices", "dtypes", "estimate_bytes", "map_chunked"]

# Default fraction of the available memory given to the input of one chunk: the computation usually needs a
# few intermediate arrays of the same size (FFT buffers, kernels, outputs).
_DEFAULT_BUDGET_FRACTION = 0.25


def dtypes(precision: str | None = None, *, backend: str | None = None) -> tuple[Any, Any]:
    """Return the ``(real, complex)`` dtypes of ``backend`` for ``precision``.

    Parameters
    ----------
    precision
        ``"double"`` (float64 / complex128) or ``"single"`` (float32 / complex64); defaults to
        ``lumos.config.precision``.
    backend
        Backend name; defaults to the current backend.
    """
    precision = config.precision if precision is None else precision
    if precision not in ("double", "single"):
        raise ValueError(f"precision must be 'double' or 'single', got {precision!r}")
    xp = get(backend).xp
    if precision == "double":
        return xp.float64, xp.complex128
    return xp.float32, xp.complex64


def estimate_bytes(shape: tuple[int, ...], dtype: Any) -> int:
    """Return the size in bytes of an array of ``shape`` and ``dtype``."""
    return math.prod(shape) * np.dtype(dtype).itemsize


def available_memory(backend: str | None = None) -> int:
    """Return the memory in bytes currently available to ``backend`` (the current backend by default).

    CPU backends report the available RAM; GPU backends report the free memory of their first device.
    """
    name = get(backend).name
    if name == "jax":
        import jax

        device = jax.devices()[0]
        stats = device.memory_stats() if device.platform != "cpu" else None
        if stats and "bytes_limit" in stats:
            return int(stats["bytes_limit"] - stats.get("bytes_in_use", 0))
    elif name == "cupy":
        import cupy  # pyright: ignore[reportMissingImports]  (optional backend)

        return int(cupy.cuda.runtime.memGetInfo()[0])
    elif name == "torch":
        import torch  # pyright: ignore[reportMissingImports]  (optional backend)

        if torch.cuda.is_available():
            return int(torch.cuda.mem_get_info()[0])
    return int(psutil.virtual_memory().available)


def chunk_slices(n: int, item_bytes: int, budget_bytes: int) -> list[slice]:
    """Split ``range(n)`` into consecutive slices of at most ``budget_bytes // item_bytes`` items.

    Every chunk holds at least one item, even when a single item exceeds the budget.
    """
    for name, value, minimum in (
        ("n", n, 0),
        ("item_bytes", item_bytes, 1),
        ("budget_bytes", budget_bytes, 1),
    ):
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}, got {value}")
    limit = max(1, budget_bytes // item_bytes)
    return [slice(start, min(start + limit, n)) for start in range(0, n, limit)]


def map_chunked(
    function: Callable[[Any], Any],
    array: Any,
    *,
    axis: int = 0,
    budget_bytes: int | None = None,
) -> Any:
    """Apply ``function`` to chunks of ``array`` along ``axis`` and concatenate the results.

    ``function`` must treat the elements along ``axis`` independently and keep their number (for example a
    2-D FFT applied to every wavelength of a cube). The result equals ``function(array)``.

    Parameters
    ----------
    function
        Computation to apply; it receives and returns arrays of the same backend.
    array
        Input array.
    axis
        Axis along which the input is cut.
    budget_bytes
        Maximum size of the input of one chunk; defaults to a quarter of the memory available to the backend.
    """
    name = backend_of(array)
    xp = get(name).xp
    axis = axis % array.ndim
    n = array.shape[axis]
    if budget_bytes is None:
        budget_bytes = max(1, int(available_memory(name) * _DEFAULT_BUDGET_FRACTION))
    item_shape = tuple(size for index, size in enumerate(array.shape) if index != axis)
    item_bytes = max(1, estimate_bytes(item_shape, array.dtype))

    results = []
    for part in chunk_slices(n, item_bytes, budget_bytes) or [slice(0, 0)]:
        index = (slice(None),) * axis + (part,)
        result = function(array[index])
        expected = len(range(n)[part])
        if result.shape[axis] != expected:
            raise ValueError(
                f"the chunked function changed the length of axis {axis} "
                f"from {expected} to {result.shape[axis]}"
            )
        results.append(result)
    return results[0] if len(results) == 1 else xp.concat(results, axis=axis)
