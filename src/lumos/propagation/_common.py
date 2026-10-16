"""Shared helpers of the propagation kernels.

Kernels are pure functions of backend arrays: ``kernel(field, grid_in, grid_out, wavelength, z, xp)`` where
``field`` has shape ``(..., ny, nx)`` (leading axes for polarisation, modes...), ``wavelength`` and ``z`` are
SI floats, and ``xp`` is the Array API namespace of the field. They return the field on ``grid_out``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from lumos.backend._core import backend_of, get
from lumos.backend.memory import dtypes
from lumos.fields.grid import Grid

__all__ = ["Kernel", "Method", "crop", "frequencies", "mft_matrix", "pad", "same_sampling"]


class Kernel(Protocol):
    """Signature of a propagation kernel."""

    def __call__(
        self, field: Any, grid_in: Grid, grid_out: Grid, wavelength: float, z: float | None, xp: Any
    ) -> Any: ...


@dataclass(frozen=True)
class Method:
    """A propagation method.

    Attributes
    ----------
    name
        Identifier used by :func:`lumos.propagation.propagate`.
    kernel
        Pure kernel computing the output field.
    any_output_grid
        Whether the method evaluates the field on any output grid (otherwise only on the input grid).
    far_field
        Whether the method handles ``distance=None`` (angular far field).
    finite_distance
        Whether the method handles finite distances.
    description
        One-line summary.
    """

    name: str
    kernel: Kernel
    any_output_grid: bool
    far_field: bool
    finite_distance: bool
    description: str


def same_sampling(a: Grid, b: Grid) -> bool:
    """Whether two grids sample the same points."""
    return a == b


def complex_dtype(field: Any) -> Any:
    """Complex dtype matching the precision of ``field``'s backend."""
    return dtypes(backend=backend_of(field))[1]


def axis_coordinates(n: int, step: float, center: float) -> np.ndarray:
    """Pixel centres ``(i - n//2)·step + center`` (§9)."""
    return (np.arange(n) - n // 2) * step + center


def mft_matrix(frequencies: np.ndarray, coordinates: np.ndarray, sign: float, xp: Any, dtype: Any) -> Any:
    """Matrix ``exp(sign · 2πi f_o x_i)`` of shape ``(n_out, n_in)`` as a backend array."""
    phase = sign * 2.0 * math.pi * np.outer(frequencies, coordinates)
    return xp.asarray(np.exp(1j * phase), dtype=dtype)


def mft_axes(
    field: Any,
    out_x: np.ndarray,
    out_y: np.ndarray,
    in_x: np.ndarray,
    in_y: np.ndarray,
    sign: float,
    xp: Any,
) -> Any:
    """Separable sum ``Σ_{i,j} F[..., j, i] exp(sign 2πi (u_o x_i + v_p y_j))`` (no area factor)."""
    dtype = complex_dtype(field)
    ex = mft_matrix(out_x, in_x, sign, xp, dtype)  # (n_out_x, n_in_x)
    ey = mft_matrix(out_y, in_y, sign, xp, dtype)  # (n_out_y, n_in_y)
    return xp.matmul(xp.matmul(ey, field), xp.matrix_transpose(ex))


def mft(field: Any, ux: np.ndarray, uy: np.ndarray, grid: Grid, sign: float, xp: Any) -> Any:
    """Matrix Fourier transform ``Σ U(x, y) exp(sign 2πi (u x + v y)) dx dy`` at frequencies ``(ux, uy)``."""
    x = axis_coordinates(grid.nx, grid.dx_si, grid.x0_si)
    y = axis_coordinates(grid.ny, grid.dy_si, grid.y0_si)
    return mft_axes(field, ux, uy, x, y, sign, xp) * (grid.dx_si * grid.dy_si)


def frequencies(n: int, step: float) -> np.ndarray:
    """Centred DFT frequencies ``(k - n//2) / (n·step)``."""
    return (np.arange(n) - n // 2) / (n * step)


def pad(field: Any, factor: int, xp: Any) -> tuple[Any, tuple[int, int]]:
    """Zero-pad the last two axes by ``factor``, keeping the central pixel at index ``N // 2``."""
    ny, nx = field.shape[-2:]
    py, px = factor * ny, factor * nx
    oy, ox = py // 2 - ny // 2, px // 2 - nx // 2
    out = xp.zeros((*field.shape[:-2], py, px), dtype=field.dtype)
    if hasattr(out, "at"):  # JAX arrays are immutable
        out = out.at[..., oy : oy + ny, ox : ox + nx].set(field)
    else:
        out[..., oy : oy + ny, ox : ox + nx] = field
    return out, (oy, ox)


def crop(field: Any, offsets: tuple[int, int], shape: tuple[int, int]) -> Any:
    """Inverse of :func:`pad`."""
    oy, ox = offsets
    return field[..., oy : oy + shape[0], ox : ox + shape[1]]


def namespace(field: Any) -> Any:
    """Array API namespace of ``field``."""
    return get(backend_of(field)).xp
