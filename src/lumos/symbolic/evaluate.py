r"""Evaluation of symbolic fields on grids (step P10.2, REQ-SYM-2).

:func:`to_wavefront` samples a :class:`~lumos.symbolic.SymbolicField` on a :class:`~lumos.fields.Grid` and
:func:`to_wavecube` on a grid and a spectral axis. The expression is compiled once with
:func:`sympy.lambdify` for the current array backend (NumPy and SciPy functions; ``jax`` and ``cupy``
printers when these backends are used), with the parameters as arguments, so that changing a parameter value
reuses the compiled function (:func:`cache_info`).

Anti-aliasing: with ``supersample = s``, every pixel is the mean of the field at :math:`s \times s` sub-pixel
centres, a midpoint quadrature of the integral of the field over the pixel. It turns sharp edges
(``Heaviside``, ``Piecewise``) into grey pixels whose sum converges to the area of the shape.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp
from astropy import units as u

from lumos.backend import current, get
from lumos.fields.grid import Grid
from lumos.fields.wavecube import Wavecube
from lumos.fields.wavefront import Wavefront
from lumos.symbolic.field import SymbolicField
from lumos.symbolic.symbols import lam, t, theta_x, theta_y, x, y
from lumos.units import require_unit, to_si

__all__ = ["CacheInfo", "cache_info", "clear_cache", "compile_field", "to_wavecube", "to_wavefront"]

logger = logging.getLogger("lumos.symbolic")

_NUMPY_MODULES: list[Any] = [{"Heaviside": lambda v, h=0.5: np.heaviside(v, h)}, "scipy", "numpy"]
_PRINTERS = {"jax": "jax", "cupy": "cupy"}


@dataclass(frozen=True)
class CacheInfo:
    """Statistics of the cache of compiled expressions."""

    hits: int
    misses: int
    size: int


_cache: dict[tuple[str, ...], Callable[..., Any]] = {}
_stats = {"hits": 0, "misses": 0}


def cache_info() -> CacheInfo:
    """Return the statistics of the cache of compiled expressions."""
    return CacheInfo(_stats["hits"], _stats["misses"], len(_cache))


def clear_cache() -> None:
    """Empty the cache of compiled expressions."""
    _cache.clear()
    _stats["hits"] = _stats["misses"] = 0


def _variables(plane: str) -> tuple[sp.Symbol, sp.Symbol]:
    return (x, y) if plane == "spatial" else (theta_x, theta_y)


def compile_field(field: SymbolicField, backend: str) -> tuple[Callable[..., Any], list[sp.Symbol]]:
    """Return the compiled function ``f(u, v, lam, t, *parameters)`` of ``field`` and its parameter order."""
    parameters = sorted(field.parameters, key=str)
    u_sym, v_sym = _variables(field.plane)
    printer = _PRINTERS.get(backend, "numpy")
    key = (sp.srepr(field.expression), field.plane, printer, *(str(p) for p in parameters))
    function = _cache.get(key)
    if function is not None:
        _stats["hits"] += 1
        return function, parameters
    _stats["misses"] += 1
    arguments = (u_sym, v_sym, lam, t, *parameters)
    modules: Any = _NUMPY_MODULES if printer == "numpy" else printer
    function = sp.lambdify(arguments, field.expression, modules=modules)
    _cache[key] = function
    return function, parameters


def _sample(
    function: Callable[..., Any],
    xp: Any,
    xx: Any,
    yy: Any,
    wavelength: float,
    time: float,
    arguments: Any,
    dx: float,
) -> Any:
    """Evaluate ``function`` at the points, taking the limits of removable singularities.

    Non-finite values (such as the 0/0 of 2J₁(v)/v on the axis) are replaced by the value at a point shifted
    by 1e-7 pixel, whose error is of the order of the square of the shift.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        sample = function(xx, yy, wavelength, time, *arguments)
        sample = sample + 0 * (xx + yy)
        bad = ~xp.isfinite(sample)
        if not bool(xp.any(bad)):
            return sample
        shifted = function(xx + 1e-7 * dx, yy + 1e-7 * dx, wavelength, time, *arguments) + 0 * (xx + yy)
    return xp.where(bad, shifted, sample)


def _evaluate(field: SymbolicField, grid: Grid, wavelength: float, *, supersample: int, time: float) -> Any:
    if grid.plane != field.plane:
        raise ValueError(f"a {field.plane} field cannot be sampled on a {grid.plane} grid (plane mismatch)")
    backend = current()
    values = field.si_parameters()
    s = max(int(supersample), 1)
    offsets = (np.arange(s) + 0.5) / s - 0.5
    names = [backend.name, "numpy"] if backend.name in _PRINTERS else ["numpy"]
    for name in names:
        selected = get(name)
        function, parameters = compile_field(field, name)
        arguments = [values[p] for p in parameters]
        xp = selected.xp
        xa, ya = grid.axes(backend=name)
        try:
            total: Any = 0
            for oy in offsets:
                for ox in offsets:
                    xx = xp.reshape(xa + ox * grid.dx_si, (1, -1))
                    yy = xp.reshape(ya + oy * grid.dy_si, (-1, 1))
                    sample = _sample(function, xp, xx, yy, wavelength, time, arguments, grid.dx_si)
                    total = total + sample + 0 * (xx + yy)  # broadcast constants to the grid
            result = total / (s * s)
        except (NameError, TypeError, AttributeError) as error:
            if name == "numpy":
                raise
            logger.info("symbolic field not compiled for %s (%s): evaluated with NumPy", name, error)
            continue
        return result if name == backend.name else backend.asarray(np.asarray(result))
    raise RuntimeError("unreachable")  # pragma: no cover


def _wavelength_si(field: SymbolicField, wavelength: u.Quantity | None) -> float:
    if wavelength is not None:
        q = require_unit(wavelength, u.m, name="wavelength", equivalencies=u.spectral())
        return float(to_si(q, u.m, equivalencies=u.spectral()))
    if field.wavelength is None:
        raise ValueError("this field depends on lambda: give the wavelength of the wavefront")
    return float(to_si(field.wavelength, u.m, equivalencies=u.spectral()))


def to_wavefront(
    field: SymbolicField,
    grid: Grid,
    *,
    wavelength: u.Quantity | None = None,
    supersample: int = 1,
    time: u.Quantity | None = None,
) -> Wavefront:
    """Sample ``field`` on ``grid`` (pixel averages over ``supersample`` × ``supersample`` sub-pixels).

    Parameters
    ----------
    field
        Symbolic field.
    grid
        Grid of the plane of the field.
    wavelength
        Wavelength (default: the wavelength of the field); needed when the expression depends on ``lam``.
    supersample
        Sub-pixels per pixel side.
    time
        Value of ``t`` (default 0 s).
    """
    lam_si = _wavelength_si(field, wavelength)
    seconds = 0.0 if time is None else float(to_si(require_unit(time, u.s, name="time")))
    values = _evaluate(field, grid, lam_si, supersample=supersample, time=seconds)
    return Wavefront(values, grid, lam_si * u.m, power_unit=field.power_unit)


def to_wavecube(
    field: SymbolicField,
    grid: Grid,
    wavelengths: u.Quantity,
    *,
    weights: Any = None,
    supersample: int = 1,
    time: u.Quantity | None = None,
) -> Wavecube:
    """Sample ``field`` on ``grid`` at each of ``wavelengths`` (an achromatic field is repeated)."""
    lam_si = np.atleast_1d(np.asarray(to_si(require_unit(wavelengths, u.m, name="wavelengths"))))
    seconds = 0.0 if time is None else float(to_si(require_unit(time, u.s, name="time")))
    backend = current()
    planes = [_evaluate(field, grid, float(value), supersample=supersample, time=seconds) for value in lam_si]
    stacked = backend.xp.stack(planes)
    return Wavecube(stacked, grid, lam_si * u.m, weights=weights, power_unit=field.power_unit)
