"""Public propagation functions: :func:`propagate` and :func:`focus` (REQ-PROP-*).

Both apply a pure kernel (:mod:`lumos.propagation.rs`, :mod:`~lumos.propagation.fraunhofer`,
:mod:`~lumos.propagation.fresnel`, :mod:`~lumos.propagation.asm`) to every polarisation component, mode and
wavelength of a signal, and return a signal of the same type on the output grid.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

import numpy as np
from astropy import units as u

from lumos.backend._core import get
from lumos.fields.grid import Grid
from lumos.fields.incoherent import IncoherentBundle
from lumos.fields.wavecube import Wavecube
from lumos.fields.wavefront import Wavefront
from lumos.propagation._common import Method, namespace
from lumos.units import require_unit, to_si

__all__ = ["focus", "methods", "plan", "propagate", "register_method"]

logger = logging.getLogger(__name__)

Signal = Wavefront | Wavecube | IncoherentBundle
_METHODS: dict[str, Method] = {}


def register_method(method: Method) -> None:
    """Register a propagation method under ``method.name``."""
    _METHODS[method.name] = method


def methods() -> dict[str, Method]:
    """Return the registered propagation methods by name (``"auto"`` is the planner, not listed)."""
    return dict(_METHODS)


def _distance(value: Any) -> float | None:
    if value is None:
        return None
    z = float(to_si(require_unit(value, u.m, name="distance")))
    if not math.isfinite(z):
        raise ValueError("the distance must be finite; use distance=None for the far field")
    return z


def _method(name: str) -> Method:
    if name not in _METHODS:
        known = ", ".join(["auto", *sorted(_METHODS)])
        raise ValueError(f"unknown propagation method {name!r}; known methods: {known}")
    return _METHODS[name]


# --- Applying a kernel to every channel of a signal --------------------------------------------------


def _rebuild(template: Wavefront | Wavecube, grid: Grid, field: Any) -> Wavefront | Wavecube:
    """Signal with the metadata of ``template`` (wavelengths, weights, unit), ``field`` on ``grid``."""
    if isinstance(template, Wavefront):
        out = object.__new__(Wavefront)
        out._init(field, grid, template.wavelength, template.power_unit)
        return out
    cube = object.__new__(Wavecube)
    cube._init(field, grid, template.wavelengths, template.weights, template.power_unit)
    return cube


def _map_channels(
    signal: Signal, grid_out: Grid, apply: Callable[[Any, float], Any], *, max_memory: int | None = None
) -> Signal:
    """Apply ``apply(field, wavelength)`` to every wavelength of ``signal`` and rebuild it on ``grid_out``.

    Polarisation components and modes are leading axes processed in one call; wavelengths are processed one
    after the other with cached kernels. With ``max_memory`` (bytes), the modes of a bundle are processed in
    chunks whose padded working arrays fit in the budget.
    """
    if isinstance(signal, Wavefront):
        return _rebuild(signal, grid_out, apply(signal.field, signal.wavelength_si))
    if isinstance(signal, Wavecube):
        xp = namespace(signal.field)
        channels = [apply(signal.field[k], lam) for k, lam in enumerate(signal.wavelengths_si)]
        return _rebuild(signal, grid_out, xp.stack(channels))
    if isinstance(signal, IncoherentBundle):
        template = signal[0]
        xp = namespace(signal.field)

        def modes(block: Any) -> Any:
            if isinstance(template, Wavefront):
                return apply(block, template.wavelength_si)  # modes are a leading axis
            return xp.stack(
                [apply(block[:, k], lam) for k, lam in enumerate(template.wavelengths_si)], axis=1
            )

        if max_memory is None:
            field = modes(signal.field)
        else:
            per_mode = 4 * int(np.prod(signal.shape[1:])) * signal.field.dtype.itemsize  # padded by 2 x 2
            chunk = max(1, int(max_memory) // max(per_mode, 1))
            blocks = [modes(signal.field[i : i + chunk]) for i in range(0, signal.n_modes, chunk)]
            field = xp.concat(blocks, axis=0)
        return IncoherentBundle(
            field, _rebuild(template, grid_out, field[0]), mode_weights=signal.mode_weights
        )
    raise TypeError(f"cannot propagate a {type(signal).__name__}")


def _check_signal(signal: Any) -> Grid:
    if not isinstance(signal, Wavefront | Wavecube | IncoherentBundle):
        raise TypeError(
            f"cannot propagate a {type(signal).__name__}; expected a Wavefront, Wavecube or bundle"
        )
    grid = signal.grid
    if grid.plane != "spatial":
        raise ValueError("propagation starts from a field on a spatial grid")
    return grid


# --- Public API -----------------------------------------------------------------------------------------


def propagate(
    signal: Signal,
    distance: u.Quantity | None,
    *,
    output_grid: Grid | None = None,
    method: str = "auto",
    max_memory: int | None = None,
) -> Signal:
    r"""Propagate a field from the plane ``z = 0`` to the plane ``z = distance`` in free space (vacuum).

    Parameters
    ----------
    signal
        :class:`~lumos.fields.Wavefront`, :class:`~lumos.fields.Wavecube` or
        :class:`~lumos.fields.IncoherentBundle` on a spatial grid.
    distance
        Propagation distance (negative to back-propagate), or ``None`` for the far field: the output is then
        the angular spectrum on an angular grid of direction cosines, normalised so that
        :math:`\sum |U|^2 \Delta\Omega` is the power.
    output_grid
        Sampling of the output plane; defaults to the input grid (required for the far field).
    method
        ``"auto"`` (the planner chooses and logs the method) or one of :func:`methods`.
    max_memory
        Optional memory budget in bytes: the modes of an incoherent bundle are then propagated in chunks.

    Returns
    -------
    Signal of the same type on ``output_grid``.
    """
    grid_in = _check_signal(signal)
    z = _distance(distance)
    if z is None:
        if output_grid is None or output_grid.plane != "angular":
            raise ValueError("the far field (distance=None) needs an angular output_grid")
    elif output_grid is None:
        output_grid = grid_in
    elif output_grid.plane != "spatial":
        raise ValueError("a finite propagation distance needs a spatial output_grid")
    assert output_grid is not None
    if method == "auto":
        from lumos.propagation.planner import choose

        chosen = choose(signal, z, output_grid)
    else:
        chosen = _method(method)
    _check_method(chosen, grid_in, output_grid, z)
    xp = namespace(signal.field)
    grid_out = output_grid

    def apply(field: Any, wavelength: float) -> Any:
        return chosen.kernel(field, grid_in, grid_out, wavelength, z, xp)

    return _map_channels(signal, output_grid, apply, max_memory=max_memory)


def plan(signal: Signal, distance: u.Quantity | None, *, output_grid: Grid | None = None) -> Any:
    """Return the :class:`~lumos.propagation.planner.Plan` that :func:`propagate` would follow."""
    from lumos.propagation.planner import plan_for

    grid_in = _check_signal(signal)
    z = _distance(distance)
    if output_grid is None:
        if z is None:
            raise ValueError("the far field (distance=None) needs an angular output_grid")
        output_grid = grid_in
    return plan_for(signal, z, output_grid)


def _check_method(method: Method, grid_in: Grid, grid_out: Grid, z: float | None) -> None:
    if z is None and not method.far_field:
        raise ValueError(f"the {method.name!r} method does not compute far fields")
    if z is not None and not method.finite_distance:
        raise ValueError(f"the {method.name!r} method needs distance=None (far field)")
    if not method.any_output_grid and grid_out != grid_in:
        raise ValueError(f"the {method.name!r} method only computes the field on the input grid")


def focus(
    signal: Signal,
    focal_length: u.Quantity,
    output_grid: Grid,
    *,
    method: str = "auto",
) -> Signal:
    r"""Field in the focal plane of a thin lens of focal length ``f`` placed in the plane of ``signal``.

    The lens multiplies the field by :math:`\exp(-ik r^2 / 2f)` and the field then propagates over :math:`f`.
    With ``method="fraunhofer"`` (the default choice of the planner) this is evaluated exactly as

    .. math:: U_f(x, y) = \frac{e^{ikf}}{i\lambda f} e^{ik(x^2 + y^2)/2f}\,
              \tilde U\!\left(\frac{x}{\lambda f}, \frac{y}{\lambda f}\right),

    the Fresnel integral whose input quadratic phase the lens cancels; other methods apply the lens phase and
    propagate. The output sampling is arbitrary (matrix Fourier transform).
    """
    grid_in = _check_signal(signal)
    f = _distance(focal_length)
    if f is None or f <= 0:
        raise ValueError("focal_length must be a positive length")
    if output_grid.plane != "spatial":
        raise ValueError("the focal plane needs a spatial output_grid")
    chosen = "fraunhofer" if method == "auto" else method
    if chosen == "fraunhofer":
        return propagate(signal, focal_length, output_grid=output_grid, method="fraunhofer")
    xx, yy = grid_in.coords(backend=signal.backend)
    xp = get(signal.backend).xp

    def lens(field: Any, wavelength: float) -> Any:
        k = 2 * math.pi / wavelength
        return field * xp.exp(-1j * k * (xx**2 + yy**2) / (2 * f))

    lensed = _map_channels(signal, grid_in, lens)
    return propagate(lensed, focal_length, output_grid=output_grid, method=chosen)
