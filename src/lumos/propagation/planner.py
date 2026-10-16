r"""Automatic choice of the propagation method (REQ-PROP-4, step P4.6, ADR-0005).

The planner inspects the field itself, not only the grids: it measures

- the **spatial support**, the smallest box holding :math:`1 - 10^{-6}` of the power;
- the **spectral support**, the highest spatial frequency below which :math:`1 - 10^{-6}` of the spectral
  power lies along each axis;

and applies the rules of ADR-0005, in order, for every wavelength of the signal:

1. far field (``distance=None``): ``"fraunhofer"``;
2. back-propagation (:math:`z < 0`): ``"asm"``, whose transfer function is exactly invertible;
3. ``"asm"`` (exact) if the spectral support lies inside the Matsushima band limit and the output plane lies
   inside the zero-padded window;
4. Fresnel methods if the neglected phase :math:`k\rho^4/8z^3 \le 0.01` rad for the largest lateral distance
   :math:`\rho` between the support and the output plane:

   - ``"fresnel_mft"`` if the chirped field is sampled (its local frequencies stay below Nyquist) and the
     output frequencies do not reach the periodic replicas;
   - otherwise ``"fresnel_ir"`` (:math:`\Delta x \le \lambda z / L`) or ``"fresnel_tf"`` on the input grid;

5. ``"fraunhofer"`` at finite distance if, in addition, the input quadratic phase
   :math:`k\rho_\mathrm{in}^2 / 2z \le 0.05` rad;
6. ``"rs"``, the direct sum, valid everywhere but expensive (a warning is logged above
   :data:`RS_WARNING_COST` operations).

The choice and its reason are logged at the INFO level on the ``lumos.propagation`` logger, and
:func:`plan` returns them without propagating.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from lumos.backend._core import get
from lumos.fields.grid import Grid
from lumos.fields.incoherent import IncoherentBundle
from lumos.fields.wavecube import Wavecube
from lumos.fields.wavefront import Wavefront
from lumos.propagation._common import Method, axis_coordinates, frequencies
from lumos.propagation.asm import band_limit

__all__ = ["RS_WARNING_COST", "Plan", "choose", "plan_for"]

logger = logging.getLogger("lumos.propagation")

#: Fraction of the power left outside the measured supports.
SUPPORT_TOLERANCE = 1e-6
#: Largest neglected Fresnel phase (rad).
FRESNEL_PHASE = 0.01
#: Largest neglected input quadratic phase for the finite-distance Fraunhofer form (rad).
FRAUNHOFER_PHASE = 0.05
#: Direct sums above this number of complex operations are logged as warnings.
RS_WARNING_COST = 1e10
_MAX_SAMPLES = 16


@dataclass(frozen=True)
class Plan:
    """Outcome of the planner.

    Attributes
    ----------
    method
        Name of the chosen method.
    reason
        Why it was chosen.
    cost
        Estimated number of complex operations per wavelength and component.
    output_grid
        Grid on which the field will be computed.
    """

    method: str
    reason: str
    cost: float
    output_grid: Grid


@dataclass(frozen=True)
class _Support:
    x: tuple[float, float]
    y: tuple[float, float]
    fx: float
    fy: float


def _samples(signal: Any) -> tuple[np.ndarray, list[float]]:
    """Up to 16 representative 2D fields (NumPy) and the wavelengths of the signal."""
    field = np.asarray(get(signal.backend).to_numpy(signal.field))
    planes = field.reshape((-1, *signal.grid.shape))
    if planes.shape[0] > _MAX_SAMPLES:
        planes = planes[np.linspace(0, planes.shape[0] - 1, _MAX_SAMPLES).astype(int)]
    if isinstance(signal, Wavefront):
        wavelengths = [signal.wavelength_si]
    elif isinstance(signal, Wavecube):
        wavelengths = list(signal.wavelengths_si)
    elif isinstance(signal, IncoherentBundle):
        member = signal[0]
        wavelengths = [member.wavelength_si] if isinstance(member, Wavefront) else list(member.wavelengths_si)
    else:  # pragma: no cover - checked by propagate()
        raise TypeError(type(signal).__name__)
    return planes, wavelengths


def _interval(weights: np.ndarray, coordinates: np.ndarray) -> tuple[float, float]:
    total = weights.sum()
    if total <= 0:
        return 0.0, 0.0
    cumulative = np.cumsum(weights) / total
    tail = SUPPORT_TOLERANCE / 2
    low = int(np.searchsorted(cumulative, tail))
    high = int(np.searchsorted(cumulative, 1 - tail))
    high = min(high, len(coordinates) - 1)
    return float(coordinates[low]), float(coordinates[high])


def _support(planes: np.ndarray, grid: Grid) -> _Support:
    intensity = (np.abs(planes) ** 2).sum(axis=0)
    x = axis_coordinates(grid.nx, grid.dx_si, grid.x0_si)
    y = axis_coordinates(grid.ny, grid.dy_si, grid.y0_si)
    spectrum = (
        np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(planes, axes=(-2, -1))), axes=(-2, -1))) ** 2
    ).sum(axis=0)
    fx = frequencies(grid.nx, grid.dx_si)
    fy = frequencies(grid.ny, grid.dy_si)
    fx_low, fx_high = _interval(spectrum.sum(axis=0), fx)
    fy_low, fy_high = _interval(spectrum.sum(axis=1), fy)
    return _Support(
        x=_interval(intensity.sum(axis=0), x),
        y=_interval(intensity.sum(axis=1), y),
        fx=max(abs(fx_low), abs(fx_high)),
        fy=max(abs(fy_low), abs(fy_high)),
    )


def _cost_fft(grid: Grid, padding: int = 2) -> float:
    n = padding**2 * grid.size
    return 5.0 * n * math.log2(max(n, 2))


def _cost_mft(grid_in: Grid, grid_out: Grid) -> float:
    return float(grid_in.ny * grid_in.nx * grid_out.nx + grid_in.ny * grid_out.nx * grid_out.ny)


def _bounds_si(grid: Grid) -> tuple[tuple[float, float], tuple[float, float]]:
    """Outer pixel edges ``((xmin, xmax), (ymin, ymax))`` in SI units."""
    x = axis_coordinates(grid.nx, grid.dx_si, grid.x0_si)
    y = axis_coordinates(grid.ny, grid.dy_si, grid.y0_si)
    return (
        (float(x[0] - grid.dx_si / 2), float(x[-1] + grid.dx_si / 2)),
        (float(y[0] - grid.dy_si / 2), float(y[-1] + grid.dy_si / 2)),
    )


def _decide(
    support: _Support, grid_in: Grid, grid_out: Grid, wavelength: float, z: float
) -> tuple[str, str, float]:
    k = 2 * math.pi / wavelength
    ox, oy = _bounds_si(grid_out)
    same = grid_out == grid_in

    # Rule 3: band-limited angular spectrum.
    limit_x = band_limit(grid_in.nx, grid_in.dx_si, wavelength, z)
    limit_y = band_limit(grid_in.ny, grid_in.dy_si, wavelength, z)
    half_x, half_y = grid_in.nx * grid_in.dx_si, grid_in.ny * grid_in.dy_si  # padded half-widths
    inside = (
        ox[0] >= grid_in.x0_si - half_x
        and ox[1] <= grid_in.x0_si + half_x
        and oy[0] >= grid_in.y0_si - half_y
        and oy[1] <= grid_in.y0_si + half_y
    )
    if support.fx <= limit_x and support.fy <= limit_y and inside:
        cost = 2 * _cost_fft(grid_in) + (
            0.0
            if same
            else _cost_mft(Grid(2 * grid_in.nx, 2 * grid_in.ny, dx=grid_in.dx, dy=grid_in.dy), grid_out)
        )
        return "asm", "the spectrum of the field lies inside the angular-spectrum band limit", cost

    # Rule 4: Fresnel.
    rho_x = max(abs(ox[1] - support.x[0]), abs(ox[0] - support.x[1]))
    rho_y = max(abs(oy[1] - support.y[0]), abs(oy[0] - support.y[1]))
    rho2 = rho_x**2 + rho_y**2
    fresnel_phase = k * rho2**2 / (8 * abs(z) ** 3)
    if fresnel_phase <= FRESNEL_PHASE:
        rho_in_x = max(abs(support.x[0]), abs(support.x[1]))
        rho_in_y = max(abs(support.y[0]), abs(support.y[1]))
        local_x = rho_in_x / (wavelength * z) + support.fx
        local_y = rho_in_y / (wavelength * z) + support.fy
        u_out_x = max(abs(ox[0]), abs(ox[1])) / (wavelength * z)
        u_out_y = max(abs(oy[0]), abs(oy[1])) / (wavelength * z)
        sampled = local_x <= 1 / (2 * grid_in.dx_si) and local_y <= 1 / (2 * grid_in.dy_si)
        no_replica = u_out_x <= 1 / grid_in.dx_si - local_x and u_out_y <= 1 / grid_in.dy_si - local_y
        if sampled and no_replica:
            return (
                "fresnel_mft",
                f"Fresnel regime (neglected phase {fresnel_phase:.1e} rad), chirp sampled",
                (_cost_mft(grid_in, grid_out)),
            )
        if same:
            critical = grid_in.nx * grid_in.dx_si**2 / wavelength
            if z >= critical:
                return (
                    "fresnel_ir",
                    "Fresnel regime beyond the critical distance N dx²/λ",
                    2 * _cost_fft(grid_in),
                )
            return "fresnel_tf", "Fresnel regime below the critical distance N dx²/λ", 2 * _cost_fft(grid_in)
        # Rule 5: Fraunhofer at finite distance.
        rho_in2 = rho_in_x**2 + rho_in_y**2
        if k * rho_in2 / (2 * z) <= FRAUNHOFER_PHASE:
            return (
                "fraunhofer",
                "Fraunhofer regime (input quadratic phase negligible)",
                _cost_mft(grid_in, grid_out),
            )

    # Rule 6: direct sum.
    return (
        "rs",
        "no fast method is valid for this geometry; direct Rayleigh-Sommerfeld sum",
        float(grid_in.size * grid_out.size),
    )


def plan_for(signal: Any, z: float | None, grid_out: Grid) -> Plan:
    """Return the :class:`Plan` for propagating ``signal`` by ``z`` (``None``: far field) to ``grid_out``."""
    grid_in = signal.grid
    if z is None:
        return Plan("fraunhofer", "far field", _cost_mft(grid_in, grid_out), grid_out)
    if z < 0:
        return Plan(
            "asm", "back-propagation needs an invertible transfer function", 2 * _cost_fft(grid_in), grid_out
        )
    planes, wavelengths = _samples(signal)
    support = _support(planes, grid_in)
    order = ["asm", "fresnel_mft", "fresnel_ir", "fresnel_tf", "fraunhofer", "rs"]
    decisions = [_decide(support, grid_in, grid_out, lam, z) for lam in wavelengths]
    # One method for every wavelength: the most conservative of the per-wavelength choices.
    method, reason, cost = max(decisions, key=lambda d: order.index(d[0]))
    if method in ("fresnel_ir", "fresnel_tf") and len({d[0] for d in decisions}) > 1:
        method, reason, cost = (
            "rs",
            "Fresnel FFT methods differ across wavelengths; direct sum",
            float(grid_in.size * grid_out.size),
        )
    return Plan(method, reason, cost, grid_out)


def choose(signal: Any, z: float | None, grid_out: Grid) -> Method:
    """Return the method chosen by the planner and log the decision."""
    from lumos.propagation.api import methods

    result = plan_for(signal, z, grid_out)
    level = logging.WARNING if result.method == "rs" and result.cost > RS_WARNING_COST else logging.INFO
    logger.log(
        level,
        "planner chose %r: %s (estimated cost %.2g operations)",
        result.method,
        result.reason,
        result.cost,
    )
    return methods()[result.method]
