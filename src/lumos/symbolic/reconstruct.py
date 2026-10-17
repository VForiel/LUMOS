r"""From grids to expressions: modal projections, fits and regressions (step P10.5, REQ-SYM-3).

Every function returns a :class:`Reconstruction`: a :class:`~lumos.symbolic.SymbolicField`, the
coefficients or parameters found, the relative error :math:`\|U - \hat U\|_2 / \|U\|_2` on the grid
(``residual``) and the method used.

- :func:`project_zernike`: phase of a field over a circular pupil on Noll's Zernike modes (Noll 1976), by
  least squares; the expression is :math:`\bar A\,\mathrm{circ}(r/R)\,\exp(i \sum_j c_j Z_j)`.
- :func:`project_hermite_gauss`: complex coefficients
  :math:`c_{nm} = \langle \mathrm{HG}_{nm}, U\rangle` on the orthonormal Hermite-Gauss modes of a waist
  :math:`w` (the modes of :mod:`lumos.symbolic.closed_forms` at their waist).
- :func:`project_fourier`: the ``n_terms`` largest terms of the discrete Fourier series of the field.
- :func:`fit_gaussian`: Gaussian beam :math:`A e^{-\rho^2/w^2} e^{ik\rho^2/2R} e^{i\phi}` around
  :math:`(x_0, y_0)`, from weighted least squares on :math:`\ln|U|` and on the phase.
- :func:`fit_airy`: Airy pattern :math:`I_0 [2J_1(v)/v]^2`, :math:`v = \pi \rho / s`, by non-linear least
  squares (``s = λf/D``).
- :func:`sparse_regression`: sequentially thresholded least squares (Brunton, Proctor & Kutz 2016, PNAS 113,
  3932) on monomials :math:`X^i Y^j` of the reduced coordinates :math:`X = x / L`, :math:`Y = y / L`.
- :func:`symbolic_regression`: genetic symbolic regression with PySR (Cranmer 2023), optional.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import sympy as sp
from astropy import units as u

from lumos.backend import to_numpy
from lumos.fields.grid import Grid
from lumos.physics.zernike import noll_indices, zernike_basis
from lumos.symbolic.field import SymbolicField
from lumos.symbolic.symbols import r, x, y
from lumos.units import require_unit, to_si

__all__ = [
    "Reconstruction",
    "fit_airy",
    "fit_gaussian",
    "project_fourier",
    "project_hermite_gauss",
    "project_zernike",
    "sparse_regression",
    "symbolic_regression",
    "zernike_expression",
]

#: Reduced coordinates of the sparse regression.
X, Y = sp.symbols("X Y", real=True)


@dataclass
class Reconstruction:
    """Result of a reconstruction.

    Attributes
    ----------
    field
        Symbolic field reconstructed.
    coefficients
        Coefficients (modal projections) or parameters (fits) found.
    residual
        Relative L2 error of the reconstruction on the grid.
    method
        Name of the method.
    """

    field: SymbolicField
    coefficients: Any
    residual: float
    method: str
    details: dict[str, Any] = dataclasses.field(default_factory=dict)


def _data(signal: Any) -> tuple[np.ndarray, Grid, Any, Any]:
    values = np.asarray(to_numpy(signal.field), dtype=complex)
    if values.ndim != 2:
        raise ValueError("reconstructions work on scalar monochromatic wavefronts")
    return values, signal.grid, signal.wavelength, signal.power_unit


def _residual(reference: np.ndarray, estimate: np.ndarray, weights: np.ndarray | None = None) -> float:
    w = 1.0 if weights is None else weights
    norm = float(np.sqrt(np.sum(w * np.abs(reference) ** 2)))
    return float(np.sqrt(np.sum(w * np.abs(reference - estimate) ** 2))) / max(norm, 1e-300)


def _evaluate(field: SymbolicField, grid: Grid) -> np.ndarray:
    from lumos.symbolic.evaluate import to_wavefront

    return np.asarray(to_numpy(to_wavefront(field, grid).field))


# --- Zernike ----------------------------------------------------------------------------------------------


def zernike_expression(j: int, radius: float) -> sp.Expr:
    """Return Noll mode ``j`` as an expression of ``x, y`` over a pupil of ``radius`` (m)."""
    # Written as a polynomial in x and y (no singularity at the centre): R_n^m(ρ) = ρ^m P(ρ²) and
    # ρ^m cos(mθ) = Re[((x + iy)/R)^m], ρ^m sin(mθ) = Im[((x + iy)/R)^m].
    n, m = noll_indices(j)
    scale = sp.Float(radius)
    rho2 = (x**2 + y**2) / scale**2
    mm = abs(m)
    polynomial = sp.Integer(0)
    for s in range((n - mm) // 2 + 1):
        coefficient = (-1) ** s * math.factorial(n - s)
        coefficient //= (
            math.factorial(s) * math.factorial((n + mm) // 2 - s) * math.factorial((n - mm) // 2 - s)
        )
        polynomial += coefficient * rho2 ** ((n - 2 * s - mm) // 2)
    if m == 0:
        return sp.sqrt(n + 1) * polynomial
    power = sp.expand(((x + sp.I * y) / scale) ** mm)
    angular = sp.re(power) if m > 0 else sp.im(power)
    return sp.sqrt(2 * (n + 1)) * polynomial * angular


def project_zernike(signal: Any, *, diameter: u.Quantity, n_modes: int) -> Reconstruction:
    """Project the phase of ``signal`` on Noll modes 1 … ``n_modes`` over a pupil of ``diameter``."""
    values, grid, wavelength, power_unit = _data(signal)
    radius = float(to_si(require_unit(diameter, u.m, name="diameter"))) / 2
    basis = zernike_basis(grid, diameter, n_modes)
    inside = basis[0] != 0
    phase = np.angle(values[inside])
    matrix = basis[:, inside].T
    coefficients, *_ = np.linalg.lstsq(matrix, phase, rcond=None)
    amplitude = float(np.mean(np.abs(values[inside])))
    expression = sum(
        (
            sp.Float(c) * zernike_expression(j + 1, radius)
            for j, c in enumerate(coefficients)
            if abs(c) > 1e-14
        ),
        sp.Integer(0),
    )
    pupil = sp.Piecewise((1, r <= sp.Float(radius * (1 + 1e-12))), (0, True))
    reconstructed = SymbolicField(
        sp.Float(amplitude) * pupil * sp.exp(sp.I * expression),
        wavelength=wavelength,
        power_unit=power_unit,
        grid=grid,
    )
    estimate = amplitude * np.exp(1j * (matrix @ coefficients))
    residual = _residual(values[inside], estimate)
    return Reconstruction(reconstructed, np.asarray(coefficients), residual, "zernike")


# --- Hermite-Gauss ---------------------------------------------------------------------------------------


def project_hermite_gauss(signal: Any, *, waist: u.Quantity, max_order: int) -> Reconstruction:
    """Project ``signal`` on the Hermite-Gauss modes of ``waist`` with ``n + m <= max_order``."""
    from scipy.special import eval_hermite

    from lumos.symbolic.closed_forms import GaussianBeamForm

    values, grid, wavelength, power_unit = _data(signal)
    w = float(to_si(require_unit(waist, u.m, name="waist")))
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    area = grid.dx_si * grid.dy_si
    coefficients: dict[tuple[int, int], complex] = {}
    estimate = np.zeros_like(values)
    expression = sp.Integer(0)
    lam = float(to_si(wavelength))
    z_r = math.pi * w**2 / lam
    for order in range(int(max_order) + 1):
        for n in range(order + 1):
            m = order - n
            norm = math.sqrt(2 / math.pi) / (
                w * math.sqrt(2 ** (n + m) * math.factorial(n) * math.factorial(m))
            )
            mode = norm * eval_hermite(n, math.sqrt(2) * xx / w) * eval_hermite(m, math.sqrt(2) * yy / w)
            mode = mode * np.exp(-(xx**2 + yy**2) / w**2)
            c = complex(np.sum(mode * values) * area)  # real modes: <mode, U>
            coefficients[(n, m)] = c
            estimate = estimate + c * mode
            form = GaussianBeamForm(complex(0, -z_r), lam, 1.0, "hg", (n, m))
            expression += (sp.Float(c.real) + sp.I * sp.Float(c.imag)) * form.expression()
    reconstructed = SymbolicField(expression, wavelength=wavelength, power_unit=power_unit, grid=grid)
    return Reconstruction(reconstructed, coefficients, _residual(values, estimate), "hermite-gauss")


# --- Fourier ------------------------------------------------------------------------------------------------


def project_fourier(signal: Any, *, n_terms: int) -> Reconstruction:
    """Keep the ``n_terms`` largest terms of the discrete Fourier series of ``signal``."""
    values, grid, wavelength, power_unit = _data(signal)
    ny, nx = values.shape
    spectrum = np.fft.fft2(values) / (nx * ny)
    order = np.argsort(np.abs(spectrum).ravel())[::-1][: int(n_terms)]
    fx = np.fft.fftfreq(nx, d=grid.dx_si)
    fy = np.fft.fftfreq(ny, d=grid.dy_si)
    expression = sp.Integer(0)
    estimate = np.zeros_like(values)
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    coefficients = {}
    for flat in order:
        iy, ix = divmod(int(flat), nx)
        # pixel index i = x / dx + N // 2: exp(2iπ k i / N) = exp(2iπ f x) exp(2iπ k (N // 2) / N)
        shift = np.exp(2j * math.pi * (ix * (nx // 2) / nx + iy * (ny // 2) / ny))
        c = complex(spectrum[iy, ix] * shift)
        x_off, y_off = grid.x0_si, grid.y0_si
        phase = (
            2 * sp.pi * (sp.Float(fx[ix]) * (x - sp.Float(x_off)) + sp.Float(fy[iy]) * (y - sp.Float(y_off)))
        )
        expression += (sp.Float(c.real) + sp.I * sp.Float(c.imag)) * sp.exp(sp.I * phase)
        estimate = estimate + c * np.exp(2j * math.pi * (fx[ix] * (xx - x_off) + fy[iy] * (yy - y_off)))
        coefficients[(float(fx[ix]), float(fy[iy]))] = c
    reconstructed = SymbolicField(expression, wavelength=wavelength, power_unit=power_unit, grid=grid)
    return Reconstruction(reconstructed, coefficients, _residual(values, estimate), "fourier")


# --- Gaussian fit -----------------------------------------------------------------------------------------


def fit_gaussian(signal: Any, *, threshold: float = 1e-3) -> Reconstruction:
    """Fit a Gaussian beam (amplitude, centre, waist, radius of curvature, phase) to ``signal``."""
    values, grid, wavelength, power_unit = _data(signal)
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    amplitude = np.abs(values)
    keep = amplitude > threshold * amplitude.max()
    weights = amplitude[keep] ** 2
    design = np.stack([np.ones(keep.sum()), xx[keep], yy[keep], xx[keep] ** 2 + yy[keep] ** 2], axis=1)
    sqrt_w = np.sqrt(weights)[:, None]
    log_fit, *_ = np.linalg.lstsq(design * sqrt_w, np.log(amplitude[keep]) * sqrt_w[:, 0], rcond=None)
    a0, bx, by, quad = log_fit
    w = math.sqrt(-1 / quad)
    x0, y0 = -bx / (2 * quad), -by / (2 * quad)
    peak = math.exp(a0 - quad * (x0**2 + y0**2))
    phase = np.unwrap(
        np.angle(
            values[keep]
            * np.exp(-1j * np.angle(values[np.unravel_index(np.argmax(amplitude), amplitude.shape)]))
        )
    )
    phase_fit, *_ = np.linalg.lstsq(design * sqrt_w, phase * sqrt_w[:, 0], rcond=None)
    lam = float(to_si(wavelength))
    k = 2 * math.pi / lam
    curvature = phase_fit[3]
    radius = math.inf if abs(curvature) < 1e-300 else k / (2 * curvature)
    reference = float(np.angle(values[np.unravel_index(np.argmax(amplitude), amplitude.shape)]))
    phi0 = reference + phase_fit[0] - curvature * (x0**2 + y0**2) + 0.0
    rho2 = (x - sp.Float(x0)) ** 2 + (y - sp.Float(y0)) ** 2
    phase_term = 0 if math.isinf(radius) else sp.Float(k / (2 * radius)) * rho2
    expression = (
        sp.Float(peak) * sp.exp(-rho2 / sp.Float(w) ** 2) * sp.exp(sp.I * (phase_term + sp.Float(phi0)))
    )
    reconstructed = SymbolicField(expression, wavelength=wavelength, power_unit=power_unit, grid=grid)
    estimate = _evaluate(reconstructed, grid)
    parameters = {
        "amplitude": peak,
        "waist": w * u.m,
        "x0": x0 * u.m,
        "y0": y0 * u.m,
        "radius": radius * u.m,
        "phase": phi0 * u.rad,
    }
    return Reconstruction(reconstructed, parameters, _residual(values, estimate), "gaussian")


# --- Airy fit ----------------------------------------------------------------------------------------------


def _airy(v: np.ndarray) -> np.ndarray:
    from scipy.special import j1

    safe = np.where(np.abs(v) > 1e-12, v, 1.0)
    return np.where(np.abs(v) > 1e-12, (2 * j1(safe) / safe) ** 2, 1.0)


def fit_airy(intensity: Any, grid: Grid) -> Reconstruction:
    """Fit an Airy pattern :math:`I_0[2J_1(v)/v]^2` to an intensity map sampled on ``grid``."""
    from scipy.optimize import least_squares

    data = np.asarray(to_numpy(intensity), dtype=float)
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    peak_index = np.unravel_index(int(np.argmax(data)), data.shape)
    peak0 = float(data[peak_index])
    x_start, y_start = float(xx[peak_index]), float(yy[peak_index])
    half = data > peak0 / 2
    width0 = 2 * math.sqrt(half.sum() * grid.dx_si * grid.dy_si / math.pi)  # FWHM ≈ 1.03 s
    scale_unit = max(grid.dx_si, grid.dy_si)

    def residuals(p: np.ndarray) -> np.ndarray:
        x0, y0, scale, peak = p[0] * scale_unit, p[1] * scale_unit, p[2] * scale_unit, p[3] * peak0
        v = math.pi * np.hypot(xx - x0, yy - y0) / scale
        return ((peak * _airy(v) - data) / peak0).ravel()

    start = [x_start / scale_unit, y_start / scale_unit, width0 / scale_unit, 1.0]
    result = least_squares(residuals, start, x_scale="jac", xtol=1e-14, ftol=1e-14, gtol=1e-14)
    x0, y0, scale = (float(v) * scale_unit for v in result.x[:3])
    peak = float(result.x[3]) * peak0
    v_expr = sp.pi * sp.sqrt((x - sp.Float(x0)) ** 2 + (y - sp.Float(y0)) ** 2) / sp.Float(scale)
    expression = sp.sqrt(sp.Float(peak)) * 2 * sp.besselj(1, v_expr) / v_expr
    reconstructed = SymbolicField(expression, wavelength=1 * u.um, grid=grid)
    estimate = peak * _airy(math.pi * np.hypot(xx - x0, yy - y0) / scale)
    parameters = {"x0": x0 * u.m, "y0": y0 * u.m, "scale": scale * u.m, "peak": peak}
    return Reconstruction(reconstructed, parameters, _residual(data, estimate), "airy")


# --- Sparse regression -------------------------------------------------------------------------------------


def sparse_regression(
    signal: Any, *, degree: int = 3, length_scale: u.Quantity, threshold: float = 1e-6, iterations: int = 10
) -> Reconstruction:
    """Find a sparse polynomial in ``X = x / L``, ``Y = y / L`` reproducing ``signal`` (STLSQ)."""
    values, grid, wavelength, power_unit = _data(signal)
    length = float(to_si(require_unit(length_scale, u.m, name="length_scale")))
    xx, yy = (np.asarray(c) / length for c in grid.coords(backend="numpy"))
    terms = [X**i * Y**j for i in range(int(degree) + 1) for j in range(int(degree) + 1 - i)]
    library = np.stack(
        [np.asarray(sp.lambdify((X, Y), term, "numpy")(xx, yy) + 0 * xx).ravel() for term in terms], axis=1
    )
    target = values.ravel()
    active = np.ones(len(terms), dtype=bool)
    coefficients = np.zeros(len(terms), dtype=complex)
    scale = float(np.abs(target).max()) or 1.0
    for _ in range(int(iterations)):
        solution, *_ = np.linalg.lstsq(library[:, active], target, rcond=None)
        coefficients[:] = 0
        coefficients[active] = solution
        small = np.abs(coefficients) < threshold * scale
        if not np.any(small & active):
            break
        active &= ~small
    found = {terms[k]: _real_if_close(coefficients[k]) for k in np.flatnonzero(active)}
    reduced = sum(
        (sp.Float(c.real) + sp.I * sp.Float(c.imag)) * term
        for term, c in ((t, complex(v)) for t, v in found.items())
    )
    expression = sp.sympify(reduced).subs({X: x / sp.Float(length), Y: y / sp.Float(length)})
    reconstructed = SymbolicField(expression + 0 * x, wavelength=wavelength, power_unit=power_unit, grid=grid)
    estimate = (library @ coefficients).reshape(values.shape)
    return Reconstruction(reconstructed, found, _residual(values, estimate), "sparse-regression")


def _real_if_close(value: complex) -> complex | float:
    return float(value.real) if abs(value.imag) <= 1e-12 * max(abs(value), 1e-300) else complex(value)


# --- PySR ---------------------------------------------------------------------------------------------------


def symbolic_regression(signal: Any, **options: Any) -> Reconstruction:
    """Search an expression of ``signal`` with PySR (optional dependency ``pysr``)."""
    try:
        from pysr import PySRRegressor  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError("symbolic_regression needs the optional package pysr (pip install pysr)") from error
    values, grid, wavelength, power_unit = _data(signal)
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    model = PySRRegressor(**options)
    features = np.stack([xx.ravel(), yy.ravel()], axis=1)
    model.fit(features, np.real(values).ravel())
    expression = cast(Any, model).sympy().subs({sp.Symbol("x0"): x, sp.Symbol("x1"): y})
    reconstructed = SymbolicField(expression, wavelength=wavelength, power_unit=power_unit, grid=grid)
    estimate = _evaluate(reconstructed, grid)
    return Reconstruction(reconstructed, {"equations": model.equations_}, _residual(values, estimate), "pysr")
