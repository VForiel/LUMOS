r"""Symbolic fields: closed-form expressions of optical fields (step P10.1, REQ-SYM-1, REQ-SIG-10).

A :class:`SymbolicField` holds a SymPy expression :math:`U(x, y[, \lambda, t])` of the complex amplitude of a
field, in the canonical variables of its plane (:mod:`lumos.symbolic.symbols`), and the values of its other
symbols, the **parameters**, as physical quantities. Inside the expression every quantity is in unscaled SI
units: a parameter ``w`` given as ``2 * u.mm`` is replaced by ``0.002`` when the field is evaluated.

The normalisation follows that of :class:`~lumos.fields.Wavefront`: :math:`|U|^2` is a power per unit area
(per steradian on angular planes) in ``power_unit`` per m². A field whose expression does not contain
:math:`\lambda` is monochromatic and needs its ``wavelength``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import sympy as sp
from astropy import units as u

from lumos.core.signal import OpticalSignal
from lumos.symbolic.symbols import CANONICAL, lam
from lumos.units import require_unit, to_si

__all__ = ["PLANES", "SymbolicField"]

PLANES: tuple[str, ...] = ("spatial", "angular")


def _symbol(key: Any) -> sp.Symbol:
    if isinstance(key, sp.Symbol):
        return key
    if isinstance(key, str):
        return sp.Symbol(key)
    raise TypeError(f"parameters are named by SymPy symbols or strings, got {key!r}")


class SymbolicField(OpticalSignal):
    r"""Closed-form expression of a field.

    Parameters
    ----------
    expression
        SymPy expression of the complex amplitude.
    plane
        ``"spatial"`` (variables ``x, y`` in m) or ``"angular"`` (``theta_x, theta_y`` in rad).
    parameters
        Values of the other symbols, by symbol or name, as quantities.
    wavelength
        Wavelength of a monochromatic field (required when the expression does not depend on ``lam``).
    power_unit
        Unit of power of :math:`\int |U|^2` (W or photon/s).
    grid
        Optional sampling hint (:class:`~lumos.fields.Grid`) used when the field is converted to numbers.
    form
        Optional structured description of a closed-form family (e.g. a Gaussian beam), used by symbolic
        propagation (:mod:`lumos.symbolic.closed_forms`).
    """

    def __init__(
        self,
        expression: Any,
        *,
        plane: str = "spatial",
        parameters: Mapping[Any, u.Quantity] | None = None,
        wavelength: u.Quantity | None = None,
        power_unit: Any = u.W,
        grid: Any = None,
        form: Any = None,
    ) -> None:
        if plane not in PLANES:
            raise ValueError(f"plane must be one of {PLANES}, got {plane!r}")
        expr = sp.sympify(expression)
        params: dict[sp.Symbol, u.Quantity] = {}
        for key, value in (parameters or {}).items():
            if not isinstance(value, u.Quantity):
                raise TypeError(f"parameter {key}: expected a Quantity, got {value!r}")
            params[_symbol(key)] = value
        allowed = CANONICAL[plane] | set(params)
        unknown = sorted(str(s) for s in expr.free_symbols - allowed)
        if unknown:
            raise ValueError(
                f"unknown symbol(s) {unknown} in a {plane} field: give their value in parameters "
                f"(canonical variables: {sorted(str(s) for s in CANONICAL[plane])})"
            )
        if wavelength is not None:
            wavelength = require_unit(wavelength, u.m, name="wavelength", equivalencies=u.spectral())
        elif lam not in expr.free_symbols:
            raise ValueError("a field that does not depend on lambda needs its wavelength")
        self._expression = expr
        self._plane = plane
        self._parameters = params
        self._wavelength = wavelength
        self._power_unit = u.Unit(power_unit)
        self.grid = grid
        self.form = form

    # --- Properties ---------------------------------------------------------------------------------

    @property
    def expression(self) -> sp.Expr:
        """Expression of the complex amplitude."""
        return self._expression

    @property
    def plane(self) -> str:
        """``"spatial"`` or ``"angular"``."""
        return self._plane

    @property
    def parameters(self) -> dict[sp.Symbol, u.Quantity]:
        """Values of the parameters (a copy)."""
        return dict(self._parameters)

    @property
    def wavelength(self) -> u.Quantity | None:
        """Wavelength of a monochromatic field (``None`` if the expression depends on ``lam``)."""
        return self._wavelength

    @property
    def power_unit(self) -> u.UnitBase:
        """Unit of power."""
        return self._power_unit

    @property
    def depends_on_wavelength(self) -> bool:
        """Whether the expression contains ``lam``."""
        return lam in self._expression.free_symbols

    def si_parameters(self) -> dict[sp.Symbol, float]:
        """Return the SI values of the parameters."""
        return {symbol: float(to_si(value)) for symbol, value in self._parameters.items()}

    def substituted(self) -> sp.Expr:
        """Return the expression with the SI values of the parameters substituted."""
        values = self.si_parameters()
        return self._expression.subs(values) if values else self._expression

    def intensity(self) -> sp.Expr:
        """Return :math:`|U|^2` (simplified)."""
        return sp.simplify(self._expression * sp.conjugate(self._expression))

    # --- Construction -----------------------------------------------------------------------------

    def with_expression(
        self,
        expression: Any,
        *,
        parameters: Mapping[Any, u.Quantity] | None = None,
        form: Any = None,
        **changes: Any,
    ) -> SymbolicField:
        """Return a field with another expression, the same settings and merged parameters."""
        merged = self._merged(parameters)
        settings = {
            "plane": self._plane,
            "wavelength": self._wavelength,
            "power_unit": self._power_unit,
            "grid": self.grid,
            **changes,
        }
        return SymbolicField(expression, parameters=merged, form=form, **settings)

    def _merged(self, parameters: Mapping[Any, u.Quantity] | None) -> dict[sp.Symbol, u.Quantity]:
        merged = dict(self._parameters)
        for key, value in (parameters or {}).items():
            symbol = _symbol(key)
            if symbol in merged and not _same(merged[symbol], value):
                raise ValueError(f"parameter {symbol} already has the value {merged[symbol]}, not {value}")
            merged[symbol] = value
        return merged

    def multiply(self, factor: Any, *, parameters: Mapping[Any, u.Quantity] | None = None) -> SymbolicField:
        """Return the field multiplied by the expression ``factor`` (a mask, a phase...)."""
        return self.with_expression(self._expression * sp.sympify(factor), parameters=parameters)

    # --- Evaluation ---------------------------------------------------------------------------------

    def to_wavefront(self, grid: Any, **options: Any) -> Any:
        """Sample the field on ``grid`` (see :func:`lumos.symbolic.evaluate.to_wavefront`)."""
        from lumos.symbolic.evaluate import to_wavefront

        return to_wavefront(self, grid, **options)

    def to_wavecube(self, grid: Any, wavelengths: u.Quantity, **options: Any) -> Any:
        """Sample the field on ``grid`` at ``wavelengths`` (:func:`lumos.symbolic.evaluate.to_wavecube`)."""
        from lumos.symbolic.evaluate import to_wavecube

        return to_wavecube(self, grid, wavelengths, **options)

    # --- Rendering ----------------------------------------------------------------------------------

    def latex(self) -> str:
        """Return the LaTeX code of the expression."""
        return sp.latex(self._expression)

    def _repr_latex_(self) -> str:
        """LaTeX representation for Jupyter."""
        return f"${self.latex()}$"

    def __repr__(self) -> str:
        """Represent the field by its expression and plane."""
        return f"SymbolicField({self._expression}, plane={self._plane!r})"


def _same(a: u.Quantity, b: u.Quantity) -> bool:
    try:
        return bool(abs(float(to_si(a)) - float(to_si(b))) <= 1e-12 * max(abs(float(to_si(a))), 1e-300))
    except (TypeError, u.UnitsError):
        return False
