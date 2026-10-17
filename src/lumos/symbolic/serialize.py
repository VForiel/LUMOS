r"""JSON form of symbolic expressions and fields, read without evaluating code (step P10.6, NF-7).

SymPy's own text forms (``srepr``, ``str``) are read back with ``eval``, which a ``.lumos`` file must never
trigger. Expressions are therefore stored as trees of JSON objects:

- ``{"op": name, "args": [...]}`` for an operation or function, ``name`` being one of :data:`FUNCTIONS`;
- ``{"symbol": name, "assumptions": {...}}`` for a symbol (its assumptions, so that ``w`` stays positive);
- ``{"int": "12"}``, ``{"rational": ["1", "3"]}``, ``{"float": "0.25"}`` and ``{"const": name}`` for numbers
  and the constants of :data:`CONSTANTS`.

Reading an unknown operation, constant or key raises :class:`ValueError`: nothing outside these lists is ever
called. Closed forms (:mod:`lumos.symbolic.closed_forms`) are stored by their fields.
"""

from __future__ import annotations

from typing import Any

import sympy as sp
from astropy import units as u
from sympy.functions.elementary.piecewise import ExprCondPair

from lumos.symbolic.field import SymbolicField
from lumos.units import quantity_from_json, quantity_to_json

__all__ = [
    "CONSTANTS",
    "FUNCTIONS",
    "expression_from_json",
    "expression_to_json",
    "field_from_json",
    "field_to_json",
]

#: Operations and functions allowed in stored expressions.
FUNCTIONS: dict[str, Any] = {
    cls.__name__: cls
    for cls in (
        sp.Add,
        sp.Mul,
        sp.Pow,
        sp.exp,
        sp.log,
        sp.sin,
        sp.cos,
        sp.tan,
        sp.sinh,
        sp.cosh,
        sp.tanh,
        sp.asin,
        sp.acos,
        sp.atan,
        sp.atan2,
        sp.sinc,
        sp.Abs,
        sp.re,
        sp.im,
        sp.conjugate,
        sp.arg,
        sp.sign,
        sp.floor,
        sp.ceiling,
        sp.Heaviside,
        sp.erf,
        sp.gamma,
        sp.factorial,
        sp.besselj,
        sp.bessely,
        sp.besseli,
        sp.besselk,
        sp.hermite,
        sp.laguerre,
        sp.assoc_laguerre,
        sp.legendre,
        sp.Max,
        sp.Min,
        sp.Piecewise,
        sp.And,
        sp.Or,
        sp.Not,
        sp.Xor,
        sp.StrictLessThan,
        sp.LessThan,
        sp.StrictGreaterThan,
        sp.GreaterThan,
        sp.Equality,
        sp.Unequality,
    )
}
FUNCTIONS["ExprCondPair"] = ExprCondPair

#: Constants allowed in stored expressions.
CONSTANTS: dict[str, Any] = {
    "I": sp.I,
    "pi": sp.pi,
    "E": sp.E,
    "oo": sp.oo,
    "-oo": -sp.oo,
    "nan": sp.nan,
    "true": sp.true,
    "false": sp.false,
}


def expression_to_json(expression: Any) -> Any:
    """Return the JSON tree of a SymPy expression."""
    expr = sp.sympify(expression)
    for name, constant in CONSTANTS.items():
        if expr == constant and type(expr) is type(constant):
            return {"const": name}
    if isinstance(expr, sp.Symbol):
        if isinstance(expr, sp.Dummy | sp.Wild):
            raise ValueError(f"dummy symbols cannot be stored: {expr}")
        assumptions = {
            key: bool(value) for key, value in sorted(expr.assumptions0.items()) if value is not None
        }
        return {"symbol": expr.name, "assumptions": assumptions}
    if isinstance(expr, sp.Integer):
        return {"int": str(int(expr))}
    if isinstance(expr, sp.Rational):
        return {"rational": [str(expr.p), str(expr.q)]}
    if isinstance(expr, sp.Float):
        return {"float": repr(float(expr))}
    name = type(expr).__name__
    if name not in FUNCTIONS:
        raise ValueError(
            f"the function {name} cannot be stored in a .lumos file (allowed: {sorted(FUNCTIONS)})"
        )
    return {"op": name, "args": [expression_to_json(arg) for arg in expr.args]}


def expression_from_json(data: Any) -> Any:
    """Return the SymPy expression of a JSON tree (only :data:`FUNCTIONS` and :data:`CONSTANTS` are used)."""
    if not isinstance(data, dict) or len(data) not in (1, 2):
        raise ValueError(f"invalid stored expression: {data!r}")
    if "op" in data:
        name = data["op"]
        if not isinstance(name, str) or name not in FUNCTIONS:
            raise ValueError(f"unknown function {name!r} in a stored expression")
        args = data.get("args")
        if not isinstance(args, list):
            raise ValueError(f"invalid arguments of {name} in a stored expression")
        return FUNCTIONS[name](*(expression_from_json(arg) for arg in args))
    if "symbol" in data:
        name, assumptions = data["symbol"], data.get("assumptions", {})
        if not isinstance(name, str) or not isinstance(assumptions, dict):
            raise ValueError(f"invalid symbol in a stored expression: {data!r}")
        if not all(isinstance(k, str) and isinstance(v, bool) for k, v in assumptions.items()):
            raise ValueError(f"invalid assumptions of the symbol {name}")
        return sp.Symbol(name, **assumptions)
    if "int" in data:
        return sp.Integer(int(str(data["int"])))
    if "rational" in data:
        p, q = data["rational"]
        return sp.Rational(int(str(p)), int(str(q)))
    if "float" in data:
        return sp.Float(float(str(data["float"])))
    if "const" in data:
        name = data["const"]
        if name not in CONSTANTS:
            raise ValueError(f"unknown constant {name!r} in a stored expression")
        return CONSTANTS[name]
    raise ValueError(f"invalid stored expression: {data!r}")


# --- Closed forms -------------------------------------------------------------------------------------------


def _form_to_json(form: Any) -> Any:
    from lumos.symbolic.closed_forms import GaussianBeamForm, PlaneWaveForm, PupilForm

    if form is None:
        return None
    if isinstance(form, GaussianBeamForm):
        return {
            "kind": "gaussian",
            "q": [form.q.real, form.q.imag],
            "wavelength": form.wavelength,
            "power": form.power,
            "family": form.family,
            "indices": list(form.indices),
            "gouy": form.gouy,
            "phase": form.phase,
            "power_unit": u.Unit(form.power_unit).to_string(),
        }
    if isinstance(form, PlaneWaveForm):
        return {"kind": "plane", "amplitude": form.amplitude, "wavelength": form.wavelength}
    if isinstance(form, PupilForm):
        return {
            "kind": "pupil",
            "shape": form.kind,
            "amplitude": form.amplitude,
            "wavelength": form.wavelength,
            "size": list(form.size),
            "focal": form.focal,
        }
    raise ValueError(f"the closed form {type(form).__name__} cannot be stored")


def _form_from_json(data: Any) -> Any:
    from lumos.symbolic.closed_forms import GaussianBeamForm, PlaneWaveForm, PupilForm

    if data is None:
        return None
    kind = data.get("kind")
    if kind == "gaussian":
        return GaussianBeamForm(
            complex(float(data["q"][0]), float(data["q"][1])),
            float(data["wavelength"]),
            float(data["power"]),
            str(data["family"]),
            (int(data["indices"][0]), int(data["indices"][1])),
            gouy=float(data["gouy"]),
            phase=float(data["phase"]),
            power_unit=u.Unit(str(data["power_unit"])),
        )
    if kind == "plane":
        return PlaneWaveForm(float(data["amplitude"]), float(data["wavelength"]))
    if kind == "pupil":
        focal = data.get("focal")
        size = data["size"]
        return PupilForm(
            str(data["shape"]),
            float(data["amplitude"]),
            float(data["wavelength"]),
            (float(size[0]), float(size[1])),
            None if focal is None else float(focal),
        )
    raise ValueError(f"unknown closed form {kind!r}")


# --- Fields -------------------------------------------------------------------------------------------------


def field_to_json(field: SymbolicField) -> dict[str, Any]:
    """Return the JSON form of a symbolic field (expression, parameters, wavelength, grid, closed form)."""
    return {
        "expression": expression_to_json(field.expression),
        "plane": field.plane,
        "parameters": [
            {"symbol": expression_to_json(symbol), "value": quantity_to_json(value)}
            for symbol, value in field.parameters.items()
        ],
        "wavelength": None if field.wavelength is None else quantity_to_json(field.wavelength),
        "power_unit": field.power_unit.to_string(),
        "grid": None if field.grid is None else field.grid.to_json(),
        "form": _form_to_json(field.form),
    }


def field_from_json(data: dict[str, Any]) -> SymbolicField:
    """Return the symbolic field of a JSON form written by :func:`field_to_json`."""
    from lumos.fields.grid import Grid

    parameters = {}
    for entry in data.get("parameters", []):
        symbol = expression_from_json(entry["symbol"])
        if not isinstance(symbol, sp.Symbol):
            raise ValueError("parameters of a stored field must be symbols")
        parameters[symbol] = quantity_from_json(entry["value"])
    wavelength = data.get("wavelength")
    grid = data.get("grid")
    return SymbolicField(
        expression_from_json(data["expression"]),
        plane=str(data.get("plane", "spatial")),
        parameters=parameters,
        wavelength=None if wavelength is None else quantity_from_json(wavelength),
        power_unit=u.Unit(str(data.get("power_unit", "W"))),
        grid=None if grid is None else Grid.from_json(grid),
        form=_form_from_json(data.get("form")),
    )
