r"""Symbolic fields inside contexts: dispatch and automatic conversion to numbers (step P10.6, REQ-SYM-5).

When an element receives a :class:`~lumos.symbolic.SymbolicField` (:meth:`lumos.core.Element.apply`):

1. if the element has a symbolic transfer and all its optical inputs are symbolic, the transfer keeps them
   symbolic (closed forms, :mod:`lumos.symbolic.closed_forms`);
2. if that transfer has no closed form for these inputs (:class:`NotImplementedError`), or the element has no
   symbolic transfer, the symbolic inputs are **converted to numbers** by :func:`to_numeric` and the numerical
   transfer runs; the journal records the conversion or the fallback.

:func:`to_numeric` samples the field on its grid hint (:attr:`SymbolicField.grid`). An expression that
depends on :math:`\lambda` gives a :class:`~lumos.fields.Wavecube` at the wavelengths of the observation of
the context, with their quadrature weights. Discontinuous expressions (``Piecewise``, ``Heaviside``,
``sign``, ``floor``...) are averaged over 4 × 4 sub-pixels (anti-aliasing, REQ-SYM-2); smooth ones are
sampled at the pixel centres.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import sympy as sp

from lumos.core.signal import OpticalSignal
from lumos.symbolic.field import SymbolicField

__all__ = ["dispatch", "is_symbolic", "to_numeric"]

logger = logging.getLogger("lumos.symbolic")

_DISCONTINUOUS = (sp.Piecewise, sp.Heaviside, sp.sign, sp.floor, sp.ceiling, sp.Abs)
_SUPERSAMPLE = 4


def is_symbolic(signal: Any) -> bool:
    """Whether ``signal`` (or one of the signals of a list) is a symbolic field."""
    if isinstance(signal, list | tuple):
        return any(isinstance(s, SymbolicField) for s in signal)
    return isinstance(signal, SymbolicField)


def _discontinuous(expression: sp.Expr) -> bool:
    return any(expression.has(kind) for kind in _DISCONTINUOUS)


def to_numeric(
    field: SymbolicField,
    *,
    grid: Any = None,
    wavelengths: Any = None,
    weights: Any = None,
    supersample: int | None = None,
) -> Any:
    """Sample a symbolic field on ``grid`` (default: its grid hint).

    Parameters
    ----------
    field
        Symbolic field.
    grid
        Sampling grid (default :attr:`SymbolicField.grid`).
    wavelengths, weights
        Wavelengths (quantity) and quadrature weights, needed when the expression depends on λ.
    supersample
        Sub-pixels per pixel side (default: 4 for discontinuous expressions, 1 otherwise).

    Returns
    -------
    Wavefront or Wavecube
        A :class:`~lumos.fields.Wavefront` for a monochromatic field, a :class:`~lumos.fields.Wavecube` when
        the expression depends on λ.
    """
    from lumos.symbolic.evaluate import to_wavecube, to_wavefront

    grid = grid if grid is not None else field.grid
    if grid is None:
        raise ValueError(
            f"{field!r} must be converted to numbers but has no sampling grid: "
            "give it one (SymbolicField(..., grid=Grid(...)) or SymbolicSource(..., grid=...))"
        )
    if supersample is None:
        supersample = _SUPERSAMPLE if _discontinuous(field.expression) else 1
    if field.depends_on_wavelength:
        if wavelengths is None:
            raise ValueError(
                f"{field!r} depends on lambda: give the wavelengths (set the observation band of the context)"
            )
        return to_wavecube(field, grid, wavelengths, weights=weights, supersample=supersample)
    return to_wavefront(field, grid, supersample=supersample)


def _observation(element: Any) -> tuple[Any, Any]:
    context = getattr(element, "context", None)
    if context is None or context.observation.band is None:
        return None, None
    return context.observation.wavelengths()


def _convert(element: Any, inputs: Mapping[str, Any]) -> dict[str, Any]:
    wavelengths, weights = _observation(element)

    def numeric(signal: Any) -> Any:
        if isinstance(signal, SymbolicField):
            return to_numeric(signal, wavelengths=wavelengths, weights=weights)
        return signal

    converted: dict[str, Any] = {}
    for name, signal in inputs.items():
        converted[name] = (
            [numeric(s) for s in signal] if isinstance(signal, list | tuple) else numeric(signal)
        )
    return converted


def dispatch(element: Any, inputs: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, dict[str, Any]]:
    """Run the symbolic transfer of ``element`` if possible, else convert its symbolic inputs.

    Returns
    -------
    tuple
        ``(outputs, None)``-like pair: the symbolic outputs (or ``None`` when the numerical transfer must run)
        and the inputs to give to the numerical transfer.
    """
    label = element.name or type(element).__name__
    signals = [
        s for value in inputs.values() for s in (value if isinstance(value, list | tuple) else [value])
    ]
    optical = [s for s in signals if isinstance(s, OpticalSignal)]
    if type(element).has_symbolic_transfer() and all(isinstance(s, SymbolicField) for s in optical):
        try:
            return type(element).transfer_symbolic(element.values(), dict(inputs)), dict(inputs)
        except NotImplementedError as error:
            logger.info("%s: no closed form (%s): numerical fallback", label, error)
        reason = "numerical fallback"
    elif type(element).has_symbolic_transfer():
        reason = "mixed symbolic and numerical inputs"
    else:
        reason = "no symbolic transfer"
    converted = _convert(element, inputs)
    logger.info("%s: symbolic input(s) converted to numbers (%s)", label, reason)
    return None, converted
