r"""Modal amplitudes of the light in one guide (REQ-SIG-7, ADR-0023).

A :class:`ModalField` is one *guided lane*: the complex amplitudes :math:`a_j` of the light in the :math:`n`
modes of one guide (``n = 1`` for a single-mode guide) at the wavelengths :math:`\lambda_k`, for
:math:`M` mutually incoherent sources. Its array has shape ``(nλ, n)`` or ``(M, nλ, n)``, and
:math:`|a_j|^2` is the power in mode :math:`j` (in :attr:`ModalField.power_unit`). Guides separated in space
are the lanes of a :class:`~lumos.fields.Bus`, never modes of one field.

:attr:`ModalField.sources` identifies the source list the sources come from: lanes interfere only if they
describe the same sources in the same order.

Scattering matrices (:mod:`lumos.fields.smatrix`) act with :meth:`ModalField.apply`. Incoherent sources add in
power and spectral channels add with the weights :math:`w_k`, like :class:`~lumos.fields.Wavecube`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from astropy import units as u

from lumos.backend._core import backend_of, convert, get, operand
from lumos.backend.memory import dtypes
from lumos.core.signal import OpticalSignal
from lumos.fields.wavecube import _wavelengths
from lumos.fields.wavefront import _array_backend, _power_unit
from lumos.units import quantity_from_json, quantity_to_json, to_si

__all__ = ["ModalField"]


def _labels(labels: Sequence[str] | None, n: int) -> tuple[str, ...] | None:
    if labels is None:
        return None
    result = tuple(str(label) for label in labels)
    if len(result) != n:
        raise ValueError(f"{n} mode labels expected, got {len(result)}")
    return result


class ModalField(OpticalSignal):
    """Complex amplitudes of the modes of one guide at several wavelengths.

    Parameters
    ----------
    amplitudes
        Complex amplitudes, shape ``(nλ, n)`` or ``(M, nλ, n)`` (``n`` modes, ``M`` incoherent sources).
    wavelengths
        ``nλ`` vacuum wavelengths.
    weights
        Spectral weights for :attr:`detected_power` (default: ones).
    labels
        Optional names of the ``n`` modes.
    power_unit
        ``u.W`` (default) or ``u.photon / u.s``: :math:`|a|^2` is a power in this unit.
    sources
        Identifier of the source list the ``M`` sources come from (``None``: unknown).
    """

    __slots__ = ("_amplitudes", "_labels", "_power_unit", "_sources", "_wavelengths", "_weights")
    __array_ufunc__ = None

    _amplitudes: Any
    _wavelengths: u.Quantity
    _weights: np.ndarray
    _labels: tuple[str, ...] | None
    _power_unit: u.UnitBase
    _sources: str | None

    def __init__(
        self,
        amplitudes: Any,
        wavelengths: u.Quantity,
        *,
        weights: Any = None,
        labels: Sequence[str] | None = None,
        power_unit: Any = u.W,
        sources: str | None = None,
    ) -> None:
        lam = _wavelengths(wavelengths)
        w = np.ones(len(lam)) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != (len(lam),):
            raise ValueError(f"weights must have one value per wavelength ({len(lam)}), got shape {w.shape}")
        target = get(_array_backend(amplitudes))
        _, complex_dtype = dtypes(backend=target.name)
        array = target.asarray(
            amplitudes if target.is_array(amplitudes) else np.asarray(amplitudes), dtype=complex_dtype
        )
        n = int(array.shape[-1]) if array.ndim else 0
        self._init(array, lam, w, _labels(labels, n), _power_unit(power_unit), sources)

    def _init(
        self,
        amplitudes: Any,
        wavelengths: u.Quantity,
        weights: np.ndarray,
        labels: tuple[str, ...] | None,
        power_unit: u.UnitBase,
        sources: str | None,
    ) -> None:
        shape = tuple(amplitudes.shape)
        if len(shape) not in (2, 3) or shape[-2] != len(wavelengths) or shape[-1] < 1:
            raise ValueError(f"the amplitude shape {shape} must be (nλ={len(wavelengths)}, n) or (M, nλ, n)")
        self._amplitudes = amplitudes
        self._wavelengths = wavelengths
        self._weights = weights
        self._labels = labels
        self._power_unit = power_unit
        self._sources = None if sources is None else str(sources)

    def _with(self, amplitudes: Any, labels: tuple[str, ...] | None) -> ModalField:
        new = object.__new__(type(self))
        new._init(amplitudes, self._wavelengths, self._weights, labels, self._power_unit, self._sources)
        return new

    # --- Attributes -------------------------------------------------------------------------------

    @property
    def amplitudes(self) -> Any:
        """Complex amplitudes, a backend array of shape ``(nλ, n)`` or ``(M, nλ, n)``."""
        return self._amplitudes

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of :attr:`amplitudes`."""
        return tuple(self._amplitudes.shape)

    @property
    def n_sources(self) -> int:
        """Number of incoherent sources ``M`` (1 without a source axis)."""
        return int(self.shape[0]) if len(self.shape) == 3 else 1

    @property
    def n_wavelengths(self) -> int:
        """Number of spectral channels."""
        return len(self._wavelengths)

    @property
    def n_modes(self) -> int:
        """Number of guided modes ``n``."""
        return int(self.shape[-1])

    @property
    def wavelengths(self) -> u.Quantity:
        """Vacuum wavelengths."""
        return self._wavelengths

    @property
    def wavelengths_si(self) -> np.ndarray:
        """Vacuum wavelengths in metres."""
        return np.asarray(to_si(self._wavelengths), dtype=float)

    @property
    def weights(self) -> np.ndarray:
        """Spectral weights (a copy)."""
        return self._weights.copy()

    @property
    def labels(self) -> tuple[str, ...] | None:
        """Names of the modes, if given."""
        return self._labels

    @property
    def sources(self) -> str | None:
        """Identifier of the source list described by the source axis, or ``None``."""
        return self._sources

    @property
    def power_unit(self) -> u.UnitBase:
        """Unit of :math:`|a|^2`."""
        return self._power_unit

    @property
    def backend(self) -> str:
        """Name of the backend holding the amplitudes."""
        return backend_of(self._amplitudes)

    # --- Powers ---------------------------------------------------------------------------------

    def _numpy_power(self) -> np.ndarray:
        xp = get(self.backend).xp
        power = xp.real(self._amplitudes * xp.conj(self._amplitudes))
        if len(self.shape) == 3:
            power = xp.sum(power, axis=0)
        return np.asarray(get(self.backend).to_numpy(power), dtype=float)

    @property
    def mode_power(self) -> u.Quantity:
        """Power per wavelength and mode, summed over the incoherent sources, shape ``(nλ, n)``."""
        return u.Quantity(self._numpy_power(), self._power_unit)

    @property
    def total_power(self) -> u.Quantity:
        """Power summed over the modes, per wavelength, shape ``(nλ,)``."""
        return u.Quantity(self._numpy_power().sum(axis=-1), self._power_unit)

    @property
    def detected_power(self) -> u.Quantity:
        r"""Spectrally weighted power per mode :math:`\sum_k w_k P_{kj}`, shape ``(n,)``."""
        return u.Quantity(np.einsum("k,kj->j", self._weights, self._numpy_power()), self._power_unit)

    # --- Operations -------------------------------------------------------------------------------

    def apply(self, matrix: Any, *, labels: Sequence[str] | None = None) -> ModalField:
        """Apply a scattering matrix ``(n_out, n_in)`` or ``(nλ, n_out, n_in)`` to each source and wavelength.

        ``labels`` names the output modes (the input labels are kept for a square matrix by default).
        """
        target = get(self.backend)
        _, complex_dtype = dtypes(backend=target.name)
        s = target.asarray(matrix if target.is_array(matrix) else np.asarray(matrix), dtype=complex_dtype)
        shape = tuple(s.shape)
        valid = (len(shape) == 2 or (len(shape) == 3 and shape[0] == self.n_wavelengths)) and (
            shape[-1] == self.n_modes
        )
        if not valid:
            raise ValueError(
                f"the matrix shape {shape} must be (n_out, {self.n_modes}) or "
                f"({self.n_wavelengths}, n_out, {self.n_modes})"
            )
        xp = target.xp
        out = xp.matmul(s, self._amplitudes[..., None])[..., 0]
        n_out = int(shape[-2])
        if labels is not None:
            new_labels = _labels(labels, n_out)
        else:
            new_labels = self._labels if n_out == self.n_modes else None
        return self._with(out, new_labels)

    def to_backend(self, name: str) -> ModalField:
        """Return the same amplitudes held by backend ``name``."""
        return self._with(convert(self._amplitudes, name), self._labels)

    def __mul__(self, other: Any) -> ModalField:
        """Multiply by a scalar or per-mode factors (broadcast on the last axis)."""
        if isinstance(other, OpticalSignal):
            return NotImplemented
        return self._with(self._amplitudes * operand(other, self.backend), self._labels)

    __rmul__ = __mul__

    def __repr__(self) -> str:
        """Summarise the modal field."""
        return (
            f"ModalField(shape={self.shape}, wavelengths={self._wavelengths}, labels={self._labels}, "
            f"power_unit={self._power_unit}, sources={self._sources!r})"
        )

    # --- JAX pytree -------------------------------------------------------------------------------

    def _tree_flatten(self) -> tuple[tuple[Any], tuple[Any, ...]]:
        aux = (
            tuple(float(v) for v in np.asarray(self._wavelengths.value)),
            self._wavelengths.unit,
            tuple(float(v) for v in self._weights),
            self._labels,
            self._power_unit,
            self._sources,
        )
        return (self._amplitudes,), aux

    @classmethod
    def _tree_unflatten(cls, aux: tuple[Any, ...], children: Any) -> ModalField:
        values, unit, weights, labels, power_unit, sources = aux
        new = object.__new__(cls)
        new._amplitudes = children[0]
        new._wavelengths = u.Quantity(np.asarray(values), unit)
        new._weights = np.asarray(weights)
        new._labels = labels
        new._power_unit = power_unit
        new._sources = sources
        return new

    # --- Serialisation ----------------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        """Encode the modal field as a strict-JSON-compatible dictionary."""
        return {
            "wavelengths": quantity_to_json(self._wavelengths),
            "weights": quantity_to_json(u.Quantity(self._weights)),
            "labels": list(self._labels) if self._labels is not None else None,
            "power_unit": self._power_unit.to_string(),
            "sources": self._sources,
            "amplitudes": quantity_to_json(u.Quantity(get(self.backend).to_numpy(self._amplitudes))),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ModalField:
        """Decode a dictionary produced by :meth:`to_json` into the current backend."""
        return cls(
            quantity_from_json(data["amplitudes"]).value,
            quantity_from_json(data["wavelengths"]),
            weights=quantity_from_json(data["weights"]).value,
            labels=cast(Any, data.get("labels")),
            power_unit=u.Unit(data["power_unit"]),
            sources=data.get("sources"),
        )
