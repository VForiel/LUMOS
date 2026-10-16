r"""Sets of mutually incoherent modes (REQ-SIG-5).

An :class:`IncoherentBundle` holds :math:`M` fields :math:`U_m` (the *modes*) that are mutually incoherent:
point sources of a scene, samples of an extended object, coherent modes of a partially coherent beam or the
two orthogonal polarisations of natural light. Each mode is a :class:`~lumos.fields.Wavefront` or a
:class:`~lumos.fields.Wavecube` and all modes share the same grid, wavelengths and power unit, so the bundle
is one array of shape ``(M, *mode_shape)`` that elements process in a single vectorised operation.

Since the modes do not interfere (Born & Wolf, *Principles of Optics*, §10.3; Goodman, *Statistical Optics*,
§5.2), detected quantities are **sums of intensities** with mode weights :math:`v_m` (default 1):

.. math:: I = \sum_m v_m I_m, \qquad P = \sum_m v_m P_m ,

where :math:`I_m` and :math:`P_m` are the (spectrally integrated) intensity and power of mode :math:`m`.

A point source at infinity in the direction :math:`(\theta_x, \theta_y)` illuminates a pupil with the plane
wave :math:`\exp[ik(x\sin\theta_x + y\sin\theta_y)]` (:meth:`IncoherentBundle.point_sources`).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, cast

import numpy as np
from astropy import units as u

from lumos.backend._core import backend_of, convert, get, operand
from lumos.backend.memory import dtypes
from lumos.core.signal import OpticalSignal
from lumos.fields.grid import Grid
from lumos.fields.wavecube import Wavecube
from lumos.fields.wavefront import Wavefront
from lumos.units import quantity_from_json, quantity_to_json, require_unit, to_si

__all__ = ["IncoherentBundle"]

Member = Wavefront | Wavecube


class IncoherentBundle(OpticalSignal):
    """Mutually incoherent modes sharing a grid, wavelengths and power unit.

    Build bundles with :meth:`from_signals`, :meth:`point_sources` or :meth:`unpolarized`.

    Notes
    -----
    Bundles are immutable JAX pytrees whose only leaf is the field of shape ``(M, *mode_shape)``.
    """

    __slots__ = ("_field", "_member_aux", "_member_cls", "_mode_weights")
    __array_ufunc__ = None

    _field: Any
    _member_cls: type[Wavefront] | type[Wavecube]
    _member_aux: tuple[Any, ...]
    _mode_weights: np.ndarray

    def __init__(self, field: Any, template: Member, *, mode_weights: Any = None) -> None:
        """Wrap ``field`` of shape ``(M, *template.shape)``; ``template`` gives grid, wavelengths and unit."""
        if not isinstance(template, Wavefront | Wavecube):
            raise TypeError("template must be a Wavefront or a Wavecube")
        target = get(template.backend)
        _, complex_dtype = dtypes(backend=target.name)
        array = target.asarray(field if target.is_array(field) else np.asarray(field), dtype=complex_dtype)
        n = int(array.shape[0]) if array.ndim else 0
        weights = np.ones(n) if mode_weights is None else np.asarray(mode_weights, dtype=float)
        self._init(array, type(template), template._tree_flatten()[1], weights, template.shape)

    def _init(
        self,
        field: Any,
        member_cls: type[Wavefront] | type[Wavecube],
        member_aux: tuple[Any, ...],
        mode_weights: np.ndarray,
        mode_shape: tuple[int, ...],
    ) -> None:
        if tuple(field.shape[1:]) != mode_shape or field.shape[0] < 1:
            raise ValueError(
                f"the field shape {tuple(field.shape)} must be (M, *{mode_shape}) with M >= 1 modes"
            )
        if mode_weights.shape != (field.shape[0],) or not np.all(np.isfinite(mode_weights)):
            raise ValueError(
                f"mode_weights must hold {field.shape[0]} finite values, got {mode_weights.shape}"
            )
        self._field = field
        self._member_cls = member_cls
        self._member_aux = member_aux
        self._mode_weights = mode_weights

    def _with_field(self, field: Any) -> IncoherentBundle:
        new = object.__new__(type(self))
        new._init(field, self._member_cls, self._member_aux, self._mode_weights, tuple(field.shape[1:]))
        return new

    @property
    def _template(self) -> Member:
        """Mode 0, which carries the metadata shared by every mode."""
        return self[0]

    # --- Factories --------------------------------------------------------------------------------

    @classmethod
    def from_signals(cls, signals: Sequence[Member], *, weights: Any = None) -> IncoherentBundle:
        """Bundle wavefronts (or wavecubes) sharing grid, wavelengths, power unit and polarisation."""
        if not signals:
            raise ValueError("an IncoherentBundle needs at least one mode")
        first = signals[0]
        for signal in signals[1:]:
            if type(signal) is not type(first):
                raise ValueError("cannot mix Wavefront and Wavecube modes")
            if signal.grid != first.grid:
                raise ValueError("every mode must share the same grid")
            if not np.array_equal(_wavelengths_si(signal), _wavelengths_si(first)):
                raise ValueError("every mode must share the same wavelength(s)")
            if signal.power_unit != first.power_unit or signal.polarized != first.polarized:
                raise ValueError("every mode must share the same power unit and polarisation")
            if isinstance(signal, Wavecube) and not np.array_equal(
                signal.weights, cast(Wavecube, first).weights
            ):
                raise ValueError("every mode must share the same spectral weights")
        xp = get(first.backend).xp
        field = xp.stack([convert(signal.field, first.backend) for signal in signals])
        return cls(field, first, mode_weights=weights)

    @classmethod
    def point_sources(
        cls,
        grid: Grid,
        wavelengths: u.Quantity,
        *,
        theta_x: u.Quantity,
        theta_y: u.Quantity,
        power: u.Quantity,
        weights: Any = None,
        mode_weights: Any = None,
        polarization: Any = None,
    ) -> IncoherentBundle:
        """Plane waves of point sources at infinity seen by a pupil sampled on ``grid``.

        Parameters
        ----------
        grid
            Spatial pupil grid.
        wavelengths
            One wavelength (modes are wavefronts) or several (modes are wavecubes).
        theta_x, theta_y
            Directions of the ``M`` sources (angles from the optical axis in the x-z and y-z planes).
        power
            Power of each source collected by the grid (per spectral channel for several wavelengths).
        weights
            Spectral weights (several wavelengths only).
        mode_weights
            Weights of the modes in the incoherent sum.
        polarization
            Optional Jones vector or state name making polarised modes.
        """
        tx = np.atleast_1d(np.asarray(to_si(require_unit(theta_x, u.rad, name="theta_x")), dtype=float))
        ty = np.atleast_1d(np.asarray(to_si(require_unit(theta_y, u.rad, name="theta_y")), dtype=float))
        if not isinstance(power, u.Quantity):
            raise TypeError("power must be a Quantity in W or photon / s")
        powers = cast(u.Quantity, np.atleast_1d(power))
        if not (tx.shape == ty.shape == powers.shape) or tx.ndim != 1:
            raise ValueError("theta_x, theta_y and power must be 1D and of the same length")
        lam = require_unit(wavelengths, u.m, name="wavelengths", equivalencies=u.spectral())
        members: list[Member] = []
        for m in range(len(tx)):
            p = cast(u.Quantity, powers[m])
            if np.ndim(lam) == 0:
                wf = Wavefront.plane(grid, lam, power=p, polarization=polarization)
                members.append(wf.tilt(tx[m] * u.rad, ty[m] * u.rad))
            else:
                wc = Wavecube.plane(grid, lam, power=p, weights=weights, polarization=polarization)
                members.append(wc.tilt(tx[m] * u.rad, ty[m] * u.rad))
        return cls.from_signals(members, weights=mode_weights)

    @classmethod
    def unpolarized(cls, signal: Member) -> IncoherentBundle:
        """Natural light with the spatial and spectral content of a scalar ``signal``: two orthogonal modes.

        The power of ``signal`` is split equally between the H and V polarisations, which do not interfere.
        """
        if signal.polarized:
            raise ValueError("unpolarized() needs a scalar signal")
        if isinstance(signal, Wavefront):
            half = signal * (0.5**0.5)
            return cls.from_signals([half.with_polarization("H"), half.with_polarization("V")])
        xp = get(signal.backend).xp
        zeros = xp.zeros_like(signal.field)
        scaled = signal.field * (0.5**0.5)
        h = xp.stack([scaled, zeros], axis=1)
        v = xp.stack([zeros, scaled], axis=1)
        return cls.from_signals([signal._with_field(h), signal._with_field(v)])

    # --- Attributes -------------------------------------------------------------------------------

    @property
    def field(self) -> Any:
        """Complex amplitudes of the modes, shape ``(M, *mode_shape)``."""
        return self._field

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of :attr:`field`."""
        return tuple(self._field.shape)

    @property
    def n_modes(self) -> int:
        """Number of modes ``M``."""
        return int(self._field.shape[0])

    @property
    def mode_weights(self) -> np.ndarray:
        """Weights of the modes in the incoherent sum (a copy)."""
        return self._mode_weights.copy()

    @property
    def grid(self) -> Grid:
        """Common grid of the modes."""
        return self._template.grid

    @property
    def polarized(self) -> bool:
        """Whether the modes hold Jones components."""
        return self._template.polarized

    @property
    def polychromatic(self) -> bool:
        """Whether the modes are wavecubes."""
        return isinstance(self._template, Wavecube)

    @property
    def power_unit(self) -> u.UnitBase:
        """Normalisation of the modes."""
        return self._template.power_unit

    @property
    def intensity_unit(self) -> u.UnitBase:
        """Unit of :attr:`intensity`."""
        return self._template.intensity_unit

    @property
    def backend(self) -> str:
        """Name of the backend holding the field."""
        return backend_of(self._field)

    def __len__(self) -> int:
        """Return the number of modes."""
        return self.n_modes

    def __getitem__(self, m: int) -> Member:
        """Mode ``m`` as a wavefront or a wavecube."""
        return self._member_cls._tree_unflatten(self._member_aux, (self._field[m],))

    def members(self) -> Iterator[Member]:
        """Iterate over the modes."""
        for m in range(self.n_modes):
            yield self[m]

    # --- Derived quantities -----------------------------------------------------------------------

    @property
    def _xp(self) -> Any:
        return get(self.backend).xp

    def _weights_array(self, values: np.ndarray, ndim: int) -> Any:
        real, _ = dtypes(backend=self.backend)
        return get(self.backend).asarray(values.reshape((-1,) + (1,) * (ndim - 1)), dtype=real)

    @property
    def mode_intensity(self) -> Any:
        """Intensity of each mode (spectrally integrated for wavecubes), shape ``(M, ny, nx)``."""
        xp = self._xp
        density = xp.real(self._field * xp.conj(self._field))
        if self.polarized:
            density = xp.sum(density, axis=-3)
        if self.polychromatic:
            spectral = cast(Wavecube, self._template).weights
            density = xp.sum(density * self._weights_array(spectral, 3)[None], axis=1)
        return density

    @property
    def intensity(self) -> Any:
        r"""Incoherent sum :math:`\sum_m v_m I_m`, shape ``grid.shape``."""
        return self._xp.sum(self._weights_array(self._mode_weights, 3) * self.mode_intensity, axis=0)

    @property
    def power(self) -> u.Quantity:
        """Power of each mode (spectrally integrated for wavecubes), shape ``(M,)``."""
        totals = get(self.backend).to_numpy(self._xp.sum(self.mode_intensity, axis=(1, 2)))
        grid = self.grid
        return u.Quantity(np.asarray(totals, dtype=float) * grid.dx_si * grid.dy_si, self.power_unit)

    @property
    def total_power(self) -> u.Quantity:
        r"""Weighted total power :math:`\sum_m v_m P_m`."""
        return u.Quantity(float(np.sum(self._mode_weights * self.power.value)), self.power_unit)

    # --- Operations -------------------------------------------------------------------------------

    def map(self, function: Any) -> IncoherentBundle:
        """Apply ``function`` (member signal -> member signal) to every mode and bundle the results."""
        return IncoherentBundle.from_signals(
            [function(member) for member in self.members()], weights=self._mode_weights
        )

    def with_power_unit(self, unit: Any) -> IncoherentBundle:
        """Return the same light in another power unit."""
        return self.map(lambda member: member.with_power_unit(unit))

    def tilt(self, theta_x: u.Quantity, theta_y: u.Quantity) -> IncoherentBundle:
        """Tilt every mode by the same angles."""
        return self.map(lambda member: member.tilt(theta_x, theta_y))

    def to_backend(self, name: str) -> IncoherentBundle:
        """Return the same bundle held by backend ``name``."""
        return self._with_field(convert(self._field, name))

    def __mul__(self, other: Any) -> IncoherentBundle:
        """Multiply every mode by a scalar or a mask (broadcast like the member signals)."""
        if isinstance(other, IncoherentBundle | Wavefront | Wavecube):
            return NotImplemented
        other = operand(other, self.backend)
        if hasattr(other, "shape") and len(other.shape) == 3 and self.polarized and self.polychromatic:
            other = other[:, None]
        return self._with_field(self._field * other)

    __rmul__ = __mul__

    def __repr__(self) -> str:
        """Summarise the bundle."""
        return f"IncoherentBundle(n_modes={self.n_modes}, mode={self._template!r})"

    # --- JAX pytree -------------------------------------------------------------------------------

    def _tree_flatten(self) -> tuple[tuple[Any], tuple[Any, ...]]:
        aux = (self._member_cls, self._member_aux, tuple(float(v) for v in self._mode_weights))
        return (self._field,), aux

    @classmethod
    def _tree_unflatten(cls, aux: tuple[Any, ...], children: Any) -> IncoherentBundle:
        member_cls, member_aux, weights = aux
        new = object.__new__(cls)
        new._field = children[0]
        new._member_cls = member_cls
        new._member_aux = member_aux
        new._mode_weights = np.asarray(weights)
        return new

    # --- Serialisation ----------------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        """Encode the bundle as a strict-JSON-compatible dictionary."""
        return {
            "mode": {"kind": type(self._template).__name__, **self._template.to_json()},
            "mode_weights": quantity_to_json(u.Quantity(self._mode_weights)),
            "field": quantity_to_json(u.Quantity(get(self.backend).to_numpy(self._field))),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> IncoherentBundle:
        """Decode a dictionary produced by :meth:`to_json` into the current backend."""
        mode = dict(data["mode"])
        kind = mode.pop("kind")
        member_cls: type[Wavefront] | type[Wavecube] = {"Wavefront": Wavefront, "Wavecube": Wavecube}[kind]
        template = member_cls.from_json(mode)
        return cls(
            quantity_from_json(data["field"]).value,
            template,
            mode_weights=quantity_from_json(data["mode_weights"]).value,
        )


def _wavelengths_si(signal: Member) -> np.ndarray:
    if isinstance(signal, Wavecube):
        return signal.wavelengths_si
    return np.array([signal.wavelength_si])
