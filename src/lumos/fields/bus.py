r"""Buses: N lanes of the same kind of signal, separated in space (ADR-0023).

A :class:`Bus` groups the signals of guides or beams that are separated in space, so that different
components can be placed on each of them: the waveguides of a photonic chip, the arms of an interferometer.
Each *lane* is one signal: a guided lane is a :class:`~lumos.fields.ModalField` of one guide (its modes), a
free-space lane a :class:`~lumos.fields.Wavefront`, :class:`~lumos.fields.Wavecube` or
:class:`~lumos.fields.IncoherentBundle`. Spectral channels, incoherent sources and the modes of one guide are
never lanes: they share the same beam or guide.

A bus behaves as a sequence (``len(bus)``, ``bus[i]``, iteration) and names its lanes (``T1``, ``T2``... by
default); it may also carry the projected position of the telescope feeding each lane.

Implicit packing
----------------
When its lanes share their metadata, a bus stores them in one array, so that a component acting on all the
lanes works on that array without copying them:

- guided lanes with the same sources, wavelengths, spectral weights and power unit are concatenated along the
  mode axis, ``(sources, λ, Σ modes)``, the lanes possibly having different numbers of modes
  (:attr:`Bus.modal`, :attr:`Bus.mode_counts`);
- free-space lanes of the same class with identical metadata (grid, sources, wavelengths, weights,
  polarisation, power unit) and shape are stacked, ``(lanes, *lane_shape)`` (:attr:`Bus.field`).

Otherwise the bus keeps the list of its lanes; :meth:`Bus.packing_problem` says why. Both forms have the
same interface.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
from astropy import units as u

from lumos.backend._core import backend_of, convert, get
from lumos.core.signal import OpticalSignal
from lumos.fields.incoherent import IncoherentBundle
from lumos.fields.modal import ModalField
from lumos.fields.wavecube import Wavecube
from lumos.fields.wavefront import Wavefront

__all__ = ["Bus"]

_LANE_CLASSES: dict[str, type[Any]] = {
    cls.__name__: cls for cls in (ModalField, Wavefront, Wavecube, IncoherentBundle)
}


def _wavelengths_si(lane: Any) -> np.ndarray:
    if isinstance(lane, IncoherentBundle):
        lane = lane[0]
    if isinstance(lane, Wavefront):
        return np.array([lane.wavelength_si])
    return np.asarray(lane.wavelengths_si, dtype=float)


def _weights(lane: Any) -> np.ndarray | None:
    if isinstance(lane, IncoherentBundle):
        lane = lane[0]
    return None if isinstance(lane, Wavefront) else np.asarray(lane.weights)


def _guided_problem(first: ModalField, lane: ModalField) -> str | None:
    if lane.sources != first.sources or lane.n_sources != first.n_sources:
        return (
            f"the lanes describe different sources ({first.n_sources} source(s) '{first.sources}' and "
            f"{lane.n_sources} source(s) '{lane.sources}')"
        )
    if not np.array_equal(lane.wavelengths_si, first.wavelengths_si):
        return "the lanes have different wavelengths"
    if not np.array_equal(lane.weights, first.weights):
        return "the lanes have different spectral weights"
    if lane.power_unit != first.power_unit:
        return f"the lanes have different power units ({first.power_unit} and {lane.power_unit})"
    return None


def _free_space_problem(first: Any, lane: Any) -> str | None:
    if getattr(lane, "sources", None) != getattr(first, "sources", None):
        return f"the lanes describe different sources ('{first.sources}' and '{lane.sources}')"
    if lane.grid != first.grid:
        return "the lanes are sampled on different grids"
    if not np.array_equal(_wavelengths_si(lane), _wavelengths_si(first)):
        return "the lanes have different wavelengths"
    weights, first_weights = _weights(lane), _weights(first)
    if weights is not None and first_weights is not None and not np.array_equal(weights, first_weights):
        return "the lanes have different spectral weights"
    if lane.power_unit != first.power_unit:
        return f"the lanes have different power units ({first.power_unit} and {lane.power_unit})"
    if lane.polarized != first.polarized:
        return "some lanes are polarised and others not"
    if tuple(lane.shape) != tuple(first.shape):
        return f"the lanes have different shapes ({tuple(first.shape)} and {tuple(lane.shape)})"
    if lane._tree_flatten()[1] != first._tree_flatten()[1]:
        return "the lanes have different metadata"
    return None


def _default_names(n: int) -> tuple[str, ...]:
    return tuple(f"T{i + 1}" for i in range(n))


def _names(names: Sequence[str] | None, n: int) -> tuple[str, ...]:
    if names is None:
        return _default_names(n)
    result = tuple(str(name) for name in names)
    if len(result) != n:
        raise ValueError(f"{n} lane names expected, got {len(result)}")
    return result


def _positions(positions: Any, n: int) -> np.ndarray | None:
    if positions is None:
        return None
    if not isinstance(positions, u.Quantity):
        raise TypeError("the positions of the lanes are a length quantity, e.g. [[0, 0], [47, 0]] * u.m")
    array = np.asarray(positions.to_value(u.m), dtype=float)
    if array.shape != (n, 2):
        raise ValueError(f"the positions of the lanes must have the shape ({n}, 2), got {array.shape}")
    return array


def _quantity(positions: np.ndarray | None) -> u.Quantity | None:
    return None if positions is None else positions * u.m


def _hashable(positions: np.ndarray | None) -> tuple[tuple[float, float], ...] | None:
    return None if positions is None else tuple((float(x), float(y)) for x, y in positions)


class Bus(OpticalSignal):
    """N lanes of the same signal class, separated in space, packed in one array when possible.

    Parameters
    ----------
    lanes
        The signals of the lanes, all of the same class (``ModalField``, ``Wavefront``, ``Wavecube`` or
        ``IncoherentBundle``).
    names
        Names of the lanes (default ``T1``... ``Tn``).
    positions
        Projected positions ``(x, y)`` of the telescopes feeding the lanes, shape ``(N, 2)``, as a length
        quantity; ``NaN`` rows for unknown positions.
    pack
        Store the lanes in one array when they share their metadata (default). ``False`` keeps a list, which
        behaves the same.
    """

    __slots__ = ("_counts", "_data", "_form", "_lane_aux", "_lane_cls", "_names", "_positions")
    __array_ufunc__ = None

    _form: str  # "guided", "stacked" or "list"
    _data: Any
    _lane_cls: type[Any]
    _lane_aux: Any
    _counts: tuple[int, ...]
    _names: tuple[str, ...]
    _positions: np.ndarray | None

    def __init__(
        self,
        lanes: Sequence[Any],
        *,
        names: Sequence[str] | None = None,
        positions: Any = None,
        pack: bool = True,
    ) -> None:
        lanes = list(lanes)
        if not lanes:
            raise ValueError("a bus needs at least one lane")
        first = lanes[0]
        if isinstance(first, Bus) or type(first) not in _LANE_CLASSES.values():
            raise TypeError(
                f"a lane is a ModalField, Wavefront, Wavecube or IncoherentBundle, got {type(first).__name__}"
            )
        for lane in lanes[1:]:
            if type(lane) is not type(first):
                raise TypeError(
                    f"every lane of a bus has the same signal class: {type(first).__name__} and "
                    f"{type(lane).__name__}"
                )
        self._names = _names(names, len(lanes))
        self._positions = _positions(positions, len(lanes))
        self._lane_cls = type(first)
        self._lane_aux = None
        self._counts = tuple(lane.n_modes for lane in lanes) if isinstance(first, ModalField) else ()
        if pack and Bus.packing_problem(lanes) is None:
            backend = first.backend
            xp = get(backend).xp
            if isinstance(first, ModalField):
                self._form = "guided"
                amplitudes = xp.concat([convert(lane.amplitudes, backend) for lane in lanes], axis=-1)
                labels = None  # unnamed modes of a lane are "" among named ones
                if any(lane.labels is not None for lane in lanes):
                    labels = [label for lane in lanes for label in (lane.labels or ("",) * lane.n_modes)]
                self._data = ModalField(
                    amplitudes,
                    first.wavelengths,
                    weights=first.weights,
                    labels=labels,
                    power_unit=first.power_unit,
                    sources=first.sources,
                )
            else:
                self._form = "stacked"
                self._lane_aux = first._tree_flatten()[1]
                self._data = xp.stack([convert(lane._tree_flatten()[0][0], backend) for lane in lanes])
        else:
            self._form = "list"
            self._data = tuple(lanes)

    # --- Factories --------------------------------------------------------------------------------

    @classmethod
    def guided(
        cls,
        amplitudes: Any,
        wavelengths: u.Quantity,
        *,
        modes: Sequence[int] | None = None,
        names: Sequence[str] | None = None,
        positions: Any = None,
        weights: Any = None,
        labels: Sequence[str] | None = None,
        power_unit: Any = u.W,
        sources: str | None = None,
    ) -> Bus:
        """Build a packed bus of guided lanes from the amplitudes of all their modes.

        ``amplitudes`` has the shape ``(nλ, Σ modes)`` or ``(sources, nλ, Σ modes)``; ``modes`` gives the
        number of modes of each lane (default: one single-mode lane per column). ``labels`` name the modes.
        """
        modal = ModalField(
            amplitudes, wavelengths, weights=weights, labels=labels, power_unit=power_unit, sources=sources
        )
        counts = (1,) * modal.n_modes if modes is None else tuple(int(m) for m in modes)
        if any(m < 1 for m in counts) or sum(counts) != modal.n_modes:
            raise ValueError(
                f"the numbers of modes of the lanes {list(counts)} must be positive and add up to the "
                f"{modal.n_modes} modes of the amplitudes"
            )
        n = len(counts)
        return cls._packed_guided(modal, counts, _names(names, n), _positions(positions, n))

    @classmethod
    def _packed_guided(
        cls, modal: ModalField, counts: tuple[int, ...], names: tuple[str, ...], positions: np.ndarray | None
    ) -> Bus:
        new = object.__new__(cls)
        new._form = "guided"
        new._data = modal
        new._lane_cls = ModalField
        new._lane_aux = None
        new._counts = counts
        new._names = names
        new._positions = positions
        return new

    @staticmethod
    def packing_problem(lanes: Sequence[Any]) -> str | None:
        """Return why ``lanes`` cannot be packed in one array, or ``None`` if they can."""
        lanes = list(lanes)
        if not lanes:
            return "there is no lane"
        first = lanes[0]
        for lane in lanes[1:]:
            if type(lane) is not type(first):
                classes = f"{type(first).__name__}, {type(lane).__name__}"
                return f"the lanes are signals of different classes ({classes})"
            check = _guided_problem if isinstance(first, ModalField) else _free_space_problem
            problem = check(first, lane)
            if problem is not None:
                return problem
        return None

    # --- Sequence ---------------------------------------------------------------------------------

    @property
    def n_lanes(self) -> int:
        """Number of lanes."""
        return len(self._names)

    def __len__(self) -> int:
        """Return the number of lanes."""
        return len(self._names)

    def __getitem__(self, index: int) -> Any:
        """Return lane ``index`` as a signal."""
        n = len(self._names)
        if not -n <= index < n:
            raise IndexError(f"lane {index} of a bus of {n} lanes")
        index %= n
        if self._form == "list":
            return self._data[index]
        if self._form == "stacked":
            return self._lane_cls._tree_unflatten(self._lane_aux, (self._data[index],))
        start = sum(self._counts[:index])
        stop = start + self._counts[index]
        modal: ModalField = self._data
        labels = None if modal.labels is None else modal.labels[start:stop]
        if labels is not None and not any(labels):
            labels = None
        return modal._with(modal.amplitudes[..., start:stop], labels)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the lanes."""
        for index in range(len(self._names)):
            yield self[index]

    @property
    def lanes(self) -> tuple[Any, ...]:
        """The signals of the lanes."""
        return tuple(self)

    @property
    def names(self) -> tuple[str, ...]:
        """Names of the lanes."""
        return self._names

    @property
    def positions(self) -> u.Quantity | None:
        """Projected positions of the telescopes feeding the lanes, ``(N, 2)``, or ``None``."""
        return None if self._positions is None else self._positions * u.m

    @property
    def lane_type(self) -> type[Any]:
        """Signal class of the lanes."""
        return self._lane_cls

    @property
    def packed(self) -> bool:
        """Whether the lanes are stored in one array."""
        return self._form != "list"

    @property
    def backend(self) -> str:
        """Name of the backend holding the lanes."""
        if self._form == "stacked":
            return backend_of(self._data)
        if self._form == "guided":
            return self._data.backend
        return self._data[0].backend

    @property
    def sources(self) -> str | None:
        """Identifier of the source list shared by every lane, or ``None``."""
        tags = {getattr(lane, "sources", None) for lane in self}
        return tags.pop() if len(tags) == 1 else None

    # --- Guided buses -----------------------------------------------------------------------------

    @property
    def mode_counts(self) -> tuple[int, ...]:
        """Number of modes of each guided lane."""
        if self._lane_cls is not ModalField:
            raise TypeError(f"a bus of {self._lane_cls.__name__} has no guided modes")
        return self._counts

    @property
    def modal(self) -> ModalField:
        """All the modes of all the lanes of a packed guided bus, ``(sources, λ, Σ modes)``."""
        if self._lane_cls is not ModalField:
            raise TypeError(f"a bus of {self._lane_cls.__name__} has no guided modes")
        if self._form != "guided":
            raise ValueError(f"the lanes of this bus cannot be packed: {Bus.packing_problem(self._data)}")
        return self._data

    def with_modal(self, modal: ModalField, counts: Sequence[int], names: Sequence[str]) -> Bus:
        """Return a packed guided bus of the modes ``modal``, lanes of ``counts`` modes named ``names``.

        Positions are kept for the lanes whose name is unchanged.
        """
        counts = tuple(int(c) for c in counts)
        names = tuple(names)
        positions = None
        if self._positions is not None:
            known = dict(zip(self._names, self._positions, strict=True))
            positions = np.array([known.get(name, (np.nan, np.nan)) for name in names], dtype=float)
        return Bus._packed_guided(modal, counts, names, positions)

    # --- Free-space buses -------------------------------------------------------------------------

    @property
    def field(self) -> Any:
        """Stacked arrays of the lanes of a packed free-space bus, ``(lanes, *lane_shape)``."""
        if self._form != "stacked":
            if self._lane_cls is ModalField:
                raise TypeError("a guided bus has no field; use modal")
            raise ValueError(f"the lanes of this bus cannot be packed: {Bus.packing_problem(self._data)}")
        return self._data

    # --- Operations -------------------------------------------------------------------------------

    def select(self, indices: Sequence[int]) -> Bus:
        """Return the bus of the lanes ``indices``, in this order, with their names and positions."""
        indices = [int(i) % len(self) for i in indices]
        names = [self._names[i] for i in indices]
        positions = None if self._positions is None else self._positions[indices]
        if self._form == "guided":
            starts = np.cumsum((0, *self._counts))
            columns = [c for i in indices for c in range(starts[i], starts[i + 1])]
            modal: ModalField = self._data
            labels = None if modal.labels is None else tuple(modal.labels[c] for c in columns)
            xp = get(modal.backend).xp
            amplitudes = xp.take(modal.amplitudes, xp.asarray(columns), axis=-1)
            counts = tuple(self._counts[i] for i in indices)
            return Bus._packed_guided(modal._with(amplitudes, labels), counts, tuple(names), positions)
        return Bus([self[i] for i in indices], names=names, positions=_quantity(positions))

    @staticmethod
    def concatenate(buses: Sequence[Bus]) -> Bus:
        """Put the lanes of ``buses`` side by side (packed again when possible), without mixing them."""
        buses = list(buses)
        if not buses:
            raise ValueError("nothing to concatenate")
        lanes = [lane for bus in buses for lane in bus]
        names = [name for bus in buses for name in bus.names]
        positions = None
        if any(bus._positions is not None for bus in buses):
            unknown = [np.full((len(bus), 2), np.nan) for bus in buses]
            positions = np.concatenate(
                [
                    bus._positions if bus._positions is not None else nan
                    for bus, nan in zip(buses, unknown, strict=True)
                ]
            )
        return Bus(lanes, names=names, positions=_quantity(positions))

    def map(self, function: Callable[[Any], Any]) -> Bus:
        """Apply ``function`` (lane -> lane) to every lane; names and positions are kept."""
        return Bus([function(lane) for lane in self], names=self._names, positions=self.positions)

    def to_backend(self, name: str) -> Bus:
        """Return the same lanes held by backend ``name``."""
        if self._form == "guided":
            return Bus._packed_guided(self._data.to_backend(name), self._counts, self._names, self._positions)
        return Bus(
            [lane.to_backend(name) for lane in self],
            names=self._names,
            positions=self.positions,
            pack=self.packed,
        )

    def __repr__(self) -> str:
        """Summarise the bus."""
        form = "packed" if self.packed else "list"
        return f"Bus({len(self)} × {self._lane_cls.__name__}, {form}, names={self._names})"

    # --- JAX pytree -------------------------------------------------------------------------------

    def _tree_flatten(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        meta = (
            self._form,
            self._lane_cls,
            self._lane_aux,
            self._counts,
            self._names,
            _hashable(self._positions),
        )
        children = self._data if self._form == "list" else (self._data,)
        return tuple(children), meta

    @classmethod
    def _tree_unflatten(cls, aux: tuple[Any, ...], children: Any) -> Bus:
        form, lane_cls, lane_aux, counts, names, positions = aux
        new = object.__new__(cls)
        new._form = form
        new._data = tuple(children) if form == "list" else children[0]
        new._lane_cls = lane_cls
        new._lane_aux = lane_aux
        new._counts = counts
        new._names = names
        new._positions = None if positions is None else np.asarray(positions, dtype=float)
        return new

    # --- Serialisation ----------------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        """Encode the bus as a strict-JSON-compatible dictionary (lanes listed one by one)."""
        positions = None
        if self._positions is not None:
            positions = [[None if np.isnan(v) else float(v) for v in row] for row in self._positions]
        return {
            "names": list(self._names),
            "positions_m": positions,
            "packed": self.packed,
            "lanes": [{"kind": self._lane_cls.__name__, **lane.to_json()} for lane in self],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Bus:
        """Decode a dictionary produced by :meth:`to_json` into the current backend."""
        lanes = []
        for entry in data["lanes"]:
            entry = dict(entry)
            lanes.append(_LANE_CLASSES[entry.pop("kind")].from_json(entry))
        rows = data.get("positions_m")
        positions = None
        if rows is not None:
            positions = (
                np.array([[np.nan if v is None else v for v in row] for row in rows], dtype=float) * u.m
            )
        return cls(lanes, names=data["names"], positions=positions, pack=bool(data.get("packed", True)))
