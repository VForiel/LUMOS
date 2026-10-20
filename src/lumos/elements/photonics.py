r"""Integrated photonics (§5.9, steps P6.8 and P15.6): components, chips and recombiner presets.

Every component maps a :class:`~lumos.fields.Bus` of guided lanes (one guide each, ADR-0023) to a new bus
through its scattering matrix :math:`S(\lambda)` (:mod:`lumos.fields.smatrix`), acting on the lanes listed in
its ``channels`` setting (all lanes when empty) and leaving the others unchanged; a single
:class:`~lumos.fields.ModalField` is a bus of one lane. A diagonal component (waveguide, phase shifter) acts
on each selected lane; a mixing component applies its matrix to the vector of all the modes of its selected
lanes, which must describe the same sources (otherwise they cannot interfere). Components with as many outputs
as input modes write their outputs on the same lanes; the others (Y-junctions, non-square matrices) put
single-mode output lanes where their first selected lane was. A chip is an
:class:`~lumos.core.group.ElementGroup` chaining components; :func:`bracewell` and :func:`kernel_nuller` build
the recombiners of Bracewell (1978) and Martinache & Ireland (2018).

Components
----------
- :class:`Waveguide`: :math:`10^{-\alpha L/20} e^{2i\pi n_\mathrm{eff} L/\lambda}` on each channel.
- :class:`PhaseShifter`: :math:`e^{i\phi}`, with :math:`\phi = \phi_0\lambda_0/\lambda` for a thermo-optic
  shifter (an optical path) or :math:`\phi_0` if achromatic.
- :class:`DirectionalCoupler`: :math:`\begin{pmatrix} t & i\kappa \\ i\kappa & t\end{pmatrix}`,
  :math:`\kappa^2` the power coupling ratio.
- :class:`YJunction`: splitter :math:`(1, 1)^T/\sqrt{2}` or combiner :math:`(1, 1)/\sqrt{2}`, which radiates
  the antisymmetric part of its inputs.
- :class:`Crossing`: transmission :math:`\sqrt{1 - x}` across and crosstalk :math:`i\sqrt{x}`.
- :class:`MMI`: :math:`N \times N` multimode interference coupler computed with the slab mode solver of
  :mod:`lumos.physics.photonics`; with ``length = 0`` it is the general-interference coupler of length
  :math:`3 L_\pi / N` at the reference wavelength (Soldano & Pennings 1995).
- :class:`TransferMatrix`: a simplified MMI given by its matrix :math:`S_{ij} = A_{ij} e^{i\phi_{ij}}` (output
  :math:`i`, input :math:`j`) at a wavelength :math:`\lambda_0`; the same at every wavelength, or with phases
  that are optical paths (:math:`\phi\lambda_0/\lambda`). It may have more or fewer outputs than inputs; a
  passive component has a largest singular value at most 1 (a matrix that amplifies is refused).
"""

from __future__ import annotations

import functools
import itertools
import math
from typing import Any

import numpy as np
from astropy import units as u

from lumos.core.element import Element, Values
from lumos.core.group import ElementGroup
from lumos.core.param import Param, Setting
from lumos.core.port import Input, Output
from lumos.core.registry import register
from lumos.fields import smatrix
from lumos.fields.bus import Bus
from lumos.fields.modal import ModalField
from lumos.physics.photonics import beat_length, effective_width, mmi_matrix, port_positions
from lumos.units import to_si

__all__ = [
    "KERNEL_PAIRS",
    "MMI",
    "Crossing",
    "DirectionalCoupler",
    "PhaseShifter",
    "PhotonicComponent",
    "TransferMatrix",
    "Waveguide",
    "YJunction",
    "bracewell",
    "kernel_nuller",
    "kernels",
]

#: SI value of 1 dB/cm (astropy stores logarithmic units in dex).
_DB_PER_CM = float(to_si(1 * u.dB / u.cm))


def _as_bus(signal: Bus | ModalField) -> Bus:
    """Return ``signal`` as a bus (a single guided lane is a bus of one lane)."""
    return signal if isinstance(signal, Bus) else Bus([signal])


def _selected(values: Values, n_lanes: int) -> list[int]:
    """Return the indices of the lanes listed in ``channels`` (all lanes when empty), in their order."""
    lanes = [int(c) for c in values.channels] if values.channels else list(range(n_lanes))
    if len(set(lanes)) != len(lanes) or not all(0 <= c < n_lanes for c in lanes):
        raise ValueError(f"channels {lanes} must be distinct lane indices below {n_lanes}")
    return lanes


def _modes(bus: Bus, selected: list[int], n_in: int | None) -> ModalField:
    """Return all the modes of the ``selected`` lanes, which must be able to interfere."""
    lanes = [bus[i] for i in selected]
    names = ", ".join(bus.names[i] for i in selected)
    problem = Bus.packing_problem(lanes)
    if problem is not None:
        raise ValueError(f"the lanes {names} cannot interfere: {problem}")
    modal = Bus(lanes).modal
    if n_in is not None and modal.n_modes != n_in:
        raise ValueError(
            f"this component acts on {n_in} mode(s), but the lanes {names} (channels={selected}) carry "
            f"{modal.n_modes}: list the lanes it acts on in the setting channels"
        )
    return modal


def _replace(bus: Bus, selected: list[int], out: ModalField) -> Bus:
    """Put the modes ``out`` of the outputs of a component back into ``bus``.

    With as many output modes as input modes, the outputs replace the selected lanes (same numbers of
    modes); otherwise they are single-mode lanes put where the first selected lane was, named after it.
    """
    lanes: list[Any] = list(bus)
    names = list(bus.names)
    known = bus.positions
    positions = None if known is None else [(float(x), float(y)) for x, y in np.asarray(known.to_value(u.m))]
    counts = [lanes[i].n_modes for i in selected]
    if out.n_modes == sum(counts):
        start = 0
        for i, count in zip(selected, counts, strict=True):
            labels = None if out.labels is None else out.labels[start : start + count]
            lanes[i] = out._with(out.amplitudes[..., start : start + count], labels)
            start += count
        return Bus(lanes, names=names, positions=bus.positions)
    first = min(selected)
    base = names[first]
    new_lanes = [out._with(out.amplitudes[..., k : k + 1], None) for k in range(out.n_modes)]
    rebuilt_lanes: list[Any] = []
    rebuilt_names: list[str] = []
    rebuilt_positions: list[tuple[float, float]] = []
    for index, lane in enumerate(lanes):
        if index == first:
            rebuilt_lanes += new_lanes
            rebuilt_names += [f"{base}.{k}" for k in range(out.n_modes)]
            rebuilt_positions += [(np.nan, np.nan)] * out.n_modes
        elif index not in selected:
            rebuilt_lanes.append(lane)
            rebuilt_names.append(names[index])
            rebuilt_positions.append(positions[index] if positions is not None else (np.nan, np.nan))
    kept = None if positions is None else np.array(rebuilt_positions, dtype=float) * u.m
    return Bus(rebuilt_lanes, names=rebuilt_names, positions=kept)


class PhotonicComponent(Element):
    """Base class: a scattering matrix applied to some lanes of a bus of guides."""

    channels = Setting(list, default=[], doc="indices of the lanes the component acts on ([]: all)")
    in_ = Input(ModalField, bus=True, doc="bus of guided lanes (or one lane)")
    out = Output(ModalField, bus=True, doc="bus of guided lanes")

    #: Number of input modes (None: any number, the matrix is diagonal and acts on each lane).
    n_inputs: int | None = None

    @classmethod
    def _block(cls, values: Values, wavelengths: np.ndarray, n_in: int) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def _transfer(cls, values: Values, inputs: Any) -> dict[str, Any]:
        bus = _as_bus(inputs["in"])
        selected = _selected(values, len(bus))
        if cls.n_inputs is None:
            lanes = list(bus)
            for i in selected:
                lane: ModalField = lanes[i]
                lanes[i] = lane.apply(cls._block(values, lane.wavelengths_si, lane.n_modes))
            return {"out": Bus(lanes, names=bus.names, positions=bus.positions)}
        modes = _modes(bus, selected, cls.n_inputs)
        return {
            "out": _replace(
                bus, selected, modes.apply(cls._block(values, modes.wavelengths_si, modes.n_modes))
            )
        }


def _diagonal(factors: np.ndarray, n_in: int) -> np.ndarray:
    """(nλ,) factors → (nλ, n_in, n_in) diagonal blocks."""
    return factors[:, None, None] * np.eye(n_in)[None]


def _constant(matrix: np.ndarray, n_wavelengths: int) -> np.ndarray:
    return np.broadcast_to(np.asarray(matrix, dtype=complex), (n_wavelengths, *np.shape(matrix)))


@register("lumos.photonics.Waveguide")
class Waveguide(PhotonicComponent):
    """Straight waveguide: propagation phase and loss."""

    length = Param(unit=u.mm, default=1 * u.mm, bounds=(0 * u.m, None), doc="length of the waveguide")
    effective_index = Param(default=1.5, bounds=(1, None), doc="effective index of the guided mode")
    loss = Param(
        unit=u.dB / u.cm, default=0 * u.dB / u.cm, bounds=(0 * u.dB / u.cm, None), doc="propagation loss"
    )

    @classmethod
    def _block(cls, values: Values, wavelengths: np.ndarray, n_in: int) -> np.ndarray:
        decibels = values.loss / _DB_PER_CM * values.length * 100
        phase = 2 * math.pi * values.effective_index * values.length / wavelengths
        return _diagonal(10 ** (-decibels / 20) * np.exp(1j * phase), n_in)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Propagate the selected channels."""
        return Waveguide._transfer(values, inputs)


@register("lumos.photonics.PhaseShifter")
class PhaseShifter(PhotonicComponent):
    """Phase shifter (thermo-optic: the phase is an optical path, set at the reference wavelength)."""

    phase = Param(unit=u.rad, default=0 * u.rad, doc="phase shift at the reference wavelength")
    reference_wavelength = Param(
        unit=u.nm,
        default=1550 * u.nm,
        bounds=(0 * u.m, None),
        inclusive=(False, True),
        doc="wavelength at which the phase is given",
    )
    achromatic = Setting(bool, default=False, doc="True: the same phase at every wavelength")

    @classmethod
    def _block(cls, values: Values, wavelengths: np.ndarray, n_in: int) -> np.ndarray:
        scale = 1.0 if values.achromatic else values.reference_wavelength / wavelengths
        return _diagonal(np.exp(1j * values.phase * np.broadcast_to(scale, wavelengths.shape)), n_in)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Shift the phase of the selected channels."""
        return PhaseShifter._transfer(values, inputs)


@register("lumos.photonics.DirectionalCoupler")
class DirectionalCoupler(PhotonicComponent):
    """Lossless symmetric directional coupler between two channels."""

    n_inputs = 2
    ratio = Param(default=0.5, bounds=(0, 1), doc="power coupling ratio")

    @classmethod
    def _block(cls, values: Values, wavelengths: np.ndarray, n_in: int) -> np.ndarray:
        return _constant(smatrix.coupler(values.ratio), len(wavelengths))

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Couple the two selected channels."""
        return DirectionalCoupler._transfer(values, inputs)


@register("lumos.photonics.YJunction")
class YJunction(PhotonicComponent):
    """Symmetric Y-junction: splits one channel in two, or combines two channels into one."""

    mode = Setting(
        str,
        default="split",
        choices=("split", "combine"),
        doc="'split' one input into two outputs, or 'combine' two into one",
    )

    @classmethod
    def _transfer(cls, values: Values, inputs: Any) -> dict[str, Any]:
        bus = _as_bus(inputs["in"])
        split = values.mode == "split"
        selected = _selected(values, len(bus))
        modes = _modes(bus, selected, 1 if split else 2)
        matrix = np.array([[1.0], [1.0]]) if split else np.array([[1.0, 1.0]])
        block = _constant(matrix / math.sqrt(2), modes.n_wavelengths)
        return {"out": _replace(bus, selected, modes.apply(block))}

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Split or combine the selected channels."""
        return YJunction._transfer(values, inputs)


@register("lumos.photonics.Crossing")
class Crossing(PhotonicComponent):
    """Waveguide crossing with a power crosstalk."""

    n_inputs = 2
    crosstalk = Param(default=0.0, bounds=(0, 1), doc="power fraction staying in the other guide")

    @classmethod
    def _block(cls, values: Values, wavelengths: np.ndarray, n_in: int) -> np.ndarray:
        x = values.crosstalk
        matrix = np.array([[1j * math.sqrt(x), math.sqrt(1 - x)], [math.sqrt(1 - x), 1j * math.sqrt(x)]])
        return _constant(matrix, len(wavelengths))

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Cross the two selected channels."""
        return Crossing._transfer(values, inputs)


@functools.lru_cache(maxsize=256)
def _mmi_block(
    n: int,
    width: float,
    port_width: float,
    n_core: float,
    n_cladding: float,
    length: float,
    reference: float,
    wavelength: float,
) -> np.ndarray:
    if length <= 0:
        length = 3 * beat_length(width, n_core, n_cladding, reference) / n
    ports = port_positions(n, effective_width(width, n_core, n_cladding, reference))
    return mmi_matrix(
        width=width,
        length=length,
        n_core=n_core,
        n_cladding=n_cladding,
        port_width=port_width,
        inputs=ports,
        outputs=ports,
        wavelength=wavelength,
    )


@register("lumos.photonics.MMI")
class MMI(PhotonicComponent):
    """N × N multimode interference coupler computed with a slab mode solver."""

    width = Param(
        unit=u.um,
        default=10 * u.um,
        bounds=(0 * u.m, None),
        inclusive=(False, True),
        doc="width of the multimode section",
    )
    port_width = Param(
        unit=u.um,
        default=3 * u.um,
        bounds=(0 * u.m, None),
        inclusive=(False, True),
        doc="width of the access waveguides",
    )
    length = Param(unit=u.um, default=0 * u.um, bounds=(0 * u.m, None), doc="0: 3 L_pi / N at the reference")
    n_core = Param(default=1.6, bounds=(1, None), doc="effective index of the multimode section")
    n_cladding = Param(default=1.45, bounds=(1, None), doc="refractive index of the cladding")
    reference_wavelength = Param(
        unit=u.nm,
        default=1550 * u.nm,
        bounds=(0 * u.m, None),
        inclusive=(False, True),
        doc="design wavelength of the self-imaging length",
    )
    n_ports = Setting(int, default=2, doc="input and output ports")

    @classmethod
    def _transfer(cls, values: Values, inputs: Any) -> dict[str, Any]:
        bus = _as_bus(inputs["in"])
        n = int(values.n_ports)
        selected = _selected(values, len(bus))
        modes = _modes(bus, selected, n)
        block = np.stack(
            [
                _mmi_block(
                    n,
                    float(values.width),
                    float(values.port_width),
                    float(values.n_core),
                    float(values.n_cladding),
                    float(values.length),
                    float(values.reference_wavelength),
                    float(lam),
                )
                for lam in modes.wavelengths_si
            ]
        )
        return {"out": _replace(bus, selected, modes.apply(block))}

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Mix the selected channels."""
        return MMI._transfer(values, inputs)


_HALF = 1 / math.sqrt(2)

#: Gain tolerated by the check of passivity: the rounding of values typed with 5 or 6 significant digits
#: (0.707107 √2 = 1.0000004).
PASSIVITY_TOLERANCE = 1e-4


@register("lumos.photonics.TransferMatrix")
class TransferMatrix(PhotonicComponent):
    """Simplified MMI: a component given by its complex transfer matrix (amplitude and phase) at a wavelength.

    Row ``i`` of the matrix is the output ``i``, column ``j`` the input ``j``: the output amplitudes are
    ``a_out = S a_in`` with ``S_ij = amplitudes[i][j] exp(i phases[i][j])``. The default is the ideal 3-dB
    coupler ``(1/√2) [[1, i], [i, 1]]``.
    """

    amplitudes = Param(
        shape=None,
        default=[[_HALF, _HALF], [_HALF, _HALF]],
        bounds=(0, None),
        doc="moduli |S_ij| (rows: outputs, columns: inputs)",
    )
    phases = Param(
        unit=u.rad,
        shape=None,
        default=[[0.0, math.pi / 2], [math.pi / 2, 0.0]] * u.rad,
        doc="phases arg S_ij at the wavelength of the matrix (rows: outputs, columns: inputs)",
    )
    wavelength = Param(
        unit=u.nm,
        default=1550 * u.nm,
        bounds=(0 * u.m, None),
        inclusive=(False, True),
        doc="wavelength at which the matrix is given",
    )
    dispersion = Setting(
        str,
        default="achromatic",
        choices=("achromatic", "optical_path"),
        doc="achromatic: the same matrix at every wavelength; optical_path: phases scaled as λ0/λ",
    )

    @staticmethod
    def _problems(amplitudes: np.ndarray, phases: np.ndarray) -> list[str]:
        if amplitudes.ndim != 2 or amplitudes.size == 0:
            return [
                f"the amplitudes must be a matrix (outputs × inputs), got the shape {list(amplitudes.shape)}"
            ]
        if phases.shape != amplitudes.shape:
            return [
                f"the phases have the shape {list(phases.shape)}, the amplitudes {list(amplitudes.shape)}: "
                "give a phase per amplitude"
            ]
        largest = float(np.linalg.norm(amplitudes * np.exp(1j * phases), 2))
        if largest > 1 + PASSIVITY_TOLERANCE:
            return [
                f"the matrix amplifies light (largest singular value {largest:.4g} > 1): a passive component "
                "has a largest singular value at most 1"
            ]
        return []

    def problems(self) -> list[str]:
        """Return why the matrix cannot be used: not a matrix, phases of another shape, or amplification."""
        return self._problems(np.asarray(self.si("amplitudes"), float), np.asarray(self.si("phases"), float))

    @classmethod
    def _transfer(cls, values: Values, inputs: Any) -> dict[str, Any]:
        bus = _as_bus(inputs["in"])
        amplitudes = np.asarray(values.amplitudes, dtype=float)
        phases = np.asarray(values.phases, dtype=float)
        problems = cls._problems(amplitudes, phases)
        if problems:
            raise ValueError(problems[0])
        selected = _selected(values, len(bus))
        modes = _modes(bus, selected, amplitudes.shape[1])
        wavelengths = np.asarray(modes.wavelengths_si, dtype=float)
        scale = (
            np.ones_like(wavelengths)
            if values.dispersion == "achromatic"
            else float(values.wavelength) / wavelengths
        )
        block = amplitudes[None] * np.exp(1j * phases[None] * scale[:, None, None])
        return {"out": _replace(bus, selected, modes.apply(block))}

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Apply the matrix to the selected channels."""
        return TransferMatrix._transfer(values, inputs)


# --- Chips ---------------------------------------------------------------------------------------------


def _chain(name: str, components: list[Element]) -> ElementGroup:
    chip = ElementGroup(name=name)
    chip.add(*components)
    for left, right in itertools.pairwise(components):
        chip.connect(left.port("out"), right.port("in"))
    chip.expose("in", components[0].port("in"))
    chip.expose("out", components[-1].port("out"))
    return chip


def _hadamard(i: int, j: int) -> list[Element]:
    """Real 2 × 2 Hadamard combiner ``[[1, 1], [1, -1]] / √2`` from a coupler and two phase shifters."""
    quarter = -math.pi / 2 * u.rad
    return [
        PhaseShifter(phase=quarter, achromatic=True, channels=[j]),
        DirectionalCoupler(ratio=0.5, channels=[i, j]),
        PhaseShifter(phase=quarter, achromatic=True, channels=[j]),
    ]


def bracewell(name: str = "bracewell") -> ElementGroup:
    """Two-input Bracewell nuller: output 0 is the dark (null) output, output 1 the bright one."""
    return _chain(
        name,
        [
            PhaseShifter(phase=math.pi / 2 * u.rad, achromatic=True, channels=[1]),
            DirectionalCoupler(ratio=0.5, channels=[0, 1]),
        ],
    )


#: Pairs of outputs of :func:`kernel_nuller` whose differences are the kernels.
KERNEL_PAIRS: tuple[tuple[int, int], ...] = ((1, 3), (2, 5), (4, 6))


def kernel_nuller(name: str = "kernel_nuller") -> ElementGroup:
    """Four-input kernel nuller (Martinache & Ireland 2018): output 0 is bright, outputs 1-6 are dark.

    A real Hadamard nuller sends the on-axis light to output 0 and makes three dark outputs; each dark output
    is split in two and the copies are combined pairwise in quadrature by 50/50 couplers. The kernels are the
    differences of the outputs of each coupler, listed in :data:`KERNEL_PAIRS` (:func:`kernels`).
    """
    components: list[Element] = []
    for i, j in ((0, 1), (2, 3), (0, 2), (1, 3)):
        components += _hadamard(i, j)
    components += [
        YJunction(mode="split", channels=[1]),
        YJunction(mode="split", channels=[3]),
        YJunction(mode="split", channels=[5]),
    ]
    components += [DirectionalCoupler(ratio=0.5, channels=list(pair)) for pair in KERNEL_PAIRS]
    return _chain(name, components)


def kernels(power: Any) -> np.ndarray:
    """Return the kernels, differences of the paired output powers of :func:`kernel_nuller`, ``(..., 3)``."""
    p = np.asarray(power)
    return np.stack([p[..., a] - p[..., b] for a, b in KERNEL_PAIRS], axis=-1)
