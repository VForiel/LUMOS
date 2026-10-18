r"""Integrated photonics (§5.9, step P6.8): components, chips and recombiner presets.

Every component maps a :class:`~lumos.fields.ModalField` to a new one through its scattering matrix
:math:`S(\lambda)` (:mod:`lumos.fields.smatrix`), acting on the channels listed in its ``channels`` setting
(all channels when empty) and leaving the others unchanged. Components with as many outputs as inputs write
their outputs on the same channels; the others (Y-junctions) put their outputs where their first input was.
A chip is an :class:`~lumos.core.group.ElementGroup` chaining components; :func:`bracewell` and
:func:`kernel_nuller` build the recombiners of Bracewell (1978) and Martinache & Ireland (2018).

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
    "Waveguide",
    "YJunction",
    "bracewell",
    "kernel_nuller",
    "kernels",
]

#: SI value of 1 dB/cm (astropy stores logarithmic units in dex).
_DB_PER_CM = float(to_si(1 * u.dB / u.cm))


def _selected(values: Values, n_channels: int, n_in: int | None) -> list[int]:
    channels = [int(c) for c in values.channels] if values.channels else list(range(n_channels))
    if n_in is not None and len(channels) != n_in:
        raise ValueError(f"this component acts on {n_in} channel(s), got channels={channels}")
    if len(set(channels)) != len(channels) or not all(0 <= c < n_channels for c in channels):
        raise ValueError(f"channels {channels} must be distinct indices below {n_channels}")
    return channels


def _embed(field: ModalField, selected: list[int], block: np.ndarray) -> ModalField:
    """Apply ``block`` (nλ, n_out, n_in) to the ``selected`` channels of ``field``."""
    n = field.n_channels
    n_out = block.shape[1]
    labels = field.labels
    if n_out == len(selected):
        rows: list[tuple[str, int]] = [("pass", c) for c in range(n)]
        for k, c in enumerate(selected):
            rows[c] = ("out", k)
    else:
        first = min(selected)
        rows = []
        for c in range(n):
            if c == first:
                rows += [("out", k) for k in range(n_out)]
            elif c not in selected:
                rows.append(("pass", c))
    full = np.zeros((block.shape[0], len(rows), n), dtype=complex)
    new_labels: list[str] | None = [] if labels is not None else None
    for r, (kind, index) in enumerate(rows):
        if kind == "pass":
            full[:, r, index] = 1.0
            if new_labels is not None and labels is not None:
                new_labels.append(labels[index])
        else:
            full[:, r, selected] = block[:, index, :]
            if new_labels is not None and labels is not None:
                base = labels[selected[index]] if n_out == len(selected) else labels[min(selected)]
                new_labels.append(base if n_out == len(selected) else f"{base}.{index}")
    return field.apply(full, labels=new_labels)


class PhotonicComponent(Element):
    """Base class: a scattering matrix applied to some channels of a modal field."""

    channels = Setting(list, default=[], doc="indices of the channels the component acts on ([]: all)")
    in_ = Input(ModalField)
    out = Output(ModalField)

    #: Number of input channels (None: any number, the matrix is diagonal).
    n_inputs: int | None = None

    @classmethod
    def _block(cls, values: Values, wavelengths: np.ndarray, n_in: int) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def _transfer(cls, values: Values, inputs: Any) -> dict[str, Any]:
        field: ModalField = inputs["in"]
        selected = _selected(values, field.n_channels, cls.n_inputs)
        block = cls._block(values, np.asarray(field.wavelengths_si, dtype=float), len(selected))
        return {"out": _embed(field, selected, block)}


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
        field: ModalField = inputs["in"]
        split = values.mode == "split"
        selected = _selected(values, field.n_channels, 1 if split else 2)
        matrix = np.array([[1.0], [1.0]]) if split else np.array([[1.0, 1.0]])
        block = _constant(matrix / math.sqrt(2), field.n_wavelengths)
        return {"out": _embed(field, selected, block)}

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
        field: ModalField = inputs["in"]
        n = int(values.n_ports)
        selected = _selected(values, field.n_channels, n)
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
                for lam in field.wavelengths_si
            ]
        )
        return {"out": _embed(field, selected, block)}

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Mix the selected channels."""
        return MMI._transfer(values, inputs)


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
