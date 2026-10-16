r"""Planar waveguides and multimode interference (MMI) couplers.

Mode solver
-----------
Integrated components are reduced to two dimensions with the effective index method: a slab of width
:math:`W` and index :math:`n_\mathrm{co}` in a cladding :math:`n_\mathrm{cl}` guides TE modes
:math:`E(x)\,e^{i\beta z}` solving

.. math:: \frac{d^2 E}{dx^2} + k^2 n(x)^2 E = \beta^2 E ,

discretised here with second-order finite differences on a uniform grid with zero field at its edges (the
window is several times wider than the guide). Guided modes have :math:`k n_\mathrm{cl} < \beta < k
n_\mathrm{co}`; for a symmetric step-index slab they obey :math:`\tan(\kappa W/2) = \gamma/\kappa` (even) or
:math:`-\cot(\kappa W/2) = \gamma/\kappa` (odd), with :math:`\kappa^2 = k^2 n_\mathrm{co}^2 - \beta^2` and
:math:`\gamma^2 = \beta^2 - k^2 n_\mathrm{cl}^2` (Marcuse, *Theory of Dielectric Optical Waveguides*, §1.3).

Multimode interference
----------------------
The field launched by an access waveguide into the multimode section is expanded on the guided modes
:math:`\psi_m` of the section, propagated as :math:`\sum_m c_m \psi_m e^{i\beta_m L}` and projected on the
modes of the output waveguides (radiation modes are neglected): this gives the scattering matrix of the
component at each wavelength. Self-images form at multiples of the beat length
:math:`L_\pi = \pi / (\beta_0 - \beta_1)`; an :math:`N \times N` coupler with ports at
:math:`x_i = W_e (2i + 1 - N) / 2N` (:func:`effective_width`) makes :math:`N` images of equal power
at :math:`L = 3 L_\pi / N`
(Soldano & Pennings 1995, J. Lightwave Technol. 13, 615; Bachmann, Besse & Melchior 1994, Appl. Opt. 33,
3905).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh_tridiagonal

__all__ = ["SlabModes", "beat_length", "effective_width", "mmi_matrix", "port_positions", "slab_modes"]


@dataclass(frozen=True)
class SlabModes:
    """Guided TE modes of a slab: propagation constants ``beta`` (rad/m) and fields ``(n_modes, nx)``.

    Fields are real and normalised so that ``sum(field**2) * dx = 1``.
    """

    x: np.ndarray
    beta: np.ndarray
    fields: np.ndarray

    @property
    def dx(self) -> float:
        """Sampling step of the fields."""
        return float(self.x[1] - self.x[0])


def slab_modes(
    x: np.ndarray, index: np.ndarray, wavelength: float, *, n_cladding: float, max_modes: int | None = None
) -> SlabModes:
    """Return the guided TE modes of the index profile ``index`` sampled on the uniform grid ``x``."""
    x = np.asarray(x, dtype=float)
    dx = float(x[1] - x[0])
    k = 2 * math.pi / wavelength
    diagonal = -2.0 / dx**2 + (k * np.asarray(index, dtype=float)) ** 2
    off = np.full(len(x) - 1, 1.0 / dx**2)
    eigenvalues, vectors = eigh_tridiagonal(diagonal, off)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    guided = eigenvalues > (k * n_cladding) ** 2
    beta = np.sqrt(eigenvalues[guided])
    fields = vectors[:, guided].T / math.sqrt(dx)
    signs = np.sign(fields[np.arange(len(fields)), np.argmax(np.abs(fields), axis=1)])
    fields = fields * signs[:, None]
    if max_modes is not None:
        beta, fields = beta[:max_modes], fields[:max_modes]
    return SlabModes(x=x, beta=beta, fields=fields)


def _profile(
    x: np.ndarray, centres: list[float], width: float, n_core: float, n_cladding: float
) -> np.ndarray:
    index = np.full_like(x, n_cladding)
    for centre in centres:
        index[np.abs(x - centre) <= width / 2] = n_core
    return index


def _window(width: float, port_width: float) -> tuple[np.ndarray, float]:
    half = 1.5 * width + 2 * port_width
    samples = int(2 * half / (port_width / 40)) | 1
    samples = max(samples, 2001)
    return np.linspace(-half, half, samples), half


def port_positions(n: int, width: float) -> list[float]:
    """Positions of ``n`` equally spaced ports of the general-interference coupler (centre at 0)."""
    return [width * (2 * i + 1 - n) / (2 * n) for i in range(n)]


def beat_length(width: float, n_core: float, n_cladding: float, wavelength: float) -> float:
    """Beat length ``π / (β0 - β1)`` of the two first modes of the multimode section."""
    x, _ = _window(width, width / 10)
    modes = slab_modes(x, _profile(x, [0.0], width, n_core, n_cladding), wavelength, n_cladding=n_cladding)
    if len(modes.beta) < 2:
        raise ValueError("the multimode section guides fewer than two modes")
    return math.pi / float(modes.beta[0] - modes.beta[1])


def mmi_matrix(
    *,
    width: float,
    length: float,
    n_core: float,
    n_cladding: float,
    port_width: float,
    inputs: list[float],
    outputs: list[float],
    wavelength: float,
) -> np.ndarray:
    """Scattering matrix ``(n_out, n_in)`` of a multimode interference coupler at ``wavelength``."""
    x, _ = _window(width, port_width)
    dx = float(x[1] - x[0])
    section = slab_modes(x, _profile(x, [0.0], width, n_core, n_cladding), wavelength, n_cladding=n_cladding)
    access = slab_modes(
        x, _profile(x, [0.0], port_width, n_core, n_cladding), wavelength, n_cladding=n_cladding, max_modes=1
    )
    if len(access.beta) == 0:
        raise ValueError("the access waveguides guide no mode")
    base = access.fields[0]

    def port_field(centre: float) -> np.ndarray:
        return np.interp(x - centre, x, base, left=0.0, right=0.0)

    launch = np.array([port_field(c) for c in inputs])  # (n_in, nx)
    receive = np.array([port_field(c) for c in outputs])  # (n_out, nx)
    coefficients = launch @ section.fields.T * dx  # (n_in, n_modes)
    propagated = (coefficients * np.exp(1j * section.beta * length)) @ section.fields  # (n_in, nx)
    return (receive @ propagated.T) * dx  # (n_out, n_in)


def effective_width(width: float, n_core: float, n_cladding: float, wavelength: float) -> float:
    """Return the effective width ``W + (λ/π) (n_co² - n_cl²)^(-1/2)`` of a TE multimode section.

    The Goos-Hänchen penetration of the modes in the cladding widens the section (Soldano & Pennings 1995).
    """
    return width + wavelength / math.pi / math.sqrt(n_core**2 - n_cladding**2)
