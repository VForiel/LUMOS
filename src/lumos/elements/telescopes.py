r"""Telescopes and telescope arrays (§5.9, step P6.4).

A :class:`Telescope` turns the sources at infinity it receives (:class:`~lumos.fields.SourceList`) into the
plane waves sampled on its pupil, one mode of an :class:`~lumos.fields.IncoherentBundle` per source: channel
:math:`k` of source :math:`m` is

.. math:: U_{mk}(x, y) = \sqrt{F_{mk} b_k}\; P(x, y)\, \exp\!\left[\frac{2i\pi}{\lambda_k}
          (x\,\theta_{x,m} + y\,\theta_{y,m})\right],

with :math:`P` the anti-aliased pupil transmission, so that the power of each channel is the photon rate
:math:`A F_{mk} b_k` collected by the pupil area :math:`A` (``photon / s``). Since grey edge pixels transmit
the power :math:`P^2 < P`, the pupil is scaled so that :math:`\sum P^2 \Delta x \Delta y` equals the
anti-aliased area :math:`\sum P \Delta x \Delta y`, which converges to the exact area.

A :class:`TelescopeArray` models an interferometer whose beams are injected into single-mode guides with an
ideal coupling: it returns a :class:`~lumos.fields.Bus` of single-mode guided lanes (``T1``... ``Tn``, with
the projected positions of the telescopes, ADR-0023), in which telescope :math:`j` at the projected position
:math:`(u_j, v_j)` delivers the amplitude

.. math:: a_{mkj} = \sqrt{\eta A F_{mk} b_k}\; \exp\!\left[\frac{2i\pi}{\lambda_k}
          (u_j\,\theta_{x,m} + v_j\,\theta_{y,m})\right]

(:math:`\eta` the throughput), the
geometric delay of the phase centre being compensated by the delay lines. Angles are direction cosines on the
sky, :math:`\theta_x` towards the East (u) and :math:`\theta_y` towards the North (v).

Projected baselines
-------------------
In the ``"ground"`` frame, station positions are local (East, North, Up) coordinates at the latitude
:math:`\varphi`. They are expressed in the equatorial frame and projected on the plane of the sky of a target
of declination :math:`\delta` at hour angle :math:`H` (Thompson, Moran & Swenson, *Interferometry and
Synthesis in Radio Astronomy*, 3rd ed., eqs. 4.15 and 4.1):

.. math::
    X = -\sin\varphi\,N + \cos\varphi\,U, \quad Y = E, \quad Z = \cos\varphi\,N + \sin\varphi\,U,

    u = \sin H\,X + \cos H\,Y, \qquad v = -\sin\delta\cos H\,X + \sin\delta\sin H\,Y + \cos\delta\,Z .

The hour angle grows with the context clock at the sidereal rate (one turn per 86164.0905 s, IERS
Conventions 2010). In the ``"sky"`` frame (space arrays), positions are given in the plane of the sky and the
array rotates about the line of sight at ``rotation_rate``.

Presets: :func:`vlti` (the four 8.2 m unit telescopes of Paranal), :func:`life` (the four 2 m collectors of
the LIFE X-array, Quanz et al. 2022, A&A 664, A21).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from astropy import units as u

from lumos.core.element import Element, Values
from lumos.core.param import Param, Setting
from lumos.core.port import Input, Output
from lumos.core.registry import register
from lumos.fields.bus import Bus
from lumos.fields.grid import Grid
from lumos.fields.incoherent import IncoherentBundle
from lumos.fields.modal import ModalField
from lumos.fields.sources import SourceList
from lumos.fields.wavecube import Wavecube
from lumos.physics import apertures as shapes

__all__ = ["Telescope", "TelescopeArray", "life", "vlti"]

#: Length of the sidereal day in seconds (IERS Conventions 2010).
SIDEREAL_DAY = 86164.0905
_PHOTON_RATE = u.photon / u.s


def combine(sources: Sequence[SourceList]) -> SourceList:
    """Concatenate the source lists received on a multiple input."""
    if not sources:
        raise ValueError("no source list received")
    total = sources[0]
    for other in sources[1:]:
        total = total + other
    return total


def _arrays(sources: SourceList) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def si(quantity: u.Quantity, unit: Any) -> np.ndarray:
        return np.asarray(quantity.to_value(unit), dtype=float)

    lam, bw = si(sources.wavelengths, u.m), si(sources.bandwidths, u.m)
    flux = si(sources.flux, u.photon / u.s / u.m**3)
    return si(sources.theta_x, u.rad), si(sources.theta_y, u.rad), lam, bw, flux


# --- Single telescope ------------------------------------------------------------------------------


@register("lumos.telescopes.Telescope")
class Telescope(Element):
    """Telescope pupil sampling the plane waves of the sources at infinity (one bundle mode per source)."""

    diameter = Param(
        unit=u.m,
        default=1 * u.m,
        bounds=(0 * u.m, None),
        inclusive=(False, True),
        doc="diameter of the primary mirror",
    )
    obstruction = Param(unit=u.m, default=0 * u.m, bounds=(0 * u.m, None), doc="central obstruction diameter")
    spider_width = Param(unit=u.m, default=0 * u.m, bounds=(0 * u.m, None), doc="width of each spider vane")
    spider_rotation = Param(unit=u.deg, default=0 * u.deg, doc="angle of the first spider vane, from +x")
    n_spiders = Setting(int, default=0, doc="number of spider vanes (0: none)")
    n_pixels = Setting(int, default=128, doc="pupil grid size")
    oversize = Setting(float, default=1.05, doc="width of the pupil grid in diameters (> 1: margin)")
    supersample = Setting(int, default=4, doc="sub-pixels per pixel side for anti-aliasing")
    in_ = Input(SourceList, multiple=True)
    out = Output(IncoherentBundle)

    @staticmethod
    def pupil_grid(values: Values) -> Grid:
        """Return the sampling of the pupil."""
        return Grid(values.n_pixels, dx=values.diameter * values.oversize / values.n_pixels * u.m)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Sample the plane waves of the received sources on the pupil."""
        from lumos.elements.apertures import _with_spiders

        sources = combine(inputs["in"])
        tx, ty, lam, bw, flux = _arrays(sources)
        grid = Telescope.pupil_grid(values)
        mask = shapes.annulus(grid, values.diameter, values.obstruction, supersample=values.supersample)
        mask = _with_spiders(values, grid, mask)
        mask = mask * np.sqrt(mask.sum() / max(float((mask**2).sum()), 1e-300))  # power = anti-aliased area
        x, y = (np.asarray(c) for c in grid.coords(backend="numpy"))
        k = 2 * math.pi / lam[None, :, None, None]
        phase = k * (tx[:, None, None, None] * x[None, None] + ty[:, None, None, None] * y[None, None])
        amplitude = np.sqrt(flux * bw[None, :])[:, :, None, None]
        field = amplitude * mask[None, None] * np.exp(1j * phase)
        template = Wavecube(field[0], grid, lam * u.m, power_unit=_PHOTON_RATE)
        return {"out": IncoherentBundle(field, template, sources=sources.identifier)}


# --- Arrays -----------------------------------------------------------------------------------------


def _positions(values: Values) -> np.ndarray:
    positions = np.atleast_2d(np.asarray(values.positions, dtype=float))
    if positions.ndim != 2 or positions.shape[1] not in (2, 3):
        raise ValueError(f"positions must have shape (n, 2) or (n, 3), got {positions.shape}")
    return positions


def _projected(values: Values) -> np.ndarray:
    positions = _positions(values)
    time = values.time
    if values.frame == "sky":
        angle = values.rotation + values.rotation_rate * time
        c, s = math.cos(angle), math.sin(angle)
        x, y = positions[:, 0], positions[:, 1]
        return np.stack([c * x - s * y, s * x + c * y], axis=1)
    east, north = positions[:, 0], positions[:, 1]
    up = positions[:, 2] if positions.shape[1] == 3 else np.zeros_like(east)
    lat, dec = values.latitude, values.declination
    h = values.hour_angle + 2 * math.pi * time / SIDEREAL_DAY
    x = -math.sin(lat) * north + math.cos(lat) * up
    y = east
    z = math.cos(lat) * north + math.sin(lat) * up
    uu = math.sin(h) * x + math.cos(h) * y
    vv = -math.sin(dec) * math.cos(h) * x + math.sin(dec) * math.sin(h) * y + math.cos(dec) * z
    return np.stack([uu, vv], axis=1)


@register("lumos.telescopes.TelescopeArray")
class TelescopeArray(Element):
    """Array of identical telescopes injecting each beam in a single-mode guide (ideal coupling)."""

    time_dependent = True
    positions = Param(
        unit=u.m,
        shape=None,
        default=[[0.0, 0.0]] * u.m,
        doc="(East, North[, Up]) or sky (x, y), shape (n, 2|3)",
    )
    diameter = Param(
        unit=u.m,
        default=1 * u.m,
        bounds=(0 * u.m, None),
        inclusive=(False, True),
        doc="diameter of each telescope",
    )
    obstruction = Param(unit=u.m, default=0 * u.m, bounds=(0 * u.m, None), doc="central obstruction diameter")
    throughput = Param(default=1.0, bounds=(0, 1), doc="power transmission up to the guides")
    latitude = Param(unit=u.deg, default=0 * u.deg, bounds=(-90 * u.deg, 90 * u.deg), doc="site latitude")
    declination = Param(
        unit=u.deg,
        default=0 * u.deg,
        bounds=(-90 * u.deg, 90 * u.deg),
        doc="declination of the target (projection of the baselines)",
    )
    hour_angle = Param(unit=u.hourangle, default=0 * u.hourangle, doc="hour angle at the start of the clock")
    rotation = Param(unit=u.deg, default=0 * u.deg, doc="sky frame: array rotation at the start of the clock")
    rotation_rate = Param(unit=u.deg / u.hour, default=0 * u.deg / u.hour, doc="sky frame: rotation rate")
    frame = Setting(str, default="ground", choices=("ground", "sky"), doc="frame of the positions")
    in_ = Input(SourceList, multiple=True)
    out = Output(ModalField, bus=True, doc="one single-mode guided lane per telescope")

    @property
    def n_telescopes(self) -> int:
        """Number of telescopes."""
        return len(_positions(self.values()))

    @property
    def area(self) -> u.Quantity:
        """Collecting area of one telescope."""
        values = self.values()
        return math.pi * (values.diameter**2 - values.obstruction**2) / 4 * u.m**2

    def ground_positions(self) -> u.Quantity:
        """Station positions as given, shape ``(n, 2)`` or ``(n, 3)``."""
        return _positions(self.values()) * u.m

    def projected_positions(self) -> u.Quantity:
        """Positions ``(u, v)`` projected on the plane of the sky at the current time, shape ``(n, 2)``."""
        return _projected(self.values()) * u.m

    def baselines(self) -> u.Quantity:
        """Projected baselines ``p_j - p_i`` for every pair ``i < j``, shape ``(n (n - 1) / 2, 2)``."""
        uv = _projected(self.values())
        pairs = list(itertools.combinations(range(len(uv)), 2))
        return np.array([uv[j] - uv[i] for i, j in pairs]).reshape(-1, 2) * u.m

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the bus of the amplitudes injected by every telescope for every source and channel."""
        sources = combine(inputs["in"])
        tx, ty, lam, bw, flux = _arrays(sources)
        uv = _projected(values)
        area = math.pi * (values.diameter**2 - values.obstruction**2) / 4
        modulus = np.sqrt(values.throughput * area * flux * bw[None, :])  # (M, nλ)
        opd = tx[:, None] * uv[None, :, 0] + ty[:, None] * uv[None, :, 1]  # (M, n)
        phase = 2 * math.pi * opd[:, None, :] / lam[None, :, None]
        amplitudes = modulus[:, :, None] * np.exp(1j * phase)
        bus = Bus.guided(
            amplitudes, lam * u.m, positions=uv * u.m, power_unit=_PHOTON_RATE, sources=sources.identifier
        )
        return {"out": bus}


# --- Presets -----------------------------------------------------------------------------------------

#: Latitude of Paranal (ESO).
PARANAL_LATITUDE = -24.6272 * u.deg

#: (East, North) positions of the VLTI unit-telescope stations UT1-UT4 relative to the array centre, in m.
#: Their separations are the published baselines 56.6, 102.4, 130.2, 46.6, 89.4 and 62.5 m.
VLTI_UT_STATIONS = [
    [-9.925, -20.335],
    [14.887, 30.502],
    [44.915, 66.183],
    [103.306, 43.999],
]


def vlti(name: str = "VLTI", **values: Any) -> TelescopeArray:
    """Return the four 8.2 m unit telescopes of the VLTI at Paranal (other fields passed in ``values``)."""
    fields: dict[str, Any] = {
        "positions": VLTI_UT_STATIONS * u.m,
        "diameter": 8.2 * u.m,
        "obstruction": 1.116 * u.m,
        "latitude": PARANAL_LATITUDE,
        "frame": "ground",
    }
    fields.update(values)
    return TelescopeArray(name=name, **fields)


def life(name: str = "LIFE", *, nulling_baseline: u.Quantity = 15 * u.m, **values: Any) -> TelescopeArray:
    """Return the LIFE X-array: four 2 m collectors on a rectangle of 1 by 6 nulling baselines.

    LIFE adapts the nulling baseline (10 to 100 m) to each target (Quanz et al. 2022); 15 m is an example.
    """
    b = float(np.asarray(nulling_baseline.to_value(u.m)))
    fields: dict[str, Any] = {
        "positions": [[-b / 2, -3 * b], [b / 2, -3 * b], [b / 2, 3 * b], [-b / 2, 3 * b]] * u.m,
        "diameter": 2 * u.m,
        "frame": "sky",
    }
    fields.update(values)
    return TelescopeArray(name=name, **fields)
