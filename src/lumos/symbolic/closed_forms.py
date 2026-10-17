r"""Closed-form symbolic propagation (step P10.4, REQ-SYM-4, REQ-SYM-5).

Symbolic fields of known families carry a **form** (:attr:`~lumos.symbolic.SymbolicField.form`) that
elements update exactly:

Gaussian beams (:class:`GaussianBeamForm`)
    Hermite-Gauss :math:`\mathrm{HG}_{nm}` and Laguerre-Gauss :math:`\mathrm{LG}_{p\ell}` modes are
    described by the complex beam parameter :math:`q`, with :math:`1/q = 1/R + i\lambda/(\pi w^2)` in
    the :math:`e^{+ikz}` convention of LUMOS (a beam at a distance :math:`z` after its waist has
    :math:`q = z - i z_R`). Through an ABCD system :math:`q \to (Aq + B)/(Cq + D)`, and the field gains
    the Gouy phase :math:`\arg(A + B/q)` per unit of mode order :math:`N + 1` (:math:`N = n + m` or
    :math:`2p + |\ell|`) (Kogelnik & Li 1966, Appl. Opt. 5, 1550; Siegman, *Lasers*, ch. 20). Free
    space of length :math:`d` is :math:`(1, d, 0, 1)` and adds the phase :math:`kd`; a thin lens is
    :math:`(1, 0, -1/f, 1)`. The fields are

    .. math::
        U_{nm} = \sqrt{P}\,\sqrt{\frac{2}{\pi}}\frac{1}{w\sqrt{2^{n+m} n!\,m!}}
                 H_n\!\left(\frac{\sqrt2 x}{w}\right) H_m\!\left(\frac{\sqrt2 y}{w}\right)
                 e^{ik r^2/2q}\,e^{-i(n+m+1)\psi}\,e^{i\phi},

        U_{p\ell} = \sqrt{P}\,\sqrt{\frac{2\,p!}{\pi (p+|\ell|)!}}\frac{1}{w}
                    \left(\frac{\sqrt2 r}{w}\right)^{|\ell|} L_p^{|\ell|}\!\left(\frac{2r^2}{w^2}\right)
                    e^{i\ell\varphi}\,e^{ik r^2/2q}\,e^{-i(2p+|\ell|+1)\psi}\,e^{i\phi},

    normalised to the power :math:`P` (:math:`\psi` accumulated Gouy phase, :math:`\phi` accumulated phase
    :math:`kd`).

Uniform pupils (:class:`PupilForm`)
    A plane wave of amplitude :math:`A` through a circular or rectangular aperture, then a thin lens of focal
    length :math:`f`, gives in the focal plane (:math:`d = f`) the Fraunhofer pattern of the pupil
    (Goodman, *Introduction to Fourier Optics*, §5.2):
    :math:`U = \frac{e^{ikf}}{i\lambda f} e^{ikr^2/2f}\,
    \tilde P\!\left(\frac{x}{\lambda f}, \frac{y}{\lambda f}\right)`
    with :math:`\tilde P = A\,\frac{\pi D^2}{4}\,\frac{2J_1(v)}{v}`, :math:`v = \pi D r/(\lambda f)` (circle)
    or :math:`A\,ab\,\mathrm{sinc}(ax/\lambda f)\,\mathrm{sinc}(by/\lambda f)` (rectangle).

Fallback
    Any other propagation samples the field on its grid hint (:attr:`SymbolicField.grid`), propagates it
    numerically (:func:`lumos.propagation.propagate`) and returns a :class:`~lumos.fields.Wavefront`; the
    journal records it.
"""

from __future__ import annotations

import cmath
import logging
import math
from dataclasses import dataclass, replace
from typing import Any, cast

import sympy as sp
from astropy import units as u

from lumos.symbolic.field import SymbolicField
from lumos.symbolic.symbols import r, x, y
from lumos.units import require_unit, to_si

__all__ = [
    "GaussianBeamForm",
    "PlaneWaveForm",
    "PupilForm",
    "gaussian_beam",
    "plane_wave",
    "propagate_symbolic",
    "rectangular_pupil",
]

logger = logging.getLogger("lumos.symbolic")


def _si(quantity: Any, unit: Any, name: str) -> float:
    return float(to_si(require_unit(quantity, unit, name=name)))


@dataclass(frozen=True)
class GaussianBeamForm:
    """State of a Hermite-Gauss or Laguerre-Gauss beam (see the module documentation)."""

    q: complex
    wavelength: float
    power: float
    family: str = "hg"
    indices: tuple[int, int] = (0, 0)
    gouy: float = 0.0
    phase: float = 0.0
    power_unit: Any = u.W

    @property
    def order(self) -> int:
        """Mode order N: ``n + m`` (HG) or ``2p + |l|`` (LG)."""
        a, b = self.indices
        return a + b if self.family == "hg" else 2 * a + abs(b)

    def width(self) -> float:
        """Beam radius :math:`w` at the current plane (m)."""
        return math.sqrt(self.wavelength / (math.pi * (1 / self.q).imag))

    def abcd(self, a: float, b: float, c: float, d: float, *, path: float = 0.0) -> GaussianBeamForm:
        """Return the beam after the ABCD system ``(a, b, c, d)`` of optical path ``path``."""
        factor = a + b / self.q
        q = (a * self.q + b) / (c * self.q + d)
        k = 2 * math.pi / self.wavelength
        return replace(self, q=q, gouy=self.gouy + cmath.phase(factor), phase=self.phase + k * path)

    def expression(self) -> sp.Expr:
        """Return the expression of the field at the current plane."""
        w = self.width()
        k = 2 * math.pi / self.wavelength
        q = sp.Float(self.q.real) + sp.I * sp.Float(self.q.imag)
        common = sp.exp(sp.I * k * r**2 / (2 * q)) * sp.exp(
            sp.I * sp.Float(self.phase - (self.order + 1) * self.gouy)
        )
        root_power = math.sqrt(self.power)
        if self.family == "hg":
            n, m = self.indices
            norm = (
                root_power
                * math.sqrt(2 / math.pi)
                / (w * math.sqrt(2 ** (n + m) * math.factorial(n) * math.factorial(m)))
            )
            shape = sp.hermite(n, math.sqrt(2) * x / w) * sp.hermite(m, math.sqrt(2) * y / w)
            return sp.Float(norm) * shape * common
        p, ell = self.indices
        norm = root_power * math.sqrt(2 * math.factorial(p) / (math.pi * math.factorial(p + abs(ell)))) / w
        radial = (math.sqrt(2) * r / w) ** abs(ell) * sp.assoc_laguerre(p, abs(ell), 2 * r**2 / w**2)
        azimuth = ((x + sp.I * y) / r) ** ell if ell else sp.Integer(1)
        return sp.Float(norm) * radial * azimuth * common

    def field(self, grid: Any = None) -> SymbolicField:
        """Return the symbolic field of the beam."""
        return SymbolicField(
            self.expression(),
            wavelength=self.wavelength * u.m,
            power_unit=self.power_unit,
            grid=grid,
            form=self,
        )


def gaussian_beam(
    *,
    waist: u.Quantity,
    wavelength: u.Quantity,
    power: u.Quantity,
    distance_from_waist: u.Quantity = 0 * u.m,
    family: str = "hg",
    indices: tuple[int, int] = (0, 0),
    grid: Any = None,
) -> SymbolicField:
    r"""Return a Gaussian beam (HG\ :sub:`nm` or LG\ :sub:`pl`) at ``distance_from_waist`` after its waist."""
    if family not in ("hg", "lg"):
        raise ValueError(f"family must be 'hg' or 'lg', got {family!r}")
    w0 = _si(waist, u.m, "waist")
    lam = float(
        to_si(
            require_unit(wavelength, u.m, name="wavelength", equivalencies=u.spectral()),
            u.m,
            equivalencies=u.spectral(),
        )
    )
    z = _si(distance_from_waist, u.m, "distance_from_waist")
    unit = cast(u.Quantity, power).unit
    value = float(cast(Any, power).value)
    z_r = math.pi * w0**2 / lam
    form = GaussianBeamForm(
        complex(0, -z_r), lam, value, family, (int(indices[0]), int(indices[1])), power_unit=unit
    )
    if z:
        form = form.abcd(1, z, 0, 1)
    return form.field(grid)


@dataclass(frozen=True)
class PlaneWaveForm:
    """Uniform plane wave of amplitude ``amplitude`` (SI) at normal incidence."""

    amplitude: float
    wavelength: float


@dataclass(frozen=True)
class PupilForm:
    """Uniformly illuminated pupil, optionally followed by a thin lens of focal length ``focal``."""

    kind: str  # "circ" or "rect"
    amplitude: float
    wavelength: float
    size: tuple[float, float]  # (diameter, 0) or (width, height)
    focal: float | None = None


def plane_wave(*, amplitude: u.Quantity, wavelength: u.Quantity, grid: Any = None) -> SymbolicField:
    """Return a uniform plane wave (``amplitude`` in sqrt(W)/m or sqrt(photon/s)/m)."""
    lam = float(
        to_si(
            require_unit(wavelength, u.m, name="wavelength", equivalencies=u.spectral()),
            u.m,
            equivalencies=u.spectral(),
        )
    )
    value = float(cast(Any, amplitude).value)
    unit = cast(Any, amplitude).unit ** 2 * u.m**2
    form = PlaneWaveForm(value, lam)
    return SymbolicField(sp.Float(value) + 0 * x, wavelength=lam * u.m, power_unit=unit, grid=grid, form=form)


def circular_pupil_field(field: SymbolicField, diameter: float) -> SymbolicField:
    """Return a plane wave through a circular pupil of ``diameter`` (m), with its closed form."""
    form = cast(PlaneWaveForm, field.form)
    expression = sp.Float(form.amplitude) * sp.Piecewise((1, r <= sp.Float(diameter / 2)), (0, True))
    pupil = PupilForm("circ", form.amplitude, form.wavelength, (diameter, 0.0))
    return field.with_expression(expression, form=pupil)


def rectangular_pupil(
    *, width: u.Quantity, height: u.Quantity, amplitude: u.Quantity, wavelength: u.Quantity, grid: Any = None
) -> SymbolicField:
    """Return a plane wave through a centred rectangular pupil of ``width`` × ``height``."""
    wave = plane_wave(amplitude=amplitude, wavelength=wavelength, grid=grid)
    a, b = _si(width, u.m, "width"), _si(height, u.m, "height")
    form = cast(PlaneWaveForm, wave.form)
    inside = (sp.Abs(x) <= sp.Float(a / 2)) & (sp.Abs(y) <= sp.Float(b / 2))
    expression = sp.Float(form.amplitude) * sp.Piecewise((1, inside), (0, True))
    return wave.with_expression(expression, form=PupilForm("rect", form.amplitude, form.wavelength, (a, b)))


def through_lens(field: SymbolicField, focal: float) -> SymbolicField:
    """Return ``field`` after a thin lens of focal length ``focal`` (m), keeping its closed form if any."""
    from lumos.symbolic.symbols import lam

    form = field.form
    if isinstance(form, GaussianBeamForm):
        return form.abcd(1, 0, -1 / focal, 1).field(field.grid)
    phase = sp.exp(-sp.I * sp.pi * r**2 / (lam * sp.Float(focal)))
    if isinstance(form, PupilForm) and form.focal is None:
        return field.with_expression(field.expression * phase, form=replace(form, focal=focal))
    return field.multiply(phase)


def _focal_field(form: PupilForm, field: SymbolicField) -> SymbolicField:
    f, lam = cast(float, form.focal), form.wavelength
    k = 2 * math.pi / lam
    prefactor = (
        sp.exp(sp.I * sp.Float(k * f))
        / (sp.I * sp.Float(lam * f))
        * sp.exp(sp.I * sp.Float(k / (2 * f)) * r**2)
    )
    if form.kind == "circ":
        diameter = form.size[0]
        v = sp.pi * sp.Float(diameter) * r / sp.Float(lam * f)
        airy = 2 * sp.besselj(1, v) / v
        spectrum = sp.Float(form.amplitude * math.pi * diameter**2 / 4) * airy
    else:
        a, b = form.size
        spectrum = (
            sp.Float(form.amplitude * a * b)
            * sp.sinc(sp.pi * sp.Float(a / (lam * f)) * x)
            * sp.sinc(sp.pi * sp.Float(b / (lam * f)) * y)
        )
    return field.with_expression(prefactor * spectrum, form=None)


def propagate_symbolic(
    field: SymbolicField, distance: float, *, output_grid: Any = None, method: str = "auto"
) -> Any:
    """Propagate ``field`` over ``distance`` (m): closed form when known, numerical fallback otherwise."""
    form = field.form
    if isinstance(form, GaussianBeamForm):
        return form.abcd(1, distance, 0, 1, path=distance).field(output_grid or field.grid)
    if (
        isinstance(form, PupilForm)
        and form.focal is not None
        and math.isclose(distance, form.focal, rel_tol=1e-12)
    ):
        return _focal_field(form, field)
    if field.grid is None:
        raise ValueError(
            "no closed form for this propagation and no sampling grid for the numerical fallback: "
            "set SymbolicField.grid"
        )
    from lumos.propagation import propagate
    from lumos.symbolic.evaluate import to_wavefront

    logger.info(
        "no closed form for the propagation of %s over %.6g m: numerical fallback on %s",
        field.expression,
        distance,
        field.grid,
    )
    sampled = to_wavefront(field, field.grid)
    return propagate(sampled, distance * u.m, output_grid=output_grid, method=method)
