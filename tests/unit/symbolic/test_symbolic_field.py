"""Tests for symbolic fields (step P10.1, REQ-SYM-1, REQ-SYM-6, REQ-SIG-10).

Requirements: a ``SymbolicField`` is a SymPy expression of the field in canonical variables (``x, y``
or ``θx, θy``, ``λ``, ``t``) and of parameters bound to physical quantities, with its domain and plane; it
renders in LaTeX.

Independent references: SymPy itself (``free_symbols``, ``latex``, ``simplify``); ``|A e^{iφ}|² = A²`` for
real A and φ.
"""

import pytest
import sympy as sp
from astropy import units as u

from lumos.core.signal import OpticalSignal
from lumos.symbolic import SymbolicField, lam, r, t, theta_x, theta_y, x, y


def gaussian():
    w = sp.Symbol("w", positive=True)
    return SymbolicField(sp.exp(-(x**2 + y**2) / w**2), parameters={w: 2 * u.mm}, wavelength=1 * u.um), w


def test_symbolic_field_is_an_optical_signal():
    field, _ = gaussian()
    assert isinstance(field, OpticalSignal)
    assert field.plane == "spatial"


def test_parameters_are_quantities_with_si_values():
    field, w = gaussian()
    assert field.parameters[w] == 2 * u.mm
    assert field.si_parameters()[w] == pytest.approx(2e-3)
    numeric = field.substituted()
    assert numeric.free_symbols == {x, y}
    assert float(numeric.subs({x: 2e-3, y: 0})) == pytest.approx(float(sp.exp(-1)))


def test_parameters_can_be_named_by_string():
    field = SymbolicField(
        sp.Symbol("a") * x, parameters={"a": 3 * u.dimensionless_unscaled}, wavelength=1 * u.um
    )
    assert field.si_parameters()[sp.Symbol("a")] == 3


def test_unknown_symbols_are_refused():
    with pytest.raises(ValueError, match="unknown symbol"):
        SymbolicField(sp.Symbol("q") * x, wavelength=1 * u.um)


def test_angular_fields_use_angular_variables():
    field = SymbolicField(sp.cos(theta_x) * sp.cos(theta_y), plane="angular", wavelength=1 * u.um)
    assert field.plane == "angular"
    with pytest.raises(ValueError, match="unknown symbol"):
        SymbolicField(x, plane="angular", wavelength=1 * u.um)


def test_radius_helper():
    assert sp.simplify(r**2 - (x**2 + y**2)) == 0


def test_wavelength_dependence():
    field, _ = gaussian()
    assert not field.depends_on_wavelength
    chromatic = SymbolicField(sp.exp(sp.I * 2 * sp.pi * x / lam))
    assert chromatic.depends_on_wavelength
    assert chromatic.wavelength is None


def test_monochromatic_fields_need_a_wavelength():
    with pytest.raises(ValueError, match="wavelength"):
        SymbolicField(sp.exp(-(x**2)))


def test_time_is_a_canonical_variable():
    field = SymbolicField(sp.cos(2 * sp.pi * t) * x, wavelength=1 * u.um)
    assert t in field.expression.free_symbols


def test_intensity_of_a_phase_factor():
    a = sp.Symbol("A", positive=True)
    field = SymbolicField(
        a * sp.exp(sp.I * 2 * sp.pi * x / lam), parameters={a: 2 * u.dimensionless_unscaled}
    )
    assert sp.simplify(field.intensity() - a**2) == 0


def test_product_with_an_expression_merges_parameters():
    field, w = gaussian()
    radius = sp.Symbol("R", positive=True)
    masked = field.multiply(sp.Heaviside(radius - r), parameters={radius: 1 * u.mm})
    assert set(masked.parameters) == {w, radius}
    assert masked.wavelength == 1 * u.um


def test_conflicting_parameters_are_refused():
    field, w = gaussian()
    with pytest.raises(ValueError, match="w"):
        field.multiply(sp.exp(-x / w), parameters={w: 3 * u.mm})


def test_latex_rendering():
    field, _ = gaussian()
    text = field.latex()
    assert "e^{" in text
    assert "w" in text
    assert field._repr_latex_().startswith("$")


def test_power_unit():
    field, _ = gaussian()
    assert field.power_unit == u.W
    photons = SymbolicField(x * 0 + 1, wavelength=1 * u.um, power_unit=u.photon / u.s)
    assert photons.power_unit == u.photon / u.s
