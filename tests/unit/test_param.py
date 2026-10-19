"""Tests for lumos.core.param and lumos.core.priors (step P1.2).

Requirements: REQ-EL-1 (declarative parameters, validation, SI storage, invalidation), REQ-EL-2 (the
user's unit is kept for display), REQ-EL-3 (free parameters with optional priors).
Reference values come from SI definitions and the exact speed of light.
"""

import numpy as np
import pytest
from astropy import units as u
from hypothesis import given
from hypothesis import strategies as st

from lumos.core.param import Param, ParamHost, Setting, State
from lumos.core.priors import LogUniform, Normal, Uniform


class Lens(ParamHost):
    focal_length = Param(u.m, default=1 * u.m, bounds=(0 * u.m, None), inclusive=(False, True), doc="Focal")
    wavelength = Param(u.m, default=550 * u.nm, equivalencies=u.spectral())
    transmission = Param(default=1.0, bounds=(0, 1))
    offsets = Param(u.m, default=[0, 0] * u.m, shape=(2,))
    profile = Param(u.m, default=np.zeros(3) * u.m, shape=None)
    jones = Param(dtype="complex", default=(1 + 0j) * u.dimensionless_unscaled)
    modes = Setting(int, default=1)
    method = Setting(str, default="auto", choices=("auto", "fft", "mft"))
    position = State(u.mm, default=0 * u.mm, bounds=(-50 * u.mm, 50 * u.mm))

    def __init__(self, **values):
        self.changes = []
        super().__init__(**values)

    def _on_field_changed(self, name):
        self.changes.append(name)


class Required(ParamHost):
    diameter = Param(u.m)


# --- Values and units ------------------------------------------------------------


def test_value_keeps_user_unit_exactly():
    lens = Lens(focal_length=100 * u.mm)
    assert lens.focal_length.unit == u.mm
    assert lens.focal_length.value == 100


def test_si_value_is_in_unscaled_si_unit():
    lens = Lens(focal_length=100 * u.mm)
    assert lens.si("focal_length") == pytest.approx(0.1, rel=1e-15)


def test_defaults_are_used():
    lens = Lens()
    assert lens.wavelength == 550 * u.nm
    assert lens.modes == 1
    assert lens.method == "auto"


def test_missing_required_parameter_raises_type_error():
    with pytest.raises(TypeError, match="diameter"):
        Required()


def test_unknown_keyword_raises_type_error():
    with pytest.raises(TypeError, match="focal_lenght"):
        Lens(focal_lenght=1 * u.m)


def test_wrong_unit_raises_units_error():
    lens = Lens()
    with pytest.raises(u.UnitsError, match="focal_length"):
        lens.focal_length = 1 * u.s


def test_plain_number_for_dimensional_param_raises_type_error():
    lens = Lens()
    with pytest.raises(TypeError, match="focal_length"):
        lens.focal_length = 1.0


def test_dimensionless_param_accepts_plain_numbers():
    lens = Lens(transmission=0.5)
    assert lens.transmission == 0.5 * u.dimensionless_unscaled
    assert lens.si("transmission") == 0.5


def test_equivalencies_allow_frequency_for_wavelength():
    lens = Lens(wavelength=1 * u.THz)
    assert lens.wavelength == 1 * u.THz  # displayed as given
    assert lens.si("wavelength") == pytest.approx(299_792_458 / 1e12, rel=1e-15)


@given(
    value=st.floats(min_value=1e-12, max_value=1e12, allow_nan=False),
    unit=st.sampled_from([u.m, u.mm, u.um, u.nm, u.km, u.au, u.pc]),
)
def test_any_length_round_trips(value, unit):
    lens = Lens(focal_length=value * unit)
    assert lens.focal_length.unit == unit
    assert lens.focal_length.value == value
    assert lens.si("focal_length") == pytest.approx((value * unit).to_value(u.m), rel=1e-15)


def test_instances_are_independent():
    a, b = Lens(focal_length=1 * u.m), Lens(focal_length=2 * u.m)
    assert a.focal_length == 1 * u.m
    assert b.focal_length == 2 * u.m


# --- Bounds ----------------------------------------------------------------------


def test_exclusive_lower_bound_rejects_boundary():
    with pytest.raises(ValueError, match="focal_length"):
        Lens(focal_length=0 * u.m)


def test_bounds_are_compared_in_si():
    with pytest.raises(ValueError, match="position"):
        Lens(position=6 * u.cm)  # 60 mm > 50 mm
    assert Lens(position=5 * u.cm).position == 5 * u.cm  # inclusive upper bound


def test_bounds_apply_to_every_array_element():
    class Host(ParamHost):
        x = Param(u.m, default=[0, 0] * u.m, shape=(2,), bounds=(0 * u.m, None))

    with pytest.raises(ValueError):
        Host(x=[1, -1] * u.m)


def test_nan_is_rejected_when_bounds_are_set():
    with pytest.raises(ValueError):
        Lens(transmission=float("nan"))


def test_incompatible_bound_unit_fails_at_declaration():
    with pytest.raises(u.UnitsError):
        Param(u.m, bounds=(0 * u.s, None))


def test_invalid_default_fails_at_declaration():
    with pytest.raises(ValueError):
        Param(u.m, default=-1 * u.m, bounds=(0 * u.m, None))


# --- Shapes and dtypes -----------------------------------------------------------


def test_fixed_shape_is_enforced():
    with pytest.raises(ValueError, match="offsets"):
        Lens(offsets=[1, 2, 3] * u.m)


def test_scalar_param_rejects_arrays():
    with pytest.raises(ValueError, match="focal_length"):
        Lens(focal_length=[1, 2] * u.m)


def test_any_shape_when_shape_is_none():
    assert Lens(profile=np.ones((2, 3)) * u.m).profile.shape == (2, 3)


def test_real_param_rejects_complex_values():
    with pytest.raises(TypeError, match="focal_length"):
        Lens(focal_length=(1 + 1j) * u.m)


def test_complex_param_accepts_complex_values():
    lens = Lens(jones=(0.5 - 0.5j) * u.dimensionless_unscaled)
    assert lens.si("jones") == 0.5 - 0.5j


def test_arrays_are_copied_and_read_only():
    data = np.array([1.0, 2.0])
    lens = Lens(offsets=data * u.m)
    data[0] = 99.0
    assert lens.offsets[0] == 1 * u.m
    with pytest.raises(ValueError):
        lens.offsets.value[0] = 5.0
    with pytest.raises(ValueError):
        lens.si("offsets")[0] = 5.0


# --- Settings and state ----------------------------------------------------------


@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_int_setting_rejects_other_types(value):
    with pytest.raises(TypeError, match="modes"):
        Lens(modes=value)


def test_setting_choices_are_enforced():
    with pytest.raises(ValueError, match="method"):
        Lens(method="fresnel")


def test_state_behaves_like_a_param():
    lens = Lens(position=12 * u.mm)
    assert lens.si("position") == pytest.approx(0.012, rel=1e-15)


def test_declared_fields_in_definition_order_with_kinds():
    fields = Lens.declared()
    assert list(fields) == [
        "focal_length",
        "wavelength",
        "transmission",
        "offsets",
        "profile",
        "jones",
        "modes",
        "method",
        "position",
    ]
    assert fields["focal_length"].kind == "param"
    assert fields["modes"].kind == "setting"
    assert fields["position"].kind == "state"


def test_inherited_fields_come_first():
    class Thick(Lens):
        thickness = Param(u.mm, default=5 * u.mm)

    names = list(Thick.declared())
    assert names[0] == "focal_length"
    assert names[-1] == "thickness"


def test_class_access_returns_the_field_declaration():
    field = Lens.focal_length
    assert isinstance(field, Param)
    assert field.name == "focal_length"
    assert field.unit == u.m
    assert field.doc == "Focal"


# --- Invalidation hook -------------------------------------------------------------


def test_hook_is_not_called_during_init():
    assert Lens(focal_length=2 * u.m).changes == []


def test_hook_is_called_on_every_change():
    lens = Lens()
    lens.focal_length = 2 * u.m
    lens.modes = 3
    lens.position = 1 * u.mm
    assert lens.changes == ["focal_length", "modes", "position"]


def test_failed_assignment_does_not_change_value_nor_call_hook():
    lens = Lens(focal_length=2 * u.m)
    with pytest.raises(ValueError):
        lens.focal_length = -1 * u.m
    assert lens.focal_length == 2 * u.m
    assert lens.changes == []


# --- Free parameters and priors ---------------------------------------------------


def test_params_are_fixed_by_default():
    assert not Lens().is_free("focal_length")


def test_declared_free_param_with_prior():
    class Host(ParamHost):
        x = Param(u.m, default=1 * u.m, free=True, prior=Uniform(0 * u.m, 2 * u.m))

    host = Host()
    assert host.is_free("x")
    assert host.prior("x") == Uniform(0 * u.m, 2 * u.m)


def test_freeing_a_param_per_instance():
    a, b = Lens(), Lens()
    a.set_free("focal_length", prior=Normal(1 * u.m, 10 * u.mm))
    assert a.is_free("focal_length")
    assert a.prior("focal_length") == Normal(1 * u.m, 10 * u.mm)
    assert not b.is_free("focal_length")
    a.set_free("focal_length", False)
    assert not a.is_free("focal_length")
    assert a.prior("focal_length") is None


def test_prior_unit_must_match_param_unit():
    with pytest.raises(u.UnitsError):
        Lens().set_free("focal_length", prior=Uniform(0 * u.s, 1 * u.s))


def test_only_numeric_settings_can_be_freed():
    with pytest.raises(TypeError):
        Lens().set_free("method")  # a choice
    lens = Lens()
    lens.set_free("modes", True, Uniform(0.5, 3.5))  # a number: free since ADR-0022
    assert lens.is_free("modes")


@pytest.mark.parametrize(
    "make",
    [
        lambda: Uniform(1 * u.m, 1 * u.m),
        lambda: Uniform(2 * u.m, 1 * u.m),
        lambda: LogUniform(0 * u.m, 1 * u.m),
        lambda: Normal(0 * u.m, 0 * u.m),
        lambda: Normal(0 * u.m, -1 * u.m),
    ],
)
def test_invalid_priors_raise_value_error(make):
    with pytest.raises(ValueError):
        make()


def test_prior_bounds_must_share_dimension():
    with pytest.raises(u.UnitsError):
        Uniform(0 * u.m, 1 * u.s)
