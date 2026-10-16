"""Tests for lumos.units (step P1.1).

Requirements: REQ-EL-1, REQ-EL-2 (Quantity at the boundary, SI floats inside) and §10 of the specs
(quantities serialised as {"value", "unit"}; arrays inline as base64 + dtype + shape).

Reference values come from the SI definitions (1 nm = 1e-9 m, 1 Jy = 1e-26 W m^-2 Hz^-1), the exact
speed of light (299 792 458 m/s) and the IEEE-754 encoding of 1.0 as a little-endian float64.
"""

import base64
import json
import math

import numpy as np
import pytest
from astropy import units as u
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from lumos.units import quantity_from_json, quantity_to_json, require_unit, si_unit, to_si

# --- require_unit --------------------------------------------------------------


def test_require_unit_accepts_convertible_quantity_unchanged():
    q = 550 * u.nm
    assert require_unit(q, u.m) is q


def test_require_unit_accepts_equivalent_unit_with_equivalencies():
    q = 1 * u.THz
    assert require_unit(q, u.m, equivalencies=u.spectral()) is q


def test_require_unit_rejects_incompatible_unit_with_name_in_message():
    with pytest.raises(u.UnitsError, match="focal_length"):
        require_unit(3 * u.s, u.m, name="focal_length")


def test_require_unit_rejects_plain_number_for_dimensional_unit():
    with pytest.raises(TypeError, match="focal_length"):
        require_unit(3.0, u.m, name="focal_length")


def test_require_unit_rejects_non_numeric():
    with pytest.raises(TypeError):
        require_unit("3 m", u.m)


@pytest.mark.parametrize("value", [0.3, 2, np.array([0.1, 0.2])])
def test_require_unit_wraps_plain_numbers_for_dimensionless(value):
    q = require_unit(value, u.dimensionless_unscaled)
    assert isinstance(q, u.Quantity)
    assert q.unit == u.dimensionless_unscaled
    np.testing.assert_array_equal(q.value, value)


# --- si_unit -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        (u.nm, u.m),
        (u.km / u.s, u.m / u.s),
        (u.deg, u.rad),
        (u.percent, u.dimensionless_unscaled),
        (u.dimensionless_unscaled, u.dimensionless_unscaled),
        (u.Jy, u.kg / u.s**2),
        (u.W / u.m**2 / u.um, u.kg / u.m / u.s**3),
    ],
)
def test_si_unit_is_unscaled_si_unit_with_same_dimension(unit, expected):
    result = si_unit(unit)
    assert result == expected
    assert result.scale == 1.0  # type: ignore[union-attr]


# --- to_si ---------------------------------------------------------------------


def test_to_si_scalar_returns_python_float():
    value = to_si(550 * u.nm)
    assert type(value) is float
    assert value == pytest.approx(550e-9, rel=1e-15)


def test_to_si_array_returns_ndarray():
    value = to_si(np.array([1.0, 2.5]) * u.km)
    assert isinstance(value, np.ndarray)
    np.testing.assert_allclose(value, [1000.0, 2500.0], rtol=1e-15)


def test_to_si_with_target_unit_and_equivalencies():
    # λ = c / ν with c = 299 792 458 m/s exactly.
    value = to_si(1 * u.THz, u.m, equivalencies=u.spectral())
    assert value == pytest.approx(299_792_458 / 1e12, rel=1e-15)


def test_to_si_angle():
    assert to_si(180 * u.deg) == pytest.approx(math.pi, rel=1e-15)


def test_to_si_rejects_incompatible_target():
    with pytest.raises(u.UnitsError):
        to_si(1 * u.s, u.m)


# --- JSON serialisation ----------------------------------------------------------


def test_scalar_keeps_user_unit():
    data = quantity_to_json(550 * u.nm)
    assert data["value"] == 550.0
    assert u.Unit(data["unit"]) == u.nm


@pytest.mark.parametrize(("value", "encoded"), [(math.nan, "nan"), (math.inf, "inf"), (-math.inf, "-inf")])
def test_non_finite_scalars_are_strings(value, encoded):
    data = quantity_to_json(value * u.m)
    assert data["value"] == encoded
    json.dumps(data, allow_nan=False)  # must be strict JSON


def test_complex_scalar():
    data = quantity_to_json((1.0 - 2.0j) * u.dimensionless_unscaled)
    assert data["value"] == {"re": 1.0, "im": -2.0}


def test_array_encoding_is_base64_little_endian_c_order():
    data = quantity_to_json(np.array([1.0]) * u.m)
    assert data["value"] == {"dtype": "<f8", "shape": [1], "data": "AAAAAAAA8D8="}


def test_array_encoding_matches_independent_decoding():
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    data = quantity_to_json(array * u.s)
    raw = base64.b64decode(data["value"]["data"])
    decoded = np.frombuffer(raw, dtype=data["value"]["dtype"]).reshape(data["value"]["shape"])
    np.testing.assert_array_equal(decoded, array)


def test_output_is_strict_json():
    q = np.array([[1 + 1j, math.nan]]) * u.W
    json.dumps(quantity_to_json(q), allow_nan=False)


@pytest.mark.parametrize(
    "data",
    [
        {"unit": "m"},
        {"value": 1.0},
        {"value": 1.0, "unit": "not_a_unit"},
        {"value": "one", "unit": "m"},
        {"value": {"dtype": "<f8", "shape": [2], "data": "AAAAAAAA8D8="}, "unit": "m"},  # wrong size
    ],
)
def test_invalid_json_raises_value_error(data):
    with pytest.raises(ValueError):
        quantity_from_json(data)


UNITS = [
    u.m,
    u.nm,
    u.dimensionless_unscaled,
    u.deg,
    u.arcsec,
    u.K,
    u.Jy,
    u.W / u.m**2 / u.um / u.sr,
    u.m / u.s**2,
    u.ph / u.s / u.cm**2,
]


def assert_same_quantity(a: u.Quantity, b: u.Quantity):
    assert a.unit == b.unit
    assert np.shape(a) == np.shape(b)
    if isinstance(a.value, np.ndarray) and a.ndim > 0:
        assert a.value.dtype == b.value.dtype
    np.testing.assert_array_equal(a.value, b.value)  # treats NaN == NaN


@given(value=st.floats(allow_nan=True, allow_infinity=True), unit=st.sampled_from(UNITS))
def test_round_trip_real_scalar(value, unit):
    q = value * unit
    assert_same_quantity(quantity_from_json(json.loads(json.dumps(quantity_to_json(q)))), q)


@given(
    re=st.floats(allow_nan=False, allow_infinity=False),
    im=st.floats(allow_nan=False, allow_infinity=False),
    unit=st.sampled_from(UNITS),
)
def test_round_trip_complex_scalar(re, im, unit):
    q = complex(re, im) * unit
    assert_same_quantity(quantity_from_json(json.loads(json.dumps(quantity_to_json(q)))), q)


@given(
    array=hnp.arrays(
        dtype=st.sampled_from([np.float32, np.float64, np.complex64, np.complex128]),
        shape=hnp.array_shapes(min_dims=1, max_dims=3, max_side=5),
    ),
    unit=st.sampled_from(UNITS),
)
def test_round_trip_array(array, unit):
    q = u.Quantity(array, unit, dtype=array.dtype)
    assert_same_quantity(quantity_from_json(json.loads(json.dumps(quantity_to_json(q)))), q)
