"""Tests for typed catalogue records (step P9.2, REQ-CAT-1, REQ-CAT-4).

Requirements: typed objects (``StarRecord``, ``ExoplanetRecord``, ``SolarSystemRecord``,
``ExtendedObjectRecord``) with units, uncertainties and provenance per field; values completed physically are
flagged as estimated.
"""

import pytest
from astropy import units as u

from lumos.catalogs.records import (
    ExoplanetRecord,
    ExtendedObjectRecord,
    Measurement,
    SolarSystemRecord,
    StarRecord,
    record_from_json,
)


def vega():
    return StarRecord(
        "Vega",
        identifiers=["* alf Lyr", "HD 172167"],
        parallax=Measurement(130.23 * u.mas, 0.36 * u.mas, source="simbad", reference="2007A&A...474..653V"),
        teff=Measurement(9602 * u.K, source="simbad", reference="2012ApJ...761...30Y"),
        spectral_type=Measurement("A0V", source="simbad"),
        mag_V=Measurement(0.03 * u.mag, source="simbad"),
    )


def test_fields_carry_values_uncertainties_and_provenance():
    star = vega()
    assert star.parallax == 130.23 * u.mas
    assert star["parallax"].uncertainty == 0.36 * u.mas
    assert star.provenance("parallax") == ("simbad", "2007A&A...474..653V")
    assert star.spectral_type == "A0V"
    assert star.mag_V == 0.03 * u.mag


def test_fields_have_checked_units():
    with pytest.raises(u.UnitsError):
        StarRecord("x", parallax=Measurement(3 * u.km))
    with pytest.raises(u.UnitsError):
        StarRecord("x", teff=Measurement(5000 * u.m))


def test_unknown_fields_are_refused():
    with pytest.raises(AttributeError, match="no field"):
        StarRecord("x", colour=Measurement(1.0))


def test_missing_fields():
    star = vega()
    assert "radius" not in star
    assert star.get("radius") is None
    with pytest.raises(AttributeError, match="radius"):
        _ = star.radius


def test_magnitudes_of_any_band():
    star = StarRecord("x", mag_Ks=Measurement(4.2 * u.mag), mag_Gaia_G=Measurement(5.0 * u.mag))
    assert set(star.magnitudes()) == {"Ks", "Gaia_G"}


def test_distance_from_the_parallax_is_estimated():
    star = vega()
    distance = star.distance_estimate()
    assert distance.value.to_value(u.pc) == pytest.approx(1000 / 130.23)
    assert distance.estimated is True
    assert distance.uncertainty.to_value(u.pc) == pytest.approx(1000 * 0.36 / 130.23**2)


def test_set_and_replace_fields():
    star = vega()
    star.set("radius", Measurement(2.36 * u.R_sun, source="literature"))
    assert star.radius == 2.36 * u.R_sun
    star.set("radius", Measurement(2.8 * u.R_sun, estimated=True))
    assert star["radius"].estimated


@pytest.mark.parametrize(
    "record",
    [
        vega(),
        ExoplanetRecord(
            "HD 209458 b",
            host="HD 209458",
            orbital_period=Measurement(3.52474859 * u.day, source="nasa_exoplanet_archive"),
            radius=Measurement(1.38 * u.R_jup),
            mass=Measurement(0.69 * u.M_jup),
        ),
        SolarSystemRecord("Mars", delta=Measurement(1.5 * u.au), phase_angle=Measurement(30 * u.deg)),
        ExtendedObjectRecord("M42", major_axis=Measurement(65 * u.arcmin), object_type=Measurement("HII")),
    ],
)
def test_json_round_trip(record):
    data = record.to_json()
    again = record_from_json(data)
    assert type(again) is type(record)
    assert again.name == record.name
    assert again.to_json() == data


def test_kinds():
    assert StarRecord.kind == "star"
    assert ExoplanetRecord.kind == "exoplanet"
    assert SolarSystemRecord.kind == "solar_system"
    assert ExtendedObjectRecord.kind == "extended"


def test_unknown_kinds_are_refused():
    with pytest.raises(ValueError, match="kind"):
        record_from_json({"kind": "galaxy-cluster", "name": "x", "fields": {}})


def test_summary_lists_fields_with_sources():
    text = vega().summary()
    assert "parallax" in text
    assert "130.23" in text
    assert "simbad" in text
