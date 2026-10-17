"""Units offered by the forms of the web interface (step P14.4, REQ-WEB-2, REQ-WEB-3).

Requirements: the palette gives, for each unit used by a parameter, the units a user may type instead with
their conversion factor, so that the forms convert values without code per element. Units of another dimension
are never offered.

Expected factors are the definitions of the SI prefixes and of the angle units (BIPM SI brochure, 9th ed.):
1 µm = 1e-6 m, 1 nm = 1e-9 m, 1 W = 1000 mW, 1 rad = 180/π deg, 1 arcsec = 1/3600 deg, 1 mas = 1e-3 arcsec.
"""

import math

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from lumos.web import create_app


@pytest.fixture(scope="module")
def units():
    with TestClient(create_app(), base_url="http://localhost") as client:
        return client.get("/api/palette").json()["units"]


def factors(units, unit):
    return dict(units[unit])


def test_every_unit_of_a_parameter_has_its_choices(units):
    with TestClient(create_app(), base_url="http://localhost") as client:
        palette = client.get("/api/palette").json()
    used = {
        field["x-lumos-unit"]
        for item in palette["types"]
        for field in item["schema"]["properties"]["params"]["properties"].values()
        if "x-lumos-unit" in field
    }
    assert used <= set(units)
    for unit in used:
        assert factors(units, unit)[unit] == pytest.approx(1.0)  # the unit itself is always offered


def test_lengths(units):
    meters = factors(units, "m")
    assert meters["um"] == pytest.approx(1e-6)
    assert meters["nm"] == pytest.approx(1e-9)
    assert meters["mm"] == pytest.approx(1e-3)
    assert "s" not in meters
    assert "deg" not in meters


def test_powers(units):
    milliwatts = factors(units, "mW")
    assert milliwatts["W"] == pytest.approx(1e3)
    assert milliwatts["uW"] == pytest.approx(1e-3)


def test_angles(units):
    degrees = factors(units, "deg")
    assert degrees["rad"] == pytest.approx(180 / math.pi)
    assert degrees["arcsec"] == pytest.approx(1 / 3600)
    assert degrees["mas"] == pytest.approx(1e-3 / 3600)
