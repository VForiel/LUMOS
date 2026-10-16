"""Tests for the stellar, exoplanet and archive providers (step P9.3, REQ-CAT-1, REQ-CAT-2).

The responses are recorded from the services (``tests/fixtures/catalogs``, see ``conftest.py``). Expected
values are those of the catalogues, with their references:

- Vega (SIMBAD, 2026-10-16): parallax 130.23 ± 0.36 mas and proper motion (200.94, 286.23) mas/yr from the
  new Hipparcos reduction (van Leeuwen 2007, A&A 474, 653, ``2007A&A...474..653V``); radial velocity
  -13.5 ± 0.4 km/s (``2004A&A...420..183A``); spectral type A0V; V = 0.03 mag (``2002yCat.2237....0D``);
  most recent effective temperature of the mesFe_H table, 9375 K (``2026ApJ..1006..228K``);
- TRAPPIST-1 (Gaia DR3, Gaia Collaboration 2023, A&A 674, A1): source 2635476908753563008, parallax
  80.2123 ± 0.0716 mas, G = 15.6226 mag;
- HD 209458 b (NASA Exoplanet Archive, Planetary Systems Composite Parameters): P = 3.52474859 ± 3.8e-7 d,
  a = 0.04707 au, R = 15.58051 R⊕, M = 232.0159 M⊕, T_eq = 1459 K, host at 48.3016 pc; the seven planets of
  TRAPPIST-1 with P_b = 1.510826 d and P_h = 18.772866 d;
- HD 209458 b (exoplanet.eu): P = 3.52472 d, R = 1.38 R_J, M = 0.69 M_J, a = 0.04747 au;
- Vega photometry (VizieR SED service): 340 points, among which Johnson V measurements;
- Vega spectrum (CALSPEC ``alpha_lyr_stis_012``): F_λ(5556 Å) = 3.46e-9 erg s⁻¹ cm⁻² Å⁻¹ within 2 %
  (Bohlin 2014, AJ 147, 127, §4: 3.47e-9 at 5556 Å);
- TRAPPIST-1 in the 2MASS point-source catalogue (IRSA): J = 11.354, H = 10.718, Ks = 10.296 mag;
- observations of TRAPPIST-1 in the ESO (HAWK-I) and MAST (HST STIS and WFC3) archives.
"""

import numpy as np
import pytest
from astropy import units as u

from lumos.catalogs import providers
from lumos.catalogs.archives import Eso, Irsa, Mast
from lumos.catalogs.calspec import Calspec
from lumos.catalogs.exoplanets import ExoplanetEU, NasaExoplanetArchive
from lumos.catalogs.gaia import GaiaDR3
from lumos.catalogs.records import ExoplanetRecord, StarRecord
from lumos.catalogs.simbad import Simbad
from lumos.catalogs.spectra import Spectrum
from lumos.catalogs.vizier import VizierPhotometry

FLAM = u.erg / u.s / u.cm**2 / u.AA


# --- SIMBAD -------------------------------------------------------------------------------------------------


def test_simbad_star(replay):
    vega = Simbad().star("Vega")
    assert isinstance(vega, StarRecord)
    assert vega.name == "* alf Lyr"
    assert "HD 172167" in vega.identifiers
    assert vega.parallax.to_value(u.mas) == pytest.approx(130.23)
    assert vega["parallax"].uncertainty.to_value(u.mas) == pytest.approx(0.36, rel=1e-6)
    assert vega.provenance("parallax") == ("simbad", "2007A&A...474..653V")
    assert vega.pm_ra.to_value(u.mas / u.yr) == pytest.approx(200.94)
    assert vega.pm_dec.to_value(u.mas / u.yr) == pytest.approx(286.23)
    assert vega.radial_velocity.to_value(u.km / u.s) == pytest.approx(-13.5)
    assert vega.spectral_type == "A0V"
    assert vega.ra.to_value(u.deg) == pytest.approx(279.234734787025)
    assert vega.mag_V.to_value(u.mag) == pytest.approx(0.03, abs=1e-6)
    assert vega.provenance("mag_V") == ("simbad", "2002yCat.2237....0D")
    assert vega.teff.to_value(u.K) == pytest.approx(9375)
    assert vega.provenance("teff")[1] == "2026ApJ..1006..228K"


def test_simbad_object_type_of_a_nebula(replay):
    nebula = Simbad().basic("M42")
    assert nebula["object_type"] == "HII"
    assert nebula["main_id"] == "M  42"


def test_simbad_quotes_are_escaped(replay):
    with pytest.raises(AssertionError, match="unexpected"):
        Simbad().star("O'Brien")  # not recorded: the request is refused by the replay transport
    query = replay.calls[-1]["data"]["QUERY"]
    assert "'O''Brien'" in query


# --- Gaia ---------------------------------------------------------------------------------------------------


def test_gaia_dr3_source(replay):
    star = GaiaDR3().source(2635476908753563008)
    assert star.parallax.to_value(u.mas) == pytest.approx(80.2123, rel=1e-5)
    assert star["parallax"].uncertainty.to_value(u.mas) == pytest.approx(0.0716, rel=1e-3)
    assert star.mag_Gaia_G.to_value(u.mag) == pytest.approx(15.6226, abs=1e-4)
    assert star.provenance("parallax")[0] == "gaia_dr3"
    assert "teff" not in star  # masked in the archive


# --- Exoplanets ---------------------------------------------------------------------------------------------


def test_nasa_exoplanet_archive_planet(replay):
    planet = NasaExoplanetArchive().planet("HD 209458 b")
    assert isinstance(planet, ExoplanetRecord)
    assert planet.host == "HD 209458"
    assert planet.orbital_period.to_value(u.day) == pytest.approx(3.52474859)
    assert planet["orbital_period"].uncertainty.to_value(u.day) == pytest.approx(3.8e-7)
    assert planet.semi_major_axis.to_value(u.au) == pytest.approx(0.04707)
    assert planet.radius.to_value(u.R_earth) == pytest.approx(15.58051)
    assert planet.mass.to_value(u.M_earth) == pytest.approx(232.0159)
    assert planet.equilibrium_temperature.to_value(u.K) == pytest.approx(1459)
    assert planet.host_distance.to_value(u.pc) == pytest.approx(48.3016)
    assert planet.discovery_method == "Radial Velocity"


def test_nasa_exoplanet_archive_system(replay):
    planets = NasaExoplanetArchive().system("TRAPPIST-1")
    assert [p.name for p in planets] == [f"TRAPPIST-1 {letter}" for letter in "bcdefgh"]
    assert planets[0].orbital_period.to_value(u.day) == pytest.approx(1.510826)
    assert planets[-1].orbital_period.to_value(u.day) == pytest.approx(18.772866)


def test_exoplanet_eu_planet(replay):
    planet = ExoplanetEU().planet("HD 209458 b")
    assert planet.orbital_period.to_value(u.day) == pytest.approx(3.52472)
    assert planet.radius.to_value(u.R_jup) == pytest.approx(1.38)
    assert planet.mass.to_value(u.M_jup) == pytest.approx(0.69)
    assert planet.semi_major_axis.to_value(u.au) == pytest.approx(0.04747)
    assert planet.host == "HD 209458"
    assert planet.provenance("radius")[0] == "exoplanet_eu"


def test_unknown_planet_raises(replay):
    from lumos.catalogs import CatalogError

    with pytest.raises(CatalogError, match="not found"):
        ExoplanetEU().planet("Nonexistent b")


# --- Photometry and spectra --------------------------------------------------------------------------------


def test_vizier_photometry(replay):
    sed = VizierPhotometry().sed("Vega")
    assert len(sed) == 340
    assert np.all(sed.flux_density.to_value(u.Jy) > 0)
    assert sed.wavelengths.unit.is_equivalent(u.m)
    assert any(name == "Johnson:V" for name in sed.filters)
    assert sed.catalogues[0].startswith("I/")


def test_calspec_spectrum_of_vega(replay):
    spectrum = Calspec().spectrum("alpha_lyr")
    assert isinstance(spectrum, Spectrum)
    assert spectrum.source == "calspec"
    assert "alpha_lyr_stis_012" in spectrum.reference
    flux = spectrum.at(5556 * u.AA).to_value(FLAM)
    assert flux == pytest.approx(3.46e-9, rel=2e-2)


# --- Archives -----------------------------------------------------------------------------------------------


def test_irsa_two_mass_photometry(replay):
    photometry = Irsa().two_mass(346.6223683 * u.deg, -5.0413975 * u.deg, radius=7.2 * u.arcsec)
    assert photometry["J"].value.to_value(u.mag) == pytest.approx(11.354, abs=1e-6)
    assert photometry["H"].value.to_value(u.mag) == pytest.approx(10.718, abs=1e-6)
    assert photometry["Ks"].value.to_value(u.mag) == pytest.approx(10.296, abs=1e-6)
    assert photometry["J"].source == "irsa"


@pytest.mark.parametrize(
    ("archive", "instruments"), [(Eso, {"HAWKI"}), (Mast, {"STIS/FUV-MAMA", "WFC3/UVIS"})]
)
def test_archive_observations(replay, archive, instruments):
    table = archive().observations("TRAPPIST-1", limit=5)
    assert len(table) == 5
    assert set(table["instrument_name"]) == instruments
    for column in ("obs_id", "dataproduct_type", "s_ra", "s_dec", "em_min", "em_max", "access_url"):
        assert column in table.colnames


# --- Registry -----------------------------------------------------------------------------------------------


def test_providers_are_registered():
    names = set(providers())
    assert {"simbad", "gaia_dr3", "nasa_exoplanet_archive", "exoplanet_eu", "vizier", "calspec"} <= names
    assert {"eso", "mast", "irsa"} <= names
    for cls in providers().values():
        if cls.url:
            assert cls.url.startswith("https://")
