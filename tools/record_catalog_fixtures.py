"""Record the responses of catalogue services used as offline test fixtures (phase 9).

The requests are made with ``requests`` directly, independently of the LUMOS code under test; each response
is stored in ``tests/fixtures/catalogs/<name>.json`` with the URL, the request and a list of ``match``
strings that the replaying test transport looks for in the requests of LUMOS. Run when a service changes::

    python tools/record_catalog_fixtures.py [name ...]
"""

from __future__ import annotations

import base64
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import requests

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "catalogs"

SIMBAD = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"
GAIA = "https://gea.esac.esa.int/tap-server/tap/sync"
NASA = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
EXOPLANET_EU = "https://voparis-tap-planeto.obspm.fr/tap/sync"
IRSA = "https://irsa.ipac.caltech.edu/TAP/sync"
ESO = "https://archive.eso.org/tap_obs/sync"
MAST = "https://mast.stsci.edu/vo-tap/api/v0.1/caom/sync"


def tap(url: str, query: str) -> dict:
    return {
        "method": "POST",
        "url": url,
        "data": {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable", "QUERY": query},
    }


def simbad_basic(name: str) -> str:
    return (
        "SELECT b.main_id, b.otype, b.ra, b.dec, b.plx_value, b.plx_err, b.plx_bibcode, b.pmra, b.pmdec, "
        "b.pm_bibcode, b.rvz_radvel, b.rvz_err, b.rvz_bibcode, b.sp_type, b.sp_bibcode, i.ids "
        "FROM basic AS b JOIN ident AS n ON n.oidref = b.oid JOIN ids AS i ON i.oidref = b.oid "
        f"WHERE n.id = '{name}'"
    )


def simbad_fluxes(name: str) -> str:
    return (
        "SELECT f.filter, f.flux, f.flux_err, f.bibcode FROM flux AS f JOIN ident AS n ON n.oidref = f.oidref "
        f"WHERE n.id = '{name}'"
    )


def simbad_parameters(name: str) -> str:
    return (
        "SELECT m.teff, m.log_g, m.fe_h, m.bibcode FROM mesFe_h AS m JOIN ident AS n ON n.oidref = m.oidref "
        f"WHERE n.id = '{name}' AND m.teff IS NOT NULL ORDER BY bibcode DESC"
    )


NASA_COLUMNS = (
    "pl_name, hostname, pl_orbper, pl_orbpererr1, pl_orbsmax, pl_orbeccen, pl_orbincl, pl_orblper, pl_rade, "
    "pl_radeerr1, pl_bmasse, pl_bmasseerr1, pl_eqt, discoverymethod, sy_dist, st_teff, st_rad, st_mass, ra, dec"
)

REQUESTS: dict[str, dict] = {
    "simbad_vega_basic": {**tap(SIMBAD, simbad_basic("Vega")), "match": ["FROM basic", "'Vega'"]},
    "simbad_vega_fluxes": {**tap(SIMBAD, simbad_fluxes("Vega")), "match": ["FROM flux", "'Vega'"]},
    "simbad_vega_parameters": {**tap(SIMBAD, simbad_parameters("Vega")), "match": ["FROM mesFe_h", "'Vega'"]},
    "simbad_trappist_basic": {
        **tap(SIMBAD, simbad_basic("TRAPPIST-1")),
        "match": ["FROM basic", "'TRAPPIST-1'"],
    },
    "simbad_trappist_fluxes": {
        **tap(SIMBAD, simbad_fluxes("TRAPPIST-1")),
        "match": ["FROM flux", "'TRAPPIST-1'"],
    },
    "simbad_trappist_parameters": {
        **tap(SIMBAD, simbad_parameters("TRAPPIST-1")),
        "match": ["FROM mesFe_h", "'TRAPPIST-1'"],
    },
    "simbad_m42_basic": {**tap(SIMBAD, simbad_basic("M42")), "match": ["FROM basic", "'M42'"]},
    "gaia_trappist": {
        **tap(
            GAIA,
            "SELECT source_id, ra, dec, parallax, parallax_error, pmra, pmdec, phot_g_mean_mag, phot_bp_mean_mag, "
            "phot_rp_mean_mag, teff_gspphot, radial_velocity FROM gaiadr3.gaia_source "
            "WHERE source_id = 2635476908753563008",
        ),
        "match": ["gaiadr3.gaia_source", "2635476908753563008"],
    },
    "nasa_hd209458b": {
        **tap(NASA, f"SELECT {NASA_COLUMNS} FROM pscomppars WHERE pl_name = 'HD 209458 b'"),
        "match": ["pscomppars", "pl_name = 'HD 209458 b'"],
    },
    "nasa_trappist_system": {
        **tap(NASA, f"SELECT {NASA_COLUMNS} FROM pscomppars WHERE hostname = 'TRAPPIST-1'"),
        "match": ["pscomppars", "hostname = 'TRAPPIST-1'"],
    },
    "exoplanet_eu_catalog": {
        "method": "GET",
        "url": "https://exoplanet.eu/catalog/csv/",
        "match": [],
        # excerpt: the header and the rows of the planets used by the tests (the full file is 3.7 MB)
        "keep_rows": ["HD 209458 b", "TRAPPIST-1 "],
    },
    "vizier_sed_vega": {
        "method": "GET",
        "url": "https://vizier.cds.unistra.fr/viz-bin/sed",
        "params": {"-c": "Vega", "-c.rs": "1"},
        "match": ["Vega"],
    },
    "irsa_2mass_trappist": {
        **tap(
            IRSA,
            "SELECT designation, ra, dec, j_m, j_msig, h_m, h_msig, k_m, k_msig FROM fp_psc WHERE "
            "CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 346.6223683, -5.0413975, 0.002)) = 1",
        ),
        "match": ["fp_psc"],
    },
    "eso_trappist": {
        **tap(
            ESO,
            "SELECT TOP 5 obs_id, target_name, instrument_name, dataproduct_type, s_ra, s_dec, em_min, em_max, "
            "t_min, access_url FROM ivoa.ObsCore WHERE target_name = 'TRAPPIST-1'",
        ),
        "match": ["ivoa.ObsCore", "'TRAPPIST-1'"],
    },
    "mast_trappist": {
        **tap(
            MAST,
            "SELECT TOP 5 obs_id, target_name, instrument_name, dataproduct_type, s_ra, s_dec, em_min, em_max, "
            "t_min, access_url FROM ivoa.obscore WHERE target_name = 'TRAPPIST-1'",
        ),
        "match": ["ivoa.obscore", "'TRAPPIST-1'"],
    },
    "horizons_mars": {
        "method": "GET",
        "url": "https://ssd.jpl.nasa.gov/api/horizons.api",
        "params": {
            "format": "json",
            "COMMAND": "'499'",
            "OBJ_DATA": "'YES'",
            "MAKE_EPHEM": "'YES'",
            "EPHEM_TYPE": "'OBSERVER'",
            "CENTER": "'500@399'",
            "START_TIME": "'2026-01-01 00:00'",
            "STOP_TIME": "'2026-01-02 00:00'",
            "STEP_SIZE": "'1 d'",
            "QUANTITIES": "'1,13,19,20,24'",
            "ANG_FORMAT": "'DEG'",
            "CSV_FORMAT": "'YES'",
        },
        "match": ["'499'"],
    },
    "svo_johnson_v": {
        "method": "GET",
        "url": "https://svo2.cab.inta-csic.es/theory/fps/fps.php",
        "params": {"ID": "Generic/Johnson.V"},
        "match": ["Generic/Johnson.V"],
    },
    "calspec_index": {
        "method": "GET",
        "url": "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/",
        "match": [],
    },
    "calspec_vega": {
        "method": "GET",
        "url": "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/alpha_lyr_stis_012.fits",
        "match": [],
    },
    "calspec_sun": {
        "method": "GET",
        "url": "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/sun_reference_stis_002.fits",
        "match": [],
    },
    "pickles_index": {
        "method": "GET",
        "url": "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/pickles/dat_uvk/pickles_uk.fits",
        "match": [],
    },
    "pickles_a0v": {
        "method": "GET",
        "url": "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/pickles/dat_uvk/pickles_uk_9.fits",
        "match": [],
    },
    "skyview_m42": {
        "method": "GET",
        "url": "https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl",
        "params": {"Position": "M42", "Survey": "DSS", "Pixels": "64", "Size": "0.5", "Return": "FITS"},
        "match": ["M42", "DSS"],
    },
    "hips2fits_m42": {
        "method": "GET",
        "url": "https://alasky.cds.unistra.fr/hips-image-services/hips2fits",
        "params": {
            "hips": "CDS/P/DSS2/red",
            "ra": "83.8221",
            "dec": "-5.3911",
            "fov": "0.5",
            "width": "64",
            "height": "64",
            "projection": "TAN",
            "format": "fits",
        },
        "match": ["CDS/P/DSS2/red"],
    },
}


def record(name: str, spec: dict) -> None:
    session = requests.Session()
    session.headers["User-Agent"] = "lumos-maxima (https://github.com/VForiel/LUMOS)"
    method = spec["method"]
    try:
        reply = session.request(
            method, spec["url"], params=spec.get("params"), data=spec.get("data"), timeout=120
        )
    except requests.RequestException as error:
        print(f"{name}: request failed: {error}")
        return
    if not reply.ok:
        print(f"{name}: HTTP {reply.status_code}: {reply.text[:600]}")
        return
    content = reply.content
    if "keep_rows" in spec:
        lines = reply.text.splitlines()
        kept = [lines[0]] + [
            line for line in lines[1:] if any(line.strip('"').startswith(key) for key in spec["keep_rows"])
        ]
        content = ("\n".join(kept) + "\n").encode("utf-8")
    fixture = {
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "method": method,
        "url": spec["url"],
        "params": spec.get("params", {}),
        "data": spec.get("data", {}),
        "match": spec.get("match", []),
        "status": reply.status_code,
        "content_type": reply.headers.get("Content-Type", ""),
        "content": base64.b64encode(content).decode("ascii"),
        "excerpt": "keep_rows" in spec,
    }
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / f"{name}.json").write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"{name}: {len(content)} bytes ({fixture['content_type']})")


def main(names: list[str]) -> None:
    for name in names or list(REQUESTS):
        record(name, REQUESTS[name])


if __name__ == "__main__":
    main(sys.argv[1:])
