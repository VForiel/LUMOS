# ADR-0012: Astronomical catalogues

- **Status**: accepted
- **Date**: 2026-10-17
- **Plan step**: P9.1 to P9.9 (REQ-CAT-1 to REQ-CAT-7, NF-7)

## Context

LUMOS reads a dozen services (SIMBAD, VizieR, Gaia, CALSPEC, ESO, IRSA, MAST, the NASA Exoplanet Archive,
exoplanet.eu, JPL Horizons, the SVO Filter Profile Service, SkyView, hips2fits) and must stay safe (TLS always
verified, timeouts), usable offline, testable without network, and traceable (provenance of every value).

## Decision

- **One network layer** (`lumos.catalogs.http.fetch`): `https` only, TLS verification that cannot be turned
  off, a timeout on every request, a JSON user cache keyed by method, URL, parameters and body (7-day
  lifetime), an offline mode, and an injectable transport (`requests` by default). `LUMOS_CATALOGS_CACHE` and
  `LUMOS_CATALOGS_OFFLINE` set the cache and the offline mode for notebooks and CI.
- **Thin providers on standard protocols**: IVOA TAP (ADQL, VOTable read with astropy) for SIMBAD, Gaia DR3,
  the NASA Exoplanet Archive (`pscomppars`), IRSA (2MASS), ESO and MAST (ObsCore); plain HTTP for the
  VizieR SED service, CALSPEC and Pickles files (STScI), the SVO FPS, JPL Horizons (JSON API), SkyView and
  hips2fits. `astroquery` and `pyvo` are not needed: the `[catalogs]` extra only holds `requests`.
- **exoplanet.eu** is read from its CSV catalogue (its TAP server refused connections on 2026-10-16); the
  file (3.7 MB) is cached like any answer.
- **Solar reference spectrum**: CALSPEC `sun_reference_stis` (Bohlin et al. 2001) instead of the ASTM E490
  table, which is distributed as a spreadsheet; both agree within a few percent.
- **Typed records** (`StarRecord`, `ExoplanetRecord`, `SolarSystemRecord`, `ExtendedObjectRecord`): every
  field is a `Measurement` (value with unit, uncertainty, provider, reference, `estimated` flag).
- **SED fusion rules**: one point per filter, priority catalogues first (`DEFAULT_PRIORITIES`), then the
  smallest relative uncertainty, then the median; completion by a blackbody fit, flagged as estimated; a
  measured effective temperature is kept and only the solid angle is fitted.
- **Tests on recorded answers**: `tools/record_catalog_fixtures.py` records the real answers with `requests`
  (independently of LUMOS) in `tests/fixtures/catalogs`; a replay transport answers the requests of LUMOS
  and refuses any other. Expected values are those of the catalogues, cited in the tests.

## Alternatives considered

- `astroquery` everywhere: convenient, but heavy, with its own caches and network settings that would bypass
  the guarantees of the network layer (TLS, timeouts, offline mode).
- Mocking parsed results instead of recording raw answers: the tests would not check the parsing of the real
  formats.

## Consequences

- Service changes (columns, formats) are caught by re-recording the fixtures; network tests can be added
  under the `network` marker.
- The cache of `examples/catalog_cache` must be re-recorded when the notebook `06` changes its queries.
