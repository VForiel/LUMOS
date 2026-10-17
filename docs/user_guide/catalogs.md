# Astronomical catalogues

`lumos.catalogs` turns names into typed records and records into elements of a scene.

```python
from lumos.catalogs import resolve
from lumos.elements.sources import Planet, Star
from lumos.scenes import MappedComponent, Scene

vega = resolve("Vega")  # StarRecord from SIMBAD
vega.parallax, vega["parallax"].uncertainty, vega.provenance("parallax")
print(vega.summary())

star = Star.from_catalog("Vega")  # temperature, V magnitude, distance from the parallax
planet = Planet.from_catalog("HD 209458 b")  # star and orbit from the NASA Exoplanet Archive
system = Scene.from_catalog("TRAPPIST-1")  # the star and its seven planets
nebula = MappedComponent.from_survey("M42", survey="DSS", flux_density=100 * u.Jy)
```

## Providers

| Provider | Data |
|----------|------|
| `Simbad` | identifiers, type, position, parallax, proper motion, radial velocity, spectral type, magnitudes, Teff/log g/[Fe/H] |
| `GaiaDR3` | astrometry, G/BP/RP photometry, Teff, radial velocity of a source |
| `NasaExoplanetArchive`, `ExoplanetEU` | planets and systems: orbits, radii, masses, host stars |
| `VizierPhotometry` | every photometric point of the VizieR catalogues (SED) |
| `Calspec`, `PicklesLibrary` | flux standards and stellar library spectra |
| `Horizons` | ephemerides of Solar-System bodies; `solar_spectrum()` |
| `SvoFilterService` | filter transmission curves and zero points |
| `Eso`, `Mast` | observations in the ESO and MAST archives (ObsCore) |
| `Irsa` | 2MASS photometry |
| `SkyView`, `Hips2Fits` | survey images, turned into mapped components by `survey_component` |

Providers are registered by name (`lumos.catalogs.providers()`); plugins add theirs with the
`lumos.catalogs` entry-point group.

## Records and provenance

Each field of a record is a `Measurement`: value with unit, uncertainty, provider, bibliographic reference
and whether LUMOS estimated it. `StarRecord.distance_estimate()` gives `1/parallax`; `lumos.catalogs.sed`
fuses multi-catalogue photometry (one point per filter, priority catalogues, smallest relative uncertainty,
median) and completes it with a blackbody fit; `complete_star` fills a missing temperature, distance or
radius, flagged as estimated.

## Network, cache and offline use

Every request uses HTTPS with verified certificates and a timeout, and is cached for 7 days in the user cache
directory:

```python
from lumos.catalogs import settings

settings.ttl = 30 * 86400  # seconds
settings.offline = True  # read the cache only
settings.cache_dir = "my_cache"
```

The environment variables `LUMOS_CATALOGS_CACHE` and `LUMOS_CATALOGS_OFFLINE=1` set the same defaults: the
notebook *Targets from catalogues* runs offline on the answers stored in `examples/catalog_cache`. See
[ADR-0012](../adr/0012-catalogues.md).
