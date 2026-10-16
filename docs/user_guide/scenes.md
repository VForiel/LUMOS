# Scenes and extended objects

A **scene** describes what an instrument observes: point sources (stars, planets) and diffuse objects
(disks, nebulae, galaxies, backgrounds, maps), placed in a frame.

```python
from astropy import units as u
from lumos import Context
from lumos.elements.sources import Star
from lumos.scenes import Scene
from lumos.scenes.diffuse import InclinedRing

ctx = Context(seed=1)
ctx.observation.band = (1.5 * u.um, 1.8 * u.um)
sky = ctx.add(
    Scene(
        name="sky", frame="celestial", ra=83.8 * u.deg, dec=-5.4 * u.deg, n_pixels=64, pixel_scale=10 * u.mas
    )
)
sky.add(Star(name="star", temperature=5800 * u.K, distance=10 * u.pc))
sky.add(
    InclinedRing(
        name="disk",
        inner_radius=100 * u.mas,
        outer_radius=150 * u.mas,
        inclination=60 * u.deg,
        flux_density=1 * u.mJy,
    )
)
result = ctx.run(["sky.sources", "sky.radiance"])
```

## Frames

| Frame | Coordinates | Objects |
|-------|-------------|---------|
| `celestial` | ICRS pointing `(ra, dec)`, angles on the sky (East towards +x, North towards +y) | at infinity |
| `terrestrial` | site `(latitude, longitude, altitude)`, angles seen from the instrument | at infinity |
| `laboratory` | metres, at `distance` from the pupil | finite distance |

Laboratory maps keep their radiance (radiance is invariant along rays) and are seen with pixels of
`dx / distance` radians.

## Components

- **Point components**: any element producing a `SourceList` (`PointSource`, `Star`, `PlanetarySystem`).
- **Parametric diffuse components** (`lumos.scenes.diffuse`): `UniformDisk`, `GaussianComponent`,
  `SersicComponent`, `InclinedRing` (debris and exo-zodiacal disks), `UniformBackground`, `ZodiacalLight`;
  each with a total flux density (or a surface brightness) at a reference wavelength and a flat or blackbody
  spectrum.
- **Maps** (`lumos.scenes.mapped.MappedComponent`): `from_fits` (images and cubes with a WCS, `BUNIT` in
  Jy/pixel, MJy/sr, Jy/arcsec², W m⁻² sr⁻¹ m⁻¹...), `from_array`, `from_image` (PNG, JPEG, TIFF with a
  calibration).
- **Procedural components** (`lumos.scenes.procedural`): `GaussianRandomField` (log-normal texture with a
  power-law spectrum), `SiemensStar`, `Checkerboard`, `BarTarget`; their geometry is in pixels, their grid
  angular (`pixel_scale`) or spatial (`pixel_size`, laboratory charts).

Diffuse components are evaluated on their own grids and resampled onto the scene grid, conserving the flux.

## Radiometry

`lumos.physics.radiometry` gives the blackbody radiance, photon energies and photon ↔ energy conversions,
solid angles of cones, étendue, Lambertian exitance, the irradiance of a uniform disk, the power collected by
a pupil and spectral integrals; magnitudes are in `lumos.physics.photometry`.

## Imaging an extended scene

`image_scene` images a radiance map through the part of a context between an input receiving point sources
and an output field:

```python
from lumos.scenes import ImagingSystem, image_scene

system = ImagingSystem(instrument_ctx, entrance="telescope.in", exit="focal_plane.out")
image = image_scene(radiance, system)  # strategy chosen automatically
image.strategy, image.reason  # e.g. "isoplanatic", "the PSF is shift-invariant..."
camera_input = image.to_wavecube()  # intensities for a detector
```

| Strategy | Principle | When |
|----------|-----------|------|
| `isoplanatic` | convolution of the scene by the on-axis PSF | shift-invariant systems (default for them) |
| `anisoplanatic` | PSFs on a grid of field points, interpolated | slowly varying PSFs (on request) |
| `decomposition` | one incoherent point source per pixel | coronagraphs, interferometers, fibres |

The automatic choice probes the system with an off-axis point; it is logged. Inside a graph, the element
`SceneToSources` decomposes a scene: `sky.radiance >> SceneToSources() >> telescope`. See
[ADR-0011](../adr/0011-extended-scene-imaging.md) and the notebook *Extended scenes*.
