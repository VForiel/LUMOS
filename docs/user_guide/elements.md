# Element library

LUMOS ships a library of elements covering the path of light from the source to the detector. Every element
declares physical parameters with units, typed ports and a pure transfer function; it can be added to a
context, connected with `>>`, saved in a `.lumos` file and rebuilt by the generated Python script.

| Family | Module | Elements |
|--------|--------|----------|
| Sources | `lumos.elements.sources` | `PointSource`, `Star`, `PlanetarySystem`, `GaussianLaser`, `IncoherentLamp` |
| Apertures | `lumos.elements.apertures` | `CircularAperture`, `PolygonAperture`, `HexSegmentedAperture`; presets `vlt()`, `jwst()`, `elt()` |
| Bulk optics | `lumos.elements.optics` | `FreeSpace`, `ThinLens`, `ThinMirror`, `BeamSplitter`, `DelayLine`, `SpectralFilter`, `Polarizer`, `WavePlate`, `PhasePlate` |
| Telescopes | `lumos.elements.telescopes` | `Telescope`, `TelescopeArray`; presets `vlti()`, `life()` |
| Atmosphere and AO | `lumos.elements.atmosphere` | `Atmosphere`, `DeformableMirror`, `WavefrontSensor`, `AOController` |
| Coronagraphs | `lumos.elements.coronagraphs` | `VortexCoronagraph`, `FourQuadrantPhaseMask`, `LyotCoronagraph` |
| Guided optics | `lumos.elements.fibers` | `Fiber` |
| Integrated photonics | `lumos.elements.photonics` | `Waveguide`, `PhaseShifter`, `DirectionalCoupler`, `YJunction`, `Crossing`, `MMI`, `TransferMatrix` (a simplified MMI given by its matrix); chips `bracewell()`, `kernel_nuller()` |
| Detectors | `lumos.elements.detectors` | `Camera`, `Photodiode`, `PowerMeter` |

The physics of each element, with its references, is given in the docstring of its module (see the
[API reference](../api/index.md)).

## Sources and telescopes

Sources at infinity produce a `SourceList` sampled at the spectral channels of the context, so the context
needs an observation band. A `Telescope` turns the sources into the plane waves sampled on its pupil, one mode
of an `IncoherentBundle` per source, carrying the photon rate collected by the pupil:

```python
from astropy import units as u
from lumos import Context
from lumos.elements.optics import FreeSpace, ThinLens
from lumos.elements.sources import Star
from lumos.elements.telescopes import Telescope
from lumos.fields import Grid

ctx = Context()
ctx.observation.band = (1.5 * u.um, 1.8 * u.um)
ctx.observation.n_wavelengths = 4

star = ctx.add(Star(name="star", band="H", magnitude=5.0))
focal = Grid(128, dx=2 * u.um)
chain = star >> Telescope(name="scope", diameter=8 * u.m, n_pixels=256)
chain = chain >> ThinLens(name="lens", focal_length=100 * u.m)
chain >> FreeSpace(name="gap", distance=100 * u.m, method="fresnel_mft", output_grid=focal)
image = ctx.run(["gap.out"])["gap.out"]
```

A `TelescopeArray` models an interferometer whose beams are injected into single-mode guides: its output is a
`Bus` of single-mode guided lanes, one per telescope, carrying the projected position of each telescope. In the `"ground"` frame the station positions are projected on
the sky for the declination and hour angle of the target, and the hour angle follows the clock of the context;
in the `"sky"` frame (space arrays such as LIFE) the array rotates about the line of sight.

## Optics, atmosphere and coronagraphs

Bulk optics act on wavefronts, wavecubes and incoherent bundles alike, channel by channel and mode by mode. A
`FreeSpace` element propagates with any method of `lumos.propagation` to an arbitrary output grid.

`Atmosphere` adds the optical path of von Kármán turbulence layers carried by the wind (frozen flow), drawn
from the reproducible random stream of the element: the same seed gives the same screens. Its
`corrected_modes` setting removes the first Zernike modes over the pupil, a simple model of adaptive optics.
`DeformableMirror` is a grid of actuators whose commands can be fitted to an optical path map.

A **closed adaptive optics loop** uses two more elements and a feedback of the context (see
[Feedback loops](context.md#feedback-loops)). `WavefrontSensor` is an ideal Shack-Hartmann: it measures the
mean gradient of the optical path in each sub-aperture (in radians, independent of the wavelength, even for
wavefronts of many waves) and the RMS residual optical path over the pupil (valid below about λ/4).
`AOController` turns the slopes into command increments of a mirror of the same geometry, `Δc = -g R s`, with
the regularised pseudo-inverse of the interaction matrix of the mirror (computed from its influence functions).

```python
dm = ctx.add(DeformableMirror(name="dm", n_actuators=17, diameter=8.2 * u.m))
wfs = ctx.add(WavefrontSensor(name="wfs", diameter=8.2 * u.m, n_subapertures=16))
ctrl = ctx.add(AOController(name="ctrl", n_actuators=17, diameter=8.2 * u.m, gain=0.4))
atmosphere >> dm >> wfs
ctx.connect(wfs.slopes, ctrl.slopes)
ctx.feedback("ctrl.commands", "dm.commands", mode="add")  # the integrator
steps = ctx.loop(30, period=2 * u.ms, outputs=["wfs.residual", "camera.out"])
```

The error of an aberration the mirror can reproduce decreases as `(1 - g)^k`; on turbulence the loop settles
at the fitting error of the mirror, about `0.3 (d/r0)^(5/3)` rad² for a pitch `d`. The *Adaptive optics loop*
preset is a complete example.

Coronagraphs receive the pupil field, apply their focal-plane mask and return the field after the Lyot stop.
Ideal masks (even-charge vortex, four-quadrant phase mask) reject an on-axis star down to the sampling error
of the pupil.

## Guided and integrated optics

A `Fiber` projects the field on its face onto its LP modes, propagates them and outputs both the modal
amplitudes (a `ModalField`, one lane) and the near field at its output; on a bus of beams it acts on every
lane. Photonic components transform buses of guided lanes with their scattering matrices, on the lanes listed
in their `channels` setting (all lanes when empty); a mixing component (coupler, MMI, transfer matrix) acts on
all the modes of its lanes, which must describe the same sources. Chips are element groups:

```python
from lumos.elements.detectors import PowerMeter
from lumos.elements.photonics import kernel_nuller, kernels
from lumos.elements.sources import PointSource
from lumos.elements.telescopes import TelescopeArray

ctx = Context()
ctx.observation.band = (3.5 * u.um, 4.0 * u.um)
star = ctx.add(PointSource(name="star", flux_density=1 * u.Jy))
array = ctx.add(
    TelescopeArray(
        name="array", frame="sky", diameter=2 * u.m, positions=[[0, 0], [30, 10], [55, -20], [80, 25]] * u.m
    )
)
star >> array >> kernel_nuller(name="chip") >> PowerMeter(name="meter")
power = ctx.run(["meter.out"])["meter.out"].data
print(kernels(power.value))  # three kernel outputs, zero for an on-axis star
```

## Detectors

A `Camera` reads the field sampled on its pixels: quantum efficiency (constant or a curve), photon and dark
noise, full-well saturation, charge binning, read noise, gain, bias and analog-to-digital conversion. Its
`expose()` action lasts the exposure time and draws a new frame; with `noise=False` it returns the expected
image. A `Photodiode` gives a photocurrent with shot noise and a `PowerMeter` the power in watts; both read a
beam or a guided lane (all its modes) as one value, and a bus as one value per lane.

## Random streams

Elements that draw random numbers (atmosphere, camera, photodiode) are *stochastic*: they receive the key of
their random stream in the context (`ctx.seed` and their identifier). Results are reproducible for a given
seed, independent between elements, and recomputed when `ctx.seed` changes (ADR-0007).
