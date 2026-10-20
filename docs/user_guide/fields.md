# Fields and data

LUMOS represents light and detected data with a small set of signal classes. They all live in
`lumos.fields`, store their numbers in arrays of the current backend (NumPy, JAX...) without units, and expose
physical quantities as `astropy.units.Quantity` objects.

| Class | Describes | Array shape |
|-------|-----------|-------------|
| `Grid` | Sampling of a spatial (m) or angular (rad) plane | — |
| `Wavefront` | Coherent field at one wavelength | `(ny, nx)` or `(2, ny, nx)` |
| `Wavecube` | Coherent field at several wavelengths | `(nλ, [2,] ny, nx)` |
| `IncoherentBundle` | Mutually incoherent modes | `(M, *mode_shape)` |
| `RadianceField` | Spectral radiance of an incoherent scene | `(nλ, [4,] ny, nx)` |
| `ModalField` | Amplitudes of the modes of one guide | `([M,] nλ, n)` |
| `Bus` | Lanes separated in space (guides, telescope arms) | one array when packed |
| `DataProduct` | Detected or processed data | any |

## Grids

A `Grid` has `nx × ny` pixels of steps `dx, dy` around a centre `(x0, y0)`. The unit of `dx` decides the
plane: a length gives a spatial plane (pupil, detector), an angle gives an angular plane (sky, far field).

```python
from astropy import units as u
from lumos.fields import Grid

pupil = Grid(256, dx=20 * u.um)
sky = Grid(128, 64, dx=5 * u.mas)
focal_plane = sky.to_spatial(10 * u.m)  # x = f·θ
```

Arrays are indexed `[..., y, x]`, and the central pixel is index `N // 2` for even and odd `N`: the pixel
coordinates are `x_i = (i - N//2)·dx + x0`. This is also the origin of the centred Fourier transforms of
`lumos.backend.fft`, so a field and its spectrum share the same central pixel.

## Wavefronts

A `Wavefront` is a complex amplitude `U` sampled on a grid at one vacuum wavelength. Its normalisation is
chosen so that `Σ |U|² dx dy` is the power, in watts or in photons per second (`power_unit`).

```python
from lumos.fields import Wavefront

beam = Wavefront.gaussian(pupil, 1.55 * u.um, waist=1 * u.mm, power=2 * u.mW)
beam.power  # 2 mW
beam.with_power_unit(u.photon / u.s).power
tilted = beam.tilt(10 * u.arcsec, 0 * u.arcsec)
```

Fields follow the convention `E = Re{U e^{-iωt}}`: a wave travelling towards +z varies as `e^{+ikz}`.
Multiplying a wavefront by an array applies a transmission mask; adding two wavefronts superposes them
coherently.

## Polarisation

A polarised wavefront holds the Jones components `(Ex, Ey)` in the laboratory basis. The module
`lumos.fields.jones` provides basis states, Jones matrices (polarisers, retarders, wave plates, rotators) and
Stokes parameters:

```python
from lumos.fields import jones

light = Wavefront.plane(pupil, 633 * u.nm, power=1 * u.mW, polarization="H")
out = jones.apply(jones.quarter_wave_plate(45 * u.deg), light)
jones.stokes(out)  # circular polarisation: |S3| = S0
```

The right circular state is `R = (1, i)/√2` and has `S3 = +S0`.

## Polychromatic light

A `Wavecube` stacks wavefronts at several wavelengths on the same grid, with spectral weights. Fields at
different wavelengths do not interfere, so the detected intensity is the weighted sum of the monochromatic
intensities. `spectral_samples` samples a band with the midpoint or Gauss-Legendre rule:

```python
from lumos.fields import Wavecube, spectral_samples

wavelengths, weights = spectral_samples(1.6 * u.um, 0.3 * u.um, 7, rule="gauss")
cube = Wavecube.plane(pupil, wavelengths, power=1 * u.mW, weights=weights)
cube.total_power
```

## Incoherent light

An `IncoherentBundle` gathers modes that do not interfere: the stars of a field, samples of an extended
object, or the two polarisations of natural light. Elements process all the modes at once, and the detected
intensity is the sum of the mode intensities.

```python
from lumos.fields import IncoherentBundle

binary = IncoherentBundle.point_sources(
    pupil, 1 * u.um, theta_x=[0, 50] * u.mas, theta_y=[0, 0] * u.mas, power=[1, 0.1] * u.nW
)
natural = IncoherentBundle.unpolarized(beam)
```

A point source at infinity in the direction `(θx, θy)` illuminates the pupil with the plane wave
`exp[ik(x sin θx + y sin θy)]`.

## Radiance maps

A `RadianceField` describes a scene without phase: its spectral radiance `L(x, y, λ)`, optionally with the
Stokes parameters. The radiance may be given per wavelength or per frequency, in energy or photon units
(`W m⁻² sr⁻¹ m⁻¹`, `Jy sr⁻¹`...). `resample` changes the grid while conserving the flux.

## Guided light

A `ModalField` holds the complex amplitudes of light in the modes of **one** guide (one mode for a
single-mode guide), for each wavelength and each incoherent source. Its `sources` attribute identifies the
source list they come from. Scattering matrices act on its modes; `lumos.fields.smatrix` provides directional
couplers and phase shifters.

```python
from lumos.fields import ModalField, smatrix

light = ModalField([[1, 0]], [1.55] * u.um)  # one guide with two modes
mzi = smatrix.coupler(0.5) @ smatrix.phase_shifter([0.3 * u.rad, 0 * u.rad]) @ smatrix.coupler(0.5)
light.apply(mzi).mode_power
```

## Buses

Guides separated in space — the waveguides of a chip, the arms of an interferometer — are the **lanes** of a
`Bus`. A bus behaves as a sequence of signals of one class (`len(bus)`, `bus[i]`, iteration), names its lanes
(`T1`, `T2`... by default) and may carry the projected position of the telescope feeding each lane. Spectral
channels, incoherent sources and the modes of one guide are never lanes: they share the same beam or guide.

```python
from lumos.fields import Bus

guides = Bus.guided([[1, 0, 0, 0]], [1.55] * u.um)  # four single-mode lanes
len(guides), guides.names  # 4, ('T1', 'T2', 'T3', 'T4')
guides[0].mode_power
```

When its lanes share their metadata (same sources, wavelengths, spectral weights and power unit for guided
lanes; same grid and metadata for beams), a bus stores them in one array (`bus.packed`): `bus.modal` holds all
the modes of all the guided lanes, `bus.field` the stacked beams. Otherwise it keeps a list of lanes, with the
same interface, and `Bus.packing_problem(lanes)` says why. Guided lanes interfere only if they describe the
same sources in the same order: a component mixing lanes that cannot be packed together refuses them.

An element whose input does not declare a bus acts on every lane of a bus it receives, as N identical copies
(a stochastic element draws an independent stream per lane): a `Fiber` on a bus of beams returns a bus of
guided lanes. Elements producing data (cameras, processing blocks) need one beam: recombine the lanes first.

To put different components on the lanes, split a bus with `Demux(n_lanes=N)` (ports `out0`... `out{N-1}`, one
lane each, keeping its name and position) and gather lanes with `Mux(n_lanes=N)` (ports `in0`...), which
stacks them in the order of its ports without mixing them; an unconnected input of a `Mux` is an empty lane.

```python
from lumos.elements.buses import Demux, Mux

demux = ctx.add(Demux(name="arms", n_lanes=4))
mux = ctx.add(Mux(name="chip_in", n_lanes=4))
ctx.connect(array.port("out"), demux.port("in"))
ctx.connect(demux.port("out0"), mux.port("in1"))  # swap the first two arms
ctx.connect(demux.port("out1"), mux.port("in0"))
```

## Data products

A `DataProduct` is detected data with physical axes, a mask, uncertainties and acquisition metadata. It is
written to FITS (with a WCS for regular axes), NPZ or HDF5 files:

```python
from lumos.fields import DataProduct

image = DataProduct.from_grid(counts * u.electron, detector_grid, meta={"exposure_time": 2 * u.s})
image.to_fits("image.fits")
```

## Plots

`lumos.plotting` draws intensities, phases (with a cyclic colour map), radiance maps and data products with
physical axes and the origin at the bottom left.
