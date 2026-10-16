# Wave propagation

LUMOS propagates a field sampled on one grid to a field sampled on another grid, in another plane. This page
explains the methods, their domains of validity and how the planner chooses between them.

```python
from astropy import units as u
from lumos.fields import Grid, Wavefront
from lumos.propagation import focus, plan, propagate

pupil = Grid(512, dx=10 * u.um)
beam = Wavefront.gaussian(pupil, 1 * u.um, waist=0.5 * u.mm, power=1 * u.mW)

near = propagate(beam, 20 * u.cm)  # planner chooses the method
far = propagate(beam, None, output_grid=Grid(128, dx=20 * u.urad))  # far field
image = focus(beam, 1 * u.m, Grid(64, dx=5 * u.um))  # focal plane of a thin lens
plan(beam, 20 * u.cm).reason  # why that method
```

## Conventions

The field is the complex amplitude `U` of `E = Re{U e^{-iωt}}`: a wave travelling towards +z varies as
`e^{+ikz}`, with `k = 2π/λ`. Every method includes this propagation phase. In the far field, the angular grid
holds direction cosines and the field is normalised so that `Σ |U|² ΔΩ` is the power.

## Methods

**Rayleigh-Sommerfeld (`rs`).** The first Rayleigh-Sommerfeld solution is summed directly over the input
pixels for each output point. It makes no approximation and works between any grids, but its cost is the
product of the numbers of input and output pixels. It is the reference against which the other methods are
validated.

**Angular spectrum (`asm`).** The field is decomposed into plane waves by a Fourier transform; each plane wave
of spatial frequency `(fx, fy)` advances by `exp(2πi z sqrt(1/λ² - fx² - fy²))`. This is exact, including
evanescent waves, as long as the transfer function is sampled finely enough. Sampling it on the grid of a
window of width `2S` aliases its chirp at long distances, so LUMOS keeps only the frequencies below the band
limit of Matsushima and Shimobaba (2009). On another output grid, the inverse transform is evaluated as a
matrix Fourier transform at the output coordinates.

**Fresnel (`fresnel_tf`, `fresnel_ir`, `fresnel_mft`).** In the paraxial approximation the propagation is a
convolution with a quadratic-phase kernel. The approximation holds when the neglected phase
`k ρ⁴ / 8z³` is small for the largest lateral distance `ρ`. The transfer-function form (`tf`) suits short
distances and the impulse-response form (`ir`) long ones; the two are separated by the critical distance
`N dx² / λ`. The one-step form (`mft`) expands the kernel and evaluates a single Fourier integral with a matrix
Fourier transform, which gives the field on any output grid.

**Fraunhofer (`fraunhofer`).** Far from the input plane, or in the focal plane of a lens, the field is the
Fourier transform of the input field. Matrix Fourier transforms (Soummer et al. 2007) compute it at any
sampling and centre, at a cost proportional to `N² M + N M²` for `N²` input and `M²` output pixels.

| Method | Output grid | Use it for |
|--------|-------------|------------|
| `rs` | any | reference, small grids, extreme near field |
| `asm` | any | most free-space propagation, back-propagation |
| `fresnel_tf` | input grid | short paraxial distances |
| `fresnel_ir` | input grid | long paraxial distances |
| `fresnel_mft` | any | paraxial propagation with resampling |
| `fraunhofer` | any | far field, focal planes, coronagraph planes |

## Sampling

A sampled field represents a band-limited function: the grid step `dx` sets the highest spatial frequency
`1/(2dx)`, hence the largest angle `λ/(2dx)` the field can carry, and the window width `L = N dx` sets the
frequency resolution `1/L`. Every method then has its own conditions:

- the Fourier methods are periodic: light leaving the window comes back on the other side unless the window
  is padded (LUMOS pads by a factor 2);
- a quadratic phase (Fresnel kernels, lenses) must change by less than π between neighbouring samples where
  the field is not negligible;
- a matrix Fourier transform evaluates the periodic spectrum of the sampled field: output frequencies must
  stay away from its replicas, one period `1/dx` apart.

## The planner

With `method="auto"` (the default), the planner looks at the field itself. It measures the region holding
all but `1e-6` of the power and the band holding all but `1e-6` of the spectral power, then applies ordered
rules: the far field uses `fraunhofer`, back-propagation uses `asm`, then `asm` if the spectrum fits inside its
band limit, then a Fresnel method if the paraxial phase error is below 0.01 rad and its sampling conditions
hold, then `fraunhofer` at finite distance, and finally the direct sum. The choice and its reason are logged on
the `lumos.propagation` logger and returned by `plan()`; passing `method=` forces a method. The rules are
recorded in [ADR-0005](../adr/0005-propagation-planner.md).

## Validation

Each method is tested against analytic results before being implemented: Airy zeros and peak intensity,
the sampled and continuous transforms of a square aperture, the Gaussian beam waist, curvature and Gouy phase,
the exact on-axis field of a circular aperture, an explicit Rayleigh-Sommerfeld sum, Parseval's theorem,
back-propagation and reciprocity. Far-field and focal-plane results agree with POPPY and HCIPy to round-off.
Gradients computed by JAX through every method agree with finite differences.
