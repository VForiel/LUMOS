# Symbolic optics

`lumos.symbolic` represents fields as closed-form SymPy expressions. Sources of symbolic fields enter a
context like any other source; elements keep the fields symbolic as long as they have a closed form, and
LUMOS converts them to sampled fields where needed, saying so in the journal.

## Symbolic fields

A {class}`~lumos.symbolic.SymbolicField` holds an expression of the canonical variables of its plane
(`x, y` in metres, or `theta_x, theta_y` in radians), of the wavelength `lam` and of the time `t`, plus named
parameters given as quantities:

```python
import sympy as sp
from astropy import units as u
from lumos.symbolic import SymbolicField, x, y

w = sp.Symbol("w", positive=True)
field = SymbolicField(sp.exp(-(x**2 + y**2) / w**2), parameters={w: 1 * u.mm}, wavelength=633 * u.nm)
field.latex()  # LaTeX code; Jupyter renders the field directly
field.to_wavefront(grid)  # sampled on a Grid (compiled with lambdify, cached)
```

As for {class}`~lumos.fields.Wavefront`, $|U|^2$ is a power per unit area in `power_unit` (W by default).
An expression that depends on `lam` is sampled into a {class}`~lumos.fields.Wavecube`.

## Closed forms

`gaussian_beam` builds Hermite-Gauss and Laguerre-Gauss beams described by their complex parameter $q$.
Free space and thin lenses apply their ABCD matrices exactly (Kogelnik & Li 1966), with the Gouy phase. A plane
wave through a circular or rectangular pupil, then a lens and a propagation over its focal length, gives the
Airy or sinc pattern in closed form (Goodman, §5.2). Delay lines, beam splitters, spectral filters and
apertures without spiders multiply the expression by their transmission.

## In a context

{class}`~lumos.symbolic.SymbolicSource` emits a symbolic field:

```python
from lumos import Context
from lumos.elements.detectors import PowerMeter
from lumos.elements.optics import FreeSpace, ThinLens
from lumos.fields import Grid
from lumos.symbolic import SymbolicSource

ctx = Context()
beam = ctx.add(
    SymbolicSource.gaussian_beam(
        name="beam", waist=2 * u.mm, wavelength=1064 * u.nm, power=10 * u.mW, grid=Grid(128, dx=10 * u.um)
    )
)
(
    beam
    >> ctx.add(ThinLens(name="lens", focal_length=0.5 * u.m))
    >> ctx.add(FreeSpace(name="gap", distance=0.499 * u.m))
    >> ctx.add(PowerMeter(name="meter"))
)
result = ctx.run(["gap.out", "meter.out"])  # gap.out is a SymbolicField
```

When an element has no symbolic transfer (detectors, most complex elements), or no closed form for its input
(a pupil with spiders, a propagation of an arbitrary expression), the field is **sampled on its grid hint**
(the `grid` given to the source) at the wavelengths of the observation if it depends on `lam`, the numerical
transfer runs, and the journal records it:

```text
meter: symbolic input(s) converted to numbers (no symbolic transfer)
pupil: no closed form (circular apertures with spiders have no symbolic transfer): numerical fallback
```

Discontinuous expressions (pupils) are averaged over 4 × 4 sub-pixels to avoid aliasing; removable
singularities, such as the centre of the Airy pattern, take their limit. Without a grid hint, the conversion
fails with a message saying which grid to give. {func}`~lumos.symbolic.to_numeric` performs the same conversion
by hand.

Symbolic sources and outputs are saved in `.lumos` files and archives as trees of allowed SymPy operations
(`lumos.symbolic.serialize`): reading them never evaluates code.

## From grids to expressions

`lumos.symbolic.reconstruct` goes the other way. Every function returns a `Reconstruction` with the symbolic
field, the coefficients or parameters found and the relative error `residual` on the grid:

| Function | Model |
| --- | --- |
| `project_zernike` | phase over a circular pupil on Noll's Zernike modes |
| `project_hermite_gauss`, `project_laguerre_gauss` | Hermite-Gauss and Laguerre-Gauss modes of a waist |
| `project_lp` | LP modes of a step-index fibre |
| `project_fourier` | largest terms of the Fourier series |
| `fit_gaussian` | Gaussian beam with centre, waist and radius of curvature |
| `fit_airy` | Airy pattern (scale λf/D, centre, peak) |
| `sparse_regression` | sparse polynomial in reduced coordinates (STLSQ) |
| `symbolic_regression` | PySR, optional (`pip install pysr`) |

See the notebook [Symbolic optics](../examples/notebooks/07_symbolic_optics.ipynb).
