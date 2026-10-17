# Processing and detection

`lumos.processing` reduces detected data and evaluates detection capabilities. Its functions take and return
{class}`~lumos.fields.DataProduct` objects and propagate their uncertainties (GUM); its **elements** wrap them
with `DataProduct` ports, so that a pipeline is built in the same graph as the optics.

## Reduction and photometry

| Function | Element | What it does |
| --- | --- | --- |
| `reduction.subtract_dark` | `DarkSubtraction` (inputs `in`, `dark`) | $S - D$, $\sigma^2 = \sigma_S^2 + \sigma_D^2$ |
| `reduction.flat_field` | `FlatField` (inputs `in`, `flat`) | division by the flat normalised to a mean of 1 |
| `reduction.stack` | `Stack` | mean ($\sigma/\sqrt N$), median ($\sqrt{\pi/2}\,\sigma/\sqrt N$) or sigma-clipped mean |
| `reduction.crop`, `rebin`, `resample` | `Crop`, `Rebin` | geometry; rebinning and resampling conserve the flux |
| `photometry.aperture_photometry` | `AperturePhotometry` | exact disk overlap per pixel, median background annulus |
| `photometry.ccd_snr` | | CCD equation, with the error of the background estimate |
| `photometry.contrast_curve` | `ContrastCurve` | 5σ limit with the small-sample penalty of Mawet et al. (2014) |
| | `RadialProfile`, `SignalToNoise` | azimuthal mean with its error; data / uncertainty |

```python
from lumos.processing import reduction, photometry

master_dark = reduction.stack(darks, method="median")
reduced = reduction.subtract_dark(reduction.stack(frames, method="sigma_clip"), master_dark)
result = photometry.aperture_photometry(
    reduced, center=(0 * u.um, 0 * u.um), radius=80 * u.um, annulus=(150 * u.um, 220 * u.um)
)
result.flux, result.uncertainty
```

Positions and radii are given in the unit of the image axes (metres or micrometres for a camera of LUMOS,
pixels for plain arrays).

## Interferometry

`interferometry.null_depth` (element `NullDepth`) divides a dark output by a bright one;
`interferometry.kernel_observables` (`KernelObservables`) forms the kernels of the kernel nuller, robust to
second order in the piston errors; `interferometry.transmission_map` moves a point source of a context over a
grid of sky positions and returns the transmission of one output.

## Detection statistics

`detection.matched_filter` is the Neyman-Pearson test of a known signal in Gaussian noise, distributed as
$\mathcal N(0, 1)$ without signal and $\mathcal N(d, 1)$ with it; `detection.energy` the energy detector
($\chi^2_N$). `roc_curve` gives the empirical ROC of samples of a statistic, `roc_matched_filter` and
`roc_energy` the analytic ones, `threshold` the threshold of a false-alarm probability. For a companion whose
deflection grows with its contrast, `contrast_limit` returns the contrast detected with a given probability and
`detection_capability` the map of the detection probability versus separation and contrast.

## Calibration

- `calibration.fit_phases(measure, initial=..., method=...)` minimises a measured quantity (a null depth...)
  over phase settings, by gradient (`"gradient"`), trial and error (`"scan"`) or a genetic algorithm
  (`"genetic"`).
- `calibration.mtf_slanted_edge` (element `MTFMeasurement`) measures the MTF of the image of a slanted edge
  (ISO 12233).
- `calibration.measure_psf` measures the centroid, peak, FWHM, encircled energy and Strehl ratio of a PSF.

## Script blocks

A function of yours becomes an element with declared ports ({doc}`../adr/0013-script-blocks`):

```python
from lumos.processing import script


@script(name="mylab.pipeline.Normalise", inputs={"in": DataProduct}, outputs={"out": DataProduct})
def normalise(frame, *, level: u.Quantity = 1 * u.adu):
    """Scale a frame to a peak of ``level``."""
    return DataProduct(frame.data / frame.data.max() * level)


ctx.add(normalise(name="normalise", level=100 * u.adu))
```

Its keyword arguments become parameters with units. A `.lumos` file stores the name of the script and the
parameter values, never its code: define the script (import its module) before loading the file.

See the notebooks [Data pipeline](../examples/notebooks/08_data_pipeline.ipynb) and
[Detection capability](../examples/notebooks/09_detection_capability.ipynb).
