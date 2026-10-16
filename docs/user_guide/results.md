# Figures, archives and results

Every LUMOS result can be traced back to the exact system and environment that produced it: figures follow a
common style with physical units, are saved as images **and** as editable PltEdit files next to their data,
and each run can be archived next to the script or notebook that produced it.

## Figure style

`lumos.plotting.style` gives all figures the same look:

- colour maps by role (`style.CMAPS`): perceptually uniform maps for intensities (`inferno`), radiances
  (`magma`) and data (`viridis`), the cyclic `twilight` map for phases, `RdBu_r` for signed quantities;
- axis labels with units, `style.axis_label("wavelength", u.um)` → `wavelength (µm)`;
- publication widths: `style.figure("single")` (88 mm, one column) and `style.figure("double")` (180 mm);
- 300 dpi output, small readable fonts, images with the origin at the bottom left.

```python
from lumos.plotting import style

style.apply()  # for the whole session, or:
with style.context():  # for a block
    fig, ax = style.figure("single")
```

## Plots

| Function | Shows |
|----------|-------|
| `plot_intensity`, `plot_phase`, `plot_field` | intensity and phase (cyclic map) of a field |
| `plot_psf` | a point-spread function normalised to its peak, optionally in λ/D units |
| `plot_radial_profile` | the azimuthal average of an intensity (`lumos.processing.radial_profile`) |
| `plot_spectrum` | spectra of sources, wavecubes or data products |
| `plot_transmission_map` | a 2D transmission map |
| `plot_chip_outputs` | the power on each output of a photonic chip |
| `plot_context`, `save_context_diagram` | the graph of a context (SVG, PNG, PDF) |
| `plot_radiance`, `plot_data` | radiance fields and data products |

## Exporting figures and data

```python
from lumos.results import save_figure, load_data

save_figure(fig, "figures/psf", data={"r": radii, "profile": profile})
# -> figures/psf.png, figures/psf.plt (reopen with `pltedit figures/psf.plt`), figures/psf.npz
```

Data files are NumPy `.npz` archives of plain arrays; quantities keep their unit and are read back as
quantities by `load_data`.

## Archives

`ctx.run(outputs, archive="name")` creates an archive with
[`vfst.archive`](https://github.com/VForiel/VFSciTools) in an `archives/` directory **next to the
executed script or notebook** (set `lumos.config.archive_root` to change it):

```text
archives/2026-10-17_1a2b3c4/0001-name/
├── context.lumos     the context, reloadable with lumos.io.load
├── metadata.yml      UTC date, OS, Python and package versions, git commit and branch, hardware,
│                     backend and precision, seed, durations, mode, twin, clock, origin
├── lumos.log         the journal of the context, one JSON object per line
├── figures/          archive.save_figure(fig, "name"): .png and .plt
└── data/             the computed outputs and archive.save_data(...) files
```

```python
result = ctx.run(["camera.out"], archive="first-light")
result.archive.save_figure(fig, "image", data={"counts": counts})

loaded = lumos.load_archive(result.archive.path)
loaded.context  # the Context
loaded.metadata  # metadata.yml
loaded.data["camera.out"]
loaded.figures["image"]  # path of image.png
```

Reading an archive never unpickles or imports anything: `.plt` files are only listed, open them with
`pltedit.load` if you trust their origin.
