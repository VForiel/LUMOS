# User guide

This guide will explain the concepts of LUMOS:

- **elements**, their parameters with physical units, their ports and their actions;
- **element groups**, which behave like single elements;
- the **context**, which holds the graph of elements, the clock and the operating mode (simulation or operation);
- the **signals** that flow between elements (wavefronts, wavecubes, incoherent bundles, radiance fields, data);
- the `.lumos` file format;
- the **element library**: sources, optics, telescopes, atmosphere, coronagraphs, fibres, photonic chips and
  detectors;
- **scenes**: point and extended objects in a frame, and the imaging of extended scenes;
- **catalogues**: typed records from SIMBAD, Gaia, exoplanet archives, surveys... and scenes built from them;
- **symbolic optics**: closed-form fields carried through the graph, and expressions recovered from grids;
- **processing**: reduction, photometry, interferometric observables, detection statistics, calibration and
  script blocks;
- **results**: figures, data files and archives of runs;
- **procedures**: sequences of actions, measurements and processing (a dark, then measurements);
- **your own components**: elements defined without code, from templates.

```{toctree}
:maxdepth: 1

fields
context
elements
scenes
catalogs
symbolic
processing
results
procedures
components
stability
```
