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
- **results**: figures, data files and archives of runs.

```{toctree}
:maxdepth: 1

fields
context
elements
scenes
catalogs
results
```
