# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Project skeleton: `lumos-maxima` distribution, `lumos` package, MIT license.
- Development tooling: ruff, pyright, pytest, Hypothesis, pre-commit.
- Continuous integration and test-integrity check.
- Sphinx documentation skeleton (Breeze theme, English and French).
- `lumos.units`: unit validation, SI conversion and strict-JSON encoding of quantities.
- `lumos.core.param`: declarative `Param`, `State` and `Setting` fields with units, bounds, shapes,
  invalidation hook and per-instance free parameters; `lumos.core.priors` (uniform, log-uniform, normal).
- `lumos.core.signal` and `lumos.core.port`: abstract signal types and typed input/output ports.
- `lumos.core.element`: `Element` base class with pure static transfer functions, `describe()`,
  Jupyter representation and JSON Schema.
- `lumos.core.registry`: element type registry with stable identifiers, versions, migrations and plugins.
- `lumos.io`: `.lumos` documents for single elements (save/load, version checks, migrations).
- `lumos.backend`: NumPy, JAX, CuPy and PyTorch backends through the Array API, loaded lazily, with a
  thread-local current backend and conversions keeping dtypes.
- `lumos.backend.fft`: 2-D FFTs on any backend, centred transforms (origin at `N // 2`), multi-threaded on CPU.
- `lumos.hardware_report()` and automatic backend selection (`lumos.config.backend = "auto"`).
- `lumos.config`: validated settings (`backend`, `precision`, `fft_workers`).
- `lumos.backend.memory`: dtypes per precision, memory estimates and chunked computation.
- `lumos.jit`: JAX compilation on JAX inputs; `Values` snapshots are JAX pytrees.
- `lumos.fields.Grid`: rectangular spatial or angular sampling, centre at `N // 2`, lazy coordinates.
- `lumos.fields.Wavefront`: scalar or polarised coherent field with power normalisation in W or photon/s.
- `lumos.fields.jones`: Jones vectors and matrices (polarisers, retarders, rotators), Stokes parameters.
- `lumos.fields.Wavecube` and `spectral_samples`: polychromatic stacks with spectral weights and quadrature.
- `lumos.fields.IncoherentBundle`: mutually incoherent modes, off-axis point sources, natural light.
- `lumos.fields.RadianceField`: spectral radiance maps with radiometric units and flux-conserving resampling.
- `lumos.fields.ModalField` and `lumos.fields.smatrix`: guided amplitudes and scattering matrices.
- `lumos.fields.DataProduct`: detected data with axes, mask, uncertainties; FITS, NPZ and HDF5 files.
- `lumos.plotting`: intensity, phase (cyclic colour map), radiance and data plots with physical axes.
- `lumos.propagation`: `propagate`, `focus` and `plan` between arbitrary grids with direct Rayleigh-Sommerfeld,
  Fraunhofer (matrix Fourier transform), Fresnel (transfer function, impulse response, one-step MFT) and
  band-limited angular spectrum methods; automatic, logged choice of the method (ADR-0005); vectorised over
  polarisation, wavelengths and modes; kernel cache; chunking under a memory budget; differentiable with JAX.
- Validation suite of the propagation methods against analytic solutions, POPPY and HCIPy (`[validation]`).
- `lumos.Context`: graph of elements with typed connections, pull execution with a per-element cache and
  downstream invalidation, `>>` shorthand, access by dotted paths, per-element random streams, observation
  band (ADR-0006).
- `lumos.core.group.ElementGroup`: composite elements with exposed ports, nested freely.
- Clock with timed actions (`@action(duration=...)`), `ctx.wait`, `ctx.parallel()` and time-dependent elements.
- Simulation and operation modes, `Driver` interface, `SimulatedDriver`, digital twin skeleton.
- Structured journal with action origins (Python, web, MCP) and a JSON-lines formatter.
- `.lumos` documents of kind `group` and `context`, `.npz` sidecar for large arrays, published JSON Schema.
- `ctx.to_python()`: equivalent Python script; `ctx.sweep` and `ctx.sample` (Monte-Carlo from priors).
- `lumos.elements.sources`: point sources, blackbody stars scaled by radius and distance or by magnitude,
  planetary systems on Keplerian orbits, Gaussian lasers and incoherent lamps; `lumos.physics` (Planck's law,
  photometric bands, Lambert phase function, orbits).
- `lumos.elements.apertures`: anti-aliased circular, polygonal and hexagonal segmented pupils with spiders;
  VLT, JWST and ELT presets.
- `lumos.elements.optics`: free space, thin lens and mirror, beam splitter, delay line, spectral filter,
  polariser, wave plate and Zernike phase plate; `GridSetting` for grids saved in `.lumos` files.
- `lumos.elements.telescopes`: telescopes, interferometric arrays with projected baselines (ground or sky
  frame), VLTI and LIFE presets.
- `lumos.elements.atmosphere`: multi-layer von Kármán turbulence in frozen flow with ideal adaptive optics
  correction, deformable mirrors; `lumos.physics.zernike` and `lumos.physics.turbulence`.
- Stochastic elements drawing from reproducible context streams (`Element.stochastic`,
  `lumos.core.element.generator`, ADR-0007); changing `ctx.seed` recomputes them.
- `lumos.elements.coronagraphs`: vortex, four-quadrant phase mask and Lyot coronagraphs with a multi-scale
  focal-plane computation (`lumos.physics.coronagraphy`).
- `lumos.elements.fibers`: step-index fibres with exact LP modes or a Gaussian mode, injection, propagation
  and exit field (`lumos.physics.fibers`).
- `lumos.elements.photonics`: waveguides, phase shifters, directional couplers, Y-junctions, crossings and
  MMI couplers from a slab mode solver (`lumos.physics.photonics`); Bracewell and kernel-nuller chips.
- `lumos.elements.detectors`: camera (QE curve, photon, dark and read noises, saturation, binning, timed
  exposures), photodiode with shot noise, power meter.

### Fixed

- The JAX import hook is no longer consumed by `importlib.util.find_spec("jax")`, so `Values` is registered as
  a pytree when JAX is first imported through the backend.
- GPU probes log the warnings of GPU libraries instead of failing on them (CuPy without `CUDA_PATH` hid the
  GPU).
- Signals held by CuPy accept NumPy masks and factors (`lumos.backend.operand`).
- JAX no longer reserves 75 % of the GPU memory when LUMOS loads it (ADR-0008).
- Turbulent phase screens restore their low frequencies (structure function within a few % of von Kármán).
