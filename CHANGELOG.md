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
- `lumos.plotting.style`: publication figure style, colour maps by role, unit labels and journal widths.
- `lumos.plotting`: PSF (λ/D units), radial profile, spectra, transmission maps, chip outputs and diagrams of
  the graph of a context (SVG, PNG); `lumos.processing.radial_profile`.
- `lumos.results`: figures saved as `.png` and `.plt` (PltEdit) with their data in `.npz` files; run archives
  (`ctx.run(..., archive=...)`, `vfst.archive`) next to the script or notebook with the context, complete
  metadata, journal, figures and outputs; `lumos.load_archive` (ADR-0010).
- Example notebooks `01_first_simulation` to `04_interferometry` and shared presets in `examples/presets/`.
- `lumos.scenes`: scenes in celestial, terrestrial and laboratory frames (`Scene`, a registered group type
  saved with `"group_type"` in `.lumos` files); parametric diffuse components (uniform disk, Gaussian, Sérsic,
  inclined ring, background, zodiacal light); mapped components from FITS images and cubes with a WCS, arrays
  and calibrated images; procedural components (random textures, Siemens star, checkerboard, bar target).
- `lumos.physics.radiometry`: étendue, solid angles, irradiance, exitance, photon conversions, spectra.
- Imaging of extended scenes by convolution, field-dependent PSFs or decomposition into point sources, chosen
  automatically by probing the shift invariance of the system and logged (`image_scene`, `SceneToSources`,
  ADR-0011); notebook `05_extended_scenes` and the `debris_disk` preset.
- `lumos.catalogs` (ADR-0012): HTTPS layer with verified TLS, timeouts, user cache and offline mode;
  typed records with provenance per field; SIMBAD, Gaia DR3, NASA Exoplanet Archive, exoplanet.eu, VizieR
  photometry, CALSPEC, Pickles, SVO filters, JPL Horizons, solar spectrum, ESO, MAST, IRSA, SkyView and
  hips2fits providers; SED fusion and blackbody completion; `resolve`, `Star.from_catalog`,
  `Planet.from_catalog`, `Scene.from_catalog`, `MappedComponent.from_survey`; notebook
  `06_targets_from_catalogs` running offline on recorded answers.
- `lumos.symbolic`: symbolic fields with parameters and LaTeX rendering; evaluation on grids (cached
  `lambdify`, anti-aliasing); symbolic transfers of lenses, mirrors, delay lines, splitters, filters and
  apertures; Gaussian, HG and LG beams through ABCD systems and tabulated circ/rect transforms with logged
  numerical fallback; reconstructions from grids (Zernike, HG, LG, LP and Fourier projections, Gaussian and
  Airy fits, sparse regression, optional PySR); `SymbolicSource` and automatic, logged conversion to numbers in
  contexts; safe JSON form in `.lumos` files and archives; notebook `07_symbolic_optics` and the
  `symbolic_focus` preset.
- `lumos.processing`: dark subtraction, flat field, stacking (mean, median, sigma clipping), cropping, rebinning
  and flux-conserving resampling with propagated uncertainties; aperture photometry with exact pixel overlaps,
  CCD equation, contrast curves (Mawet et al. 2014), radial profiles and SNR maps; null depth, kernel observables
  and transmission maps; matched-filter and energy statistics, ROC curves, contrast limits and detection
  capability maps; phase fitting (gradient, scan, genetic), slanted-edge MTF and PSF measurements; processing
  elements with `DataProduct` ports and script blocks (ADR-0013); notebooks `08_data_pipeline` and
  `09_detection_capability`.
- Operation mode (ADR-0014): drivers check software limits and confirmations for every command, support dry
  runs, reconnect after lost connections, stop from any thread and reach a safe state (`ctx.stop_all()`,
  `ctx.operation_guard()`); `lumos.hardware` with TCP, serial, VISA and in-process transports, the LumosLabs
  ASCII protocol, emulators of nine devices with fault injection (`lumos emulators start`), their drivers,
  discovery and association with elements, driver plugins (`lumos.drivers` entry points); lab elements
  (`lumos.elements.lab`); digital twin with measured, simulated and residual outputs and emulators fed by the
  twin; `ctx.link`; recorded sequences (`ctx.record()`, `ctx.replay()`); guides "Operating a bench" and
  "Writing a driver"; notebook `10_bench_operation`.
- `PowerMeter(channel=k)` reads one guided channel.
- `lumos.elements.sources.Planet`: one planet of a star on a Keplerian orbit.
- Source lists can be archived and read back.
- `lumos.service` (ADR-0015): the service layer shared by the web backend and the MCP server — sessions with
  an origin, contexts and presets, editing with quantities written as text (`"0.5 mm"`), validation with
  hints, runs in the background with progress and quotas, port summaries with figures, catalogues, archives
  and the gated operation of equipment; errors say how to fix the request; groups created in one call.
- `lumos.mcp` and `lumos mcp` (ADR-0016): MCP server over stdio for AI assistants, with tools to discover
  element types and documentation, build and edit contexts, validate them, run them with progress and read
  their results, resolve catalogue objects, build processing pipelines, evaluate detection limits and read
  archives; the context defaults to the last one used. Resources (`lumos://docs`, `lumos://elements`,
  `lumos://presets`, `lumos://context/current`, journals, `lumos://archives`) and prompts for four use cases
  (`design_instrument`, `estimate_detection`, `calibrate_bench`, `diagnose_simulation`).
- Operation through MCP (`lumos mcp --enable-operation`): tools `execute_action`, `list_devices`,
  `device_status`, `stop_all`, and `discover_devices` and `associate_devices` once enabled; every action on
  real equipment is confirmed by the user through an MCP elicitation asked by the server; limits and the
  emergency stop are enforced by the server; actions are journaled with the origin `mcp`.
- Responses for language models: `inspect_port` and `get_result` return a short summary with units and the
  figure as an image; runs answer with one line; summaries of incoherent bundles (modes, power, grid).
- The service always asks for the unit of a physical quantity (a bare number is refused); the values it
  shows, arrays (`"[1, 2] nm"`) and grids included, are accepted back by `set_param`.
- `lumos mcp --transport http`: the MCP server over streamable HTTP (`lumos.mcp.create_http_app` gives the
  ASGI application for the web instance); local only unless `--allow-remote`, never with the operation.
- Guide "AI assistant": setting up Claude Code, Claude Desktop and other MCP clients, example conversations,
  reference of the tools, resources and prompts, safety; the server is tested against the MCP specification
  with a raw JSON-RPC client.
- Evaluation of the MCP server (`lumos.mcp.evaluation`, `tools/mcp_eval.py`, ADR-0017): scenarios UC1 (nuller
  transmission), UC8 (propagation methods) and UC9 (coronagraph contrast) with analytic expected values,
  scripted reference solutions run by the tests, and runs on demand with Claude Code
  (`tools/mcp_eval.py --claude-code`, no API key).
- Tool `radial_profile` (azimuthal mean of the intensity of a field) and the 1/e² beam radius in the summaries
  of fields.
- The journal of a context records each run (`kind: "run"`: outputs and duration).
- `lumos.web` and `lumos web` (ADR-0018): HTTP backend of the web interface, a FastAPI adapter of the service
  layer — JSON API under `/api` (OpenAPI page at `/api/docs`), one service session per browser behind an
  HttpOnly cookie, runs and journals pushed over the WebSocket `/api/ws`, `localhost` by default with checks
  of the host name and of the origin of WebSockets; `Service.default_outputs`, `list_jobs`, `journal_since`.
- Authentication and authorisation of the web instance (ADR-0019): users file with Argon2id password hashes
  and SSH public keys (`lumos users add/remove/list`), login with a password or by signing a single-use
  challenge with `ssh-keygen -Y sign -n lumos` (`lumos login`, bearer tokens), roles viewer, simulator and
  operator on every route, sessions renewed at login and expiring, rate limits, audit journal, HTTPS required
  to log in or operate outside the local machine; routes of the operation (discovery, association, confirmed
  actions, emergency stop).

### Changed

- Sizes of the test targets (`SiemensStar`, `Checkerboard`, `BarTarget`) may be angles or lengths; plain
  numbers keep meaning pixels.
- VFSciTools is no longer a dependency: the PltEdit `.plt` writer and `vfst.archive` are shipped in
  `lumos._vendor`, so LUMOS installs from PyPI, on Read the Docs and in CI.
- Cached propagation kernels stay on the device: GPU backends no longer copy them from the host at every
  propagation.

### Fixed

- The JSON Schema of an element gives the default of a quantity setting (`Crop.x`...) as `{value, unit}`.
- `search_docs` no longer fails on a page matched by its title only.
- `explain_sampling` reports the propagation method forced by an element (it reported the planner's choice).
- `Crop` centred on the origin by default fails no more on images with axes in lengths or angles.
- Random streams of stochastic elements inside groups are keyed by their path `group.child`.

- The JAX import hook is no longer consumed by `importlib.util.find_spec("jax")`, so `Values` is registered as
  a pytree when JAX is first imported through the backend.
- GPU probes log the warnings of GPU libraries instead of failing on them (CuPy without `CUDA_PATH` hid the
  GPU).
- Signals held by CuPy accept NumPy masks and factors (`lumos.backend.operand`).
- JAX no longer reserves 75 % of the GPU memory when LUMOS loads it (ADR-0008).
- Turbulent phase screens restore their low frequencies (structure function within a few % of von Kármán).
- In single precision, the chirps of the Fraunhofer and one-step Fresnel propagations and the lens phase of
  `focus` are computed in double precision and rounded once (phases such as `k z` were lost in float32).
- The propagation benchmarks wait for CuPy results; the CuPy times first documented were too short.
