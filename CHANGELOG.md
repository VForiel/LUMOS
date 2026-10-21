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
- Deployment profiles of the web instance: `local` (localhost only, its user has every right) and `public`
  (anonymous simulators, operation disabled, quotas); TOML configuration files (`lumos web --config`) with
  command-line overrides; unsafe configurations refused at start-up. Compute quotas in the service, shared
  with the MCP server: `max_grid_pixels`, `max_memory_mb`, `max_concurrent_runs`, and the time quota applied
  to inspections.
- Web frontend (`web/`, ADR-0018): Vite, React, `@xyflow/react`, Tailwind CSS and i18next; palette, blocks,
  ports coloured by signal and typed connections generated from `/api/palette` (schemas, signal classes, units
  with their factors); English and French; light and dark themes; the last context is restored after a
  restart of the server. Unit tests with Vitest, browser tests with Playwright (`pytest -m e2e`).
- Editing in the web interface: forms generated from the schemas (units with conversion, bounds, defaults),
  undo and redo of every edit, renaming, groups made from a selection (expandable, ungroupable), copy and
  paste, reusable blocks (`.lumos` files of kind `group`), quick add by name or from a dropped wire, keyboard
  shortcuts, `.lumos` files opened and saved, equivalent Python script. Service: `undo`, `redo`,
  `update_ui`, `rename_element`, `group_elements`, `ungroup`, `copy_elements`, `paste_elements`,
  `export_element`, `import_element`, `set_context`; summaries give the inside of groups and the undo state.
- Runs in the web interface: progress, cancellation, archives; results with figures and statistics; any port
  inspected without a run; downloads as PNG, PltEdit, NPZ and JSON; live journal; problems leading to the
  element. Service: `port_file`, `result_file`, `archive_zip`, `cancel_job` (status `cancelled`).
- Presets and tour in the web interface: gallery of the presets (title, description, types of elements) on a
  first visit and from the toolbar; step-by-step guided tour overlaid on the interface (driver.js, English and
  French) from a preset to its results. `PresetInfo` gives `title` and `elements`.
- Operation panel of the web interface: devices and their state, discovery and association (operators),
  actions of the elements (simulated when the operation is disabled), confirmation dialog naming the device,
  emergency stop in the toolbar; login dialog (password, or SSH challenge with the `ssh-keygen` command) and
  role shown in the toolbar.
- MCP server of the web instance at `/mcp` (streamable HTTP, same authentication and same session as the
  interface; tools of the operation for operators only; actions journaled with the origin `mcp`) and assistant
  panel: Claude called from the browser with the API key of the user (kept in the tab), using the tools of the
  instance on the open context. `create_server(session=..., can_operate=...)`, `lumos.service.service.acting_as`.
- Deployment of the web instance: Docker image (`deploy/Dockerfile`, local profile with `--listen-all` on a
  port published locally, unprivileged user, health check `/api/health`), compose files for a single machine and
  for a public instance behind Caddy (TLS), example configuration with quotas, hosting guide; wheels ship the
  built interface.
- Tests of the web interface: HTTP API, security (access refused without authentication, elevation of role,
  replay of SSH challenges), browser tests in English and French (Playwright, run in CI), parity of the Python
  API, the web interface and the MCP server.
- Ergonomics of the web editor (P14.12): pointer mode (move the view or select a zone, keys H and V); badge
  and confirmation for unsaved changes (new context, file, preset, tour, closing the page); removal of a link
  alone; palette, inspector and bottom panel resized from their edge and hidden, sizes kept by the browser.
- Studies (P14.13, REQ-CTX-10): noise realizations of a context (`Context.realization`, independent and
  reproducible streams); `Context.sample` draws a new realization per draw (`noise="independent"`) and repeats
  the run with independent noise without free parameters; `Context.sweep(noise=...)`; progress of both. Service
  `set_context(seed=)`, `set_free`, `run_study`, `get_study`, `study_csv` (quota `max_study_points`), HTTP
  routes, MCP tools `set_seed`, `set_free_parameter`, `run_sweep`, `run_monte_carlo`, `get_study`; in the editor,
  the seed of the context, free parameters with priors, the study dialog and tab (curve, histogram, table,
  CSV); preset *Monte-Carlo tolerancing*.
- Feedback loops (P14.14, ADR-0020): `Context.feedback`, `remove_feedback`, `feedbacks` and `Context.loop`
  (time iterations, feedbacks applied between them); `.lumos` format 1.1 (1.0 still read) and its JSON schema;
  `WavefrontSensor` (ideal Shack-Hartmann) and `AOController` (integrator with the regularised pseudo-inverse of
  the interaction matrix of its mirror); service `add_feedback`, `remove_feedback`, `run_loop`, HTTP routes and
  MCP tools; in the editor, feedback curves, the feedback section of the inspector and loops in the study
  dialog; preset *Adaptive optics loop*.
- Assistant providers (P14.15): Claude, OpenAI, Google Gemini, OpenRouter (key or login with OAuth PKCE) and any
  OpenAI-compatible endpoint (tool loop of the chat completions API); connector links (`/mcp/<secret>`,
  `/api/assistant/link`) to use a Claude or ChatGPT subscription by adding LUMOS as a connector.
- Editor (P14.16, review of 19/10/2026): the label of a port is its handle (drag it, or click two ports, to
  connect) and it grows sideways in place when pointed at; an *Inspect* button in the header of the blocks;
  the categories of the palette start collapsed; selecting a block shows a hidden inspector.
- The interface follows the assistant (P14.18): every context has a `revision` and every session a focused
  context (`Service.focus`, `get_focus`, `/api/focus`), pushed on the WebSocket; the tools work by default on
  the context open in the interface and the interface opens the contexts the assistant creates; MCP tool
  `get_view_link` and route `/view/<secret>` opening a context of the session in another browser.
- Free targets (P14.17, ADR-0022): numeric settings and components of array parameters (`"field[i]"`,
  `"field[i, j]"`) can be free, swept and drawn (`ParamHost.free_target`, `free_names`, `get_target`,
  `set_target`, `lumos.core.param.parse_target`); `.lumos` format 1.2; in the editor, dice on numeric settings
  and on arrays (component chosen in the editor of the prior), intermediate outputs in the study dialog, and
  an explanation of the feedback loops in the inspector.
- `TransferMatrix` (P14.19): a simplified MMI given by the moduli and phases of its transfer matrix at a
  wavelength, achromatic or with phases that are optical paths, square or not, refused when it amplifies;
  `Element.problems()`, reported by the validation of the contexts; matrix editor in the inspector (cells,
  inputs and outputs added or removed, lossless, lossy or amplifying).
- Results browser (P14.20): `run_study` and `run_loop` take `record`, keeping the figure of every drawable
  output at every point (`StudyResult.recorded`, `Service.get_recording`, `/api/jobs/{job}/points/{k}/figure.png`);
  `lumos.processing.Recorder` taps a port and records it in every study (one point out of `every`); in the
  editor, a window listing the runs and studies of the session, with a slider and a play button over the
  points of a study.
- Storage of the sessions (P14.21): `session_storage`, `max_session_bytes`, `min_free_disk_bytes` and
  `max_jobs_kept` in `ServiceConfig` (public profile: per-session archives deleted with the session, 1 GB per
  session, 2 GB kept free, 20 runs in memory); `Service.storage` and `/api/storage`; job warnings; the
  results panel shows the data of the session.
- Buses (P15.6, ADR-0023): `lumos.fields.Bus` groups lanes separated in space (guides of a chip, arms of an
  interferometer), named and with optional telescope positions, packed in one array when they share their
  metadata (`Bus.guided`, `modal`, `field`, `select`, `concatenate`, `packing_problem`); ports declare buses
  (`Input(..., bus=True)`, `x-lumos-ports[].bus` in the schemas); an element whose input does not declare a bus
  acts on every lane of a bus (one random stream per lane); `SourceList.identifier` and the `sources` of
  `ModalField` and `IncoherentBundle`, checked before lanes interfere; `.lumos` format 1.3.
- `Demux` and `Mux` (P15.7, category *Buses*): split a bus into one-lane ports and gather lanes into a bus
  without mixing them, ports following `n_lanes` (published as `in{i}` / `out{i}` patterns in the schema;
  wires of removed ports dropped), optional inputs (`Input(..., optional=True)`; an unconnected `Mux` input is
  an empty lane, a validation warning); `Element.port_lanes` and `ElementSummary.lanes`; the editor draws the
  pins of bus ports gathered by a brace and bus wires thicker.
- Free-space telescope arrays (P15.8): `TelescopeArray` (type version 2) returns a bus of pupil-plane beams,
  one per telescope; `Fiber.injection_focal_length` injects a pupil-plane field through an ideal lens (real
  coupling, antenna lobe); the former model is `IdealInjectionArray`, and documents holding a
  `TelescopeArray` of type version 1 are read as such (`lumos.io.lumos_format.REPLACED_TYPES`); reading a
  document older than 1.3 warns about photonic components listing channels of the modes of a fibre; the
  Bracewell and kernel-nuller presets inject explicitly.
- `FizeauCombiner` (P15.9): recombines a bus of pupil-plane beams in one homothetic output pupil (Traub 1986),
  each lane at the projected position of its telescope (sub-pixel Fourier shift, power kept).
- Splitting a bus port (P15.10): `Service.split_port` and `POST /api/contexts/{id}/split` insert a `Mux` before
  a bus input or a `Demux` after a bus output, attached to the block (`ui.attached`, renamed and deleted with
  it); the inspector offers **Split** and the editor draws the split against its port; generic outputs
  (`Output(..., generic=True)`, those of `Demux` and `Mux`) may feed any optical input, checked at run time.
- Procedures (P15.11, ADR-0024): `lumos.core.procedure` (declarative steps action, measure, repeat, for_each,
  process, set, use, wait, if, save, show; a fixed table of processing functions; checks before running;
  progress, cancellation, confirmation of the actions on equipment), `Context.procedures` and
  `Context.run_procedure`, `Context.run(feeds=...)`; `.lumos` documents of kind `procedure` and the procedures
  of a context in its document (format 1.3); service, web API and MCP tools (`set_procedure`,
  `list_procedures`, `remove_procedure`, `run_procedure`), jobs of kind `procedure`.
- Procedures tab (P15.12): blocks (Blockly) next to the graph, quick buttons per step, problems, run and named
  results, export to Python (`Procedure.to_python`, `GET .../procedures/{name}/export.py`); script mode
  (`{"name", "script"}`) run only with `ServiceConfig.scripts_enabled` (local profile; refused on a public
  instance open to anonymous visitors); preset `dark_then_measure` and notebook 15.
- Studies on procedures (P15.13): `ctx.sweep(..., procedure=...)` and `ctx.sample(..., procedure=...)` run a
  procedure at every point on a copy of the context, the named results being the outputs, with `workers` points
  at a time; `for_each` steps with `"parallel": true`; `Service.run_study(procedure=...)`, the Study dialog and
  the MCP tools `run_sweep` / `run_monte_carlo` take a procedure.
- Declarative components (P15.14, ADR-0025): `lumos.components` (documents of kind `component`, templates
  `transfer_matrix`, `gain_phase`, `spectral_transmission` and `composite`, `register_component`, `create`,
  `check_component` running the checks of the component), `lumos component check FILE`, definitions embedded
  in the documents that use them (format 1.3), category *Components* of the palette.
- Component editor (P15.15): the button *New component* of *My blocks* opens a form (identifier, texts,
  license, parameters, template and behaviour, checks) whose problems are listed while it is typed, the
  document it makes, and, where scripts run, a code view defining element types in Python; a saved component
  is a type of the palette of the session (composite ones are placed from *My blocks*).
  `Service.check_component`, `save_component`, `list_components`, `get_component`, `add_component`,
  `define_code_component`, the routes `/api/components...` and the MCP tools `list_components`,
  `check_component`, `save_component`, `add_component`.
- Review of 20/10, continued (P15.18): the switch Graph / Procedures is in the top bar and changes the mode of
  the whole interface (the editor of procedures alone, without palette nor inspector); `ctx.to_python()`
  includes the procedures of the context, each one a function returning its named results
  (`Procedure.to_function`, `procedures=False` to leave them out), an option of the Python dialog, of
  `GET .../export.py?procedures=` and of the MCP tool `export_python`.
- Community catalogue (P15.16, ADR-0026): the `community/` folder (components, shared presets, `index.json`
  built by `tools/build_catalogue.py`), shipped in the wheel as `lumos/community`; `lumos.catalogue`
  (`build_index`, `check_catalogue`, `read_index`); the workflow *Community catalogue*, run on changes of
  `community/` only (the main CI ignores them); `Service.list_catalogue`, `use_catalogue_component`,
  `open_catalogue_preset`, the routes `/api/catalogue...`, the MCP tools `list_catalogue` and `use_catalogue`
  and the *Catalogue* tab of the palette; the service loads the element packages of the entry point
  `lumos.elements` when it starts; the guide page "Create and share a component". A component appears in the
  palette under its title.
- Performance (P15.3), results unchanged: the Lyot-plane transform of coronagraphs works on batches a few
  fields at a time, skips the FFTs of the zero padding and computes its matrix Fourier transforms as large
  matrix products (debris-disk preset 5.9 s → 3.2 s); the LP modes of fibres are found with a vectorised
  scan and kept per normalised frequency (identical fibres and repeated runs solve them once; kernel nuller
  over 50 wavelengths 2.4 s → 0.9 s for a first run, 0.11 s for the next ones); the automatic choice of backend no longer imports
  JAX when no JAX accelerator plugin is installed (about 2 s saved on the first run of every process);
  `FizeauCombiner` shifts its beams with threaded FFTs; benchmarks of every preset
  (`pytest -m benchmark tests/benchmarks`).
- Quality and security (P15.4, ADR-0019 amendment): every route to code (configuration, service, web API,
  MCP server, studies, opened documents) is pinned refused on a public instance; the readers of untrusted
  components, procedures and `.lumos` documents are fuzzed (they never execute anything); a procedure step
  without element or with fields of the wrong type is reported as a problem instead of crashing the checks;
  a malformed `.lumos` field gives a `LumosFormatError`; a users file whose entries are not tables is
  refused cleanly; strict type checking of the authentication, sessions, configuration, catalogue, schema
  and MCP modules; a coverage floor of 88 % in CI (90 % measured); blocks added in quick succession (or while
  the server is slow) no longer land on the same spot.
- Public instance (P15.5): `deploy/public/update.sh` pulls and rebuilds the image with the version of the
  code (`tools/release_version.py`, the scheme of the package), so the instance no longer shows 0.0.0.
- Top bar for every window (P15.19): no control cut off whatever the width; the commands without room go into a
  **More** menu.
- Review of 20/10 (P15.17): the archive button opens the archives of the session (switch, list, download,
  files, preview of the data with its figure, deletion; `Service.archive_files`, `archive_data`,
  `archive_file`, `delete_archive` and their routes); the bottom tab "Results" is named "Outputs"; in a narrow
  window one side column at a time (both closed at load) and the bottom tabs in a menu.

### Changed

- Guided light (P15.6, ADR-0023): a `ModalField` is the modes of one guide (`n_sources`, `n_modes`,
  `mode_power` replace `n_modes`, `n_channels`, `channel_power`); photonic components take and return buses
  of guided lanes and their `channels` designate lanes; `TelescopeArray` returns a packed bus with the
  positions of its telescopes; a `Fiber` on a bus of beams acts on every lane; photodiodes and power meters
  read a lane (all its modes) as one value and a bus as one value per lane.
- Review of the public API before its 1.x freeze (P15.1, ADR-0021): American spelling of the identifiers
  (`FlatField` and `KernelObservables` settings and the keyword of `flat_field` and `kernel_observables`
  renamed `normalize`, type version 2 reading the older documents; `subaperture_centers`); the element class
  `PSFMeasurementElement` is `PSFMeasurement`, like its type, and the result of `measure_psf` is `PSFResult`;
  every field of every element type has a description; `lumos.core` and `lumos.hardware` list their modules;
  the API reference documents `lumos.symbolic` and `lumos.hardware`; versions and stability page.
- Sizes of the test targets (`SiemensStar`, `Checkerboard`, `BarTarget`) may be angles or lengths; plain
  numbers keep meaning pixels.
- VFSciTools is no longer a dependency: the PltEdit `.plt` writer and `vfst.archive` are shipped in
  `lumos._vendor`, so LUMOS installs from PyPI, on Read the Docs and in CI.
- Cached propagation kernels stay on the device: GPU backends no longer copy them from the host at every
  propagation.

### Fixed

- Connector links (P14.15) outlive their browser session and the restarts of the server (90 days,
  `connector_link_days`; `connector_links_file` keeps their digests): a connector added to Claude.ai no
  longer breaks when the session expires.

- Security: the archives of a public instance were listed and downloadable by every visitor; they are now
  those of the session only (`session_storage`).
- Editor: adding or removing a block no longer moves the other blocks of a context laid out automatically
  (they piled up in one column); the handles of the ports no longer slide off the edge of the block when
  pointed at; the content of the result cards no longer spills out of them in a low bottom panel.
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
