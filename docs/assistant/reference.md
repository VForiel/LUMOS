# Reference of the MCP server

The tools, resources and prompts offered by `lumos mcp`. Their descriptions, as the assistant sees them, are
also listed by any MCP client (`/mcp` in Claude Code).

## Conventions

- **Quantities** are written as text with their unit: `"0.5 mm"`, `"1.6 um"`, `"2 mW"`, `"10 mas"`; arrays as
  `"[1, 2, 3] nm"`. A number alone is only accepted for dimensionless values.
- **Ports** are written `"element.port"` (`"laser.out"`, `"meter.in"`); children of groups `"group.child"`.
- **Current context**: the context in the foreground of the session: the last one created, loaded, imported
  or named, or the one open in the web interface when the assistant works in its session (connector link,
  assistant panel). Tools that take `context` use it when the argument is omitted.
- **Errors** give the cause, a `hint:` saying how to fix the request and a `code:` (`invalid`, `not_found`,
  `forbidden`, `quota`, `failed`).
- **Results** are a short text with units, a figure for signals, and structured data (JSON) for programs.

## Tools

### Discovery

| Tool | Purpose |
|------|---------|
| `list_element_types` | Element types with their ports, filtered by words, a page at a time. |
| `get_element_schema` | Parameters of a type with their units, bounds and defaults; its ports. |
| `search_docs` | Excerpts of the documentation matching a query. |

### Contexts and construction

| Tool | Purpose |
|------|---------|
| `list_presets` | Ready-made instruments and benches. |
| `load_preset` | Load a preset into a new context. |
| `create_context` | Create an empty context, in simulation or operation mode. |
| `list_contexts` | Contexts of the conversation. |
| `describe_context` | Elements with their values and units, connections, mode, observation band. |
| `close_context` | Forget a context. |
| `get_view_link` | Link opening a context in the web interface, to give to the user (servers of a web instance). |
| `add_element` | Add an element with parameter values. |
| `remove_element` | Remove an element and its connections. |
| `connect` | Connect an output to an input. |
| `disconnect` | Remove a connection. |
| `set_param` | Set a parameter or a setting. |
| `set_observation` | Set the observation band and its number of wavelengths. |
| `create_group` | Add a group of elements behaving as one element. |
| `build_pipeline` | Chain processing elements after an output. |
| `import_lumos` | Create a context from a `.lumos` document. |
| `export_lumos` | The `.lumos` document of a context. |
| `export_python` | The equivalent Python script. |

### Verification

| Tool | Purpose |
|------|---------|
| `validate_context` | Problems that would make a run fail, with fixes. |
| `explain_sampling` | Method of a propagation (forced or chosen by the planner), reason, grids. |
| `plan_propagation` | Same as `explain_sampling`. |

### Execution and results

| Tool | Purpose |
|------|---------|
| `run` | Compute outputs, with progress notifications; the main outputs of a preset by default. |
| `get_job` | State of a run. |
| `wait_job` | Wait for the end of a run. |
| `get_result` | Summary of a computed output, optionally with its figure. |
| `inspect_port` | Compute an output and return its summary and figure. |
| `radial_profile` | Azimuthal mean of the intensity of a field, ring by ring (contrast curves). |
| `get_journal` | Last entries of the journal of a context. |
| `list_archives` | Archived runs. |
| `read_archive_metadata` | Metadata of an archived run. |

### Studies

| Tool | Purpose |
|------|---------|
| `set_seed` | Seed of every random process of a context (noise, turbulence, Monte-Carlo draws). |
| `set_free_parameter` | Free a parameter with a prior (uniform, log-uniform, normal), or fix it again. |
| `run_sweep` | Run the context for listed values or ranges of some parameters (grid or zip). |
| `run_monte_carlo` | Draw the free parameters from their priors `n` times, each draw with its own noise. |
| `get_study` | Points of a sweep, Monte-Carlo run or loop as a CSV table, with the structured data. |
| `add_feedback` | Close a loop: an output (`ctrl.commands`) drives a field (`dm.commands`) at the next step. |
| `remove_feedback` | Open a loop again. |
| `run_loop` | Run the context as a closed loop of `steps` iterations of `period`. |

### Procedures

| Tool | Purpose |
|------|---------|
| `list_procedures` | Procedures of a context, with their problems and the equipment they act on. |
| `set_procedure` | Add or replace a procedure (steps: action, measure, repeat, for_each, process, set, use, wait, if, save, show). |
| `remove_procedure` | Remove a procedure. |
| `run_procedure` | Run a procedure; its named results are the outputs of the job (`get_result`). On equipment, only with `confirmed=true` after the user confirmed. |

### Catalogues and processing

| Tool | Purpose |
|------|---------|
| `resolve_object` | Properties of an astronomical object, with units and provenance. |
| `evaluate_detection` | 5σ contrast curve of a residual image. |

### Operation

| Tool | Purpose |
|------|---------|
| `execute_action` | Execute an action of an element; on real equipment, after your confirmation. |
| `list_devices` | Devices attached to the elements of a context. |
| `device_status` | State of the device of an element. |
| `stop_all` | Emergency stop of every device of a context. |
| `discover_devices` | Find devices on TCP, serial or VISA (only with `--enable-operation`). |
| `associate_devices` | Attach the discovered devices to elements (only with `--enable-operation`). |

## Resources

| URI | Content |
|-----|---------|
| `lumos://docs` | Index of the documentation pages. |
| `lumos://docs/{+path}` | A documentation page (Markdown). |
| `lumos://elements` | Every element type. |
| `lumos://elements/{type_id}` | JSON Schema of an element type. |
| `lumos://presets` | Preset contexts. |
| `lumos://presets/{name}` | `.lumos` document of a preset. |
| `lumos://context/current` | The current context. |
| `lumos://context/current/journal` | Journal of the current context. |
| `lumos://contexts/{id}` | A context by identifier. |
| `lumos://contexts/{id}/journal` | Journal of a context. |
| `lumos://archives` | Archived runs. |
| `lumos://archives/{+path}` | Metadata of an archived run. |

## Prompts

| Prompt | Arguments | Workflow |
|--------|-----------|----------|
| `design_instrument` | `goal` | Design an instrument or a bench for a goal, check it, simulate it. |
| `estimate_detection` | `target`, `instrument` | Decide whether an instrument can detect a target. |
| `calibrate_bench` | `bench` | Calibrate a bench step by step, with confirmed actions on equipment. |
| `diagnose_simulation` | `problem`, `context` | Find why a simulation is wrong, slow or suspicious. |

## Evaluation

The scenarios of `lumos.mcp.evaluation` check that an assistant reaches verifiable results through the server:
the transmission map of a Bracewell nuller (UC1), a Gaussian beam propagated with two methods (UC8) and the
raw contrast of a vortex coronagraph (UC9), each compared with its analytic solution. `tools/mcp_eval.py`
solves them with scripted tool calls, as the tests do; `tools/mcp_eval.py --claude-code` lets Claude Code solve
them with your subscription (no API key) and checks its answers.

## Safety

- **Operation disabled by default.** Without `--enable-operation`, the assistant cannot discover or drive
  equipment; actions on elements in operation mode are refused. Actions on simulated elements are allowed.
- **Human confirmation.** With `--enable-operation`, every action on real equipment is confirmed by you: the
  server itself asks the question through the client (MCP elicitation), so the assistant cannot answer it. A
  client that cannot ask you cannot act on equipment.
- **Limits enforced by the server.** Software limits, dry runs and confirmations of the drivers apply whatever
  the request ([Operating a bench](../hardware/operation.md)); `stop_all` is always available and never asks.
- **Traceability.** Every run and action is recorded in the journal of its context with the origin `mcp`, and
  archives record it too.
- **Quotas.** Contexts per session, runs at a time and run duration are limited by the service.
- **Network.** The HTTP transport listens on your machine only, unless `--allow-remote`; remote clients can
  never drive equipment.
