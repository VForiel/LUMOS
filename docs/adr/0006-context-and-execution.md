# ADR-0006: Context, graph and execution model

- **Status**: accepted
- **Date**: 2026-10-16
- **Plan step**: P5.1–P5.10

## Context

Phase 5 turns isolated elements into systems (REQ-CTX-1 to 10): a graph of elements connected by typed
ports, executed on demand with caching, a clock driven by timed actions, two operating modes with a digital
twin, a structured journal, files and generated scripts. These choices shape every later phase (web editor,
MCP server, hardware drivers), so they are recorded here.

## Decision

- **Graph.** A `Graph` holds elements under unique identifiers (explicit, from the name, or class name and
  counter) and connections between `PortRef`s. Connections are type-checked, single inputs take one
  connection, and cycles are refused at connection time.
- **Pull execution.** `run(outputs)` computes the required sub-graph in a stable topological order. Outputs
  are cached **per element** (all outputs of an element come from one transfer call). Elements notify
  listeners when a field changes; the graph then drops the cache of the element and of everything
  downstream. Connections and removals invalidate the same way.
- **Containers.** An element belongs to at most one container, a `Context` or an `ElementGroup`. Groups are
  elements with **instance-level ports** (`instance_ports()`), feeding their exposed inputs to the internal
  graph through `Graph.run(..., feeds=...)`; child changes propagate up as field changes of the group.
- **Clock and actions.** Time advances only through actions and waits, never with the wall clock in
  simulation, which keeps simulations reproducible. `@action(duration=...)` computes the duration before the
  action runs; `ctx.parallel()` blocks advance by the longest duration. Time-dependent elements receive
  `values.time` and are invalidated when the clock moves.
- **Modes.** The context executes every action: in simulation the method body updates the virtual state; in
  operation a `Command` goes to the element's `Driver`, the clock advances by the measured duration, and the
  twin either keeps the simulated state next to the measured one or the measured state is written back.
  Overrides keep chosen elements simulated.
- **Journal.** Entries are dataclasses kept in `ctx.journal` and attached to `logging` records
  (`record.lumos`); records of other LUMOS loggers are captured during runs. The origin (`python`, `web`,
  `mcp`) is a context-managed stack set by the service layer.
- **Files.** Context documents follow §10 of the specifications; arrays above 64 kB go to a `.npz` sidecar
  next to the file, referenced by name only (no paths). A JSON Schema of the envelope is shipped with the
  package and published with the documentation.

## Alternatives considered

- Caching per port instead of per element: no gain, since a transfer function computes all its outputs at
  once.
- Push execution (recompute on every change): wasteful for interactive editing, where parameters change many
  times before a result is needed.
- Wall-clock time in simulation: makes runs depend on the machine and breaks reproducibility.
- Storing large arrays as base64 only: JSON files of hundreds of megabytes for pupil maps or phase screens.

## Consequences

- The web interface and the MCP server drive the same `Context` API and only set the origin.
- Hardware drivers (phase 12) implement `Driver._send` and return `Reply` objects; nothing else changes in
  the context.
- Compiled pipelines bypass the graph and call pure transfer functions directly (ADR-0002); the graph is an
  orchestration layer, not a compiled one.
- Plan numbering: the ADR planned as ADR-0006 for extended-scene imaging becomes ADR-0007, and the later ones
  shift accordingly.
