# ADR-0014: Operation mode, LumosLabs equipment and the digital twin

- **Status**: accepted
- **Date**: 2026-10-17
- **Plan step**: P12.1 to P12.10 (REQ-HW-1 to REQ-HW-8, REQ-EL-7)

## Context

In operation mode the actions of elements are sent to real equipment (ADR-0006, step P5.6 gave the `Driver`
interface and `SimulatedDriver`). Phase 12 must make this path safe and testable without hardware: bounds,
emergency stop, dry-run, confirmations, timeouts and recovery (REQ-HW-4), transports (serial, TCP, VISA), a
reference family of virtual devices, LumosLabs, whose emulators speak a documented protocol over a real
transport (REQ-HW-3), discovery, the digital twin with real and simulated outputs side by side (REQ-EL-7) and
recorded sequences.

## Decision

- **Package** `lumos.hardware`: `transport` (`Transport` with `TCPTransport`, `SerialTransport` on pyserial URLs,
  including `loop://` and `socket://`, `VisaTransport` on pyvisa, and an in-process `PipeTransport`),
  `protocol` (the LumosLabs ASCII protocol), `emulators` (one class per device and a threaded TCP server),
  `drivers` (one driver per device, plus the plugin registry), `discovery`, `twin` and `sequence`.
  pyserial and pyvisa stay optional (extra `[hardware]`); TCP and pipes need nothing.
- **Driver safety in the base class**, so that every driver gets it: software limits checked in `send` for every
  command, including commands sent directly to the driver; `dry_run` validates and logs without sending;
  actions listed as dangerous need a confirmation callback (refused when absent); timeouts and a bounded number
  of reconnections on transport errors; `stop()` usable from another thread, interrupting a wait in progress;
  `safe_state()` (laser off, shutter closed, motion stopped). `Context.stop_all()` stops every driver, and
  `Context.operation_guard()` brings every driver to its safe state on `KeyboardInterrupt` or any exception.
- **Protocol**: one ASCII line per command and per reply, `\n`-terminated; replies `OK [value [unit]]` or
  `ERR <code> <message>`; common commands `*IDN?`, `*RST`, `STATUS?`, `BUSY?`, `STOP`, `LIMITS? <name>`;
  quantities always carry their unit. Long operations (moves, exposures) return at once and are followed by
  `BUSY?` polling, which lets `STOP` interrupt them.
- **Emulators** run in-process (tests, notebooks) or as TCP servers (`lumos emulators start`). They model
  latencies, noise, limits, homing, drifts and **fault injection** (disconnection, delay, out-of-range value,
  stuck motion). `LL-CAM` and `LL-PM` measure what the twin sends them through `SIM:` commands of the protocol
  (`SIM:POWER`, `SIM:IMAGE`), then add their own noise and defects: the real and simulated outputs then differ in
  a controlled way.
- **Lab elements** (`lumos.elements.lab`): stages, piezo controller, shutter, filter wheel, laser source, phase
  controller; `ctx.link("stage.position", "gap.distance", offset=...)` lets an actuator drive a parameter of the
  optical model, so that the twin follows the bench.
- **Twin outputs**: in operation, detectors attached to a driver return the measured `DataProduct`; the result of
  `ctx.run` also holds the simulated outputs (`result.twin`) and the residuals (`result.residuals`).
- **Sequences**: `ctx.record()` records the actions as commands with their arguments; a sequence replays in any
  mode and its commands are identical in simulation and in operation.
- **Plugins**: drivers of other packages register through the `lumos.drivers` entry points.

## Alternatives considered

- Driving the emulators by direct method calls: simpler, but the drivers would not go through transports,
  errors and latencies (REQ-HW-3 requires the same code path as real equipment).
- asyncio drivers: the rest of LUMOS is synchronous; threads (a stop from another thread, a polling loop) are
  enough for the bench equipment considered.
- SCPI as the protocol: close in spirit, but its full grammar is much larger than needed; the LumosLabs protocol
  borrows `*IDN?`/`*RST` and keeps units explicit.

## Consequences

- Every new driver inherits the safety mechanisms; the emulators double as the reference implementation of the
  protocol.
- The notebook `10_bench_operation` and the tests run against the emulators without hardware.
- Vendor drivers (Thorlabs first) follow the same interface; the coverage of their SDKs is documented.
