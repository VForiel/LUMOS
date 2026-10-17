# ADR-0015: Service layer

- **Status**: accepted
- **Date**: 2026-10-17
- **Plan step**: P13.1 (REQ-SVC-1, REQ-MCP-5); the plan calls it "ADR-0007", a number already used by
  "Stochastic elements"

## Context

The web interface (phase 14) and the MCP server (phase 13) must offer the same operations as the Python API,
with the same results (objective O4). Writing them twice would let them drift apart. Both also need
safeguards that the Python API leaves to the user: quotas, the gating of the operation of equipment, errors that
explain how to fix the request.

## Decision

- `lumos.service.Service` is the single implementation of the high-level operations: sessions (each with the
  origin of its actions, `python`, `web` or `mcp`), contexts (create, presets, `.lumos` import and export,
  Python export), editing (elements, connections, parameters, observation band, processing chains),
  verification (validation with hints, propagation plans), runs in a thread pool with progress, inspection of
  ports (compact summaries and PNG figures), catalogues, archives and operation.
- Inputs and outputs are Pydantic models (`lumos.service.models`); every quantity is written as text with its
  unit (`"0.5 mm"`), parsed by astropy (no evaluation of code).
- Errors are `ServiceError` with a message, a hint (how to fix it) and a code (`invalid`, `not_found`,
  `forbidden`, `quota`, `failed`).
- Safeguards in the service, whatever the adapter: quotas (contexts per session, runs at a time, run
  duration); actions on equipment refused unless the operation is enabled by the configuration and the call is
  confirmed (`confirmed=True`, set by the adapter after asking a human); the emergency stop is always allowed;
  archives are only read inside the archive directory.
- The web backend and the MCP server translate their protocol into these calls and nothing else.

## Alternatives considered

- Adapters calling the Python API directly: no parity by construction, duplicated safeguards.
- An HTTP API used by the MCP server: an extra hop and a running server even for local stdio use.
- asyncio throughout: the core is synchronous and CPU-bound; a thread pool keeps the event loops of the
  adapters free.

## Consequences

- Each new feature of LUMOS reaches the web and the assistant once added to the service.
- A context is not thread-safe: the service serialises the runs and edits of a session with a lock.
- Presets are shipped in the wheel (`lumos/presets`) so that installed servers have them.
