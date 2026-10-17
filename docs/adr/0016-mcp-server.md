# ADR-0016: MCP server

- **Status**: accepted
- **Date**: 2026-10-17
- **Plan step**: P13.2 (REQ-MCP-1, REQ-MCP-2); completed by P13.3 to P13.6

## Context

AI assistants (Claude Code, Claude Desktop, other MCP clients) must build, run and analyse simulations
through the Model Context Protocol, with the same results as the Python API (objective O4) and the safeguards
of the service layer (ADR-0015). The official Python SDK changed its server API in version 2: `FastMCP` became
`MCPServer`, and the protocol revision 2026-07-28 replaced the requests sent by the server in the middle of a
call (elicitation, sampling) by results asking the client for input (`InputRequiredResult`).

## Decision

- `lumos.mcp.create_server(service)` builds an `MCPServer` of the SDK 2 (`mcp>=2.3,<3`, extra `mcp`).
  `lumos mcp` serves it over stdio; the logs go to stderr, since stdout carries the protocol.
- Each tool translates its arguments into one call of `lumos.service.Service` and returns the Pydantic model of
  the service (structured content and JSON text). The tools have the names of REQ-MCP-2; their docstrings are
  their descriptions, written for a language model (units as text, ports as `"element.port"`).
- A server holds one session of the service, of origin `mcp`, so its runs and actions are journaled as such.
- **Current context**: the last context created, loaded, imported or named by a tool. Tools take an optional
  `context` and use the current one when it is omitted, which keeps the conversations short.
- **Errors**: a `ServiceError` becomes a `ToolError` whose text holds the message, the hint and the code, so
  that the assistant can fix its request; other exceptions are reported with their type and message rather
  than the SDK's generic "Error executing tool".
- **Runs**: `run` starts a job of the service, reports its progress with MCP progress notifications while it
  waits (up to `wait_seconds`) and returns the job; longer runs continue in the background (`get_job`,
  `wait_job`). Without `outputs`, the main outputs of the context (those of its preset) are computed.
- **Resources** (P13.3) use the scheme `lumos://`: `docs` and `docs/{+path}`, `elements` and
  `elements/{type_id}`, `presets` and `presets/{name}`, `context/current` (and its `journal`),
  `contexts/{id}` and `contexts/{id}/journal`, `archives` and `archives/{+path}` (relative to the archive
  directory; the SDK rejects `..` and absolute paths). **Prompts** guide four use cases: `design_instrument`,
  `estimate_detection`, `calibrate_bench`, `diagnose_simulation`.
- **Responses** (P13.5): signals are answered with a short text summary (kind, shape, range and mean with
  their unit, grid, wavelength) and a PNG image, runs with one line naming the job and the next tool; the
  Pydantic model stays the structured content (`Annotated[CallToolResult, Model]`), without the figure bytes.
  The values shown are accepted back as they are, and a quantity without unit is refused with an example.
- Synchronous tools run in worker threads of the SDK; the service serialises the work of a session.
- Groups are created in one call (`create_group`: elements, internal connections, exposed ports), added to the
  service as `Service.create_group`.
- **Operation** (P13.4): `execute_action`, `list_devices`, `device_status` and `stop_all` are always offered;
  `discover_devices` and `associate_devices` only when the server runs with `--enable-operation`. Actions on
  simulated elements run directly. An action on real equipment is confirmed by the user: a resolver of the SDK
  (`Resolve` returning `Elicit`) asks the question, which works with both protocol revisions (input-required
  results from 2026-07-28, elicitation requests before). The flag `confirmed` of the service is set from the
  answer and never appears in the input schema of the tool, so the model cannot set it. A client without the
  elicitation capability gets a tool error and cannot act on equipment. Limits and the emergency stop stay in
  the drivers (ADR-0014); `stop_all` never asks.

## Alternatives considered

- SDK 1 (`FastMCP`): superseded; its elicitation does not work with the 2026-07-28 revision.
- One tool per element type, or a generic `call(method, args)` tool: too many tools for the context of the
  model, or no schema to guide it.
- Requiring `context` in every call: longer requests and more mistakes from the assistant.
- A blocking `run` without progress: the client times out on long runs and the user sees nothing.

## Consequences

- New operations of the service reach the assistant by adding a short tool.
- The HTTP transport (P13.6) shares one server, so one session, between its clients until the web instance
  (P14) gives each user its own session.
- The SDK 2 is required: version 1 is not supported.
