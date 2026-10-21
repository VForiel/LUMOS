# ADR-0019: Security model of the public instance

- **Status**: accepted
- **Date**: 2026-10-17
- **Plan step**: P14.2, P14.3 (REQ-WEB-9, REQ-WEB-10, REQ-MCP-4, NF-7); the plan calls it "ADR-0008", a number
  already used by "GPU memory sharing"

## Context

LUMOS will run as a public instance on a VPS (UC5: students and the general public), and on the machines of
its users, where it may drive optical equipment. A public instance must resist anonymous abuse (computations
that exhaust the server, guessing passwords) and must never let anyone drive equipment unless an administrator
decided it and the person is authenticated. The local instance must not be reachable from other machines or
from other web sites open in the same browser.

## Decision

- **Two profiles.** `local` listens on `127.0.0.1` only (refused otherwise); its user is an operator, every
  action on equipment still needs a confirmation in the interface. `public` serves anonymous visitors as
  simulators (configurable: viewers, or nobody without login) with quotas; the operation is disabled.
- **Enabling the operation off the local machine** (configuration `operation_enabled`) is refused at start-up
  unless TLS is set up (certificate, or a reverse proxy terminating TLS with `trust_proxy`), at least one user
  is an operator and anonymous visitors are not operators. Logging in and operating outside the local machine
  are refused over plain HTTP.
- **Roles**: `viewer` (read, load presets, export), `simulator` (+ edit, run, inspect, simulated actions),
  `operator` (+ discover and associate devices, actions on equipment). Each route declares the role it needs;
  the role comes from the server-side session only, never from the request. The emergency stop is allowed to
  every role.
- **Users** are listed in a TOML file managed by `lumos users`: role, Argon2id hash of the password
  (`argon2-cffi`, never the password), public SSH keys.
- **SSH keys** prove their owner by signing a server challenge with `ssh-keygen -Y sign -n lumos` (SSHSIG,
  namespace `lumos`), done by `lumos login` or pasted in the interface. Challenges are random, bound to the
  user name, valid for 2 minutes and consumed by the first attempt, successful or not (no replay). Ed25519,
  ECDSA and RSA (SHA-2) signatures are verified with `cryptography`; SHA-1 RSA signatures are refused.
- **Sessions**: opaque random tokens (cookie `HttpOnly`, `SameSite=Strict`, `Secure` over HTTPS, or
  `Authorization: Bearer` for scripts), renewed at login (no fixation), closed after an idle time and, once
  authenticated, after a maximum lifetime. Wrong credentials give one message whatever was wrong; unknown users
  cost the same time as known ones and get challenges too (no enumeration).
- **Rate limits** per client address (requests) and per address and user name (login attempts), with
  `Retry-After`; behind a trusted proxy the address is the first `X-Forwarded-For` entry.
- **Audit journal** (JSON lines, logger `lumos.audit`): logins, logouts, challenges, refusals, rate limits,
  device discovery, actions on equipment, emergency stops; never passwords, signatures or tokens.
- **Quotas** (P14.3) are enforced by the service, so the MCP server shares them: contexts per session, runs at a
  time, global concurrent runs, run duration, size of grids and estimated memory.

## Alternatives considered

- OAuth or an external identity provider: one more service to run for a small group of operators; can be put in
  front of the instance by the reverse proxy if needed.
- JWT tokens: no server-side revocation at logout; sessions are cheap to keep in memory for one instance.
- Verifying signatures with `ssh-keygen -Y verify`: needs OpenSSH on the server and an allowed-signers file per
  attempt; `cryptography` is already present with TLS.
- Allowing the operation with passwords over HTTP on a private network: a single misconfiguration would expose
  equipment; HTTPS is mandatory instead.

## Consequences

- An administrator enables the operation with three explicit settings (operation, TLS, operators); a mistake
  stops the server at start-up with a message saying what is missing.
- Sessions live in the memory of one process: restarting the server logs everybody out, and running several
  processes needs sticky sessions (not needed for the expected load).
- The MCP server over HTTP mounted in the web instance (P14.9) reuses these sessions and roles.

## Amendment (2026-10-20): durable connector links

A connector link (`/mcp/<secret>`, P14.15) is pasted once into Claude.ai or ChatGPT and used for months, but it
died with the browser session that created it (idle timeout, restart of the server at every update): the
connector then failed with "unknown or revoked connector link". A link now has a record — SHA-256 digest of
the secret, user, role, expiry (`connector_link_days`, 90 by default) — kept in memory and, with
`connector_links_file`, in a file (mode 600, digests only) that survives restarts. Used while its session is
gone, a valid link opens a new session for the same user and role. Creating a new link or revoking it deletes
the record. The secret remains the only credential, as before; its exposure lasts longer, which the
revocation in the assistant panel bounds.

## Amendment (2026-10-21): audit of the code zones and of untrusted documents (P15.4)

Procedures written as Python scripts (ADR-0024) and components with code (ADR-0025) are the only places where
a user's code runs. The audit traced every route to them and pinned each with a test
(`tests/integration/test_public_instance_security.py`):

- configuration: a public instance open to anonymous visitors refuses to start with `scripts_enabled`;
- service: running a script procedure, a study on one and defining code components check `scripts_enabled`;
- web API and MCP server: thin adapters of the service, with no other route to code (the MCP server has no
  code tool); a script procedure can be saved and read but never run;
- documents: opening a `.lumos` file never runs its scripts; a component document holds data only (bodies
  are parsed by astropy, procedure functions come from a fixed table); element types are looked up in the
  registry, and the only modules a document can trigger are those of a fixed list of built-in types;
- element packages of the `lumos.elements` entry point (ADR-0026) are code installed by whoever runs the
  instance, loaded on every profile.

Readers of untrusted JSON (components, procedures, `.lumos` documents) are fuzzed with Hypothesis
(`tests/unit/test_untrusted_documents.py`): they raise only their documented errors and never call `exec`,
`eval`, a command or a process. Fuzzing found two crashes, both fixed: a step without element broke the checks
of a procedure (malformed steps are now reported as problems), and a field of the wrong type escaped
`from_document` as a `TypeError` (now a `LumosFormatError`). Procedures run as jobs: they share the quotas of
runs (time of a run, runs at a time per session and per instance) and their loops are bounded (10 000
iterations).

The modules deciding access or reading configuration and catalogue (`web/auth`, `web/sessions`,
`web/config`, `catalogue`, `io/schema`, `mcp/server`) are type-checked in strict mode; the readers of JSON
documents validate types at run time instead, where strict mode only adds casts, and rely on the fuzzing.
