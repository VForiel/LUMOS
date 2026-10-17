# ADR-0013: Script blocks

- **Status**: accepted
- **Date**: 2026-10-17
- **Plan step**: P11.5 (REQ-PIPE-6, NF-7)

## Context

Processing pipelines need steps that LUMOS does not provide: a user function placed in the graph with declared
ports (REQ-PIPE-6). The same contexts are saved in `.lumos` files, shared, loaded by the web instance and by the
MCP server; loading a file must never run code from it (NF-7, no arbitrary imports).

## Decision

- `lumos.processing.script.script(name=..., inputs=..., outputs=...)` turns a function into an `Element`
  subclass registered under `name`, like any element type. Inputs are passed positionally in the declared
  order; the function returns a mapping of outputs (or the signal of a single output).
- Keyword-only arguments with defaults become the parameters: quantities make `Param`s with units, `bool`,
  `int`, `float` and `str` make `Setting`s. They are saved, edited and swept like other parameters.
- A `.lumos` file stores the type identifier and the parameter values only. Loading needs the script to be
  defined beforehand, by importing the module that defines it or through the `lumos.elements` entry points of
  a plugin package; otherwise loading fails with `UnknownTypeError`.
- Defining again a script with the same name and the same function (module and qualified name), as when a
  notebook cell runs twice, replaces the registration; another function under the same name is refused.

## Alternatives considered

- Storing the source code in the `.lumos` file and executing it on load: arbitrary code execution from shared
  files, forbidden by NF-7.
- Storing an import path (`module:function`) and importing it on load: an arbitrary import chosen by the file,
  also forbidden.
- Per-instance ports on a single generic `ScriptBlock` class: every element type would no longer have fixed
  ports, which the JSON Schema, the web palette and the codegen rely on.

## Consequences

- Scripts are visible in the registry, the web palette and the JSON Schema once defined.
- Sharing a context that uses scripts requires sharing the code that defines them (a module or a plugin).
- The web interface does not offer to type code in the browser; scripts come from Python.
