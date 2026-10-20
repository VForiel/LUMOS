# ADR-0024: Procedures, declarative sequences of actions and measurements

- **Status**: accepted
- **Date**: 2026-10-20
- **Plan step**: P15.11 (also P15.12, P15.13)

## Context

A bench is used through *procedures*: close the shutter, take twenty frames, average them into a dark, open
the shutter, take ten frames and subtract the dark. Such a sequence of actions and measurements in time, whose
intermediate results are reused, is neither the graph (a computation), nor a study (sweep, Monte-Carlo,
loop), nor a recorded sequence of commands (ADR-0014, no measurement nor processing). The review of 19/10/2026
and its decisions of 20/10 (`docs/specs/pistes_evolution.md`, §3) ask for procedures built with blocks in the
web interface (Blockly), run by the service as jobs, identical in simulation and in operation, usable as the
body of a study, and never running user code on the public instance.

## Decision

1. **A procedure is data.** A JSON object `{"name", "description", "steps": [...]}`; each step is an object
   whose `do` names its kind. No code, no expression language, no import: a procedure can only call what the
   engine lists.

   | `do` | fields | effect |
   |---|---|---|
   | `action` | `element`, `action`, `args` (list), `kwargs` (object) | calls an action of an element (timed, journaled) |
   | `measure` | `outputs` (list of ports), `as`, `collect` | runs the outputs; stores the value of one port (or a mapping of several) in the named result `as`; with `collect`, appends it to a list |
   | `repeat` | `times`, `steps` | runs `steps` `times` times |
   | `for_each` | `values` (list, or `{start, stop, num, log}`), `as`, `steps` | runs `steps` with the named result `as` set to each value |
   | `process` | `function`, `args` (object), `as` | applies a listed processing function; an argument `"$name"` is a named result |
   | `set` | `target` (`element.field`, free-target syntax of ADR-0022), `value` | sets a parameter, setting or component, to a quantity or a named result |
   | `use` | `result`, `port` | feeds an input port with a named result for the following measurements (`null` result: stop feeding) |
   | `wait` | `duration` | advances the clock (simulation) or waits (operation) |
   | `if` | `left`, `op` (`<`, `<=`, `>`, `>=`, `==`, `!=`), `right`, `then`, `else` | compares two scalars (quantities or named results) |
   | `save` | `results`, `archive` | writes named results to an archive |
   | `show` | `results` | marks named results to display (results browser) |

   Values are quantities written as strings (`"2 mm"`), numbers, booleans, strings or `"$name"`. The
   processing functions are a fixed table (`mean`, `median`, `sum`, `stack`, `subtract`, `subtract_dark`,
   `divide`, `scale`, `flat_field`, `crop`, `rebin`, `aperture_photometry`, `total`, `maximum`...), each with
   its arguments; a plugin cannot extend it from a document.
2. **Validation before any run** (`Procedure.problems(ctx)`): every element, port, action, target, function and
   argument exists; every named result is defined before it is used; loops are bounded (at most
   `MAX_ITERATIONS` iterations in all, 10 000, before the run starts when the counts are literal); nesting
   depth at most 16. The validation of the context reports the problems of its procedures.
3. **Engine** `lumos.core.procedure`: `Procedure` (from and to JSON), `run_procedure(ctx, procedure, *,
   progress, cancelled, confirm)` returning a `ProcedureResult` (named results, values shown, journal of the
   steps with their times). The context runs the graph for `measure`, performs actions for `action`, advances its
   clock for `wait`; everything is journaled with the origin of the run. `ctx.procedures` holds the procedures of
   a context by name; the `.lumos` context document stores them under `"procedures"` (format 1.3), and a
   procedure alone is a document of kind `"procedure"`.
4. **Service**: `set_procedure`, `remove_procedure`, `list_procedures`, `run_procedure` (a job of kind
   `procedure`: progress, cancellation between steps, journal, `max_run_seconds` and the other quotas; its named
   results are the outputs of the job, read with `get_result`); the web API and the MCP server expose the same
   operations.
5. **Operation**: a procedure acting on equipment needs one human confirmation for the whole run, listing the
   devices it drives (`run_procedure(..., confirmed=True)`); without it the run is refused, as a single action
   is (ADR-0014, ADR-0016).
6. **Python**: a procedure exports to an equivalent Python script (P15.12); the *script mode* (a Python script
   instead of blocks) runs only on local and private instances, never on the public one (P15.12, P15.4).
7. **Studies** (P15.13): a procedure may be the body of a sweep or a Monte-Carlo run, its named results being the
   outputs of each point.

## Amendment (P15.12): script mode and export

`Procedure.to_python()` exports the steps; a procedure `{"name", "script"}` is the script mode, run by `exec` with
`ctx`, `results`, `u`, `np` and `call` only when the service has `scripts_enabled`: true by default on the local
profile, false on the public one, where the configuration refuses it unless `anonymous_role = "none"` (a
private instance whose visitors all log in). Loading a document never runs a script. The web interface edits
the steps with Blockly (13.3, Apache-2.0), loaded only in the Procedures tab; the blocks are converted to and
from the JSON of the procedure by pure functions (`web/src/lib/procedures.ts`).

## Alternatives considered

- **A Python subset or an expression language in the documents**: flexible, but it is code; a document could
  then compute anything, which the public instance must refuse, and its sandboxing is a known source of
  vulnerabilities.
- **Procedures as graphs**: the graph describes a computation without time; a procedure is ordered in time
  (actions, waits) with state (named results), which a list of instructions shows directly.
- **Reusing recorded sequences** (ADR-0014): they replay commands only, without measurement, processing nor
  control flow.

## Consequences

- Procedures run identically in Python, in the web interface and through the MCP server, and are saved with
  their context.
- The table of processing functions is the vocabulary of the procedures: adding a function is a change of
  LUMOS (with its tests), not of a document.
- The Blockly editor (P15.12) only edits this JSON, which keeps the blocks and the engine independent.
