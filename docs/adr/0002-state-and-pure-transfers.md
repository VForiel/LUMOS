# ADR-0002: Element state and pure transfer functions

- **Status**: accepted
- **Date**: 2026-10-16
- **Plan step**: P1.4

## Context

Elements carry configuration (focal length, velocity...) and a mutable state changed by actions (stage
position, shutter open...). They must also compute how they act on light. Requirements REQ-EL-8 and
REQ-EL-10 ask for pure numerical transfer functions, so that they can be JIT-compiled, vectorised (`vmap`),
differentiated (JAX) and run in parallel; REQ-EL-5 and REQ-EL-7 ask for actions that change the state, in
simulation as well as in operation.

## Decision

1. **Fields** are declared on the class: `Param` (configuration), `State` (changed by actions) and
   `Setting` (non-physical). Each assignment validates the value, stores the user value and its SI form,
   and calls `_on_field_changed` (cache invalidation).
2. **Transfer functions are `staticmethod`s** with the signature `transfer(values, inputs) -> outputs`.
   - `values` is an immutable `Values` snapshot of the SI values of every field (params, state, settings);
   - `inputs` and `outputs` map port names to signals.

   Being static, a transfer function cannot read or modify the element: everything it depends on is in its
   arguments. Declaring a non-static `transfer` raises `TypeError` when the class is created.
3. **State changes only through assignments** (and later actions), never inside a transfer function.
4. `Element.apply(**inputs)` runs the transfer on the current values and checks port names and signal
   types. The execution engine (phase 5) calls `transfer` directly with snapshots it caches.
5. Elements refuse unknown public attributes (typos raise `AttributeError` with a suggestion); private
   attributes (leading underscore) remain free for implementation details.

## Alternatives considered

- **Methods reading `self`**: simpler to write, but the purity would rest on discipline only, and JAX
  transformations would capture the mutable object.
- **Pure functions defined outside the class**: equivalent purity, but scattered code and no natural link
  between an element and its behaviour.

## Consequences

- `Values` must become a JAX pytree in phase 2 (P2.5), so that snapshots can be traced and differentiated.
- Elements needing pre-computed data (masks, mode profiles) must expose it through fields or through cached
  helpers computed from `values`, not through hidden attributes read by `transfer`.
