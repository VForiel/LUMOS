# ADR-0007: Random streams of stochastic elements

- **Status**: accepted
- **Date**: 2026-10-16
- **Plan step**: P6.5

## Context

Turbulent phase screens (P6.5) and detector noise (P6.9) need random numbers inside transfer functions.
Transfers are pure static functions of their values and inputs (ADR-0002), so they cannot call
`ctx.rng(element)`. Results must still be reproducible from the seed of the context (REQ-CTX-6), streams must
be independent between elements, and a cached output must be recomputed when the seed changes.

## Decision

- An element class sets `stochastic = True` to declare that its transfer draws random numbers, like
  `time_dependent` and `observation_dependent`.
- `Element.values()` then adds `random_key = (seed, *spawn_key)`, the key of the context stream of the
  element (`(0,)` outside a context). It is a tuple of integers: hashable (usable as a cache key, e.g. for
  phase-screen spectra) and static for JAX pytrees.
- `lumos.core.element.generator(values)` builds the NumPy generator of that stream,
  `default_rng(SeedSequence(seed, spawn_key=...))`, identical to `ctx.rng(element)`.
- `Context.seed` is a property: setting it invalidates every stochastic element.
- Draws that must differ over time (e.g. successive exposures) mix the elapsed time of the clock into the
  key in the element itself.

## Alternatives considered

- **A `seed` setting per element**: simple, but duplicates the seed of the context and breaks the "one seed
  per context" rule of REQ-CTX-6.
- **Passing a `Generator` object in the values**: a generator is stateful (draws depend on call order) and
  is neither hashable nor a valid static pytree leaf.
- **Drawing in the context and feeding noise as an input signal**: makes graphs harder to read and moves
  element physics outside the element.

## Consequences

- Stochastic transfers stay pure: the same values (including the key) give the same output.
- Elements outside a context always use the same stream; tests vary realisations by creating contexts with
  different seeds.
- `random_key` is a reserved field name for stochastic elements.
