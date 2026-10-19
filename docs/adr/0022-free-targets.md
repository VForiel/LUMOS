# ADR-0022: Free settings and components of array parameters

- **Status**: accepted
- **Date**: 2026-10-19
- **Plan step**: P14.17

## Context

Sweeps and Monte-Carlo runs (REQ-CTX-10, P14.13) vary *free parameters*, which were only the scalar physical
parameters (`Param`) of the elements. A review of the web interface (19/10/2026) found that many values a user
wants to vary could not be: numeric settings without unit (`Star.magnitude`, `CircularAperture.n_spiders`,
`Telescope.oversize`, 57 of them) and the components of array parameters (the Zernike coefficients of a
`PhasePlate`, the wind speeds of the layers of an `Atmosphere`, the entries of a transfer matrix).

## Decision

1. A **free target** of an element is named by a string:
   - `"field"`: a physical parameter (as before) or a **numeric setting** (`int` or `float`, without a list
     of choices);
   - `"field[i]"` or `"field[i, j]"`: a **component** of an array parameter, indices in the order of the
     array (C order), checked against its current shape.
   `ParamHost.set_free`, `is_free`, `prior`, `free_names`, `get_target` and `set_target` take these names;
   `Context.free_parameters`, `sweep` and `sample` take paths `"element.field[i]"` the same way.
2. Priors of numeric settings are dimensionless; priors of components have the unit of their parameter. An
   **integer setting** takes the nearest integer of a drawn or swept value (`numpy.rint`), so a uniform prior
   on `[2.5, 6.5]` draws 3 to 6 with equal chances.
3. Setting a component writes a copy of the whole array with this component changed (the element sees an
   ordinary change of its parameter, caches are invalidated as usual).
4. `.lumos` documents keep these targets in the `"free"` object of the element, under the same names
   (`"n_spiders"`, `"coefficients[3]"`). This is a new feature of the format, which becomes **1.2**; LUMOS reads
   1.0, 1.1 and 1.2 documents, a 1.1 reader refuses a 1.2 document instead of dropping free targets
   (ADR-0021).

## Alternatives considered

- Free whole arrays with a prior of the same shape: the priors (`Uniform`, `LogUniform`, `Normal`) are scalar;
  a component at a time covers the needs met (a mode of a phase plate, a layer of turbulence) and keeps the
  draws reproducible and readable in the tables of the studies.
- Convert numeric settings into dimensionless `Param`: it would change the types of many elements (type
  versions and migrations) for no gain to the users of the Python API.
- Free boolean or choice settings: a prior over categories is a different object (discrete distribution);
  sweeps already take lists of values for them.

## Consequences

- The web interface frees any numeric field and any component of an array, and sweeps them.
- Format 1.2: documents written by this version are refused by readers of 1.1.
- Studies name their parameters with the same paths; their tables and CSV files show them as such.
