# ADR-0021: Public API and stability of LUMOS 1.x

- **Status**: accepted
- **Date**: 2026-10-18
- **Plan step**: P15.1

## Context

LUMOS 1.0 freezes its API and its `.lumos` format (plan, P15.1). Users write scripts and notebooks, save
`.lumos` files, call the HTTP API and let assistants call the MCP tools; all of them must keep working through
the 1.x versions. The review of the API before the freeze found few inconsistencies: two settings and one
function spelled in British English (`normalise`, `subaperture_centres`) among American identifiers
(`normalize`, `center`, `polarization`, `colorbar`), an element class named differently from its type
(`PSFMeasurementElement` for `lumos.processing.PSFMeasurement`), 110 fields of element types without
description, two subpackages without `__all__` and two absent from the API reference.

## Decision

1. **What is public** in 1.x, and covered by the rules below:
   - the names listed in `__all__` of `lumos` and of its public subpackages (`backend`, `catalogs`, `core`,
     `elements`, `fields`, `hardware`, `io`, `plotting`, `processing`, `propagation`, `results`, `scenes`,
     `service`, `symbolic`) and documented in the API reference;
   - the element types: their identifiers `lumos.<family>.<ClassName>`, fields (names, kinds, units) and ports;
   - the `.lumos` format 1.x and its JSON schema;
   - the models of the service layer, the routes of the HTTP API, the MCP tools and their arguments, the
     commands of the `lumos` command line.

   Names starting with `_`, `lumos._vendor`, the internals of `lumos.web` (other than `create_app` and its
   configuration) and the web interface itself are not.
2. **Semantic versioning**: a 1.x release adds and fixes, it does not remove or rename. A public name to be
   removed is deprecated first (`DeprecationWarning`, documented in the changelog) during at least one minor
   release, and removed in 2.0 at the earliest.
3. **Element types and files**: a change of the fields of an element type raises its type version and adds a
   migration (`registry.add_migration`), so that every `.lumos` file written by a 1.x version is read by the
   later ones. A new feature of the format raises its minor version (1.1 added the feedbacks, ADR-0020): newer
   readers read older documents; older readers refuse newer ones with a clear error.
4. **Conventions of the identifiers**: snake_case; American spelling, as NumPy, astropy and matplotlib
   (`normalize`, `center`, `color`, `polarization`), the prose of the documentation being free; physical
   quantities as `astropy.units.Quantity`; counts named `n_<things>`; every field of an element type has a
   short description (shown by the web interface, the JSON schemas and the MCP server).
5. The review renames, before the freeze: the setting `normalise` of `FlatField` and `KernelObservables`
   and the keyword of `flat_field` and `kernel_observables` (type version 2, documents of version 1 migrated);
   `subaperture_centres` to `subaperture_centers`; the element class `PSFMeasurementElement` to
   `PSFMeasurement` (its type identifier) and the result of `measure_psf` from `PSFMeasurement` to `PSFResult`.

`tests/unit/test_api_review.py` checks points 1 and 4 (documentation of the element types, identifiers,
spelling, `__all__`, API reference) at every run.

## Alternatives considered

- British spelling everywhere: most identifiers and the scientific Python ecosystem are American; renaming
  `polarization`, `center` and `colorbar` would break more than it fixes.
- Aliases keeping the old spellings: LUMOS was not released yet (0.x); migrations of the type versions keep
  the files readable, which is what matters to users.

## Consequences

- Users of 1.x can rely on the public API and on their files; the changelog lists every deprecation.
- New element types and fields must follow the conventions, or the review test fails.
- Adding to the format needs an ADR and a minor version of the format.
