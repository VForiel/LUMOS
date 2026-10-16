# ADR-0003: The `.lumos` document format

- **Status**: accepted
- **Date**: 2026-10-16
- **Plan step**: P1.6

## Context

Contexts, element groups and single elements must be exchanged between the Python API, the web interface,
the MCP server and users (REQ-EL-12, REQ-CTX-8, §10 of the specs). Documents must be readable by humans,
safe to open (NF-7), and survive the evolution of LUMOS (versions, migrations).

## Decision

1. A `.lumos` file is a **UTF-8 strict-JSON** document (no `NaN`/`Infinity` literals) with an envelope:
   `lumos_format` (`"major.minor"`, currently `"1.0"`), `kind` (`"element"` now; `"group"` and
   `"context"` with phase 5), `lumos_version` and `created_at` (UTC, ISO 8601).
2. A document of kind `element` holds one `element` entry: `type` (registry identifier), `type_version`,
   `name`, `params`, `settings`, `state` and `free` (free parameters and their priors).
3. **Quantities keep the user's unit** and use the encoding of `lumos.units.quantity_to_json`; arrays are
   inline base64 with an allow-list of dtypes. (Large arrays will move to a sidecar file in step P5.8.)
4. Element classes are found in the **registry** from `type`; reading a document never imports anything. The
   body is validated with the JSON Schema of the class, then values are checked by the class itself (units,
   bounds, shapes).
5. **Versions**:
   - a newer `lumos_format` (major or minor) is refused with `FormatVersionError`;
   - a newer `type_version` is refused; an older one is upgraded by the migrations registered with
     `Registry.add_migration(type_id, from_version)`, applied one version at a time.
6. Every reading error is a `LumosFormatError` (a `ValueError`), except unknown types (`UnknownTypeError`).

## Alternatives considered

- **YAML or TOML**: friendlier to edit by hand, but JSON is native to the web interface and the MCP server,
  and its schema tooling is standard.
- **Pickle**: refused for security reasons (arbitrary code execution) and because it is not portable.
- **Storing SI values only**: simpler, but users would lose their units when reopening a file.

## Consequences

- Each change of the serialised form of an element class requires incrementing its `type_version` and
  registering a migration.
- The document schema (envelope + element bodies) must be published in the documentation (step P5.8).
