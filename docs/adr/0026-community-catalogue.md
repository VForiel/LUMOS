# ADR-0026: Community catalogue

- **Status**: accepted
- **Date**: 2026-10-20
- **Plan step**: P15.16

## Context

Declarative components (ADR-0025) and presets are worth sharing. The review of 20/10/2026
(docs/specs/pistes_evolution.md §2) settled that the catalogue lives in the LUMOS repository, in a `community/`
folder for the contributions of the community, with a CI filtered by path, an index with one entry per
contribution (author, version, license, minimum format), and a "Catalogue" tab of the palette that imports a
declarative component in one click. Components with code are Python packages installed locally
(`lumos-component-…`), never run on the public instance.

## Decision

1. **Layout.** `community/components/<author>/<Name>.lumos` (documents of kind `component`, identifier
   `<author>.<Name>`), `community/presets/<author>/<name>.lumos` (contexts whose `meta` gives `name`,
   `description`, `authors` and `license`; the components they use are embedded), `community/README.md` (how to
   contribute) and `community/index.json`.
2. **Index.** `lumos.catalogue.build_index(root)` makes the index from the files: per entry `kind`
   (`component` or `preset`), `id` (`author.Name` or `author/name`), `title`, `description`, `authors`,
   `version` (components), `license`, `lumos_format` (the format of the file, the minimum to read it) and
   `path`. The committed index must equal the built one; `tools/build_catalogue.py` writes it.
3. **Checks.** `lumos.catalogue.check_catalogue(root)` lists what refuses a contribution: a component failing
   `check_component` (its checks carry independent reference values), a file not at the place of its
   identifier, a preset without license, authors or description, or not loading, a format newer than the
   library. The workflow `community.yml` runs it, `lumos component check` on every component and the index
   check, on changes to `community/` only; the main CI ignores changes under `community/`.
4. **Shipping.** The wheel includes `community/` as `lumos/community` (like the presets); the service finds it
   there, else in the repository, or in `ServiceConfig.catalogue_dir`.
5. **Service.** `list_catalogue(query)`, `use_catalogue_component(session, id)` (registers the component in the
   session, as the editor does) and `open_catalogue_preset(session, id)` (a new context). Identifiers are
   looked up in the index only, so nothing outside it can be read; the routes `/api/catalogue...` and the MCP
   tools `list_catalogue` and `use_catalogue` are thin adapters.

## Alternatives considered

- A separate repository for the catalogue: two CIs and two reviews, and contributions lagging behind the
  format; refused on 20/10.
- Fetching the index from GitHub at run time: needs the network, and an instance would show contributions that
  its version cannot read.
- Executable contributions in `community/`: code never runs on the public instance (ADR-0025 §5); code
  components are packages installed by whoever runs the instance.

## Consequences

- A contribution is a pull request adding one file (and the rebuilt index); reviewers see its checks.
- The catalogue of an instance is the one of its version; a new contribution reaches the public instance when
  it is redeployed.
- Shared resources of the public instance (P15.5) will join `community/` with their own kind of entry.
