# ADR-0010: Results, figures and archives

- **Status**: accepted
- **Date**: 2026-10-17
- **Plan step**: P7.3 to P7.5 (REQ-RES-1 to REQ-RES-6, NF-7)

## Context

Every run must be traceable (objective O5): the context, the environment, the journal, the figures and the
data behind them. The specifications require `vfst.archive` and `pltedit` (VFSciTools) for the archives and
the editable figures. PltEdit's `.plt` format pickles matplotlib figures, while LUMOS forbids unpickling user
files (NF-7).

## Decision

- **VFSciTools is a core dependency** (NF-2): `vfst.archive.new` creates the dated archive directory
  (`<root>/<date>_<commit>/<NNNN>-<name>/`) and records the UTC date, Python version and git commit and
  branch; LUMOS adds the other REQ-RES-2 fields to `metadata.yml` (OS, versions of LUMOS and of the key
  dependencies, hardware report probed once per process, backend, precision, seed, durations, mode, twin,
  clock, origin, outputs).
- **Layout**: `context.lumos`, `metadata.yml`, `lumos.log` (journal as JSON lines), `figures/` (`.png` and
  `.plt`), `data/`.
- **Root**: `lumos.config.archive_root`, else an `archives/` directory next to the running notebook (VS Code's
  `__vsc_ipynb_file__`, Jupyter's `JPY_SESSION_NAME`), else next to the executed script (`__main__.__file__`),
  else the current directory.
- **Data files**: data products as `.npz` with a JSON header, optical signals as `.json` (`{"signal": class,
  "data": to_json()}`), other arrays as `.npz` where a quantity `name` is stored with a string entry
  `__unit__name`. Every `.npz` is written and read with `allow_pickle=False`.
- **Reading back** (`lumos.load_archive`): JSON signals are decoded by a fixed list of classes; `.plt` files
  are listed, never loaded, since loading them unpickles. Users open them with PltEdit when they trust them.
- `ctx.run(outputs, archive=...)` archives the run and every computed output; `result.archive` gives access
  to the archive to add figures.

## Alternatives considered

- A LUMOS-specific archive directory without VFSciTools: duplicates `vfst.archive` and breaks the convention
  shared with the author's other tools.
- HDF5 for all data: an extra dependency for the core; `.npz` with plain arrays is enough and safe.
- Loading `.plt` figures in `load_archive`: convenient but unpickles files that may come from someone else.

## Consequences

- ~~VFSciTools must be published on PyPI before LUMOS can be installed from PyPI or tested in CI.~~ See the
  amendment below.
- The archive tests probe the real git repository and hardware; the hardware report is cached per process.

## Amendment (2026-10-17): VFSciTools is vendored

Installing LUMOS on Read the Docs and in CI failed because `vfscitools` is not on PyPI. At the author's request,
the two parts LUMOS uses are copied into `lumos._vendor` (MIT license, same author): `pltedit` (`save`, `load`,
`set_style`; same `.plt` format, the files still open in the PltEdit application) and `vfst_archive`
(`vfst.archive.new`). They are kept as upstream except for their imports, excluded from ruff and pyright, and
updated by copying them again. `vfscitools` is no longer a dependency; once it is on PyPI, LUMOS may depend on it
again and drop the copies.

## Amendment (2026-10-28): LRFutils replaces the vendored copies

VFSciTools is merged into LRFutils, published on PyPI. LRFutils 0.2 ships the same code (`LRFutils.archive.new`,
`LRFutils.pltedit.save`, `load`, `set_style`; same archive layout and `.plt` format), so `LRFutils>=0.2` is a core
dependency and `lumos._vendor` is removed. `metadata.yml` records the version of `LRFutils` instead of `vfscitools`.
