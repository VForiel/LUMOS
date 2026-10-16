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

- VFSciTools must be published on PyPI before LUMOS can be installed from PyPI or tested in CI (tracked in
  `todo.md`).
- The archive tests probe the real git repository and hardware; the hardware report is cached per process.
