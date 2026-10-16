# LUMOS — instructions for coding agents

LUMOS (*Lumos is a Universal and Modular Optical Simulator & Operating System*) is a Python library
(distribution `lumos-maxima`, import `lumos`) with a web interface and an MCP server.

## Read first

- Requirements: [docs/specs/cahier_des_charges.md](docs/specs/cahier_des_charges.md) (French).
- Step-by-step plan: [docs/specs/plan_developpement.md](docs/specs/plan_developpement.md) (French). Work one step `Px.y` at a time and tick its box when done.
- Architecture decisions: [docs/adr/](docs/adr/). Write a new ADR (copy `0000-template.md`) for every structural decision.

## Development cycle of every step: specify → test → implement → document

1. **Specify**: reread the requirements targeted by the step; clarify the expected behaviour.
2. **Test first**: write the tests from the specification and commit them alone: `test(Px.y): …`. They must fail.
   - Expected values come from **independent sources**: analytical solutions, published results (cite them in the test), established independent libraries (astropy, POPPY, HCIPy, …) or measurements.
   - **Never** generate an expected value by running the code under test. Snapshot/regression tests are allowed only after an independent validation and must be marked as such.
   - Prefer property-based tests (Hypothesis) for invariants: energy conservation, unitarity, reciprocity, round-trips.
3. **Implement** until green **without touching the tests**: `feat(Px.y): …` / `fix(Px.y): …`.
4. **Document**: NumPy-style docstrings explaining the physics, doc pages, `CHANGELOG.md`, ADR if needed.

### Changing an existing test

Only allowed when the objective changed (recorded in the specs or an ADR) or the test is proven wrong or limited
(wrong reference value, physically unjustified tolerance, wrong assumption). Then:

- use a **dedicated** commit `test(Px.y)!: …` whose body explains why and cites the authoritative source;
- never mix it with implementation code;
- never loosen a tolerance just to get green.

`tools/check_test_integrity.py` enforces these rules in CI (see its docstring).

## Conventions

- **Language**: code, identifiers, docstrings, comments, notebooks, log and error messages in **English**. User docs and web UI are written in English and translated to French (gettext).
- **Units**: every user-facing physical quantity is an `astropy.units.Quantity`; internal storage is SI floats/arrays. Never accept raw floats for physical quantities in public APIs.
- **Arrays**: backend arrays (NumPy, JAX, …) through the Array API (`lumos.backend.current().xp`); no units inside performance-critical arrays; compile pure kernels with `lumos.jit`.
- **Pure transfers**: element transfer functions are pure (state passed explicitly) so they can be JIT-compiled and differentiated.
- **No** `print` (use `logging`), **no** `pickle` for user files, **no** disabling TLS verification, **no** arbitrary imports when loading `.lumos` files.
- **Layout**: `src/lumos/` source only, `tests/` tests only, `examples/` notebooks and presets, `tools/` dev scripts, `docs/` documentation. Do not create files at the repository root; use `tmp/` (git-ignored) as a scratchpad.
- **Commits**: Conventional Commits with the plan step as scope, e.g. `feat(P4.3): add matrix Fourier transform`.

## Commands (Windows paths; use `.venv/bin/` on Linux/macOS)

```
.venv\Scripts\python.exe -m pip install -e ".[dev,docs]"
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m pytest -m benchmark tests/benchmarks
.venv\Scripts\python.exe -m ruff check . ; .venv\Scripts\python.exe -m ruff format .
.venv\Scripts\python.exe -m pyright
.venv\Scripts\python.exe -m sphinx -W -b html docs docs/_build/html/en
.venv\Scripts\python.exe -m sphinx -W -b html -D language=fr docs docs/_build/html/fr
.venv\Scripts\python.exe tools/check_test_integrity.py origin/main..HEAD
```
