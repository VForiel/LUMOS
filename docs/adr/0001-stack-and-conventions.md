# ADR-0001: Technical stack and project conventions

- **Status**: accepted
- **Date**: 2026-10-16
- **Plan step**: P0.1

## Context

LUMOS starts from scratch. The requirements (`docs/specs/cahier_des_charges.md`, decisions D1–D13) fix the main
orientations; this ADR records the concrete tooling chosen in phase 0.

## Decision

| Topic | Choice |
|-------|--------|
| Distribution / import | `lumos-maxima` on PyPI (`lumos` is taken), `import lumos`, CLI `lumos` |
| Python versions | SPEC 0 policy. On 2026-10-16 the latest NumPy (2.5), SciPy (1.18) and JAX (0.11) all require Python ≥ 3.12, so `requires-python = ">=3.12"`; CI tests 3.12, 3.13 and 3.14 |
| Build backend | `hatchling` with `hatch-vcs` (version derived from git tags) |
| Layout | `src/` layout, typed package (`py.typed`) |
| Lint / format | `ruff` (lint and format), configured in `pyproject.toml` |
| Type checking | `pyright` in standard mode; annotations of public code enforced by ruff (`ANN` rules) in `src/` |
| Tests | `pytest`, `pytest-cov`, `hypothesis`; markers `network`, `hardware`, `gpu`, `slow` excluded by default |
| Test discipline | tests first, separate test commits, independent reference values, enforced by `tools/check_test_integrity.py` |
| Commits | Conventional Commits, scope = plan step (`feat(P4.3): …`); breaking test changes use `test(Px.y)!:` with a justification |
| Docs | Sphinx, Breeze theme, MyST, myst-nb, Mermaid; English source, French via gettext (`sphinx-intl`); ReadTheDocs (one project per language) |
| CI | GitHub Actions: lint, type checking, test matrix (Linux, Windows, macOS × 3.12–3.14), docs (EN and FR, warnings as errors), test integrity |
| Release | PyPI trusted publishing on GitHub Release |
| License | MIT |

### `vfscitools` dependency

`vfscitools` (archives and `.plt` figures, REQ-RES-*) is **not published on PyPI**, and a package published on
PyPI cannot depend on a git URL. It is therefore not declared yet. It must be published on PyPI before step P7.3,
when LUMOS starts using it.
*Superseded on 2026-10-17: the parts used are vendored in `lumos._vendor` (ADR-0010, amendment); then on 2026-10-28
by the `LRFutils` dependency, into which VFSciTools is merged (ADR-0010, second amendment).*

### Amendment (2026-10-16, step P1.1): no pyright strict mode

Strict mode was tried on `lumos.units`: it reported 17 errors, almost all caused by the incomplete typing of
astropy (`Quantity.unit` may be `None`, unknown members) and by NumPy dtype unions, not by actual defects.
Silencing them would have required casts everywhere in physics code. Pyright stays in standard mode, and ruff
enforces type annotations on every public function of `src/`.

## Alternatives considered

- **setuptools + setuptools-scm**: equivalent, but more configuration for the same result.
- **mypy**: slower, and its NumPy/JAX stub ecosystem is less convenient than pyright's.
- **Python ≥ 3.11** for a wider audience: impossible without giving up the latest NumPy, SciPy and JAX releases.

## Consequences

- Version numbers come from git tags (`v0.1.0`). Untagged commits get `0.1.devN` versions.
- `vfscitools` must be released on PyPI before P7.3.
