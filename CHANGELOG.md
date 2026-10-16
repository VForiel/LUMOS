# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Project skeleton: `lumos-maxima` distribution, `lumos` package, MIT license.
- Development tooling: ruff, pyright, pytest, Hypothesis, pre-commit.
- Continuous integration and test-integrity check.
- Sphinx documentation skeleton (Breeze theme, English and French).
- `lumos.units`: unit validation, SI conversion and strict-JSON encoding of quantities.
- `lumos.core.param`: declarative `Param`, `State` and `Setting` fields with units, bounds, shapes,
  invalidation hook and per-instance free parameters; `lumos.core.priors` (uniform, log-uniform, normal).
- `lumos.core.signal` and `lumos.core.port`: abstract signal types and typed input/output ports.
- `lumos.core.element`: `Element` base class with pure static transfer functions, `describe()`,
  Jupyter representation and JSON Schema.
- `lumos.core.registry`: element type registry with stable identifiers, versions, migrations and plugins.
- `lumos.io`: `.lumos` documents for single elements (save/load, version checks, migrations).
