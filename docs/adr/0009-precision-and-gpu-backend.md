# ADR-0009: Double precision by default and JAX as the first GPU backend

- **Status**: accepted
- **Date**: 2026-10-17
- **Plan step**: P4.11 (benchmarks, revisiting REQ-HPC-5 and ADR-0004)

## Context

The GPU benchmarks of P4.11 raised two questions: should LUMOS compute in single precision by default to be
faster, and should JAX stay the preferred GPU backend (ADR-0004) although it was measured 2 to 3 times slower
than CuPy on the same GPU?

Measurements on a GeForce GTX 1060 (consumer GPU, double precision at 1/32 of the single-precision rate) and an
8-core CPU, reported in `docs/learn/performance.md`:

- **Single precision speed**: padded FFTs (angular spectrum, Fresnel transfer function) are limited by memory
  bandwidth: single precision makes them about 2 times faster on the CPU and 3 times faster on the GPU.
  Matrix Fourier transforms are limited by arithmetic: 13 to 23 times faster on this GPU, 1.5 to 2 times on
  the CPU.
- **Single precision accuracy**: fields agree with double precision within about 1e-6 of their peak once
  every kernel is computed in double precision and rounded once (fixed for the MFT chirps in this step: a
  phase such as `k z` reaches 1e4 to 1e7 rad, which float32 resolves only to 1e-3 to 1 rad). Running the test
  suite in single precision fails 125 of its 1000 tests, whose tolerances (1e-10 to 1e-12 for energy
  conservation, unitarity, reciprocity, nulls) assume double precision.
- **JAX**: run operation by operation ("eager"), each JAX operation is dispatched separately and a LUMOS
  propagation is 1.4 (N = 1024) to 7 (N = 256) times slower than with CuPy; compiled with `jax.jit`, the same
  propagation is as fast as CuPy or faster (17.8 ms against 25.9 ms for a 1024 × 1024 angular spectrum). The gap was the eager dispatch, not JAX itself. The first CuPy times were also too short, because
  the benchmark did not wait for CuPy's asynchronous kernels.

## Decision

1. **Double precision stays the default** (`lumos.config.precision = "double"`); single precision is an
   opt-in for speed. Whatever the precision, kernels (transfer functions, chirps, MFT matrices) are computed in
   double precision and rounded once.
2. **JAX stays the first GPU backend** of `"auto"` (ADR-0004 unchanged): compiled, it matches CuPy and it is
   differentiable. CuPy remains the GPU backend on native Windows, where JAX has no GPU build.
3. Pipelines run on JAX should be compiled: users compile their own functions with `lumos.jit`, and the
   execution of a context should compile its pure transfer chain on JAX when this becomes possible (ADR-0006).

## Rationale

- LUMOS targets high-contrast instruments (nulling interferometry, coronagraphy) whose signals are 1e-5 to
  1e-10 of the starlight: the round-off of single precision (about 1e-6 of the peak field, 1e-12 in intensity
  after one propagation) is close to these levels after a chain of elements, and accumulates in long
  simulations (turbulence time series, optimisation loops, finite-difference gradients).
- The speed-up of single precision is modest where most of the time is spent (padded FFTs: 2 to 3 times)
  and large only for matrix Fourier transforms on consumer GPUs; professional GPUs (A100, H100) compute in
  double precision at half or full single-precision rate.
- A single-precision default would not simplify the tests: their tolerances would have to be loosened by
  four to six orders of magnitude, hiding real errors such as the lost `k z` phase found here.
- REQ-HPC-5 asks for a configurable precision, and the `.lumos` format records `complex128` as the default.

## Alternatives considered

- **Single precision by default**: faster on consumer GPUs, but silently inaccurate for the main use cases,
  and incompatible with the current test tolerances.
- **Precision chosen by backend** (single on GPUs, double on CPUs): results would depend on the hardware,
  contradicting REQ-HPC-1 (identical results on every backend within floating-point tolerances).
- **CuPy as the first GPU backend**: faster without compilation, but not differentiable, and it would give
  different execution paths for optimisation (JAX) and simulation (CuPy).

## Consequences

- Users who want speed on consumer GPUs set `lumos.config.precision = "single"` and accept a relative error of
  about 1e-6; the documentation says when this is safe.
- Eager LUMOS code is slower on JAX than on CuPy until context execution compiles its pipelines; the
  performance page shows how to compile with `lumos.jit`.
- New kernels must be computed in double precision on the host or with explicit float64 coordinates, never
  from coordinates in the working precision; `tests/physics/test_propagation_precision.py` checks this for
  the propagation methods.
