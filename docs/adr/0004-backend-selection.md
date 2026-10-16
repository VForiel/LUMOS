# ADR-0004: Automatic backend selection

- **Status**: accepted
- **Date**: 2026-10-16
- **Plan step**: P2.3

## Context

REQ-HPC-2 asks LUMOS to detect the hardware it runs on and to pick the array backend that exploits it
best, while letting the user force a choice. Several backends may be installed at once (NumPy always, JAX,
CuPy, PyTorch), with or without a GPU.

## Decision

1. `lumos.config.backend` is `"auto"` by default; it can be set to any backend name. `lumos.backend.use()`
   overrides it locally (per thread or task).
2. With `"auto"`, the backend is chosen once per process, the first time an array is needed, from
   `hardware_report(probe_gpus=True)` by the pure function `select_backend`:
   1. **JAX** if it sees a GPU (or TPU);
   2. otherwise **CuPy** if it sees a GPU;
   3. otherwise **PyTorch** if it sees a GPU (CUDA, ROCm or Apple Metal);
   4. otherwise **NumPy**.
3. The choice is logged (`INFO` level on the `lumos.backend._core` logger).

## Rationale

- JAX is the differentiable reference of LUMOS (decision D3): preferring it on GPUs gives the same results,
  gradients included, as on a GPU-less development machine running JAX on CPU.
- Without a GPU, NumPy (with multi-threaded `scipy.fft`) is the reference implementation, has no compilation
  overhead and no device transfer; JAX on CPU stays available on request (`use("jax")`).

## Alternatives considered

- **Benchmark at start-up**: more accurate but slow and non-deterministic; it may come later as an opt-in
  helper.
- **Always NumPy unless asked**: simpler, but contradicts REQ-HPC-2.

## Consequences

- The first array operation of a process with `"auto"` imports the installed GPU-capable libraries to list
  their devices (one-off cost of about a second); importing `lumos` itself stays light.
- Results must not depend on the backend beyond floating-point tolerances; the test suite is parametrised by
  backend to enforce it.
