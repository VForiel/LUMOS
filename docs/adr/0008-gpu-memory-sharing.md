# ADR-0008: Sharing the GPU memory

- **Status**: accepted
- **Date**: 2026-10-16
- **Plan step**: P2.2 (found while validating P4 and P6 on a GPU)

## Context

JAX reserves 75 % of the GPU memory when it creates its GPU client (`XLA_PYTHON_CLIENT_PREALLOCATE`, JAX
documentation "GPU memory allocation"). The memory reserved this way is only available to XLA: cuFFT and
cuBLAS workspaces, CuPy arrays and other processes must fit in the rest. On a 6 GB GeForce GTX 1060 that also
drives the Windows display (about 2 GB used), the test suite run under WSL with the JAX CUDA backend had 63
`CUDA_ERROR_OUT_OF_MEMORY` failures; with preallocation disabled, all tests passed.

LUMOS users run on desktop and laptop GPUs shared with the display, and LUMOS may use JAX and CuPy in the same
process (REQ-HPC-1).

## Decision

- Before LUMOS loads the JAX backend or lists the JAX devices, it sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  if the variable is not set (`lumos.backend._core.prepare_jax`). JAX then allocates memory on demand.
- A value set by the user (environment or `os.environ` before the first use of a JAX device) is kept, so
  dedicated compute nodes can still preallocate for speed.

## Alternatives considered

- **Keep the JAX default**: fastest allocation on dedicated GPUs, but out-of-memory errors on shared ones.
- **`XLA_PYTHON_CLIENT_MEM_FRACTION`**: a fixed fraction is either too large for small shared GPUs or too
  small for large computations.
- **Setting the variable when `lumos` is imported**: would also change the behaviour of JAX code that does not
  use LUMOS.

## Consequences

- JAX allocations are slightly slower (on demand) and memory may fragment on long runs; users can set
  `XLA_PYTHON_CLIENT_PREALLOCATE=true` for benchmarks on dedicated GPUs.
- If the JAX GPU client already exists when LUMOS loads JAX (the user used JAX first), the setting has no
  effect; this is documented.
