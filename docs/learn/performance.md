# Performance

LUMOS measures its numerical kernels with `pytest-benchmark`:

```bash
python -m pytest -m benchmark tests/benchmarks
```

## Propagation (CPU)

Mean time of one propagation of a Gaussian beam (milliseconds), on an AMD Ryzen 7 2700X (8 cores), NumPy
2.x with SciPy FFTs and JAX on the CPU, double precision. The FFT methods zero-pad by a factor 2, so an
`N × N` grid is transformed as `2N × 2N`.

| Method | N = 256, NumPy | N = 256, JAX | N = 1024, NumPy | N = 1024, JAX |
|--------|---------------:|-------------:|----------------:|--------------:|
| `asm` | 9.0 | 21.3 | 340 | 254 |
| `fresnel_tf` | 9.6 | 20.9 | 352 | 241 |
| `fresnel_mft` (256² output) | 10.7 | 8.9 | 93 | 51 |
| `fraunhofer` (256² output) | 5.7 | 5.8 | 36 | 28 |

NumPy is faster on small grids, where the overhead of dispatching to JAX dominates; JAX is faster on large
grids. Matrix Fourier transforms are much cheaper than padded FFTs when the output grid is smaller than the
input grid.

## GPU

The GPU speed-up is measured with the same benchmark on a machine with a CUDA GPU (CuPy, or JAX with a GPU
device): the Ryzen 7 2700X above with an NVIDIA GeForce GTX 1060 6 GB (Pascal, driver 582.28, CUDA 12
libraries), idle. Mean time of one propagation in milliseconds, in double and single precision; every run
waits for the asynchronous GPU computation to finish.

Windows 11, CuPy 14.2:

| Method | N = 256, NumPy | N = 256, CuPy | N = 256, CuPy single | N = 1024, NumPy | N = 1024, CuPy | N = 1024, CuPy single |
|--------|---------------:|--------------:|---------------------:|----------------:|---------------:|----------------------:|
| `asm` | 11.1 | 1.8 | 1.8 | 374 | 25.9 | 7.2 |
| `fresnel_tf` | 9.9 | 1.7 | 1.9 | 376 | 19.8 | 7.3 |
| `fresnel_mft` (256² output) | 6.7 | 2.9 | 0.7 | 41 | 23.3 | 1.1 |
| `fraunhofer` (256² output) | 5.7 | 2.7 | 0.4 | 29.8 | 22.9 | 1.0 |

Ubuntu 26.04 under WSL 2, JAX 0.11.2 with CUDA 12, N = 1024; "compiled" wraps the propagation in `jax.jit`:

| Method | NumPy | JAX GPU | JAX GPU compiled | JAX GPU compiled, single |
|--------|------:|--------:|-----------------:|-------------------------:|
| `asm` | 179 | 35.0 | 17.8 | 5.8 |
| `fresnel_tf` | 178 | 31.7 | 16.9 | 5.9 |
| `fresnel_mft` (256² output) | 68 | 19.5 | 19.1 | 1.4 |
| `fraunhofer` (256² output) | 43.6 | 19.1 | 18.1 | 1.4 |

- The GPU pays off on large padded FFTs: 10 to 20 times faster than NumPy for N = 1024 in double precision.
  On small grids (N = 256) the fixed cost of each GPU call dominates.
- **JAX should be compiled.** Run operation by operation, every JAX operation is dispatched separately: at
  N = 256 an eager JAX propagation (13 ms) is slower than NumPy (7.6 ms). Compiled with `lumos.jit` (or
  `jax.jit`), it takes 1.5 ms and matches or beats CuPy.
- **Single precision** makes padded FFTs, which are limited by memory bandwidth, about 2 times faster on the
  CPU and 3 times faster on the GPU. Matrix Fourier transforms are limited by arithmetic, and consumer GPUs
  such as the GTX 1060 compute in double precision at 1/32 of their single-precision rate: they become 13 to
  23 times faster.

### Choosing the precision

LUMOS computes in double precision by default (ADR-0009). `lumos.config.precision = "single"` stores fields
in complex64; kernels are still computed in double precision and rounded once, and propagated fields agree
with double precision within about 1e-6 of their peak. Single precision suits exploratory runs, imaging and
moderate contrasts. Keep double precision for high-contrast work (nulls and coronagraphic contrasts below
about 1e-8, where round-off in a chain of elements reaches the signal), for long time series and for
gradients checked by finite differences.

### Enabling an NVIDIA GPU

LUMOS sees a GPU only through an array library built for CUDA; `lumos.hardware_report()` lists the GPUs that
these libraries see, and `lumos.config.backend = "auto"` then chooses a GPU backend.

- **Windows**: install CuPy with the CUDA 12 libraries from pip, which need no system CUDA toolkit:
  `pip install cupy-cuda12x "cuda-toolkit[cublas,cudart,cufft,curand,nvrtc]==12.*"`. CUDA 12 supports GPUs
  from the Maxwell generation on (CUDA 13 needs Turing or newer).
- **Linux or WSL 2**: `pip install "jax[cuda12]"` (JAX publishes no CUDA build for native Windows), or CuPy.
- JAX reserves 75 % of the GPU memory by default; LUMOS turns this off so that JAX shares the GPU with the
  display and CuPy (ADR-0008). Set `XLA_PYTHON_CLIENT_PREALLOCATE=true` to preallocate on a dedicated GPU.
- Measure with the GPU otherwise idle: a game or another GPU application changes the results by large
  factors.

The test suite runs on NumPy by default; `LUMOS_TEST_BACKEND=jax` (or `cupy`) runs it on another backend.
