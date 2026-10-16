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
libraries), idle, double precision. Mean time of one propagation in milliseconds, and speed-up over NumPy on
the same system.

Windows 11, CuPy 14.2:

| Method | N = 256, NumPy | N = 256, CuPy | Speed-up | N = 1024, NumPy | N = 1024, CuPy | Speed-up |
|--------|---------------:|--------------:|---------:|----------------:|---------------:|---------:|
| `asm` | 9.0 | 1.9 | 4.7 | 345 | 13.2 | 26 |
| `fresnel_tf` | 9.6 | 1.9 | 5.0 | 340 | 16.1 | 21 |
| `fresnel_mft` (256² output) | 11.1 | 2.8 | 3.9 | 92 | 25.6 | 3.6 |
| `fraunhofer` (256² output) | 5.7 | 2.0 | 2.8 | 30.4 | 17.3 | 1.8 |

Ubuntu 26.04 under WSL 2, JAX 0.11.2 with CUDA 12:

| Method | N = 256, NumPy | N = 256, JAX GPU | Speed-up | N = 1024, NumPy | N = 1024, JAX GPU | Speed-up |
|--------|---------------:|-----------------:|---------:|----------------:|------------------:|---------:|
| `asm` | 7.8 | 11.0 | 0.7 | 173 | 36.3 | 4.8 |
| `fresnel_tf` | 14.4 | 10.1 | 1.4 | 186 | 33.5 | 5.5 |
| `fresnel_mft` (256² output) | 18.5 | 7.2 | 2.6 | 110 | 23.9 | 4.6 |
| `fraunhofer` (256² output) | 8.4 | 5.3 | 1.6 | 41.6 | 20.5 | 2.0 |

The GPU pays off on large padded FFTs (`asm`, `fresnel_tf`: 21 to 26 times faster with CuPy at N = 1024);
matrix Fourier transforms to a small output grid gain less. The GTX 1060 computes in double precision at
1/32 of its single-precision rate, so `lumos.config.precision = "single"` should gain more on such consumer
GPUs. With the same GPU, JAX is about 2 to 3 times slower than CuPy on the padded FFTs, and slower than NumPy
on small grids, where dispatching to the GPU dominates.

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
