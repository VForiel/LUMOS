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
device). The measurements will be added here.

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
