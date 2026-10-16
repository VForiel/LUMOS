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
