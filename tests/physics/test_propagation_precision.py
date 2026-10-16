"""Accuracy of propagation in single precision (steps P4.3 and P4.4, REQ-HPC-5).

``lumos.config.precision = "single"`` stores fields as complex64. Kernels must still be computed in double
precision and rounded once: a phase such as ``k z`` reaches 1e4 to 1e7 rad, and float32, with a relative
resolution of 6e-8, would lose it by 1e-3 to 1 rad if it were computed in single precision.

Independent references and tolerances
-------------------------------------
- **Gaussian beam** (Siegman, *Lasers*, 1986, §17.1), exact for the paraxial methods: see
  ``test_propagation_validation.py``. The reference is evaluated in double precision here.
- **Focal field of a lens** (Goodman, *Introduction to Fourier Optics*, 3rd ed., eq. 5-19): a thin lens of
  focal length f followed by a propagation over f gives
  ``U_f(x, y) = e^{ikf} e^{ik(x² + y²)/(2f)} / (iλf) · Û0(x/λf, y/λf)``, with
  ``Û0(u, v) = A π w² exp(-π² w² (u² + v²))`` for ``U0 = A exp(-r²/w²)``.
- **Tolerance**: rounding to complex64 changes every value by 6e-8 in relative terms, and the sums of an FFT
  or a matrix Fourier transform over N terms add an error of order ``6e-8 log2 N`` to ``6e-8 √N``, below 1e-6
  for these grids. The field must match the reference within 1e-5 of its peak, the double-precision
  tolerance of these exact methods being far smaller.
"""

import math

import numpy as np
import pytest
from astropy import units as u

import lumos
from lumos import backend
from lumos.fields import Grid, Wavefront
from lumos.propagation import clear_cache, focus, propagate

LAM = 1e-6  # m
K = 2 * math.pi / LAM
W0 = 20e-6
ZR = math.pi * W0**2 / LAM
BEAM_GRID = Grid(256, dx=2e-6 * u.m)


@pytest.fixture(params=["double", "single"])
def precision(request):
    previous = lumos.config.precision
    lumos.config.precision = request.param
    clear_cache()
    yield request.param
    lumos.config.precision = previous
    clear_cache()


def coordinates(grid):
    """Double-precision coordinates of the pixel centres, independent of ``lumos.config.precision``."""
    x = (np.arange(grid.nx) - grid.nx // 2) * grid.dx_si + grid.x0_si
    y = (np.arange(grid.ny) - grid.ny // 2) * grid.dy_si + grid.y0_si
    return np.meshgrid(x, y)


def as_numpy(wf):
    return np.asarray(backend.convert(wf.field, "numpy"))


def gaussian_beam(grid, z):
    """Analytic paraxial Gaussian beam of unit power (Siegman §17.1), in double precision."""
    xx, yy = coordinates(grid)
    r2 = xx**2 + yy**2
    w = W0 * math.sqrt(1 + (z / ZR) ** 2)
    curvature = 0.0 if z == 0 else 1 / (z * (1 + (ZR / z) ** 2))
    gouy = math.atan2(z, ZR)
    amplitude = math.sqrt(2 / math.pi) / w * np.exp(-r2 / w**2)
    return amplitude * np.exp(1j * (K * z + K * r2 * curvature / 2 - gouy))


@pytest.mark.parametrize("method", ["fresnel_tf", "fresnel_ir", "fresnel_mft"])
@pytest.mark.parametrize("z_over_zr", [1.0, 2.0])
def test_gaussian_beam_in_both_precisions(precision, method, z_over_zr):
    z = z_over_zr * ZR
    wf = Wavefront(gaussian_beam(BEAM_GRID, 0.0), BEAM_GRID, LAM * u.m)
    out = propagate(wf, z * u.m, method=method)
    expected = gaussian_beam(out.grid, z)
    np.testing.assert_allclose(as_numpy(out), expected, rtol=0, atol=1e-5 * np.abs(expected).max())


@pytest.mark.parametrize("method", ["fraunhofer", "fresnel_mft"])
def test_focal_field_of_a_gaussian_beam_in_both_precisions(precision, method):
    w, f = 1e-3, 1.0  # beam radius at the lens and focal length (m): k f = 6.3e6 rad
    pupil = Grid(256, dx=40e-6 * u.m)  # ±5.1 w: the truncated tail is below exp(-26)
    xx, yy = coordinates(pupil)
    wf = Wavefront(np.exp(-(xx**2 + yy**2) / w**2), pupil, LAM * u.m)
    focal = Grid(64, dx=20e-6 * u.m)  # the focal waist λf/(πw) = 318 µm spans ±2 waists
    out = focus(wf, f * u.m, output_grid=focal, method=method)
    xf, yf = coordinates(focal)
    spectrum = math.pi * w**2 * np.exp(-((math.pi * w / (LAM * f)) ** 2) * (xf**2 + yf**2))
    expected = np.exp(1j * K * f) * np.exp(1j * K * (xf**2 + yf**2) / (2 * f)) / (1j * LAM * f) * spectrum
    np.testing.assert_allclose(as_numpy(out), expected, rtol=0, atol=1e-5 * np.abs(expected).max())
