"""Validation suite of the numerical propagation methods (step P4.1).

This suite is written before any propagation code. Every method must pass the cases of its domain of
validity; ``"auto"`` (the planner, P4.6) must pass all of them.

API under test (``lumos.propagation``)::

    propagate(signal, distance, *, output_grid=None, method="auto")   # plane z=0 -> plane z=distance
    propagate(signal, None, output_grid=angular_grid, method=...)     # far field (Fraunhofer)
    focus(signal, focal_length, output_grid, method="auto")           # pupil -> focal plane of a thin lens

Conventions (§9 of the specs): ``U ∝ e^{+ikz}``; the far field on an angular grid is normalised so that
``Σ |U|² ΔΩ`` is the power; every method includes the propagation phase ``e^{ikz}``. Far-field angular
coordinates are direction cosines (θx = sin of the angle to z in the x-z plane), so that the far field is the
Fourier transform of the pupil field at the spatial frequencies ``θ / λ`` (Goodman §4.2).

Methods: ``"rs"`` (direct Rayleigh-Sommerfeld, any grid to any grid), ``"fraunhofer"`` (matrix Fourier
transform, far field or focal plane, any output sampling), ``"fresnel_tf"`` and ``"fresnel_ir"`` (Fresnel
transfer function and impulse response, output on the input grid), ``"fresnel_mft"`` (one-step Fresnel with a
matrix Fourier transform, any output grid), ``"asm"`` (band-limited angular spectrum, any output grid).

Independent references and tolerances
-------------------------------------
- **Airy pattern** of a circular pupil of diameter D (Born & Wolf, *Principles of Optics*, §8.5.2): zeros
  at θ = 1.21967, 2.23313, 3.23832 λ/D; peak intensity ``P A / λ²`` for a uniform pupil of area A. The
  pupil is digitised with 256 pixels across: the zeros are located within 0.5 % (half a pixel over the
  radius is 0.4 %).
- **Square aperture** of side a: the Fourier transform of the sampled aperture is the Dirichlet kernel
  ``dx sin(π u a) / sin(π u dx)`` per axis (exact, sum of a geometric series); it differs from the
  continuous ``a sinc(u a)`` by the relative midpoint-quadrature error ``≈ (π u dx)² / 6``.
- **Gaussian beam** (Siegman, *Lasers*, 1986, §17.1): ``U = sqrt(2P/π)/w exp(-r²/w²) exp(i[kz + kr²/(2R)
  - ζ])`` with ``w = w0 sqrt(1 + (z/zR)²)``, ``R = z (1 + (zR/z)²)``, ``ζ = atan(z/zR)``,
  ``zR = π w0² / λ``. Non-paraxial corrections scale as ``(k w0)^-2 ≈ 6e-5`` here: tolerance 1e-3 of the
  peak amplitude.
- **On-axis field of a circular aperture** under normal plane-wave illumination, exact for the first
  Rayleigh-Sommerfeld solution (Goodman, *Introduction to Fourier Optics*, 3rd ed., problem 3-12;
  Sommerfeld, *Optics*, §34): ``U(0, 0, z) = U0 [e^{ikz} - z / sqrt(z² + a²) e^{ik sqrt(z² + a²)}]``.
  The disk is anti-aliased by 16×16 supersampling; tolerance 2 % of U0 from the residual edge error.
- **Direct Rayleigh-Sommerfeld integral** (Goodman eq. 3-42, first kind), evaluated in this file by an
  explicit sum on a small case: ``h = z / (2π r²) (1/r - ik) e^{ikr}``.
- **Parseval**: transforms with ``|H| = 1`` (angular spectrum, Fresnel transfer function) and the DFT-matched
  far field conserve the power to round-off.
- **Back-propagation** by ``-z`` inverts propagation by ``+z`` (unitary transfer function), and the free-space
  Green function is **reciprocal**: ``G(p1 → p2) = G(p2 → p1)``.
"""

import math

import numpy as np
import pytest
from astropy import units as u

from lumos import backend
from lumos.fields import Grid, Wavefront
from lumos.propagation import focus, propagate

LAM = 1e-6  # m
K = 2 * math.pi / LAM


def as_numpy(wf):
    return backend.to_numpy(wf.field)


def disk(grid, radius, supersample=1):
    """Fraction of each pixel inside a disk of the given radius (m) centred on the origin."""
    x, y = grid.axes(backend="numpy")
    if supersample == 1:
        return ((x[None, :] ** 2 + y[:, None] ** 2) <= radius**2).astype(float)
    offsets = ((np.arange(supersample) + 0.5) / supersample - 0.5) * grid.dx_si
    coverage = np.zeros(grid.shape)
    for oy in offsets:
        for ox in offsets:
            coverage += ((x[None, :] + ox) ** 2 + (y[:, None] + oy) ** 2) <= radius**2
    return coverage / supersample**2


def uniform(grid, mask, power=1.0):
    wf = Wavefront(mask.astype(complex), grid, LAM * u.m)
    return wf.normalized(power * u.W)


# --- Airy pattern (far field and focal plane) --------------------------------------------------

AIRY_ZEROS = (1.21967, 2.23313, 3.23832)
D = 256e-6  # pupil diameter, m
PUPIL = Grid(320, dx=D / 256 * u.m)


def local_minima(profile):
    return [
        i for i in range(1, len(profile) - 1) if profile[i] < profile[i - 1] and profile[i] <= profile[i + 1]
    ]


@pytest.mark.parametrize("method", ["fraunhofer", "auto"])
def test_airy_zeros_in_the_far_field(method):
    wf = uniform(PUPIL, disk(PUPIL, D / 2))
    step = 0.005 * LAM / D
    cut = Grid(801, 1, dx=step * u.rad, x0=400 * step * u.rad)  # θx from 0 to 4 λ/D
    far = propagate(wf, None, output_grid=cut, method=method)
    profile = np.asarray(far.intensity)[0]
    theta = cut.x.to_value(u.rad) / (LAM / D)
    minima = theta[local_minima(profile)]
    assert len(minima) >= 3
    np.testing.assert_allclose(minima[:3], AIRY_ZEROS, rtol=5e-3)
    for zero in minima[:3]:
        assert profile[np.argmin(np.abs(theta - zero))] < 1e-3 * profile[0]


@pytest.mark.parametrize("method", ["fraunhofer", "auto"])
def test_airy_peak_intensity(method):
    mask = disk(PUPIL, D / 2)
    wf = uniform(PUPIL, mask, power=2.0)
    area = mask.sum() * PUPIL.dx_si * PUPIL.dy_si
    centre = Grid(1, dx=1e-9 * u.rad)
    far = propagate(wf, None, output_grid=centre, method=method)
    # |U(0)|² = |Σ U dA|² / λ² = P A / λ² for a uniform pupil
    assert float(np.asarray(far.intensity)[0, 0]) == pytest.approx(2.0 * area / LAM**2, rel=1e-10)
    # and A matches the continuous disk area to the digitisation error (Gauss circle problem, < 1e-3)
    assert area == pytest.approx(math.pi * D**2 / 4, rel=1e-3)


@pytest.mark.parametrize("method", ["fraunhofer", "auto"])
def test_airy_first_zero_in_the_focal_plane(method):
    f = 0.5  # m
    wf = uniform(PUPIL, disk(PUPIL, D / 2))
    step = 0.005 * LAM * f / D
    cut = Grid(401, 1, dx=step * u.m, x0=200 * step * u.m)  # x from 0 to 2 λf/D
    image = focus(wf, f * u.m, cut, method=method)
    profile = np.asarray(image.intensity)[0]
    x = cut.x.to_value(u.m) / (LAM * f / D)
    assert x[local_minima(profile)][0] == pytest.approx(AIRY_ZEROS[0], rel=5e-3)


def test_focal_plane_energy_is_conserved_on_a_large_grid():
    f = 0.5
    wf = uniform(PUPIL, disk(PUPIL, D / 2), power=1.0)
    # Nyquist-sampled focal grid covering the whole band of the pupil sampling (DFT-matched): exact Parseval.
    dx_focal = LAM * f / (PUPIL.nx * PUPIL.dx_si)
    image = focus(wf, f * u.m, Grid(PUPIL.nx, dx=dx_focal * u.m), method="fraunhofer")
    assert image.power.to_value(u.W) == pytest.approx(1.0, rel=1e-10)


# --- Square aperture ----------------------------------------------------------------------------


def test_square_aperture_far_field_is_the_sampled_dirichlet_kernel():
    grid = Grid(256, dx=1e-6 * u.m)
    mask = np.zeros(grid.shape)
    mask[64:192, 64:192] = 1.0  # side a = 128 dx
    wf = Wavefront(mask.astype(complex), grid, LAM * u.m)
    a, dx = 128e-6, 1e-6
    out = Grid(61, 41, dx=0.13 * LAM / a * u.rad, x0=0.05 * LAM / a * u.rad, y0=-0.4 * LAM / a * u.rad)
    far = propagate(wf, None, output_grid=out, method="fraunhofer")
    ux = out.x.to_value(u.rad) / LAM
    uy = out.y.to_value(u.rad) / LAM

    def dirichlet(freq):
        with np.errstate(invalid="ignore", divide="ignore"):
            value = dx * np.sin(np.pi * freq * a) / np.sin(np.pi * freq * dx)
        return np.where(np.abs(freq) < 1e-12, a, value)

    expected = (np.abs(dirichlet(uy))[:, None] * np.abs(dirichlet(ux))[None, :]) ** 2 / LAM**2
    np.testing.assert_allclose(np.asarray(far.intensity), expected, rtol=1e-9, atol=1e-9 * expected.max())

    # Continuous sinc², within the midpoint-quadrature bound (π u dx)²/6 per axis (doubled for |.|², 2 axes).
    sinc = (a * np.sinc(uy * a))[:, None] * (a * np.sinc(ux * a))[None, :]
    bound = 4 * ((np.pi * np.abs(ux)[None, :] * dx) ** 2 + (np.pi * np.abs(uy)[:, None] * dx) ** 2) / 6
    continuous = sinc**2 / LAM**2
    assert np.all(
        np.abs(np.asarray(far.intensity) - continuous) <= bound * continuous + 1e-9 * continuous.max()
    )


# --- Gaussian beam (Fresnel regime) -------------------------------------------------------------

W0 = 20e-6
ZR = math.pi * W0**2 / LAM
BEAM_GRID = Grid(256, dx=2e-6 * u.m)


def gaussian_beam(grid, z, power=1.0):
    """Analytic paraxial Gaussian beam (Siegman §17.1), complex amplitude on ``grid`` at distance z."""
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    r2 = xx**2 + yy**2
    w = W0 * math.sqrt(1 + (z / ZR) ** 2)
    curvature = 0.0 if z == 0 else 1 / (z * (1 + (ZR / z) ** 2))
    gouy = math.atan2(z, ZR)
    amplitude = math.sqrt(2 * power / math.pi) / w * np.exp(-r2 / w**2)
    return amplitude * np.exp(1j * (K * z + K * r2 * curvature / 2 - gouy))


# Domains of validity on BEAM_GRID (N dx = L = 512 µm, dx = 2 µm), with z_c = L dx / λ = 1.02 mm = 0.82 zR
# (Voelz, Computational Fourier Optics, §5.3): the impulse-response kernel is aliased and the one-step MFT
# output window λz/dx is smaller than the grid for z < z_c. The transfer function is valid at all three
# distances because its chirp is sampled over the narrow spectrum of the beam (aliasing starts at
# f = 1/(2 λ z Δf) ≥ 1.0e5 m⁻¹ where the spectrum is below exp(-41)); the angular spectrum is exact.
GAUSSIAN_CASES = [
    *[("fresnel_tf", r) for r in (0.5, 1.0, 2.0)],
    *[("fresnel_ir", r) for r in (1.0, 2.0)],
    *[("fresnel_mft", r) for r in (1.0, 2.0)],
    *[("asm", r) for r in (0.5, 1.0, 2.0)],
    *[("auto", r) for r in (0.5, 1.0, 2.0)],
]


@pytest.mark.parametrize(("method", "z_over_zr"), GAUSSIAN_CASES)
def test_gaussian_beam_propagation(method, z_over_zr):
    z = z_over_zr * ZR
    wf = Wavefront(gaussian_beam(BEAM_GRID, 0.0), BEAM_GRID, LAM * u.m)
    out = propagate(wf, z * u.m, method=method)
    expected = gaussian_beam(out.grid, z)
    peak = np.abs(expected).max()
    np.testing.assert_allclose(as_numpy(out), expected, atol=1e-3 * peak)


@pytest.mark.parametrize("method", ["rs", "fresnel_mft", "asm", "auto"])
def test_gaussian_beam_on_another_output_grid(method):
    z = ZR
    wf = Wavefront(gaussian_beam(BEAM_GRID, 0.0), BEAM_GRID, LAM * u.m)
    out_grid = Grid(24, 20, dx=5e-6 * u.m, x0=7e-6 * u.m, y0=-3e-6 * u.m)  # coarser, shifted, rectangular
    out = propagate(wf, z * u.m, output_grid=out_grid, method=method)
    assert out.grid == out_grid
    expected = gaussian_beam(out_grid, z)
    np.testing.assert_allclose(as_numpy(out), expected, atol=1e-3 * np.abs(expected).max())


@pytest.mark.parametrize("method", ["fresnel_tf", "asm", "auto"])
def test_gaussian_waist_and_gouy_phase(method):
    z = 1.5 * ZR
    wf = Wavefront(gaussian_beam(BEAM_GRID, 0.0), BEAM_GRID, LAM * u.m)
    out = propagate(wf, z * u.m, method=method)
    intensity = np.asarray(out.intensity)
    xx, _ = (np.asarray(c) for c in BEAM_GRID.coords(backend="numpy"))
    second_moment = (intensity * xx**2).sum() / intensity.sum()
    assert 2 * math.sqrt(second_moment) == pytest.approx(W0 * math.sqrt(1 + 1.5**2), rel=1e-3)
    iy, ix = BEAM_GRID.center_index
    on_axis = as_numpy(out)[iy, ix] * np.exp(-1j * K * z)
    assert np.angle(on_axis) == pytest.approx(-math.atan(1.5), abs=1e-3)


# --- Near field of a circular aperture (exact RS1 on axis) -------------------------------------


@pytest.mark.parametrize("method", ["rs", "asm", "auto"])
@pytest.mark.parametrize("z", [1e-3, 2e-3, 4e-3])  # edge phase step k a dx / z <= 0.31 rad
def test_on_axis_field_of_a_circular_aperture(method, z):
    grid = Grid(512, dx=1e-6 * u.m)
    a = 50e-6
    wf = Wavefront(disk(grid, a, supersample=16).astype(complex), grid, LAM * u.m)  # U0 = 1
    on_axis = Grid(1, dx=1e-6 * u.m)
    out = propagate(wf, z * u.m, output_grid=on_axis, method=method)
    r = math.sqrt(z**2 + a**2)
    expected = np.exp(1j * K * z) - z / r * np.exp(1j * K * r)
    assert abs(as_numpy(out)[0, 0] - expected) < 0.02


# --- Direct Rayleigh-Sommerfeld integral on a small case ---------------------------------------


def rs_reference(field, grid_in, grid_out, z):
    """Explicit Rayleigh-Sommerfeld (first kind) sum, written independently of LUMOS (Goodman eq. 3-42)."""
    xi, yi = (np.asarray(c).ravel() for c in grid_in.coords(backend="numpy"))
    xo, yo = (np.asarray(c).ravel() for c in grid_out.coords(backend="numpy"))
    out = np.zeros(xo.size, dtype=complex)
    for n in range(xo.size):
        r = np.sqrt((xo[n] - xi) ** 2 + (yo[n] - yi) ** 2 + z**2)
        kernel = z / (2 * np.pi * r**2) * (1 / r - 1j * K) * np.exp(1j * K * r)
        out[n] = np.sum(field.ravel() * kernel) * grid_in.dx_si * grid_in.dy_si
    return out.reshape(grid_out.shape)


def test_rs_matches_the_explicit_integral_between_arbitrary_grids():
    rng = np.random.default_rng(4)
    grid_in = Grid(24, 20, dx=1.5e-6 * u.m)
    field = rng.normal(size=grid_in.shape) + 1j * rng.normal(size=grid_in.shape)
    grid_out = Grid(9, 7, dx=4e-6 * u.m, x0=10e-6 * u.m, y0=-5e-6 * u.m)
    z = 40e-6
    wf = Wavefront(field, grid_in, LAM * u.m)
    out = propagate(wf, z * u.m, output_grid=grid_out, method="rs")
    expected = rs_reference(field, grid_in, grid_out, z)
    np.testing.assert_allclose(as_numpy(out), expected, rtol=1e-10, atol=1e-12 * np.abs(expected).max())


# --- Parseval, back-propagation, reciprocity ---------------------------------------------------


def random_smooth_field(grid, seed):
    rng = np.random.default_rng(seed)
    field = rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)
    # Low-pass so that the field is well inside the propagating band and away from the window edges.
    spectrum = np.fft.fft2(field)
    fy, fx = np.meshgrid(np.fft.fftfreq(grid.ny), np.fft.fftfreq(grid.nx), indexing="ij")
    spectrum[np.hypot(fx, fy) > 0.15] = 0
    smooth = np.fft.ifft2(spectrum)
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    window = np.exp(-((xx / (0.25 * grid.nx * grid.dx_si)) ** 8 + (yy / (0.25 * grid.ny * grid.dy_si)) ** 8))
    return smooth * window


@pytest.mark.parametrize("method", ["asm", "fresnel_tf"])
def test_unitary_transfer_functions_conserve_the_power(method):
    grid = Grid(128, dx=2e-6 * u.m)
    wf = Wavefront(random_smooth_field(grid, 5), grid, LAM * u.m)
    out = propagate(wf, 50e-6 * u.m, method=method)
    assert out.power.to_value(u.W) == pytest.approx(wf.power.to_value(u.W), rel=1e-10)


def test_dft_matched_far_field_conserves_the_power():
    grid = Grid(128, dx=2e-6 * u.m)
    wf = Wavefront(random_smooth_field(grid, 6), grid, LAM * u.m)
    far_grid = Grid(128, dx=LAM / (128 * 2e-6) * u.rad)  # dθ = λ / (N dx): the DFT frequencies
    far = propagate(wf, None, output_grid=far_grid, method="fraunhofer")
    assert far.power.to_value(u.W) == pytest.approx(wf.power.to_value(u.W), rel=1e-10)


@pytest.mark.parametrize("method", ["asm", "fresnel_tf", "auto"])
def test_back_propagation_restores_the_field(method):
    grid = Grid(128, dx=2e-6 * u.m)
    field = random_smooth_field(grid, 7)
    wf = Wavefront(field, grid, LAM * u.m)
    back = propagate(propagate(wf, 80e-6 * u.m, method=method), -80e-6 * u.m, method=method)
    np.testing.assert_allclose(as_numpy(back), field, atol=1e-9 * np.abs(field).max())


@pytest.mark.parametrize("method", ["rs", "asm"])
def test_free_space_propagation_is_reciprocal(method):
    grid = Grid(64, dx=2e-6 * u.m)
    p1, p2 = (20, 30), (40, 25)  # (iy, ix)
    z = 60e-6

    def response(source, target):
        field = np.zeros(grid.shape, dtype=complex)
        field[source] = 1.0
        out = propagate(Wavefront(field, grid, LAM * u.m), z * u.m, method=method)
        return as_numpy(out)[target]

    assert response(p1, p2) == pytest.approx(response(p2, p1), rel=1e-10)


def test_unknown_method_is_refused():
    wf = Wavefront(np.ones(BEAM_GRID.shape), BEAM_GRID, LAM * u.m)
    with pytest.raises(ValueError, match="method"):
        propagate(wf, 1 * u.mm, method="magic")


def test_distance_must_be_a_length():
    wf = Wavefront(np.ones(BEAM_GRID.shape), BEAM_GRID, LAM * u.m)
    with pytest.raises(TypeError):
        propagate(wf, 1e-3)
    with pytest.raises(u.UnitsError):
        propagate(wf, 1 * u.s)
