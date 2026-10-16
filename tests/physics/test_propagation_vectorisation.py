"""Vectorised propagation over polarisation, wavelengths and modes; kernel cache; chunking (step P4.7).

Requirements: REQ-SIG-4/5 (propagation vectorised over the wavelength and mode axes), REQ-HPC-4 (automatic
chunking when memory is insufficient), REQ-HPC-1 (identical results on every backend).

References: free-space propagation is linear and acts independently on each Jones component (the scalar
propagator applies to Ex and Ey separately in the paraxial and angular-spectrum descriptions, Goodman §3.10),
on each wavelength (with its own wavenumber) and on each mutually incoherent mode. Results are therefore
compared with the monochromatic scalar propagation of each component, channel and mode, which the
validation suite (P4.1) checks against analytic solutions.
"""

import importlib.util
import math

import numpy as np
import pytest
from astropy import units as u

from lumos import backend
from lumos.fields import Grid, IncoherentBundle, Wavecube, Wavefront
from lumos.propagation import cache_info, clear_cache, propagate

HAS_JAX = importlib.util.find_spec("jax") is not None
GRID = Grid(64, dx=2e-6 * u.m)
OTHER = Grid(20, 16, dx=3e-6 * u.m, x0=4e-6 * u.m)
ANGULAR = Grid(24, dx=4e-3 * u.rad)
CASES = [
    ("asm", 0.2 * u.mm, GRID),
    ("asm", 0.3 * u.mm, OTHER),
    ("fresnel_tf", 0.2 * u.mm, GRID),
    ("fresnel_mft", 1.0 * u.mm, OTHER),
    ("rs", 0.2 * u.mm, OTHER),
    ("fraunhofer", None, ANGULAR),
]


def field(seed, grid=GRID):
    rng = np.random.default_rng(seed)
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    envelope = np.exp(-(xx**2 + yy**2) / (20e-6) ** 2)
    return envelope * (1 + 0.3 * rng.normal(size=grid.shape)) * np.exp(1j * rng.normal(size=grid.shape) * 0.2)


def numpy_field(signal):
    return backend.to_numpy(signal.field)


@pytest.mark.parametrize(("method", "distance", "out_grid"), CASES)
def test_jones_components_propagate_independently(method, distance, out_grid):
    ex, ey = field(1), field(2)
    polarised = Wavefront(np.stack([ex, ey]), GRID, 1 * u.um)
    out = propagate(polarised, distance, output_grid=out_grid, method=method)
    for component, source in enumerate((ex, ey)):
        scalar = propagate(Wavefront(source, GRID, 1 * u.um), distance, output_grid=out_grid, method=method)
        np.testing.assert_allclose(numpy_field(out)[component], numpy_field(scalar), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(("method", "distance", "out_grid"), CASES)
def test_wavecube_channels_propagate_with_their_own_wavelength(method, distance, out_grid):
    lams = [0.8, 1.0, 1.3] * u.um
    channels = [field(k) for k in range(3)]
    cube = Wavecube(np.stack(channels), GRID, lams, weights=[0.2, 0.5, 0.3])
    out = propagate(cube, distance, output_grid=out_grid, method=method)
    assert isinstance(out, Wavecube)
    np.testing.assert_array_equal(out.weights, [0.2, 0.5, 0.3])
    for k in range(3):
        mono = propagate(Wavefront(channels[k], GRID, lams[k]), distance, output_grid=out_grid, method=method)
        np.testing.assert_allclose(numpy_field(out)[k], numpy_field(mono), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(("method", "distance", "out_grid"), CASES)
def test_bundle_modes_propagate_independently(method, distance, out_grid):
    members = [Wavefront(field(k), GRID, 1 * u.um) for k in range(4)]
    bundle = IncoherentBundle.from_signals(members, weights=[1, 2, 3, 4])
    out = propagate(bundle, distance, output_grid=out_grid, method=method)
    assert isinstance(out, IncoherentBundle)
    np.testing.assert_array_equal(out.mode_weights, [1, 2, 3, 4])
    for k, member in enumerate(members):
        mono = propagate(member, distance, output_grid=out_grid, method=method)
        np.testing.assert_allclose(numpy_field(out)[k], numpy_field(mono), rtol=1e-12, atol=1e-14)


def test_polychromatic_bundle():
    lams = [1.0, 1.2] * u.um
    bundle = IncoherentBundle.point_sources(
        GRID, lams, theta_x=[0, 0.01] * u.rad, theta_y=[0, 0] * u.rad, power=[1, 1] * u.W
    )
    out = propagate(bundle, None, output_grid=ANGULAR, method="fraunhofer")
    assert out.shape == (2, 2, *ANGULAR.shape)
    for m in range(2):
        for k in range(2):
            mono = Wavefront(numpy_field(bundle)[m, k], GRID, lams[k])
            ref = propagate(mono, None, output_grid=ANGULAR, method="fraunhofer")
            np.testing.assert_allclose(numpy_field(out)[m, k], numpy_field(ref), rtol=1e-12, atol=1e-14)


@pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")
@pytest.mark.parametrize(("method", "distance", "out_grid"), CASES)
def test_numpy_and_jax_give_the_same_field(method, distance, out_grid):
    wf = Wavefront(field(3), GRID, 1 * u.um)
    out_np = propagate(wf, distance, output_grid=out_grid, method=method)
    out_jax = propagate(wf.to_backend("jax"), distance, output_grid=out_grid, method=method)
    assert out_jax.backend == "jax"
    np.testing.assert_allclose(numpy_field(out_jax), numpy_field(out_np), rtol=1e-10, atol=1e-12)


def test_kernels_are_cached():
    clear_cache()
    wf = Wavefront(field(4), GRID, 1 * u.um)
    propagate(wf, 0.25 * u.mm, method="asm")
    first = cache_info()
    propagate(wf * 2.0, 0.25 * u.mm, method="asm")
    second = cache_info()
    assert second.hits > first.hits
    assert second.misses == first.misses
    clear_cache()
    assert cache_info().currsize == 0


def test_modes_are_chunked_under_a_memory_budget():
    members = [Wavefront(field(k), GRID, 1 * u.um) for k in range(6)]
    bundle = IncoherentBundle.from_signals(members)
    reference = propagate(bundle, 0.2 * u.mm, method="asm")
    one_mode = 16 * 4 * GRID.size  # bytes of one padded complex128 mode
    chunked = propagate(bundle, 0.2 * u.mm, method="asm", max_memory=3 * one_mode)
    np.testing.assert_allclose(numpy_field(chunked), numpy_field(reference), rtol=1e-12, atol=1e-14)
    assert chunked.n_modes == 6


def test_wavelength_dependence_of_the_far_field_scale():
    # The far field of the same pupil scales as λ: the first Airy-like minimum moves proportionally.
    grid = Grid(128, dx=1e-6 * u.m)
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    pupil = (xx**2 + yy**2 <= (32e-6) ** 2).astype(float)
    cube = Wavecube(np.stack([pupil, pupil]), grid, [1.0, 2.0] * u.um)
    cut = Grid(1600, 1, dx=5e-5 * u.rad, x0=800 * 5e-5 * u.rad)  # θ from 0 to 0.08, step 0.3 % of 1.22 λ/D
    out = propagate(cube, None, output_grid=cut, method="fraunhofer")
    profiles = np.asarray(out.intensity)[:, 0]
    theta = cut.x.to_value(u.rad)
    first_min = [theta[1:-1][(p[1:-1] < p[:-2]) & (p[1:-1] <= p[2:])][0] for p in profiles]
    assert first_min[1] / first_min[0] == pytest.approx(2.0, rel=0.01)
    assert math.isfinite(first_min[0])
