"""Signals held by a GPU backend accept NumPy masks and factors (steps P3.2-P3.7, REQ-HPC-1).

Masks and transmission maps are usually built with NumPy (apertures, Zernike maps, detector curves). CuPy
refuses to mix its arrays with NumPy arrays, so the signals convert such operands to their own backend.

Reference: the product computed with NumPy on the same data.
"""

import importlib.util

import numpy as np
import pytest
from astropy import units as u

from lumos import backend
from lumos.fields import Grid, IncoherentBundle, ModalField, RadianceField, Wavecube, Wavefront

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="CuPy is not installed"),
]

GRID = Grid(8, dx=1 * u.um)
MASK = np.linspace(0.0, 1.0, 64).reshape(8, 8)


def on_gpu(signal):
    return signal.to_backend("cupy")


def test_wavefront_times_numpy_mask():
    wf = Wavefront.plane(GRID, 1 * u.um, power=1 * u.W)
    out = on_gpu(wf) * MASK
    assert out.backend == "cupy"
    np.testing.assert_allclose(backend.to_numpy(out.field), backend.to_numpy(wf.field) * MASK)
    np.testing.assert_allclose(backend.to_numpy((MASK * on_gpu(wf)).field), backend.to_numpy(out.field))


def test_wavecube_times_numpy_masks():
    cube = Wavecube.plane(GRID, [1.0, 2.0] * u.um, power=1 * u.W)
    chromatic = np.stack([MASK, 1 - MASK])
    for mask in (MASK, chromatic):
        out = on_gpu(cube) * mask
        assert out.backend == "cupy"
        np.testing.assert_allclose(backend.to_numpy(out.field), backend.to_numpy(cube.field) * mask)


def test_bundle_times_numpy_mask():
    bundle = IncoherentBundle.point_sources(
        GRID, 1 * u.um, theta_x=[0, 0.01] * u.rad, theta_y=[0, 0] * u.rad, power=[1, 1] * u.W
    )
    out = on_gpu(bundle) * MASK
    assert out.backend == "cupy"
    np.testing.assert_allclose(backend.to_numpy(out.field), backend.to_numpy(bundle.field) * MASK)


def test_radiance_times_numpy_mask():
    radiance = RadianceField(np.ones((2, 8, 8)) * u.W / u.m**2 / u.sr / u.m, GRID, [1.0, 2.0] * u.um)
    out = radiance.to_backend("cupy") * MASK
    assert out.backend == "cupy"
    np.testing.assert_allclose(backend.to_numpy(out.data), np.ones((2, 8, 8)) * MASK)


def test_modal_field_times_numpy_factors():
    modal = ModalField(np.ones((1, 3)), [1.0] * u.um)
    factors = np.array([1.0, 0.5, 0.0])
    out = modal.to_backend("cupy") * factors
    assert out.backend == "cupy"
    np.testing.assert_allclose(backend.to_numpy(out.amplitudes), factors[None])
