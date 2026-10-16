"""Differentiability of the propagation kernels with JAX (step P4.8).

Requirements: REQ-PROP-6 / REQ-EL-10 (pure, differentiable transfers), ADR-0002, and the plan of P4.8 (JAX
gradients checked against finite differences).

References:

- central finite differences, ``f'(x) ≈ [f(x + h) - f(x - h)] / 2h`` with truncation error ``O(h²)``,
  computed with NumPy, independently of JAX's automatic differentiation;
- the on-axis far-field intensity of a uniform pupil with a phase ``α Z(x, y)``,
  ``I(α) = |Σ e^{iαZ} dA|² / λ²``, is stationary at ``α = 0`` and ``I''(0) = -2 N² dA² var(Z) / λ²`` for N
  pupil pixels (second-order expansion of the exponential; Mahajan, *Optical Imaging and Aberrations*, II,
  §1.4: Strehl ratio ≈ 1 - σ²).
"""

import importlib.util
import math

import numpy as np
import pytest
from astropy import units as u

import lumos
from lumos.fields import Grid, Wavefront
from lumos.propagation import propagate

pytestmark = pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX is not installed")

GRID = Grid(48, dx=2e-6 * u.m)
LAM = 1 * u.um
ON_AXIS = Grid(1, dx=1e-6 * u.rad)


def pupil():
    xx, yy = (np.asarray(c) for c in GRID.coords(backend="numpy"))
    r2 = xx**2 + yy**2
    mask = (r2 <= (20e-6) ** 2).astype(float)
    defocus = (2 * r2 / (20e-6) ** 2 - 1) * mask
    return mask, defocus


def far_peak(alpha, mask, aberration, xp):
    field = xp.asarray(mask) * xp.exp(1j * alpha * xp.asarray(aberration))
    out = propagate(Wavefront(field, GRID, LAM), None, output_grid=ON_AXIS, method="fraunhofer")
    return xp.sum(xp.abs(out.field) ** 2)


def test_strehl_is_stationary_and_has_the_analytic_curvature():
    import jax
    import jax.numpy as jnp

    mask, z = pupil()
    with lumos.backend.use("jax"):
        grad = jax.grad(lambda a: far_peak(a, mask, z, jnp))(0.0)
        hess = jax.hessian(lambda a: far_peak(a, mask, z, jnp))(0.0)
    n = mask.sum()
    da = GRID.dx_si * GRID.dy_si
    lam = 1e-6
    values = z[mask > 0]
    expected = -2 * n**2 * da**2 * values.var() / lam**2
    peak = (n * da) ** 2 / lam**2
    assert abs(float(grad)) < 1e-12 * peak
    assert float(hess) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("alpha", [0.3, 1.1])
def test_far_field_gradient_matches_finite_differences(alpha):
    import jax
    import jax.numpy as jnp

    mask, z = pupil()
    with lumos.backend.use("jax"):
        grad = float(jax.grad(lambda a: far_peak(a, mask, z, jnp))(alpha))
    h = 1e-5
    numeric = (far_peak(alpha + h, mask, z, np) - far_peak(alpha - h, mask, z, np)) / (2 * h)
    assert grad == pytest.approx(float(numeric), rel=1e-6)


@pytest.mark.parametrize(
    ("method", "out_grid"),
    [
        ("asm", GRID),
        ("asm", Grid(5, dx=5e-6 * u.m)),
        ("fresnel_mft", Grid(5, dx=5e-6 * u.m)),
        ("fresnel_tf", GRID),
        ("rs", Grid(3, dx=5e-6 * u.m)),
    ],
)
def test_near_field_gradients_match_finite_differences(method, out_grid):
    import jax
    import jax.numpy as jnp

    mask, z = pupil()
    xx, _ = (np.asarray(c) for c in GRID.coords(backend="numpy"))
    distance = 0.3 * u.mm if method != "fresnel_mft" else 2 * u.mm

    def point_intensity(params, xp):
        tilt, scale = params[0], params[1]
        field = scale * xp.asarray(mask) * xp.exp(1j * (tilt * xp.asarray(xx) / 20e-6 + 0.4 * xp.asarray(z)))
        out = propagate(Wavefront(field, GRID, LAM), distance, output_grid=out_grid, method=method)
        return xp.sum(xp.abs(out.field) ** 2)

    params = np.array([0.7, 1.3])
    with lumos.backend.use("jax"):
        grad = np.asarray(jax.grad(lambda p: point_intensity(p, jnp))(jnp.asarray(params)))
    h = 1e-6
    numeric = []
    for i in range(2):
        step = np.zeros(2)
        step[i] = h
        numeric.append((point_intensity(params + step, np) - point_intensity(params - step, np)) / (2 * h))
    np.testing.assert_allclose(grad, np.array(numeric, dtype=float), rtol=1e-5)


def test_propagation_compiles_with_lumos_jit():
    import jax.numpy as jnp

    mask, z = pupil()
    wf = Wavefront(mask * np.exp(0.5j * z), GRID, LAM).to_backend("jax")

    @lumos.jit
    def image(w):
        return propagate(w, None, output_grid=Grid(16, dx=5e-3 * u.rad), method="fraunhofer")

    compiled = image(wf)
    direct = propagate(wf, None, output_grid=Grid(16, dx=5e-3 * u.rad), method="fraunhofer")
    np.testing.assert_allclose(np.asarray(compiled.field), np.asarray(direct.field), rtol=1e-12, atol=1e-12)
    assert math.isfinite(float(jnp.sum(jnp.abs(compiled.field))))
