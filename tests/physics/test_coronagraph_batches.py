"""Coronagraph transforms on batches of fields (step P15.3: performance without changing the results).

Requirements: the Lyot-plane transform of a batch of fields (the sources of an extended scene, the
wavelengths of a band) is computed in pieces and with fewer, larger operations; this must not change it.

Independent references (invariants of the transform, not values computed by the code under test):

- the transform is linear: L(a f + b g) = a L(f) + b L(g);
- a batch is transformed field by field: L([f1, ..., fn])[k] = L(fk), whatever the size of the batch;
- an ideal even-charge vortex removes an on-axis point source from a circular pupil (Mawet et al. 2005,
  ApJ 633, 1191): the field in the Lyot stop is zero up to the residual of the pupil sampling, which
  tests/physics/test_coronagraphs.py checks against its published level.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lumos.physics.coronagraphy import lyot_plane


def vortex(charge):
    def mask(ux, uy):
        return np.exp(1j * charge * np.arctan2(uy, ux))

    return mask


def grid(n):
    x = (np.arange(n) - n // 2) / (n * 0.9)  # pupil diameter 0.9 n samples
    return x, x.copy()


def random_fields(rng, batch, n):
    x, _ = grid(n)
    pupil = (np.hypot(*np.meshgrid(x, x)) <= 0.5).astype(float)
    return pupil * (rng.standard_normal((batch, n, n)) + 1j * rng.standard_normal((batch, n, n)))


@pytest.mark.parametrize("batch", [1, 3, 40])
def test_a_batch_is_transformed_field_by_field(batch):
    rng = np.random.default_rng(batch)
    n = 24
    x, y = grid(n)
    fields = random_fields(rng, batch, n)
    together = lyot_plane(fields, x, y, vortex(2))
    assert together.shape == fields.shape
    scale = np.max(np.abs(together))
    for k in range(batch):
        alone = lyot_plane(fields[k], x, y, vortex(2))
        np.testing.assert_allclose(together[k], alone, rtol=0, atol=1e-11 * scale)


@settings(max_examples=10, deadline=None)
@given(
    a=st.complex_numbers(max_magnitude=10, allow_nan=False, allow_infinity=False),
    b=st.complex_numbers(max_magnitude=10, allow_nan=False, allow_infinity=False),
    seed=st.integers(0, 2**16),
)
def test_the_transform_is_linear(a, b, seed):
    rng = np.random.default_rng(seed)
    n = 16
    x, y = grid(n)
    f, g = random_fields(rng, 2, n)
    left = lyot_plane(a * f + b * g, x, y, vortex(4))
    right = a * lyot_plane(f, x, y, vortex(4)) + b * lyot_plane(g, x, y, vortex(4))
    scale = max(np.max(np.abs(right)), 1e-300)
    np.testing.assert_allclose(left, right, rtol=0, atol=1e-11 * scale)


def test_extra_batch_dimensions_are_kept():
    rng = np.random.default_rng(7)
    n = 16
    x, y = grid(n)
    fields = random_fields(rng, 6, n).reshape(2, 3, n, n)
    together = lyot_plane(fields, x, y, vortex(2))
    assert together.shape == (2, 3, n, n)
    np.testing.assert_allclose(
        together[1, 2],
        lyot_plane(fields[1, 2], x, y, vortex(2)),
        rtol=0,
        atol=1e-11 * np.max(np.abs(together)),
    )
