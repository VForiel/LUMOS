"""Tests for coronagraphs (step P6.6).

Requirements: §5.9 (four-quadrant phase mask, vortex and Lyot coronagraphs), objective O1 (rejection of ideal
coronagraphs), REQ-EL-8 (transfers on every coherent signal).

Independent references:

- an even-charge vortex (Mawet et al. 2005, ApJ 633, 1191; Foo et al. 2005, Opt. Lett. 30, 3308) and a
  four-quadrant phase mask (Rouan et al. 2000, PASP 112, 1479) send all the on-axis light of an unobstructed
  circular pupil outside the geometric pupil: the light through a Lyot stop smaller than the pupil vanishes.
  With a pixelated pupil the residual is a sampling error that must decrease as the sampling is refined;
  at 256 samples across the pupil it must stay below 1e-5 of the light through the stop without a mask;
- off-axis sources far from the axis (6 λ/D) are transmitted (Mawet et al. 2005, Fig. 2; Rouan et al. 2000);
- a quadrant mask whose phase step is φ instead of π is ``(1 + e^{iφ})/2 + (1 - e^{iφ})/2 · S`` with S an
  ideal mask, so it leaks the fraction ``cos²(φ/2)`` of the light; a non-achromatic step designed for λ0 gives
  ``φ = π λ0 / λ``;
- Babinet's principle with the Airy amplitude ``(π/4) 2 J1(π u)/(π u)`` of a unit pupil: behind an opaque disk
  of radius m λ/D the field at the centre of the Lyot plane is ``J0(π m)`` times the incident field
  (``∫_0^m 2π u (π/4) 2 J1(π u)/(π u) du = 1 - J0(π m)``); J0 from SciPy. A disk of physical radius
  ``m0 λ0 / D`` has the radius ``m0 λ0 / λ`` in units of λ/D at λ.
"""

import math

import numpy as np
import pytest
from astropy import units as u
from scipy.special import j0

from lumos.elements.coronagraphs import FourQuadrantPhaseMask, LyotCoronagraph, VortexCoronagraph
from lumos.fields import Grid, Wavecube, Wavefront
from lumos.physics.apertures import disk

D = 1.0


def setup(n=256):
    grid = Grid(int(n * 1.1) // 2 * 2, dx=D / n * u.m)
    pupil = disk(grid, D, supersample=8)
    stop = disk(grid, 0.95 * D, supersample=8)
    return grid, pupil, stop


def pupil_field(grid, pupil, wavelength=1 * u.um, offset=(0.0, 0.0)):
    """Uniform pupil lit by a source at ``offset`` (in λ/D)."""
    wf = Wavefront.plane(grid, wavelength, power=1 * u.W) * pupil
    lam_over_d = wavelength.to_value(u.m) / D
    return wf.tilt(offset[0] * lam_over_d * u.rad, offset[1] * lam_over_d * u.rad)


def through_stop(field, stop):
    return float(np.sum(np.abs(np.asarray(field) * stop) ** 2))


def leak(element, n=256, wavelength=1 * u.um, offset=(0.0, 0.0)):
    grid, pupil, stop = setup(n)
    wf = pupil_field(grid, pupil, wavelength, offset)
    out = element.apply(**{"in": wf})["out"]
    return through_stop(out.field, stop) / through_stop(wf.field, stop)


def vortex(charge, **kw):
    return VortexCoronagraph(charge=charge, pupil_diameter=D * u.m, lyot_stop=0.95, **kw)


def fqpm(**kw):
    return FourQuadrantPhaseMask(pupil_diameter=D * u.m, lyot_stop=0.95, **kw)


# --- Ideal rejection ---------------------------------------------------------------------------------


@pytest.mark.parametrize("charge", [2, 4, 6])
def test_even_charge_vortex_rejects_on_axis_light(charge):
    assert leak(vortex(charge)) < 1e-5


def test_four_quadrant_phase_mask_rejects_on_axis_light():
    assert leak(fqpm()) < 1e-5


@pytest.mark.parametrize("make", [lambda: vortex(2), fqpm])
def test_rejection_converges_with_the_sampling(make):
    assert leak(make(), n=256) < leak(make(), n=128) / 3


def test_odd_charge_vortex_is_not_a_perfect_coronagraph():
    assert leak(vortex(1)) > 0.1


# --- Off-axis transmission ---------------------------------------------------------------------------


def test_vortex_transmits_off_axis_sources():
    assert leak(vortex(2), offset=(6.0, 0.0)) > 0.9


def test_four_quadrant_mask_transmits_off_axis_sources_on_the_diagonals():
    assert leak(fqpm(), offset=(4.24, 4.24)) > 0.9


# --- Chromaticity ------------------------------------------------------------------------------------


def test_non_achromatic_quadrant_step_leaks_cos_squared():
    element = fqpm(achromatic=False, reference_wavelength=1 * u.um)
    phi = math.pi / 1.1
    assert leak(element, wavelength=1.1 * u.um) == pytest.approx(math.cos(phi / 2) ** 2, rel=1e-2)
    assert leak(fqpm(achromatic=True, reference_wavelength=1 * u.um), wavelength=1.1 * u.um) < 1e-5


def test_masks_act_on_every_channel_of_a_wavecube():
    grid, pupil, stop = setup()
    cube = Wavecube.plane(grid, [1.0, 1.5] * u.um, power=1 * u.W) * pupil
    out = vortex(2).apply(**{"in": cube})["out"]
    for k in range(2):
        assert through_stop(out.field[k], stop) / through_stop(cube.field[k], stop) < 1e-5


# --- Lyot coronagraph --------------------------------------------------------------------------------


@pytest.mark.parametrize("radius", [1.5, 2.5, 3.0])
def test_lyot_field_at_the_centre_follows_babinet(radius):
    grid, pupil, _ = setup()
    wf = pupil_field(grid, pupil)
    element = LyotCoronagraph(mask_radius=radius, pupil_diameter=D * u.m, reference_wavelength=1 * u.um)
    out = np.asarray(element.apply(**{"in": wf})["out"].field)
    centre = (grid.shape[0] // 2, grid.shape[1] // 2)
    ratio = out[centre] / np.asarray(wf.field)[centre]
    assert ratio.real == pytest.approx(j0(math.pi * radius), abs=2e-3)
    assert abs(ratio.imag) < 2e-3


def test_lyot_mask_has_a_physical_size():
    grid, pupil, _ = setup()
    wf = pupil_field(grid, pupil, wavelength=2 * u.um)
    element = LyotCoronagraph(mask_radius=5.0, pupil_diameter=D * u.m, reference_wavelength=1 * u.um)
    out = np.asarray(element.apply(**{"in": wf})["out"].field)
    centre = (grid.shape[0] // 2, grid.shape[1] // 2)
    assert (out[centre] / np.asarray(wf.field)[centre]).real == pytest.approx(j0(math.pi * 2.5), abs=2e-3)


def test_lyot_stop_blocks_the_light_outside_it():
    grid, pupil, _ = setup()
    out = np.asarray(vortex(2).apply(**{"in": pupil_field(grid, pupil)})["out"].field)
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    outside = np.hypot(xx, yy) > 0.95 * D / 2 + grid.dx_si
    assert np.all(out[outside] == 0)
