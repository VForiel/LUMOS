r"""Fizeau recombination of the beams of a telescope array (step P15.9, ADR-0023).

Specification: docs/specs/pistes_evolution.md §1, step 5. A ``FizeauCombiner`` receives a bus of pupil-plane
beams (from a ``TelescopeArray``, possibly after per-arm optics) and places each lane at the projected
position of its telescope in **one** output pupil, at the scale of the input pupils (a homothetic exit pupil,
the condition of Traub 1986, J. Opt. Soc. Am. A 3, 1491, which keeps a wide field). The output is a single
``IncoherentBundle`` that the propagation elements bring to a focal plane.

Independent references:

- two circular apertures of diameter D whose centres are B apart give in the focal plane of a lens of focal
  f the Airy pattern of one aperture modulated by Young fringes (Born & Wolf, *Principles of Optics*, §8.6):
  :math:`I(x) \propto [2J_1(\pi D x/\lambda f)/(\pi D x/\lambda f)]^2 \cos^2(\pi B x/\lambda f)`;
- placing the lanes moves light without creating or losing any: the output power is the sum of the powers of
  the lanes (non-overlapping apertures), for any sub-pixel position;
- with the homothetic pupil, the field of an off-axis source is one plane wave across the output pupil, so the
  fringes move with the Airy envelope: the image of a source at angle θ is the on-axis image shifted by f θ.
"""

import math

import numpy as np
import pytest
from astropy import units as u
from scipy import special

from lumos import Context
from lumos.elements.buses import Demux, Mux
from lumos.elements.optics import FreeSpace, ThinLens
from lumos.elements.sources import PointSource
from lumos.elements.telescopes import FizeauCombiner, TelescopeArray
from lumos.fields import Grid, IncoherentBundle

LAM = 1e-6
D, B, F = 1.0, 3.0, 20.0


def fizeau_context(positions, *, theta_x=0.0, n_pixels=128, focal=None):
    ctx = Context()
    ctx.observation.band = (0.999 * u.um, 1.001 * u.um)
    star = ctx.add(PointSource(name="star", flux_density=1 * u.Jy, theta_x=theta_x * u.rad))
    array = ctx.add(
        TelescopeArray(
            name="array", frame="sky", positions=positions * u.m, diameter=D * u.m, n_pixels=n_pixels
        )
    )
    combiner = ctx.add(FizeauCombiner(name="combiner"))
    ctx.connect(star.port("out"), array.port("in"))
    ctx.connect(array.port("out"), combiner.port("in"))
    if focal is not None:
        lens = ctx.add(ThinLens(name="lens", focal_length=F * u.m))
        image = ctx.add(FreeSpace(name="image", distance=F * u.m, method="fresnel_mft", output_grid=focal))
        ctx.connect(combiner.port("out"), lens.port("in"))
        ctx.connect(lens.port("out"), image.port("in"))
    return ctx


def test_the_output_is_one_pupil_holding_every_aperture():
    ctx = fizeau_context([[-B / 2, 0.0], [B / 2, 0.0]])
    result = ctx.run(["combiner.out", "array.out"])
    pupil = result["combiner.out"]
    assert isinstance(pupil, IncoherentBundle)
    assert pupil.sources == result["array.out"][0].sources
    assert pupil.grid.dx_si == pytest.approx(result["array.out"][0].grid.dx_si)
    width = pupil.grid.nx * pupil.grid.dx_si
    assert width >= B + D


@pytest.mark.parametrize("offset", [0.0, 0.37, 1.61])
def test_the_power_of_the_lanes_is_kept(offset):
    ctx = fizeau_context([[-B / 2 + offset * 0.01, 0.0], [B / 2, 0.3 + offset * 0.013]])
    result = ctx.run(["combiner.out", "array.out"])
    total = sum(float(lane.total_power.value) for lane in result["array.out"])
    assert float(result["combiner.out"].total_power.value) == pytest.approx(total, rel=1e-6)


def airy_fringes(x, lam=LAM):
    a = math.pi * D * x / (lam * F)
    envelope = np.where(np.abs(a) < 1e-12, 1.0, (2 * special.j1(a) / np.where(a == 0, 1, a)) ** 2)
    return envelope * np.cos(math.pi * B * x / (lam * F)) ** 2


def test_two_apertures_give_young_fringes_under_the_airy_pattern():
    pixel = LAM * F / D / 8  # 8 pixels per λ/D: fringes of period λF/B, 8/3 pixels
    focal = Grid(129, 1, dx=pixel * u.m)
    ctx = fizeau_context([[-B / 2, 0.0], [B / 2, 0.0]], n_pixels=256, focal=focal)
    image = ctx.run(["image.out"])["image.out"]
    intensity = np.asarray(image.intensity, dtype=float).reshape(-1)
    x = np.asarray(focal.axes(backend="numpy")[0], dtype=float)
    centre = int(np.argmin(np.abs(x)))
    measured = intensity / intensity[centre]
    inside = np.abs(x) < 1.2 * LAM * F / D  # the central lobe of the Airy pattern
    np.testing.assert_allclose(measured[inside], airy_fringes(x[inside]), atol=1e-2)


def test_an_off_axis_source_moves_the_whole_image():
    pixel = LAM * F / D / 8
    focal = Grid(129, 1, dx=pixel * u.m)
    shift_pixels = 5
    theta = shift_pixels * pixel / F
    on = fizeau_context([[-B / 2, 0.0], [B / 2, 0.0]], n_pixels=256, focal=focal)
    off = fizeau_context([[-B / 2, 0.0], [B / 2, 0.0]], n_pixels=256, focal=focal, theta_x=theta)
    a = np.asarray(on.run(["image.out"])["image.out"].intensity, dtype=float).reshape(-1)
    b = np.asarray(off.run(["image.out"])["image.out"].intensity, dtype=float).reshape(-1)
    np.testing.assert_allclose(
        b[20 + shift_pixels : 110 + shift_pixels], a[20:110], rtol=0, atol=2e-3 * a.max()
    )


def test_lanes_without_positions_or_of_different_sources_are_refused():
    ctx = fizeau_context([[-B / 2, 0.0], [B / 2, 0.0]])
    bus = ctx.run(["array.out"])["array.out"]
    parts = Demux(n_lanes=2).apply(**{"in": bus})
    plain = Mux(n_lanes=2).apply(in0=parts["out0"][0], in1=parts["out1"][0])["out"]  # positions lost
    with pytest.raises(ValueError, match="position"):
        FizeauCombiner().apply(**{"in": plain})
