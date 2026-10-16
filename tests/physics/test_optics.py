"""Tests for bulk optics elements (step P6.3).

Requirements: §5.9 (free space, thin lens and mirror, beam splitter, phase plates, polarisers, wave plates,
spectral filter, delay line), REQ-EL-8 (transfers accept wavefronts, wavecubes, polarised or not, and
incoherent bundles).

Independent references:

- a thin lens of focal length f applies ``exp(-i k r² / 2f)`` (§9 of the specs; Goodman §5.1); a uniform beam
  of diameter D focused by it has its first dark ring at ``1.21967 λ f / D`` (Born & Wolf §8.5.2);
- a collimated Gaussian beam of waist w is focused to a waist ``λ f / (π w)`` in the back focal
  plane (Siegman §17.2);
- a lossless beam splitter with ``t = sqrt(T)``, ``r = i sqrt(R)`` conserves the power; two 50/50 splitters
  with a phase φ between the arms make a Mach-Zehnder with outputs ``sin²(φ/2)`` and ``cos²(φ/2)``;
- a vacuum delay d adds the phase ``2π d / λ`` to each wavelength;
- Malus's law for a polariser element, and a half-wave plate at θ rotating a linear polarisation by 2θ
  (Hecht §8).
"""

import math

import numpy as np
import pytest
from astropy import units as u

from lumos import Context
from lumos.elements.apertures import CircularAperture
from lumos.elements.optics import (
    BeamSplitter,
    DelayLine,
    FreeSpace,
    Polarizer,
    SpectralFilter,
    ThinLens,
    ThinMirror,
    WavePlate,
)
from lumos.fields import Grid, IncoherentBundle, Wavecube, Wavefront, jones
from lumos.propagation import propagate

LAM = 1e-6


def run(element, signal, port="out"):
    return element.apply(**{"in": signal})[port]


def local_minima(profile):
    return [
        i for i in range(1, len(profile) - 1) if profile[i] < profile[i - 1] and profile[i] <= profile[i + 1]
    ]


# --- Lens, mirror, free space --------------------------------------------------------------------


def test_thin_lens_phase():
    grid = Grid(64, dx=10 * u.um)
    wf = Wavefront.plane(grid, LAM * u.m, power=1 * u.W)
    out = run(ThinLens(focal_length=0.1 * u.m), wf)
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    expected = np.exp(-1j * 2 * math.pi / LAM * (xx**2 + yy**2) / (2 * 0.1))
    np.testing.assert_allclose(np.asarray(out.field) / np.asarray(wf.field), expected, atol=1e-12)
    assert out.power.to_value(u.W) == pytest.approx(1.0, rel=1e-12)


def test_mirror_focuses_like_a_lens_of_the_same_focal_length():
    grid = Grid(32, dx=10 * u.um)
    wf = Wavefront.plane(grid, LAM * u.m, power=1 * u.W)
    np.testing.assert_allclose(
        np.asarray(run(ThinMirror(focal_length=0.2 * u.m), wf).field),
        np.asarray(run(ThinLens(focal_length=0.2 * u.m), wf).field),
    )


def test_lens_then_free_space_makes_an_airy_pattern():
    d, f = 2e-3, 0.1
    grid = Grid(512, dx=d / 256 * u.m)
    wf = run(CircularAperture(diameter=d * u.m), Wavefront.plane(grid, LAM * u.m, power=1 * u.W))
    focused = run(ThinLens(focal_length=f * u.m), wf)
    step = 0.01 * LAM * f / d
    cut = Grid(201, 1, dx=step * u.m, x0=100 * step * u.m)
    image = run(FreeSpace(distance=f * u.m, method="fresnel_mft", output_grid=cut), focused)
    profile = np.asarray(image.intensity)[0]
    x = cut.x.to_value(u.m) / (LAM * f / d)
    assert x[local_minima(profile)][0] == pytest.approx(1.21967, rel=1e-2)


def test_gaussian_beam_focal_waist():
    w, f = 1e-3, 0.5
    grid = Grid(512, dx=12e-6 * u.m)
    beam = Wavefront.gaussian(grid, LAM * u.m, waist=w * u.m, power=1 * u.W)
    focused = run(ThinLens(focal_length=f * u.m), beam)
    expected = LAM * f / (math.pi * w)  # 159 um
    focal_grid = Grid(128, dx=expected / 16 * u.m)
    image = run(FreeSpace(distance=f * u.m, method="fresnel_mft", output_grid=focal_grid), focused)
    intensity = np.asarray(image.intensity)
    xx, _ = (np.asarray(c) for c in focal_grid.coords(backend="numpy"))
    waist = 2 * math.sqrt((intensity * xx**2).sum() / intensity.sum())
    assert waist == pytest.approx(expected, rel=1e-2)


def test_free_space_element_matches_propagate():
    grid = Grid(64, dx=5 * u.um)
    beam = Wavefront.gaussian(grid, LAM * u.m, waist=40 * u.um, power=1 * u.W)
    out = run(FreeSpace(distance=1 * u.mm), beam)
    np.testing.assert_allclose(np.asarray(out.field), np.asarray(propagate(beam, 1 * u.mm).field))


# --- Beam splitter -------------------------------------------------------------------------------


@pytest.mark.parametrize("reflectance", [0.0, 0.3, 0.5, 1.0])
def test_beam_splitter_conserves_power(reflectance):
    grid = Grid(16, dx=1 * u.um)
    wf = Wavefront.plane(grid, LAM * u.m, power=2 * u.W)
    outputs = BeamSplitter(reflectance=reflectance).apply(**{"in": wf})
    assert outputs["reflected"].power.to_value(u.W) == pytest.approx(2 * reflectance, abs=1e-12)
    assert outputs["transmitted"].power.to_value(u.W) == pytest.approx(2 * (1 - reflectance), abs=1e-12)


@pytest.mark.parametrize("phi", [0.0, 0.7, math.pi / 2, 2.0, math.pi])
def test_mach_zehnder(phi):
    grid = Grid(8, dx=1 * u.um)
    wf = Wavefront.plane(grid, LAM * u.m, power=1 * u.W)
    first = BeamSplitter(reflectance=0.5).apply(**{"in": wf})
    upper = run(DelayLine(delay=phi / (2 * math.pi) * LAM * u.m), first["transmitted"])
    lower = first["reflected"]
    # second splitter: output 1 = t·upper + r·lower, output 2 = r·upper + t·lower
    t, r = math.sqrt(0.5), 1j * math.sqrt(0.5)
    out1 = upper * t + lower * r
    out2 = upper * r + lower * t
    assert out1.power.to_value(u.W) == pytest.approx(math.sin(phi / 2) ** 2, abs=1e-12)
    assert out2.power.to_value(u.W) == pytest.approx(math.cos(phi / 2) ** 2, abs=1e-12)


# --- Delay line and filter -----------------------------------------------------------------------


def test_delay_line_phase_per_wavelength():
    grid = Grid(4, dx=1 * u.um)
    lams = [1.0, 1.5, 2.0] * u.um
    cube = Wavecube.plane(grid, lams, power=1 * u.W)
    out = run(DelayLine(delay=0.3 * u.um), cube)
    phases = np.angle(np.asarray(out.field)[:, 0, 0] / np.asarray(cube.field)[:, 0, 0])
    expected = np.angle(np.exp(2j * math.pi * 0.3 / lams.to_value(u.um)))
    np.testing.assert_allclose(phases, expected, atol=1e-12)


def test_spectral_filter_transmits_its_band():
    grid = Grid(4, dx=1 * u.um)
    cube = Wavecube.plane(grid, [1.0, 1.5, 2.0] * u.um, power=1 * u.W)
    out = run(SpectralFilter(center=1.5 * u.um, width=0.2 * u.um, transmission=0.8), cube)
    np.testing.assert_allclose(out.power.to_value(u.W), [0.0, 0.8, 0.0], atol=1e-12)


def test_filter_and_delay_act_on_bundles():
    grid = Grid(8, dx=1 * u.um)
    bundle = IncoherentBundle.point_sources(
        grid, [1.0, 2.0] * u.um, theta_x=[0, 0.01] * u.rad, theta_y=[0, 0] * u.rad, power=[1, 1] * u.W
    )
    out = run(SpectralFilter(center=1 * u.um, width=0.1 * u.um), bundle)
    assert out.total_power.to_value(u.W) == pytest.approx(2.0, rel=1e-12)  # 1 W per channel kept, 2 modes
    delayed = run(DelayLine(delay=0.25 * u.um), bundle)
    ratio = np.asarray(delayed.field)[1, 0, 0, 0] / np.asarray(bundle.field)[1, 0, 0, 0]
    assert np.angle(ratio) == pytest.approx(math.pi / 2, abs=1e-12)  # 2π · 0.25 / 1


# --- Polarisation elements -------------------------------------------------------------------


@pytest.mark.parametrize("angle", [0, 30, 45, 90])
def test_polarizer_element_follows_malus(angle):
    grid = Grid(8, dx=1 * u.um)
    wf = Wavefront.plane(grid, LAM * u.m, power=1 * u.W, polarization="H")
    out = run(Polarizer(angle=angle * u.deg), wf)
    assert out.power.to_value(u.W) == pytest.approx(math.cos(math.radians(angle)) ** 2, abs=1e-12)


def test_half_wave_plate_element_rotates_by_twice_its_angle():
    grid = Grid(8, dx=1 * u.um)
    wf = Wavefront.plane(grid, LAM * u.m, power=1 * u.W, polarization="H")
    out = run(WavePlate(retardance=180 * u.deg, angle=20 * u.deg), wf)
    analysed = jones.apply(jones.polarizer(40 * u.deg), out)
    assert analysed.power.to_value(u.W) == pytest.approx(1.0, rel=1e-12)


def test_polarisation_elements_need_polarised_light():
    grid = Grid(8, dx=1 * u.um)
    with pytest.raises(ValueError, match="polari"):
        run(Polarizer(angle=0 * u.deg), Wavefront.plane(grid, LAM * u.m, power=1 * u.W))


# --- In a context ----------------------------------------------------------------------------


def test_elements_chain_in_a_context():
    grid = Grid(64, dx=10 * u.um)
    ctx = Context()

    from lumos.elements.sources import GaussianLaser

    laser = ctx.add(
        GaussianLaser(
            name="laser",
            wavelength=1 * u.um,
            power=1 * u.mW,
            waist=0.1 * u.mm,
            pixel_scale=grid.dx,
            n_pixels=64,
        )
    )
    laser >> ThinLens(name="lens", focal_length=0.1 * u.m) >> FreeSpace(name="gap", distance=1 * u.mm)
    out = ctx.run(["gap.out"])["gap.out"]
    assert out.power.to_value(u.mW) == pytest.approx(1.0, rel=1e-6)


def test_free_space_with_an_output_grid_survives_lumos_files_and_codegen():
    from lumos.core.element import Element
    from lumos.io.codegen import to_python

    cut = Grid(16, 8, dx=2 * u.um, x0=10 * u.um)
    gap = FreeSpace(name="gap", distance=1 * u.mm, method="fresnel_mft", output_grid=cut)
    loaded = Element.from_lumos(gap.to_lumos())
    assert loaded.output_grid == cut
    assert loaded.method == "fresnel_mft"

    ctx = Context()
    ctx.add(gap)
    namespace: dict = {}
    exec(to_python(ctx), namespace)
    assert namespace["ctx"]["gap"].output_grid == cut
