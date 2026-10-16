"""Tests for the imaging of extended incoherent scenes (step P8.5, REQ-SCN-6).

Requirements: strategies chosen automatically (and forcible): (a) isoplanatic, convolution of the scene by
the PSF(λ); (b) anisoplanatic, PSFs on a grid of field points, interpolated; (c) explicit decomposition of
the scene into incoherent point sources, compulsory for systems that are not shift-invariant (coronagraphs,
interferometers, fibre injection). The choice and its reason are logged.

Independent references:

- for a shift-invariant system the three strategies compute the same image (linearity of incoherent
  imaging: Goodman, *Introduction to Fourier Optics*, §6.1);
- flux conservation: the power in the image equals the photon flux of the scene collected by the pupil
  area ``π D² / 4``, minus what falls outside the detector (the encircled energy of an Airy pattern reaches
  ``1 - 2 / (π² R)`` asymptotically, Born & Wolf §8.5.2, less than 1 % missing beyond 32 λ/D);
- a uniform disk of radiance L and angular radius R imaged by a circular pupil of diameter D at the focal
  length f gives, at the image centre, the irradiance ``L (π D² / 4) / f² × EE(R)``, where
  ``EE(R) = 1 - J0²(π R D / λ) - J1²(π R D / λ)`` is the encircled energy of the Airy pattern (Born & Wolf
  §8.5.2): the convolution of the disk by the PSF at the centre is the PSF integrated over the disk. At the
  second dark ring (R = 2.233 λ/D) EE is stationary, insensitive to the pixelisation of the disk;
- an ideal vortex coronagraph removes an on-axis point source but transmits off-axis ones (Mawet et al. 2005),
  so it is not shift-invariant.
"""

import logging
import math

import numpy as np
import pytest
from astropy import units as u
from scipy.special import j0, j1

from lumos import Context
from lumos.elements.coronagraphs import VortexCoronagraph
from lumos.elements.detectors import PowerMeter
from lumos.elements.optics import FreeSpace, ThinLens
from lumos.elements.telescopes import Telescope
from lumos.fields import PHOTON_FLUX_DENSITY, Grid, SourceList
from lumos.scenes import Scene
from lumos.scenes.diffuse import GaussianComponent, UniformDisk
from lumos.scenes.imaging import ImagingSystem, SceneToSources, image_scene, scene_to_sources

LAM, D, F = 1.0e-6, 1.0, 10.0
H, C = 6.62607015e-34, 299_792_458.0
LAMBDA_OVER_D = LAM / D  # rad


def instrument(*, coronagraph=False, n_detector=64, detector_step=0.5):
    """Telescope, (vortex), lens and focal plane; the telescope input is left free (the entrance)."""
    ctx = Context(seed=1)
    ctx.observation.band = (0.9999 * u.um, 1.0001 * u.um)
    ctx.observation.n_wavelengths = 1
    scope = ctx.add(Telescope(name="telescope", diameter=D * u.m, n_pixels=64, oversize=1.1))
    chain = scope
    if coronagraph:
        chain = chain >> VortexCoronagraph(name="vortex", charge=2, pupil_diameter=D * u.m, lyot_stop=0.95)
    chain = chain >> ThinLens(name="lens", focal_length=F * u.m)
    grid = Grid(n_detector, dx=detector_step * LAM * F / D * u.m)
    chain >> FreeSpace(name="image", distance=F * u.m, method="fresnel_mft", output_grid=grid)
    return ctx


def scene_field(ctx, *components, n=32, step=0.5):
    scene = ctx.add(Scene(name="sky", n_pixels=n, pixel_scale=step * LAMBDA_OVER_D * u.rad))
    for component in components:
        scene.add(component)
    return ctx.run(["sky.radiance"])["sky.radiance"]


def blob(x0=0.0):
    return GaussianComponent(
        name="blob",
        fwhm=3 * LAMBDA_OVER_D * u.rad,
        flux_density=1 * u.Jy,
        x0=x0 * LAMBDA_OVER_D * u.rad,
        n_pixels=64,
        pixel_scale=0.25 * LAMBDA_OVER_D * u.rad,
    )


def photon_rate(radiance, area):
    """Photons per second collected by ``area`` from a radiance field (one channel)."""
    integrated = radiance.integrated()[0]  # W m-2 m-1
    lam = radiance.wavelengths[0].to_value(u.m)
    bandwidth = 2e-10  # band of the context: 0.9999-1.0001 µm
    return integrated.to_value(u.W / u.m**3) * bandwidth * lam / (H * C) * area


# --- Strategies ------------------------------------------------------------------------------------------


def test_strategies_agree_on_a_shift_invariant_system():
    ctx = instrument()
    radiance = scene_field(ctx, blob(x0=2.0))
    system = ImagingSystem(ctx, entrance="telescope.in", exit="image.out")
    images = {
        s: image_scene(radiance, system, strategy=s)
        for s in ("decomposition", "isoplanatic", "anisoplanatic")
    }
    reference = images["decomposition"].power
    peak = reference.max()
    for strategy in ("isoplanatic", "anisoplanatic"):
        np.testing.assert_allclose(images[strategy].power, reference, atol=2e-3 * peak)
        assert images[strategy].strategy == strategy


def test_flux_is_conserved_from_scene_to_image():
    ctx = instrument(n_detector=128)
    radiance = scene_field(ctx, blob())
    system = ImagingSystem(ctx, entrance="telescope.in", exit="image.out")
    image = image_scene(radiance, system, strategy="decomposition")
    expected = photon_rate(radiance, math.pi * D**2 / 4)
    assert image.power.sum() == pytest.approx(expected, rel=2e-2)
    assert image.power_unit.is_equivalent(u.photon / u.s)


def test_image_of_a_uniform_disk_matches_the_analytic_solution():
    ctx = instrument(n_detector=33, detector_step=0.5)
    radius = 2.233 * LAMBDA_OVER_D
    disk = UniformDisk(
        name="disk",
        radius=radius * u.rad,
        flux_density=1 * u.Jy,
        n_pixels=96,
        pixel_scale=0.0625 * LAMBDA_OVER_D * u.rad,
    )
    radiance = scene_field(ctx, disk, n=96, step=0.0625)
    system = ImagingSystem(ctx, entrance="telescope.in", exit="image.out")
    for strategy in ("isoplanatic", "decomposition"):
        image = image_scene(radiance, system, strategy=strategy)
        centre = image.power[0, 16, 16]
        area = math.pi * D**2 / 4
        ee = 1 - j0(math.pi * 2.233) ** 2 - j1(math.pi * 2.233) ** 2
        radiance_photons = photon_rate(radiance, 1.0) / (math.pi * radius**2)  # photon s-1 m-2 sr-1
        pixel = (0.5 * LAM * F / D) ** 2
        expected = radiance_photons * area / F**2 * ee * pixel
        assert centre == pytest.approx(expected, rel=2e-2), strategy


# --- Automatic choice --------------------------------------------------------------------------------------


def test_auto_chooses_the_convolution_for_a_shift_invariant_system(caplog):
    ctx = instrument()
    radiance = scene_field(ctx, blob())
    system = ImagingSystem(ctx, entrance="telescope.in", exit="image.out")
    with caplog.at_level(logging.INFO, logger="lumos"):
        image = image_scene(radiance, system)
    assert image.strategy == "isoplanatic"
    assert image.reason
    assert any("isoplanatic" in record.getMessage() for record in caplog.records)


def test_auto_decomposes_the_scene_for_a_coronagraph(caplog):
    ctx = instrument(coronagraph=True)
    radiance = scene_field(ctx, blob(x0=3.0))
    system = ImagingSystem(ctx, entrance="telescope.in", exit="image.out")
    with caplog.at_level(logging.INFO, logger="lumos"):
        image = image_scene(radiance, system)
    assert image.strategy == "decomposition"
    assert "shift" in image.reason
    assert any("decomposition" in record.getMessage() for record in caplog.records)


def test_convolution_is_refused_for_a_coronagraph():
    ctx = instrument(coronagraph=True)
    radiance = scene_field(ctx, blob(x0=3.0))
    system = ImagingSystem(ctx, entrance="telescope.in", exit="image.out")
    with pytest.raises(ValueError, match="shift-invariant"):
        image_scene(radiance, system, strategy="isoplanatic")


def test_unknown_strategy_is_refused():
    ctx = instrument()
    radiance = scene_field(ctx, blob())
    with pytest.raises(ValueError, match="strategy"):
        image_scene(radiance, ImagingSystem(ctx, entrance="telescope.in", exit="image.out"), strategy="magic")


# --- Point sources and graph -------------------------------------------------------------------------------


def test_point_sources_are_added_exactly():
    ctx = instrument()
    radiance = scene_field(ctx, blob())
    system = ImagingSystem(ctx, entrance="telescope.in", exit="image.out")
    lam = radiance.wavelengths
    star = SourceList([2e-6] * u.rad, [0.0] * u.rad, lam, [2e-4] * u.um, [[1e12]] * PHOTON_FLUX_DENSITY)
    alone = image_scene(radiance, system, strategy="isoplanatic")
    both = image_scene(radiance, system, strategy="isoplanatic", points=star)
    np.testing.assert_allclose(both.power - alone.power, system.response(star)[0], rtol=1e-10, atol=1e-30)


def test_scene_to_sources_conserves_the_flux():
    ctx = instrument()
    radiance = scene_field(ctx, blob())
    sources = scene_to_sources(radiance, [2e-4] * u.um)
    total = sources.photon_rate(1 * u.m**2).sum().to_value(u.photon / u.s)
    assert total == pytest.approx(photon_rate(radiance, 1.0), rel=1e-10)


def test_scene_to_sources_element_in_a_graph():
    ctx = instrument()
    scene = ctx.add(Scene(name="sky", n_pixels=32, pixel_scale=0.5 * LAMBDA_OVER_D * u.rad))
    scene.add(blob(x0=1.0))
    converter = ctx.add(SceneToSources(name="pixels"))
    ctx.connect(scene.port("radiance"), converter.port("in"))
    ctx.connect(converter.port("out"), ctx["telescope"].port("in"))
    ctx.connect(ctx["image"].port("out"), ctx.add(PowerMeter(name="meter")).port("in"))
    graph_image = ctx.run(["image.out"])["image.out"]
    total = ctx.run(["meter.out"])["meter.out"].data.to_value(u.W)
    assert total > 0
    radiance = ctx.run(["sky.radiance"])["sky.radiance"]
    alone = instrument()
    system = ImagingSystem(alone, entrance="telescope.in", exit="image.out")
    reference = image_scene(radiance, system, strategy="decomposition")
    intensity = (
        np.asarray(graph_image.intensity).sum(axis=0) * graph_image.grid.dx_si * graph_image.grid.dy_si
    )
    np.testing.assert_allclose(intensity, reference.power[0], rtol=1e-6, atol=1e-6 * reference.power.max())


def test_image_as_a_wavecube_for_detectors():
    ctx = instrument()
    radiance = scene_field(ctx, blob())
    image = image_scene(radiance, ImagingSystem(ctx, entrance="telescope.in", exit="image.out"))
    cube = image.to_wavecube()
    power = cube.power.to_value(u.photon / u.s)
    np.testing.assert_allclose(power, image.power.sum(axis=(-2, -1)), rtol=1e-10)
