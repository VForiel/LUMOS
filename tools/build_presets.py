"""Build the preset contexts of ``examples/presets/`` (shared by the notebooks and the web interface).

Each preset is a ``.lumos`` context whose ``meta`` holds a ``description`` and the list of ``outputs`` to
compute. Run after changing a preset::

    python tools/build_presets.py
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from astropy import units as u
from astropy.time import Time

import lumos.io
from lumos import Context
from lumos.elements.coronagraphs import VortexCoronagraph
from lumos.elements.detectors import Camera, PowerMeter
from lumos.elements.optics import FreeSpace, ThinLens
from lumos.elements.photonics import bracewell, kernel_nuller
from lumos.elements.sources import GaussianLaser, PointSource
from lumos.elements.telescopes import Telescope, TelescopeArray
from lumos.fields import Grid
from lumos.scenes import Scene, SceneToSources
from lumos.scenes.diffuse import InclinedRing

PRESETS = Path(__file__).resolve().parents[1] / "examples" / "presets"
START = Time("2026-01-01T00:00:00", scale="utc")


def _context(name: str, description: str, outputs: list[str], seed: int = 1) -> Context:
    ctx = Context(seed=seed, start=START, name=name)
    ctx.meta["description"] = description
    ctx.meta["outputs"] = outputs
    ctx.meta["author"] = "LUMOS"
    return ctx


def first_simulation() -> Context:
    ctx = _context(
        "First simulation",
        "A HeNe laser focused by a 200 mm lens on a camera.",
        ["camera.out", "focus.out"],
    )
    laser = ctx.add(GaussianLaser(name="laser", wavelength=633 * u.nm, power=1 * u.mW, waist=1 * u.mm))
    lens = ctx.add(ThinLens(name="lens", focal_length=200 * u.mm))
    focus = ctx.add(
        FreeSpace(name="focus", distance=200 * u.mm, method="fresnel_mft", output_grid=Grid(64, dx=2 * u.um))
    )
    camera = ctx.add(Camera(name="camera", exposure=1 * u.us, full_well=1e9 * u.electron, bits=32))
    laser >> lens >> focus >> camera
    return ctx


def propagation_methods() -> Context:
    ctx = _context(
        "Propagation methods",
        "A Gaussian beam propagated over 2 m: the planner chooses the method (see the journal).",
        ["far.out", "meter.out"],
    )
    laser = ctx.add(GaussianLaser(name="laser", waist=0.5 * u.mm, pixel_scale=20 * u.um, n_pixels=256))
    far = ctx.add(FreeSpace(name="far", distance=2 * u.m))
    laser >> far >> PowerMeter(name="meter")
    return ctx


def _band(ctx: Context, low: u.Quantity, high: u.Quantity, n: int) -> None:
    ctx.observation.band = (low, high)
    ctx.observation.n_wavelengths = n


def vlt_psf() -> Context:
    ctx = _context(
        "VLT point-spread function",
        "A star seen in the H band by an 8.2 m telescope with a central obstruction and four spiders.",
        ["image.out", "camera.out"],
    )
    _band(ctx, 1.5 * u.um, 1.8 * u.um, 3)
    star = ctx.add(PointSource(name="star", flux_density=1 * u.Jy))
    scope = ctx.add(
        Telescope(
            name="telescope",
            diameter=8.2 * u.m,
            obstruction=1.116 * u.m,
            n_spiders=4,
            spider_width=5 * u.cm,
            spider_rotation=45 * u.deg,
            n_pixels=256,
        )
    )
    lens = ctx.add(ThinLens(name="lens", focal_length=120 * u.m))
    sampling = 1.5e-6 * 120 / 8.2 / 3  # three pixels per λ/D at 1.5 µm
    image = ctx.add(
        FreeSpace(
            name="image",
            distance=120 * u.m,
            method="fresnel_mft",
            output_grid=Grid(96, dx=sampling * u.m),
        )
    )
    star >> scope >> lens >> image >> Camera(name="camera", exposure=1 * u.s, noise=False)
    return ctx


def coronagraph() -> Context:
    lam, d, f = 1.6e-6, 8.0, 100.0
    ctx = _context(
        "Vortex coronagraph",
        "A star and a companion 1e-4 fainter at 6 λ/D behind a charge-2 vortex coronagraph.",
        ["image.out", "coronagraph.out", "telescope.out"],
    )
    _band(ctx, 1.59 * u.um, 1.61 * u.um, 1)
    star = ctx.add(PointSource(name="star", flux_density=1 * u.Jy))
    planet = ctx.add(PointSource(name="planet", flux_density=1e-4 * u.Jy, theta_x=6 * lam / d * u.rad))
    scope = ctx.add(Telescope(name="telescope", diameter=d * u.m, n_pixels=256, oversize=1.1))
    ctx.connect(star.port("out"), scope.port("in"))
    ctx.connect(planet.port("out"), scope.port("in"))
    vortex = ctx.add(VortexCoronagraph(name="coronagraph", charge=2, pupil_diameter=d * u.m, lyot_stop=0.95))
    lens = ctx.add(ThinLens(name="lens", focal_length=f * u.m))
    image = ctx.add(
        FreeSpace(
            name="image",
            distance=f * u.m,
            method="fresnel_mft",
            output_grid=Grid(81, dx=lam * f / d / 4 * u.m),
        )
    )
    scope >> vortex >> lens >> image
    return ctx


def kernel_nuller_preset() -> Context:
    ctx = _context(
        "Kernel nuller",
        "Four 2 m telescopes recombined by a kernel-nuller chip (Martinache & Ireland 2018).",
        ["meter.out"],
    )
    _band(ctx, 3.5 * u.um, 4.0 * u.um, 4)
    star = ctx.add(PointSource(name="star", flux_density=1 * u.Jy))
    planet = ctx.add(
        PointSource(name="planet", flux_density=0.01 * u.Jy, theta_x=4e-8 * u.rad, theta_y=-1.5e-8 * u.rad)
    )
    array = ctx.add(
        TelescopeArray(
            name="array",
            frame="sky",
            positions=[[0.0, 0.0], [30.0, 10.0], [55.0, -20.0], [80.0, 25.0]] * u.m,
            diameter=2 * u.m,
        )
    )
    ctx.connect(star.port("out"), array.port("in"))
    ctx.connect(planet.port("out"), array.port("in"))
    array >> kernel_nuller(name="chip") >> PowerMeter(name="meter")
    return ctx


def bracewell_preset() -> Context:
    ctx = _context(
        "Bracewell nuller",
        "Two 2 m telescopes 20 m apart recombined by a Bracewell nuller at 10 µm.",
        ["meter.out"],
    )
    _band(ctx, 9.99 * u.um, 10.01 * u.um, 1)
    source = ctx.add(PointSource(name="source", flux_density=1 * u.Jy, theta_x=0.25 * 10e-6 / 20 * u.rad))
    array = ctx.add(
        TelescopeArray(
            name="array", frame="sky", positions=[[-10.0, 0.0], [10.0, 0.0]] * u.m, diameter=2 * u.m
        )
    )
    source >> array >> bracewell(name="chip") >> PowerMeter(name="meter")
    return ctx


def debris_disk() -> Context:
    lam, d, f = 1.6e-6, 8.0, 100.0
    resolution = lam / d  # rad
    ctx = _context(
        "Debris disk",
        "A star and an inclined debris disk (1e-3 of the stellar flux) behind a vortex coronagraph; "
        "the disk is decomposed into incoherent point sources (SceneToSources).",
        ["image.out", "sky.radiance"],
    )
    _band(ctx, 1.59 * u.um, 1.61 * u.um, 1)
    sky = ctx.add(Scene(name="sky", n_pixels=24, pixel_scale=0.75 * resolution * u.rad))
    sky.add(PointSource(name="star", flux_density=1 * u.Jy))
    sky.add(
        InclinedRing(
            name="disk",
            inner_radius=5 * resolution * u.rad,
            outer_radius=7 * resolution * u.rad,
            inclination=60 * u.deg,
            position_angle=30 * u.deg,
            flux_density=1e-3 * u.Jy,
            n_pixels=48,
            pixel_scale=0.375 * resolution * u.rad,
        )
    )
    pixels = ctx.add(SceneToSources(name="pixels"))
    scope = ctx.add(Telescope(name="telescope", diameter=d * u.m, n_pixels=128, oversize=1.1))
    ctx.connect(sky.port("radiance"), pixels.port("in"))
    ctx.connect(pixels.port("out"), scope.port("in"))
    ctx.connect(sky.port("sources"), scope.port("in"))
    vortex = ctx.add(VortexCoronagraph(name="coronagraph", charge=2, pupil_diameter=d * u.m, lyot_stop=0.95))
    lens = ctx.add(ThinLens(name="lens", focal_length=f * u.m))
    image = ctx.add(
        FreeSpace(
            name="image",
            distance=f * u.m,
            method="fresnel_mft",
            output_grid=Grid(64, dx=lam * f / d / 2.5 * u.m),
        )
    )
    scope >> vortex >> lens >> image
    return ctx


BUILDERS: dict[str, Callable[[], Context]] = {
    "first_simulation": first_simulation,
    "propagation_methods": propagation_methods,
    "vlt_psf": vlt_psf,
    "coronagraph": coronagraph,
    "kernel_nuller": kernel_nuller_preset,
    "bracewell": bracewell_preset,
    "debris_disk": debris_disk,
}


def main() -> None:
    PRESETS.mkdir(parents=True, exist_ok=True)
    for name, build in BUILDERS.items():
        ctx = build()
        ctx.run(ctx.meta["outputs"])  # fail early on an invalid preset
        lumos.io.save(ctx, PRESETS / f"{name}.lumos")
        print(f"wrote {name}.lumos ({len(ctx)} elements)")


if __name__ == "__main__":
    main()
