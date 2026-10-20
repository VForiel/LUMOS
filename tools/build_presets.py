"""Build the preset contexts of ``examples/presets/`` (shared by the notebooks and the web interface).

Each preset is a ``.lumos`` context whose ``meta`` holds a ``description`` and the list of ``outputs`` to
compute. Run after changing a preset::

    python tools/build_presets.py               # every preset
    python tools/build_presets.py symbolic_focus  # some of them
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

from astropy import units as u
from astropy.time import Time

import lumos.io
from lumos import Context
from lumos.core.procedure import Procedure
from lumos.elements.coronagraphs import VortexCoronagraph
from lumos.elements.detectors import Camera, PowerMeter
from lumos.elements.fibers import Fiber
from lumos.elements.lab import Shutter
from lumos.elements.optics import FreeSpace, ThinLens
from lumos.elements.photonics import bracewell, kernel_nuller
from lumos.elements.sources import GaussianLaser, PointSource
from lumos.elements.telescopes import Telescope, TelescopeArray
from lumos.fields import Grid
from lumos.scenes import Scene, SceneToSources
from lumos.scenes.diffuse import InclinedRing
from lumos.symbolic import SymbolicSource

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


def _injection(name: str, wavelength: u.Quantity, diameter: u.Quantity, focal_length: u.Quantity) -> Fiber:
    """Single-mode fibres (Gaussian mode) behind lenses of focal ``focal_length``, matched to the pupils.

    The mode imaged in the pupil has the waist R / 1.1209, the best coupling of a circular pupil (81 %,
    Shaklan & Roddier 1988), at ``wavelength``.
    """
    w_pupil = diameter / 2 / 1.1209
    waist = u.Quantity(wavelength * focal_length / (math.pi * w_pupil)).to(u.um)
    return Fiber(
        name=name,
        mode_model="gaussian",
        mode_field_diameter=2 * waist,
        length=0 * u.m,
        injection_focal_length=focal_length,
    )


def kernel_nuller_preset() -> Context:
    ctx = _context(
        "Kernel nuller",
        "Four 2 m telescopes injected into single-mode fibres and recombined by a kernel-nuller chip "
        "(Martinache & Ireland 2018).",
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
    fibers = ctx.add(_injection("fibers", 3.75 * u.um, 2 * u.m, 6 * u.m))
    ctx.connect(star.port("out"), array.port("in"))
    ctx.connect(planet.port("out"), array.port("in"))
    ctx.connect(array.port("out"), fibers.port("in"))
    fibers.port("modes") >> kernel_nuller(name="chip") >> PowerMeter(name="meter")
    return ctx


def bracewell_preset() -> Context:
    ctx = _context(
        "Bracewell nuller",
        "Two 2 m telescopes 20 m apart, injected into single-mode fibres and recombined by a Bracewell "
        "nuller at 10 µm.",
        ["meter.out"],
    )
    _band(ctx, 9.99 * u.um, 10.01 * u.um, 1)
    source = ctx.add(PointSource(name="source", flux_density=1 * u.Jy, theta_x=0.25 * 10e-6 / 20 * u.rad))
    array = ctx.add(
        TelescopeArray(
            name="array", frame="sky", positions=[[-10.0, 0.0], [10.0, 0.0]] * u.m, diameter=2 * u.m
        )
    )
    fibers = ctx.add(_injection("fibers", 10 * u.um, 2 * u.m, 6 * u.m))
    source >> array
    ctx.connect(array.port("out"), fibers.port("in"))
    fibers.port("modes") >> bracewell(name="chip") >> PowerMeter(name="meter")
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


#: The procedure of the review of 19/10/2026: a dark from frames taken shutter closed, then measurements.
DARK_THEN_MEASURE = {
    "name": "dark then measure",
    "description": "20 frames shutter closed averaged into a dark, then 10 frames shutter open minus the dark",
    "steps": [
        {"do": "action", "element": "shutter", "action": "close"},
        {
            "do": "repeat",
            "times": 20,
            "steps": [
                {"do": "action", "element": "camera", "action": "expose"},
                {"do": "measure", "outputs": ["camera.out"], "as": "darks", "collect": True},
            ],
        },
        {"do": "process", "function": "mean", "args": {"frames": "$darks"}, "as": "dark"},
        {"do": "action", "element": "shutter", "action": "open"},
        {
            "do": "repeat",
            "times": 10,
            "steps": [
                {"do": "action", "element": "camera", "action": "expose"},
                {"do": "measure", "outputs": ["camera.out"], "as": "images", "collect": True},
            ],
        },
        {"do": "process", "function": "mean", "args": {"frames": "$images"}, "as": "image"},
        {
            "do": "process",
            "function": "subtract_dark",
            "args": {"frame": "$image", "dark": "$dark"},
            "as": "science",
        },
        {"do": "show", "results": ["dark", "science"]},
    ],
}


def dark_then_measure() -> Context:
    ctx = _context(
        "dark then measure",
        "A laser beam on a camera behind a shutter: the procedure takes a dark shutter closed, then subtracts it "
        "from measurements shutter open (Procedures tab).",
        ["camera.out"],
    )
    laser = ctx.add(GaussianLaser(name="laser", power=1e-13 * u.W, n_pixels=64, pixel_scale=0.05 * u.mm))
    shutter = ctx.add(Shutter(name="shutter"))
    camera = ctx.add(
        Camera(
            name="camera",
            exposure=0.5 * u.s,
            dark_current=200 * u.electron / u.s,
            read_noise=5 * u.electron,
            bias=100 * u.adu,
        )
    )
    laser >> shutter >> camera
    ctx.procedures["dark then measure"] = Procedure.from_json(DARK_THEN_MEASURE)
    return ctx


def symbolic_focus() -> Context:
    """Gaussian beam focused by a lens, kept symbolic up to its new waist (closed form by ABCD)."""
    import math

    lam, w0, f = 1.064e-6, 2e-3, 0.5
    z_r = math.pi * w0**2 / lam
    distance = f / (1 + (f / z_r) ** 2)  # new waist (Self 1983)
    waist = lam * f / (math.pi * w0) / math.sqrt(1 + (f / z_r) ** 2)
    ctx = _context(
        "symbolic focus",
        "Nd:YAG Gaussian beam focused by a lens, propagated in closed form to its new waist; a power meter "
        "receives the field converted to numbers.",
        ["gap.out", "meter.out"],
    )
    beam = ctx.add(
        SymbolicSource.gaussian_beam(
            name="beam",
            waist=w0 * u.m,
            wavelength=lam * u.m,
            power=0.01 * u.W,  # in W, as the sampled fields (|U|² in W/m²)
            grid=Grid(128, dx=waist / 8 * u.m),
        )
    )
    lens = ctx.add(ThinLens(name="lens", focal_length=f * u.m))
    gap = ctx.add(FreeSpace(name="gap", distance=distance * u.m))
    beam >> lens >> gap >> ctx.add(PowerMeter(name="meter"))
    return ctx


BUILDERS: dict[str, Callable[[], Context]] = {
    "first_simulation": first_simulation,
    "propagation_methods": propagation_methods,
    "vlt_psf": vlt_psf,
    "coronagraph": coronagraph,
    "kernel_nuller": kernel_nuller_preset,
    "bracewell": bracewell_preset,
    "debris_disk": debris_disk,
    "symbolic_focus": symbolic_focus,
    "dark_then_measure": dark_then_measure,
}


def main(names: list[str]) -> None:
    PRESETS.mkdir(parents=True, exist_ok=True)
    for name, build in BUILDERS.items():
        if names and name not in names:
            continue
        ctx = build()
        ctx.run(ctx.meta["outputs"])  # fail early on an invalid preset
        lumos.io.save(ctx, PRESETS / f"{name}.lumos")
        print(f"wrote {name}.lumos ({len(ctx)} elements)")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
