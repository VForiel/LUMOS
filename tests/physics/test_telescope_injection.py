r"""Free-space telescope arrays and explicit injection into fibres (step P15.8, ADR-0023 §7).

Specification: docs/specs/pistes_evolution.md §1.3 (decisions of 20/10/2026).

- ``TelescopeArray`` returns a packed bus of ``IncoherentBundle`` in the pupil plane, one lane per telescope:
  the pupil of a ``Telescope`` (validated in test_telescopes.py) times the geometric phase of its arm,
  :math:`\exp[2i\pi(u_j\theta_x + v_j\theta_y)/\lambda]`; lanes carry the projected positions.
- The injection is an explicit ``Fiber``. With ``injection_focal_length`` f, a pupil-plane field is focused
  by an ideal lens on the fibre face; the coupling is the overlap of the pupil field with the mode imaged in
  the pupil, a Gaussian of waist :math:`w_p = \lambda f / (\pi w_0)` for the Gaussian model.
- The former ideal model is ``IdealInjectionArray``; documents written before (``TelescopeArray`` of type
  version 1) are read as ``IdealInjectionArray`` and give the same results.

Independent references:

- coupling of a uniform circular pupil of radius R into a Gaussian mode of pupil waist :math:`w_p`:
  :math:`\eta = 2(1 - e^{-\beta^2})^2/\beta^2`, :math:`\beta = R/w_p`, maximum 0.8145 at
  :math:`\beta = 1.1209` (Shaklan & Roddier 1988, Appl. Opt. 27, 2334; Ruilier 1998, Proc. SPIE 3350, 319);
- antenna lobe: for a plane wave tilted by θ,
  :math:`a(\theta) \propto \int_0^R e^{-r^2/w_p^2} J_0(2\pi\theta r/\lambda)\, r\, dr`, evaluated here with
  ``scipy.integrate.quad`` (independent of the 2-D overlap computed by LUMOS);
- a Bracewell nuller fed by two such fibres transmits :math:`2\eta(\theta)\,P\sin^2(\pi B\theta/\lambda)` on
  its dark output (Bracewell 1978, Nature 274, 780), :math:`P` the power collected by one telescope.
"""

import math

import numpy as np
import pytest
from astropy import units as u
from scipy import integrate, special

from lumos import Context
from lumos.core.element import Element
from lumos.elements.detectors import PowerMeter
from lumos.elements.fibers import Fiber
from lumos.elements.photonics import bracewell
from lumos.elements.sources import PointSource
from lumos.elements.telescopes import IdealInjectionArray, Telescope, TelescopeArray
from lumos.fields import Bus, IncoherentBundle

LAM = 10e-6
D = 2.0
H, C = 6.62607015e-34, 299_792_458.0


def band(ctx, low_um, high_um, n=1):
    ctx.observation.band = (low_um * u.um, high_um * u.um)
    ctx.observation.n_wavelengths = n


def gaussian_fiber(beta, *, focal_length=1.0, lam=LAM, name="fiber"):
    """A Gaussian-mode fibre whose mode imaged in a pupil of diameter D has the waist R / beta."""
    w_pupil = D / 2 / beta
    w0 = lam * focal_length / (math.pi * w_pupil)
    return Fiber(
        name=name,
        mode_model="gaussian",
        mode_field_diameter=2 * w0 * u.m,
        length=0 * u.m,
        injection_focal_length=focal_length * u.m,
    )


def pupil_coupling(beta):
    return 2 * (1 - math.exp(-(beta**2))) ** 2 / beta**2


def lobe(beta, theta, lam=LAM):
    """Relative amplitude coupled from a plane wave tilted by theta (Hankel integral, scipy)."""
    r_max, w_p = D / 2, D / 2 / beta
    k = 2 * math.pi / lam

    def amplitude(t):
        return integrate.quad(lambda r: math.exp(-(r**2) / w_p**2) * special.j0(k * t * r) * r, 0, r_max)[0]

    return amplitude(theta) / amplitude(0.0)


# --- TelescopeArray in free space ------------------------------------------------------------------


def test_the_array_returns_one_pupil_per_telescope_with_the_phase_of_its_arm():
    ctx = Context()
    band(ctx, 9.9, 10.1, 2)
    tx, ty = 2e-7, -1e-7
    star = ctx.add(PointSource(name="star", flux_density=1 * u.Jy, theta_x=tx * u.rad, theta_y=ty * u.rad))
    positions = [[0.0, 0.0], [12.0, 5.0], [-4.0, 9.0]]
    array = ctx.add(
        TelescopeArray(name="array", frame="sky", positions=positions * u.m, diameter=D * u.m, n_pixels=32)
    )
    scope = ctx.add(Telescope(name="scope", diameter=D * u.m, n_pixels=32))
    ctx.connect(star.port("out"), array.port("in"))
    ctx.connect(star.port("out"), scope.port("in"))
    result = ctx.run(["array.out", "scope.out", "star.out"])
    bus, alone = result["array.out"], result["scope.out"]
    assert isinstance(bus, Bus)
    assert bus.lane_type is IncoherentBundle
    assert bus.packed
    assert bus.names == ("T1", "T2", "T3")
    np.testing.assert_allclose(bus.positions.to_value(u.m), positions, atol=1e-12)
    assert bus[0].sources is not None
    assert bus[0].sources == alone.sources
    lam = result["star.out"].wavelengths.to_value(u.m)
    for j, (x, y) in enumerate(positions):
        phase = np.exp(2j * math.pi * (x * tx + y * ty) / lam)  # per wavelength
        expected = np.asarray(alone.field) * phase[None, :, None, None]
        np.testing.assert_allclose(np.asarray(bus[j].field), expected, rtol=1e-9, atol=1e-12)


def test_the_array_follows_the_clock_like_the_ideal_model():
    fields = {"frame": "sky", "positions": [[0, 0], [10, 0]] * u.m, "rotation_rate": 15 * u.deg / u.hour}
    free, ideal = TelescopeArray(**fields), IdealInjectionArray(**fields)
    np.testing.assert_allclose(free.projected_positions().value, ideal.projected_positions().value)
    assert free.port_lanes("out") == ideal.port_lanes("out") == 2


# --- Injection into a fibre --------------------------------------------------------------------------


@pytest.mark.parametrize("beta", [0.8, 1.1209, 1.5])
def test_coupling_of_a_circular_pupil_into_a_single_mode_fibre(beta):
    ctx = Context()
    band(ctx, 9.999, 10.001)
    star = ctx.add(PointSource(name="star", flux_density=1 * u.Jy))
    scope = ctx.add(Telescope(name="scope", diameter=D * u.m, n_pixels=256, oversize=1.02))
    fiber = ctx.add(gaussian_fiber(beta))
    ctx.connect(star.port("out"), scope.port("in"))
    ctx.connect(scope.port("out"), fiber.port("in"))
    result = ctx.run(["scope.out", "fiber.modes"])
    collected = float(result["scope.out"].total_power.value)
    coupled = float(np.sum(result["fiber.modes"].total_power.value))
    assert coupled / collected == pytest.approx(pupil_coupling(beta), abs=2e-3)


def test_the_best_coupling_is_81_percent():
    betas = np.linspace(0.9, 1.35, 46)
    assert max(pupil_coupling(b) for b in betas) == pytest.approx(0.8145, abs=1e-4)


@pytest.mark.parametrize("theta_over", [0.0, 0.3, 0.6, 1.0])
def test_antenna_lobe_of_the_fibre(theta_over):
    beta = 1.1209
    theta = theta_over * LAM / D
    ctx = Context()
    band(ctx, 9.999, 10.001)
    star = ctx.add(PointSource(name="star", flux_density=1 * u.Jy, theta_x=theta * u.rad))
    scope = ctx.add(Telescope(name="scope", diameter=D * u.m, n_pixels=256, oversize=1.02))
    fiber = ctx.add(gaussian_fiber(beta))
    ctx.connect(star.port("out"), scope.port("in"))
    ctx.connect(scope.port("out"), fiber.port("in"))
    result = ctx.run(["scope.out", "fiber.modes"])
    eta = float(np.sum(result["fiber.modes"].total_power.value)) / float(
        result["scope.out"].total_power.value
    )
    assert eta == pytest.approx(pupil_coupling(beta) * lobe(beta, theta) ** 2, abs=2e-3)


def test_a_fiber_on_the_array_gives_a_bus_of_guided_lanes():
    ctx = Context()
    band(ctx, 9.999, 10.001)
    star = ctx.add(PointSource(name="star", flux_density=1 * u.Jy))
    array = ctx.add(
        TelescopeArray(
            name="array", frame="sky", positions=[[0, 0], [20, 0]] * u.m, diameter=D * u.m, n_pixels=64
        )
    )
    fiber = ctx.add(gaussian_fiber(1.1209))
    ctx.connect(star.port("out"), array.port("in"))
    ctx.connect(array.port("out"), fiber.port("in"))
    modes = ctx.run(["fiber.modes"])["fiber.modes"]
    assert isinstance(modes, Bus)
    assert modes.packed
    assert modes.mode_counts == (1, 1)
    np.testing.assert_allclose(modes.positions.to_value(u.m), [[0, 0], [20, 0]])


@pytest.mark.parametrize("theta_over", [0.25, 0.5, 0.9])
def test_a_bracewell_nuller_sees_fringes_under_the_antenna_lobe(theta_over):
    beta, baseline = 1.1209, 20.0
    theta = theta_over * LAM / D
    ctx = Context()
    band(ctx, 9.999, 10.001)
    star = ctx.add(PointSource(name="star", flux_density=1 * u.Jy, theta_x=theta * u.rad))
    array = ctx.add(
        TelescopeArray(
            name="array",
            frame="sky",
            positions=[[-baseline / 2, 0.0], [baseline / 2, 0.0]] * u.m,
            diameter=D * u.m,
            n_pixels=256,
            oversize=1.02,
        )
    )
    fiber = ctx.add(gaussian_fiber(beta))
    ctx.connect(star.port("out"), array.port("in"))
    ctx.connect(array.port("out"), fiber.port("in"))
    fiber.port("modes") >> bracewell(name="chip") >> PowerMeter(name="meter")
    result = ctx.run(["meter.out", "array.out"])
    lane = result["array.out"][0]
    lam = (
        float(np.asarray(lane[0].wavelengths_si)[0])
        if hasattr(lane[0], "wavelengths_si")
        else lane[0].wavelength_si
    )
    collected = float(lane.total_power.value) * H * C / lam  # W, one telescope
    eta = pupil_coupling(beta) * lobe(beta, theta, lam) ** 2
    expected_dark = 2 * eta * collected * math.sin(math.pi * baseline * theta / lam) ** 2
    dark = result["meter.out"].data.to_value(u.W)[0]
    assert dark == pytest.approx(expected_dark, rel=5e-3)


# --- The ideal model and the documents written before -----------------------------------------------


def test_documents_of_the_ideal_array_are_read_as_ideal_injection_arrays():
    from lumos.io import from_document

    document = IdealInjectionArray(name="array", frame="sky", positions=[[0, 0], [10, 0]] * u.m).to_lumos()
    document["lumos_format"] = "1.2"
    document["element"]["type"] = "lumos.telescopes.TelescopeArray"
    document["element"]["type_version"] = 1
    loaded = from_document(document)
    assert isinstance(loaded, IdealInjectionArray)
    np.testing.assert_allclose(loaded.projected_positions().value, [[0, 0], [10, 0]])
    assert isinstance(Element.from_lumos(TelescopeArray(n_pixels=16).to_lumos()), TelescopeArray)


def test_a_chip_addressing_the_modes_of_a_multimode_fibre_is_reported(caplog):
    """ADR-0023 §8: channels now designate lanes, which a 1.2 document cannot have meant for fibre modes."""
    from lumos.elements.optics import FreeSpace  # noqa: F401 - registered types
    from lumos.elements.photonics import DirectionalCoupler
    from lumos.elements.sources import GaussianLaser
    from lumos.io import from_document, to_document

    ctx = Context()
    laser = ctx.add(GaussianLaser(name="laser"))
    fiber = ctx.add(Fiber(name="fiber"))
    coupler = ctx.add(DirectionalCoupler(name="coupler", channels=[0, 2]))
    ctx.connect(laser.port("out"), fiber.port("in"))
    ctx.connect(fiber.port("modes"), coupler.port("in"))
    document = to_document(ctx)
    document["lumos_format"] = "1.2"
    with caplog.at_level("WARNING"):
        from_document(document)
    assert any(
        "coupler" in record.getMessage() and "lane" in record.getMessage() for record in caplog.records
    )
