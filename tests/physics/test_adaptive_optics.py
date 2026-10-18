"""Closed-loop adaptive optics: wavefront sensor, controller and feedback loop (step P14.14, ADR-0020).

Requirements: §5.9 (adaptive optics), REQ-CTX-2 (loops as time iterations of the context), review of
18/10/2026 (an adaptive optics preset with a real feedback loop).

Independent references:

- the Noll Zernike polynomials have a unit RMS over the pupil: Z2 = 2ρcosθ is a tilt of slope 2a/R for a
  coefficient a, Z4 a defocus of RMS a (Noll 1976, JOSA 66, 207, Table I);
- a Shack-Hartmann sub-aperture measures the mean gradient of the optical path over its area, i.e. the tilt
  of the wavefront in radians, independently of the wavelength (Hardy 1998, *Adaptive Optics for Astronomical
  Telescopes*, §5.3);
- a pure integrator ``c_{k+1} = c_k - g R s_k`` acting on an aberration the mirror can reproduce, with a
  reconstructor inverting the sensor (``R D = I``), leaves the error ``(1 - g)^k`` times the initial one after
  ``k`` frames (discrete control, Hardy 1998, §8.2);
- a continuous-facesheet mirror of pitch ``d`` corrects Kolmogorov turbulence down to the fitting error
  ``σ² ≈ 0.28 (d/r0)^(5/3)`` rad² (Hudgin 1977, JOSA 67, 393; Hardy 1998, Table 9.1); an ideal Fried sensor
  adds aliasing of the same order (Rigaut et al. 1998), hence the range accepted below.
"""

import math

import numpy as np
import pytest
from astropy import units as u

from lumos import Context
from lumos.core.element import Element
from lumos.core.port import Output
from lumos.core.signal import OpticalSignal
from lumos.elements.apertures import CircularAperture
from lumos.elements.atmosphere import AOController, Atmosphere, DeformableMirror, WavefrontSensor
from lumos.elements.optics import PhasePlate
from lumos.fields import Grid, Wavefront

D = 8.0  # m
N = 128
GRID = Grid(N, dx=1.1 * D / N * u.m)


def pupil_field(wavelength=1 * u.um):
    plane = Wavefront.plane(GRID, wavelength, power=1 * u.W)
    return CircularAperture(diameter=D * u.m).apply(**{"in": plane})["out"]


def pupil():
    xx, yy = (np.asarray(c) for c in GRID.coords(backend="numpy"))
    return np.hypot(xx, yy) <= D / 2


def sensor(n=16):
    return WavefrontSensor(diameter=D * u.m, n_subapertures=n)


def measure(field, n=16):
    return sensor(n).apply(**{"in": field})


# --- Sensor -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("wavelength", [0.5, 1.65])
def test_the_sensor_measures_the_slope_of_a_tilt_at_any_wavelength(wavelength):
    a = 2e-6  # m, Noll coefficient of Z2: an optical path of 2a x / R, many waves across the pupil
    radius = 0.6 * D  # the plate covers the pupil and its grey edge pixels
    plate = PhasePlate(diameter=2 * radius * u.m, coefficients=[0, a] * u.m)
    field = plate.apply(**{"in": pupil_field(wavelength * u.um)})["out"]
    slopes = measure(field)["slopes"]
    data = slopes.data.to_value(u.rad)
    valid = ~np.asarray(slopes.mask)
    assert data.shape == (2, 16, 16)
    assert valid[0].sum() > 150
    assert not valid[0, 0, 0]  # corner of the lenslet array: outside the pupil
    np.testing.assert_allclose(data[0][valid[0]], 2 * a / radius, rtol=1e-6)
    np.testing.assert_allclose(data[1][valid[1]], 0, atol=1e-12)


def test_the_residual_is_the_rms_optical_path_over_the_pupil():
    rms = 30e-9  # m: a defocus of Noll coefficient a has an RMS of a
    plate = PhasePlate(diameter=D * u.m, coefficients=[0, 0, 0, rms] * u.m)
    field = plate.apply(**{"in": pupil_field(1.65 * u.um)})["out"]
    residual = measure(field)["residual"]
    assert residual.data.unit.is_equivalent(u.nm)
    assert float(residual.data.to_value(u.nm)) == pytest.approx(30, rel=0.02)


# --- Controller -------------------------------------------------------------------------------------------


def geometry(n_actuators=17):
    # Fried geometry: actuators at the corners of the sub-apertures
    return {"n_actuators": n_actuators, "diameter": D * u.m, "coupling": 0.15}


def test_the_controller_reconstructs_the_shape_of_its_mirror():
    rng = np.random.default_rng(3)
    shape = DeformableMirror(**geometry())
    shape.commands = rng.normal(size=17 * 17) * 20 * u.nm
    field = shape.apply(**{"in": pupil_field(1.65 * u.um)})["out"]
    controller = AOController(**geometry(), gain=1.0)
    correction = controller.apply(slopes=measure(field)["slopes"])["commands"]
    corrected = DeformableMirror(**geometry())
    corrected.commands = shape.commands + correction.data
    inside = pupil()
    before = shape.opd(GRID).to_value(u.nm)[inside]
    after = corrected.opd(GRID).to_value(u.nm)[inside]
    assert np.std(after) < 0.1 * np.std(before)


def test_the_controller_refuses_a_mirror_of_another_geometry():
    ctx = Context(seed=1)
    ctx.add(DeformableMirror(name="dm", **geometry(9)))
    ctx.add(AOController(name="ctrl", **geometry(17)))
    with pytest.raises(ValueError, match="17"):
        ctx.feedback("ctrl.commands", "dm.commands", mode="add")
    ctx.add(DeformableMirror(name="other", **geometry(17)))
    with pytest.raises(ValueError, match="commands"):
        ctx.feedback("ctrl.commands", "other.coupling")


# --- Closed loops -----------------------------------------------------------------------------------------


class Plane(Element):
    """A plane wave at 1.65 um on the grid of the tests."""

    out = Output(OpticalSignal)

    @staticmethod
    def transfer(values, inputs):
        return {"out": Wavefront.plane(GRID, 1.65 * u.um, power=1 * u.W)}


def closed_loop(ctx, aberration, gain):
    """Plane wave, pupil, ``aberration``, corrector, sensor and controller; the loop drives the corrector."""
    source = ctx.add(Plane(name="source"))
    stop = ctx.add(CircularAperture(name="pupil", diameter=D * u.m))
    corrector = ctx.add(DeformableMirror(name="dm", **geometry()))
    wfs = ctx.add(WavefrontSensor(name="wfs", diameter=D * u.m, n_subapertures=16))
    controller = ctx.add(AOController(name="ctrl", **geometry(), gain=gain))
    source >> stop >> aberration >> corrector >> wfs
    ctx.connect(wfs.slopes, controller.slopes)
    ctx.feedback("ctrl.commands", "dm.commands", mode="add")
    return corrector


def test_a_closed_loop_on_a_static_aberration_converges_geometrically():
    rng = np.random.default_rng(5)
    ctx = Context(seed=2)
    aberration = DeformableMirror(name="aberration", **geometry())
    aberration.commands = rng.normal(size=17 * 17) * 15 * u.nm
    ctx.add(aberration)
    corrector = closed_loop(ctx, aberration, gain=0.5)
    steps = ctx.loop(12, period=1 * u.ms, outputs=["wfs.residual"])
    residuals = np.array([float(s.outputs["wfs.residual"].data.to_value(u.nm)) for s in steps])
    for k in range(1, 5):
        assert residuals[k] / residuals[0] == pytest.approx(0.5**k, rel=0.15)
    assert residuals[-1] < 0.05 * residuals[0]
    inside = pupil()
    total = aberration.opd(GRID).to_value(u.nm) + corrector.opd(GRID).to_value(u.nm)
    assert np.std(total[inside]) < 0.05 * np.std(aberration.opd(GRID).to_value(u.nm)[inside])


def test_a_closed_loop_corrects_turbulence_down_to_the_fitting_error():
    ctx = Context(seed=7)
    r0_500 = 0.5  # m at 500 nm
    atmosphere = Atmosphere(
        name="atm",
        r0=r0_500 * u.m,
        wind_speed=[0.0] * u.m / u.s,
        pupil_diameter=D * u.m,
        outer_scale=1e4 * u.m,
    )
    ctx.add(atmosphere)
    closed_loop(ctx, atmosphere, gain=0.5)
    steps = ctx.loop(25, period=1 * u.ms, outputs=["wfs.residual"])
    residual = float(steps[-1].outputs["wfs.residual"].data.to_value(u.m))
    wavelength = 1.65e-6
    sigma2 = (2 * math.pi * residual / wavelength) ** 2  # rad² at the sensing wavelength
    r0 = r0_500 * (wavelength / 0.5e-6) ** 1.2
    pitch = D / 16
    coefficient = sigma2 / (pitch / r0) ** (5 / 3)
    assert 0.15 < coefficient < 0.6, coefficient
