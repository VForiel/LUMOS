"""Tests for interferometric processing (step P11.2, REQ-PIPE-3).

Requirements: null depth, kernel observables and transmission maps, as functions and as elements.

Independent references:

- a two-beam nuller with equal amplitudes and a phase error φ has the null depth
  N = I_dark / I_bright = tan²(φ/2) (from |1 - e^{iφ}|² / |1 + e^{iφ}|²; Serabyn 2000, Proc. SPIE 4006, 328,
  gives the small-error form N ≈ φ²/4);
- the transmission map of a Bracewell nuller of baseline B is sin²(π B θ_x / λ) (Bracewell 1978, Nature 274,
  780);
- the kernels of the kernel nuller vanish to second order in the piston errors of an on-axis source
  (Martinache & Ireland 2018, A&A 619, A87): they scale as ε³, while the dark outputs scale as ε²;
- ratios and differences of independent measurements propagate their uncertainties as in the GUM
  (JCGM 100:2008, §5.1).
"""

import math

import numpy as np
import pytest
from astropy import units as u

from lumos import Context
from lumos.elements.detectors import PowerMeter
from lumos.elements.photonics import PhaseShifter, bracewell, kernel_nuller
from lumos.elements.sources import PointSource
from lumos.elements.telescopes import TelescopeArray
from lumos.fields import DataProduct
from lumos.processing.interferometry import (
    KernelObservables,
    NullDepth,
    kernel_observables,
    null_depth,
    transmission_map,
)


def band(low, high, n=1):
    ctx = Context()
    ctx.observation.band = (low * u.um, high * u.um)
    ctx.observation.n_wavelengths = n
    return ctx


def bracewell_bench(phase_error=0.0, baseline=20.0):
    ctx = band(9.99, 10.01)
    source = ctx.add(PointSource(name="src", flux_density=1 * u.Jy))
    array = ctx.add(
        TelescopeArray(
            name="array",
            frame="sky",
            positions=[[-baseline / 2, 0.0], [baseline / 2, 0.0]] * u.m,
            diameter=2 * u.m,
        )
    )
    error = ctx.add(PhaseShifter(name="error", phase=phase_error * u.rad, achromatic=True, channels=[1]))
    source >> array >> error >> bracewell(name="chip") >> PowerMeter(name="meter")
    return ctx, source


@pytest.mark.parametrize("phase_error", [0.01, 0.1, 0.5])
def test_null_depth_of_a_bracewell_with_a_phase_error(phase_error):
    ctx, _ = bracewell_bench(phase_error)
    outputs = ctx.run(["meter.out"])["meter.out"]
    depth = null_depth(outputs, dark=0, bright=1)
    assert depth.data[0].value == pytest.approx(math.tan(phase_error / 2) ** 2, rel=1e-9)
    element = NullDepth(dark=0, bright=1)
    assert element.apply(**{"in": outputs})["out"].data[0].value == pytest.approx(depth.data[0].value)


def test_null_depth_uncertainty():
    outputs = DataProduct(np.array([2.0, 100.0]) * u.W, uncertainty=np.array([0.1, 3.0]) * u.W)
    depth = null_depth(outputs, dark=0, bright=1)
    assert depth.data[0].value == pytest.approx(0.02)
    assert depth.uncertainty[0].value == pytest.approx(math.sqrt((0.1 / 100) ** 2 + (2 * 3 / 100**2) ** 2))


def test_transmission_map_of_a_bracewell():
    ctx, source = bracewell_bench()
    lam, baseline = 10e-6, 20.0
    thetas = np.linspace(-lam / baseline, lam / baseline, 7) * u.rad
    tmap = transmission_map(ctx, source, "meter.out", theta_x=thetas, theta_y=thetas[:3], channel=0)
    assert tmap.shape == (3, 7)
    assert tmap.axis_names == ("theta_y", "theta_x")
    wavelength = 10e-6  # band centre: one sample at 10 µm
    expected = np.sin(math.pi * baseline * thetas.to_value(u.rad) / wavelength) ** 2
    for row in range(3):
        np.testing.assert_allclose(tmap.data.value[row], expected, atol=1e-9)
    assert source.theta_x == 0 * u.rad  # the position of the source is restored


def kernel_bench(pistons):
    ctx = band(3.5, 4.0)
    positions = np.array([[0.0, 0.0], [30.0, 10.0], [55.0, -20.0], [80.0, 25.0]])
    star = ctx.add(PointSource(name="star", flux_density=1 * u.Jy))
    array = ctx.add(TelescopeArray(name="array", frame="sky", positions=positions * u.m, diameter=2 * u.m))
    chain = star >> array
    for k, piston in enumerate(pistons):
        chain = chain >> PhaseShifter(name=f"piston{k}", phase=piston * u.rad, achromatic=True, channels=[k])
    chain >> kernel_nuller(name="chip") >> PowerMeter(name="meter")
    return ctx.run(["meter.out"])["meter.out"]


def test_kernels_are_robust_to_second_order():
    direction = np.array([0.3, -0.7, 0.2, 0.5])
    big, small = kernel_bench(1e-2 * direction), kernel_bench(0.5e-2 * direction)
    k_big, k_small = kernel_observables(big), kernel_observables(small)
    assert k_big.shape == (3,)
    ratio = np.abs(k_big.data.value / k_small.data.value)
    np.testing.assert_allclose(ratio, 8.0, rtol=0.02)  # third order
    dark_ratio = big.data.value[1:] / small.data.value[1:]
    np.testing.assert_allclose(dark_ratio, 4.0, rtol=0.02)  # second order
    normalised = kernel_observables(big, normalize=True)
    np.testing.assert_allclose(normalised.data.value, k_big.data.value / big.data.value[0])


def test_kernel_uncertainties_and_element():
    outputs = DataProduct(np.arange(7.0) * u.W, uncertainty=np.full(7, 0.2) * u.W)
    kernels = KernelObservables().apply(**{"in": outputs})["out"]
    np.testing.assert_allclose(kernels.data.value, [1 - 3, 2 - 5, 4 - 6])
    np.testing.assert_allclose(kernels.uncertainty.value, math.sqrt(2) * 0.2)
