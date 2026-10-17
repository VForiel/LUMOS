"""Tests for symbolic fields inside contexts (step P10.6, REQ-SYM-4, REQ-SYM-5, REQ-EL-9).

Requirements:

- symbolic fields flow through the graph: elements with a symbolic transfer keep them symbolic;
- they are converted to numbers automatically (on their grid hint, or at the wavelengths of the observation
  when the expression depends on λ) before an element that has no symbolic transfer or no closed form, and the
  journal records the conversion (REQ-SYM-5);
- ``to_numeric`` samples a symbolic field on its grid hint, at given wavelengths when it depends on λ;
- sources of symbolic fields are saved in ``.lumos`` files without evaluating code (NF-7).

Independent references:

- a Gaussian beam of waist w₀ at a thin lens of focal length f has its new waist at
  d' = f / (1 + (f/z_R)²) with w₀' = (λf/πw₀) / √(1 + (f/z_R)²), z_R = πw₀²/λ, and a flat phase there
  (Self 1983, Appl. Opt. 22, 658, eq. 9 and 10 for an input waist at the lens);
- the power of a beam is conserved by free space and lenses, and a power meter reads it;
- the irradiance of a Gaussian beam of power P is 2P/(πw²) exp(-2r²/w²).
"""

import logging
import math

import numpy as np
import pytest
import sympy as sp
from astropy import units as u

import lumos.io
from lumos import Context
from lumos.backend import to_numpy
from lumos.elements.apertures import CircularAperture
from lumos.elements.detectors import PowerMeter
from lumos.elements.optics import FreeSpace, SpectralFilter, ThinLens
from lumos.fields import Grid, Wavecube, Wavefront
from lumos.symbolic import SymbolicField, gaussian_beam, lam, x, y
from lumos.symbolic.evaluate import to_wavefront
from lumos.symbolic.integration import to_numeric
from lumos.symbolic.source import SymbolicSource

LAMBDA = 1.0e-6
W0 = 1.0e-3
F = 0.2
POWER = 2e-3


def focused_waist():
    z_r = math.pi * W0**2 / LAMBDA
    factor = 1 + (F / z_r) ** 2
    return F / factor, (LAMBDA * F / (math.pi * W0)) / math.sqrt(factor)


def test_gaussian_beam_through_a_lens_and_free_space_stays_symbolic():
    distance, waist = focused_waist()
    ctx = Context()
    source = ctx.add(
        SymbolicSource.gaussian_beam(name="beam", waist=W0 * u.m, wavelength=LAMBDA * u.m, power=POWER * u.W)
    )
    (
        source
        >> ctx.add(ThinLens(name="lens", focal_length=F * u.m))
        >> ctx.add(FreeSpace(name="gap", distance=distance * u.m))
    )
    out = ctx.run(["gap.out"])["gap.out"]
    assert isinstance(out, SymbolicField)
    grid = Grid(129, dx=waist / 16 * u.m)
    field = np.asarray(to_numpy(to_wavefront(out, grid).field))
    xx, yy = (np.asarray(c) for c in grid.coords(backend="numpy"))
    expected = 2 * POWER / (math.pi * waist**2) * np.exp(-2 * (xx**2 + yy**2) / waist**2)
    np.testing.assert_allclose(np.abs(field) ** 2, expected, rtol=1e-6, atol=1e-6 * expected.max())
    centre = field[64, 64]
    at_waist = field[64, 64 + 16]
    assert abs(np.angle(at_waist / centre)) < 1e-6


def test_conversion_to_numbers_before_an_element_without_symbolic_transfer(caplog):
    grid = Grid(128, dx=40 * u.um)
    ctx = Context()
    source = ctx.add(
        SymbolicSource.gaussian_beam(
            name="beam", waist=0.5 * u.mm, wavelength=633 * u.nm, power=POWER * u.W, grid=grid
        )
    )
    source >> ctx.add(ThinLens(name="lens", focal_length=1 * u.m)) >> ctx.add(PowerMeter(name="meter"))
    with caplog.at_level(logging.INFO):
        reading = ctx.run(["meter.out"])["meter.out"]
    assert float(np.sum(reading.data.to_value(u.W))) == pytest.approx(POWER, rel=1e-6)
    messages = [record.getMessage() for record in caplog.records]
    assert any("meter" in m and "converted" in m for m in messages)


def test_numerical_fallback_without_closed_form_is_logged(caplog):
    grid = Grid(64, dx=50 * u.um)
    beam = gaussian_beam(waist=0.6 * u.mm, wavelength=633 * u.nm, power=1 * u.mW, grid=grid)
    ctx = Context()
    source = ctx.add(SymbolicSource(name="beam", field=beam))
    aperture = ctx.add(
        CircularAperture(name="pupil", diameter=2 * u.mm, n_spiders=3, spider_width=0.1 * u.mm)
    )
    source >> aperture
    with caplog.at_level(logging.INFO):
        out = ctx.run(["pupil.out"])["pupil.out"]
    assert isinstance(out, Wavefront)
    assert any("pupil" in r.getMessage() and "fallback" in r.getMessage() for r in caplog.records)
    numeric = CircularAperture(diameter=2 * u.mm, n_spiders=3, spider_width=0.1 * u.mm).apply(
        **{"in": to_wavefront(beam, grid)}
    )["out"]
    np.testing.assert_allclose(
        np.asarray(to_numpy(out.field)), np.asarray(to_numpy(numeric.field)), atol=1e-12
    )


def test_conversion_needs_a_grid():
    ctx = Context()
    source = ctx.add(
        SymbolicSource.gaussian_beam(name="beam", waist=1 * u.mm, wavelength=1 * u.um, power=1 * u.W)
    )
    source >> ctx.add(PowerMeter(name="meter"))
    with pytest.raises(ValueError, match="grid"):
        ctx.run(["meter.out"])


def test_chromatic_fields_are_sampled_at_the_observation_wavelengths():
    grid = Grid(32, dx=0.1 * u.mm)
    w = sp.Symbol("w", positive=True)
    chromatic = SymbolicField(
        sp.exp(-(x**2 + y**2) / w**2) * sp.sqrt(lam / sp.Float(1e-6)), parameters={w: 1 * u.mm}, grid=grid
    )
    ctx = Context()
    ctx.observation.band = (1.0 * u.um, 1.2 * u.um)
    ctx.observation.n_wavelengths = 3
    source = ctx.add(SymbolicSource(name="src", field=chromatic))
    source >> ctx.add(PowerMeter(name="meter"))
    reading = ctx.run(["meter.out"])["meter.out"]
    # spectral filter as a symbolic element in front keeps the field symbolic
    ctx2 = Context()
    ctx2.observation.band = (1.0 * u.um, 1.2 * u.um)
    ctx2.observation.n_wavelengths = 3
    src2 = ctx2.add(SymbolicSource(name="src", field=chromatic))
    filt = ctx2.add(SpectralFilter(name="filter", center=1.1 * u.um, width=0.5 * u.um, transmission=1.0))
    src2 >> filt
    assert isinstance(ctx2.run(["filter.out"])["filter.out"], SymbolicField)
    wavelengths, weights = ctx.observation.wavelengths()
    cube = to_numeric(chromatic, wavelengths=wavelengths, weights=weights)
    assert isinstance(cube, Wavecube)
    np.testing.assert_allclose(cube.wavelengths.to_value(u.m), wavelengths.to_value(u.m))
    # |U|² ∝ λ: the power at each wavelength grows linearly, weighted by the quadrature weights
    single = math.pi * (1e-3) ** 2 / 2  # ∫ exp(-2r²/w²) dA
    expected = single * float(np.sum(np.asarray(weights) * wavelengths.to_value(u.m) / 1e-6))
    assert float(np.sum(reading.data.to_value(u.W))) == pytest.approx(expected, rel=1e-3)


def test_symbolic_sources_are_saved_and_loaded(tmp_path):
    grid = Grid(32, dx=0.1 * u.mm)
    ctx = Context()
    source = ctx.add(
        SymbolicSource.gaussian_beam(
            name="beam",
            waist=1 * u.mm,
            wavelength=1 * u.um,
            power=1 * u.mW,
            family="lg",
            indices=(1, 2),
            grid=grid,
        )
    )
    source >> ctx.add(ThinLens(name="lens", focal_length=0.5 * u.m))
    lumos.io.save(ctx, tmp_path / "ctx.lumos")
    loaded = lumos.io.load(tmp_path / "ctx.lumos")
    a = ctx.run(["lens.out"])["lens.out"]
    b = loaded.run(["lens.out"])["lens.out"]
    assert isinstance(b, SymbolicField)
    np.testing.assert_allclose(
        np.asarray(to_numpy(to_wavefront(b, grid).field)),
        np.asarray(to_numpy(to_wavefront(a, grid).field)),
        rtol=1e-12,
    )


def test_arbitrary_expressions_are_saved_and_loaded(tmp_path):
    w = sp.Symbol("w", positive=True)
    expression = sp.exp(-(x**2 + y**2) / w**2) * sp.Piecewise((1, sp.sqrt(x**2 + y**2) <= w), (0, True))
    field = SymbolicField(expression, parameters={w: 2 * u.mm}, wavelength=500 * u.nm)
    lumos.io.save(SymbolicSource(name="s", field=field), tmp_path / "s.lumos")
    loaded = lumos.io.load(tmp_path / "s.lumos")
    assert sp.simplify(loaded.field.expression - expression) == 0
    assert u.isclose(loaded.field.parameters[w], 2 * u.mm)


def test_loading_refuses_unknown_functions(tmp_path):
    import json

    field = SymbolicField(sp.exp(-(x**2)), wavelength=500 * u.nm)
    path = tmp_path / "s.lumos"
    lumos.io.save(SymbolicSource(name="s", field=field), path)
    document = json.loads(path.read_text(encoding="utf-8"))
    text = json.dumps(document).replace('"exp"', '"__import__"')
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="__import__"):
        lumos.io.load(path)
