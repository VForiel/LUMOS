"""Tests for the symbolic transfers of simple elements (step P10.3, REQ-EL-9, REQ-SYM-4).

Requirements: an element may provide a symbolic transfer function; the symbolic and numeric evaluations agree
within 1e-6 (plan, tests of phase 10).

Independent reference: the numeric transfer functions of the same elements, validated in phase 6 (thin lens
phase ``exp(-i k r² / 2f)``, delay ``exp(2iπ d / λ)``, beam splitter ``t = √(1-R)``, ``r = i√R``, top-hat
filter, aperture masks), applied to the sampled input field.
"""

import numpy as np
import pytest
import sympy as sp
from astropy import units as u

from lumos.backend import to_numpy
from lumos.elements.apertures import CircularAperture
from lumos.elements.atmosphere import Atmosphere
from lumos.elements.optics import BeamSplitter, DelayLine, SpectralFilter, ThinLens, ThinMirror
from lumos.fields import Grid
from lumos.symbolic import SymbolicField, lam, x, y
from lumos.symbolic.evaluate import to_wavecube, to_wavefront

W = sp.Symbol("w", positive=True)
GRID = Grid(64, dx=40 * u.um, x0=12 * u.um)


def beam(wavelength=633 * u.nm):
    return SymbolicField(sp.exp(-(x**2 + y**2) / W**2), parameters={W: 0.6 * u.mm}, wavelength=wavelength)


def compare(element, field, port="out", grid=GRID):
    symbolic = element.apply_symbolic(**{"in": field})[port]
    assert isinstance(symbolic, SymbolicField)
    via_symbols = np.asarray(to_numpy(to_wavefront(symbolic, grid).field))
    numeric = np.asarray(to_numpy(element.apply(**{"in": to_wavefront(field, grid)})[port].field))
    scale = np.abs(numeric).max()
    np.testing.assert_allclose(via_symbols, numeric, rtol=0, atol=1e-6 * scale)


@pytest.mark.parametrize("element", [ThinLens(focal_length=150 * u.mm), ThinMirror(focal_length=-80 * u.mm)])
def test_lenses_and_mirrors(element):
    compare(element, beam())


def test_delay_line():
    compare(DelayLine(delay=0.3 * u.um), beam())


def test_beam_splitter_outputs():
    splitter = BeamSplitter(reflectance=0.3)
    compare(splitter, beam(), port="reflected")
    compare(splitter, beam(), port="transmitted")


@pytest.mark.parametrize("center", [633 * u.nm, 800 * u.nm])
def test_spectral_filter(center):
    compare(SpectralFilter(center=center, width=50 * u.nm, transmission=0.8), beam())


def test_circular_aperture_without_anti_aliasing():
    aperture = CircularAperture(diameter=1.71 * u.mm, obstruction=0.43 * u.mm, supersample=1)
    compare(aperture, beam())


def test_chromatic_field_through_a_delay_line():
    field = SymbolicField(
        sp.exp(-(x**2 + y**2) / W**2) * sp.exp(sp.I * 2 * sp.pi * x / lam / 100), parameters={W: 0.6 * u.mm}
    )
    line = DelayLine(delay=0.4 * u.um)
    wavelengths = [0.6, 0.9] * u.um
    symbolic = line.apply_symbolic(**{"in": field})["out"]
    via_symbols = np.asarray(to_numpy(to_wavecube(symbolic, GRID, wavelengths).field))
    numeric = np.asarray(to_numpy(line.apply(**{"in": to_wavecube(field, GRID, wavelengths)})["out"].field))
    np.testing.assert_allclose(via_symbols, numeric, rtol=0, atol=1e-6 * np.abs(numeric).max())


def test_elements_without_symbolic_transfer():
    assert ThinLens.has_symbolic_transfer()
    assert not Atmosphere.has_symbolic_transfer()
    with pytest.raises(NotImplementedError, match="symbolic"):
        Atmosphere().apply_symbolic(**{"in": beam()})


def test_apertures_with_spiders_have_no_closed_form():
    aperture = CircularAperture(diameter=1 * u.mm, n_spiders=4, spider_width=10 * u.um)
    with pytest.raises(NotImplementedError):
        aperture.apply_symbolic(**{"in": beam()})


def test_symbolic_transfer_keeps_the_parameters_and_wavelength():
    out = ThinLens(focal_length=100 * u.mm).apply_symbolic(**{"in": beam()})["out"]
    assert out.parameters[W] == 0.6 * u.mm
    assert out.wavelength == 633 * u.nm
    assert "lambda" in out.latex()
