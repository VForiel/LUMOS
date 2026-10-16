"""Comparison of LUMOS propagation with POPPY and HCIPy (step P4.9, extra ``[validation]``).

Both libraries are developed independently of LUMOS (Perrin et al. 2012, Proc. SPIE 8442, 84423D; Por et
al. 2018, Proc. SPIE 10703, 1070342). The same input array is given to each library, so that only the
propagation itself is compared:

- **far field and focal plane** (matrix Fourier transforms in all three codes): the intensities agree to
  round-off, tolerance 1e-9 of the peak (POPPY works with angles in arcsec, converted with astropy);
- **near field** (angular spectrum and Fresnel): HCIPy's propagators are used as a sanity check at 1 % of
  the peak intensity. They are not exact at the 1e-3 level: on the case below, HCIPy's angular spectrum
  departs from the direct Rayleigh-Sommerfeld sum by 1.7e-3 of the peak intensity, while LUMOS's agrees
  with it to 1e-12 (the direct sum is validated independently in P4.1).

Grids have an odd number of pixels so that the centre conventions coincide (HCIPy centres even grids on a
pixel corner, LUMOS on pixel ``N // 2``).
"""

import math

import numpy as np
import pytest
from astropy import units as u

from lumos.fields import Grid, Wavefront
from lumos.propagation import focus, propagate

poppy = pytest.importorskip("poppy")
hcipy = pytest.importorskip("hcipy")

N = 127
DX = 2e-6
LAM = 1e-6
GRID = Grid(N, dx=DX * u.m)


def coordinates():
    return (np.asarray(c) for c in GRID.coords(backend="numpy"))


def disk(radius=100e-6):
    xx, yy = coordinates()
    return ((xx**2 + yy**2) <= radius**2).astype(float)


@pytest.mark.filterwarnings("ignore")
def test_far_field_matches_poppy():
    mask = disk()
    optic = poppy.ArrayOpticalElement(transmission=mask, pixelscale=DX * u.m / u.pixel, name="pupil")
    system = poppy.OpticalSystem(npix=N, oversample=1, pupil_diameter=N * DX * u.m)
    system.add_pupil(optic)
    pixel = (0.05 * LAM / (N * DX) * u.rad).to(u.arcsec)
    system.add_detector(pixelscale=pixel.value, fov_pixels=21)
    reference = system.calc_psf(wavelength=LAM, normalize="first")[0].data  # power fraction per pixel

    wf = Wavefront(mask, GRID, LAM * u.m).normalized(1 * u.W)
    far = propagate(wf, None, output_grid=Grid(21, dx=pixel), method="fraunhofer")
    ours = np.asarray(far.intensity) * pixel.to_value(u.rad) ** 2
    np.testing.assert_allclose(ours, reference, rtol=0, atol=1e-9 * reference.max())


def test_focal_plane_matches_hcipy():
    mask = disk()
    f = 0.5
    m, step = 33, 0.3 * LAM * 0.5 / 200e-6
    pupil_grid = hcipy.make_pupil_grid(N, N * DX)
    focal_grid = hcipy.make_uniform_grid([m, m], [m * step, m * step])
    field = hcipy.Wavefront(hcipy.Field(mask.ravel().astype(complex), pupil_grid), LAM)
    reference = hcipy.FraunhoferPropagator(pupil_grid, focal_grid, focal_length=f).forward(field)
    reference = np.abs(np.asarray(reference.electric_field).reshape(m, m)) ** 2

    image = focus(Wavefront(mask, GRID, LAM * u.m), f * u.m, Grid(m, dx=step * u.m), method="fraunhofer")
    np.testing.assert_allclose(np.asarray(image.intensity), reference, rtol=0, atol=1e-9 * reference.max())


@pytest.mark.parametrize(
    ("propagator", "method"), [("AngularSpectrumPropagator", "asm"), ("FresnelPropagator", "fresnel_tf")]
)
def test_near_field_agrees_with_hcipy(propagator, method):
    xx, yy = coordinates()
    field = np.exp(-(xx**2 + yy**2) / (20e-6) ** 2) * np.exp(1j * xx / 30e-6)
    z = 1.2e-3
    pupil_grid = hcipy.make_pupil_grid(N, N * DX)
    wave = hcipy.Wavefront(hcipy.Field(field.ravel(), pupil_grid), LAM)
    reference = getattr(hcipy, propagator)(pupil_grid, z).forward(wave)
    reference = np.abs(np.asarray(reference.electric_field).reshape(N, N)) ** 2

    ours = np.asarray(propagate(Wavefront(field, GRID, LAM * u.m), z * u.m, method=method).intensity)
    np.testing.assert_allclose(ours, reference, rtol=0, atol=1e-2 * reference.max())
    assert math.isclose(ours.sum(), reference.sum(), rel_tol=1e-2)
