"""Tests for telescopes and telescope arrays (step P6.4).

Requirements: §5.9 (telescope, telescope array with VLTI and LIFE presets), REQ-SCN-2 (sources at infinity
collected by a pupil), UC1 (an array feeding a recombiner).

Independent references:

- a pupil of diameter D with a central obstruction d collects ``π (D² - d²) / 4 · Σ_k F_k b_k`` photons per
  second from a source of spectral photon flux density F_k over channels of widths b_k;
- an off-axis source at angle θ is imaged at ``f θ`` in the focal plane of a lens (geometrical optics);
- projection of a ground baseline (East, North, Up) on the plane of the sky for a target of declination δ
  at hour angle H, from a site at latitude φ (Thompson, Moran & Swenson, *Interferometry and Synthesis in
  Radio Astronomy*, 3rd ed., eqs. 4.1 and 4.15): at the zenith (δ = φ, H = 0) the projected baseline is the
  ground baseline; an East-West baseline B traces the ellipse ``u = B cos H``, ``v = B sin δ sin H``;
- the hour angle grows by 15° per sidereal hour (sidereal day of 86164.0905 s, IERS Conventions 2010);
- the lengths of the six VLTI unit-telescope baselines published by ESO: UT1-UT2 56.6 m, UT1-UT3 102.4 m,
  UT1-UT4 130.2 m, UT2-UT3 46.6 m, UT2-UT4 89.4 m, UT3-UT4 62.5 m (ESO VLTI configuration overview), and the
  8.2 m diameter of the unit telescopes;
- LIFE: four 2 m collectors in an X-array whose imaging baseline is 6 times the nulling baseline (Quanz et
  al. 2022, A&A 664, A21);
- an ideal single-mode injection gives each telescope the amplitude ``sqrt(A F_k b_k)`` with the phase
  ``2π (u θ_x + v θ_y) / λ`` of the plane wave of the source at the projected position (u, v) of the
  telescope.
"""

import itertools
import math

import numpy as np
import pytest
from astropy import units as u

from lumos import Context
from lumos.elements.optics import FreeSpace, ThinLens
from lumos.elements.sources import PointSource
from lumos.elements.telescopes import Telescope, TelescopeArray, life, vlti
from lumos.fields import Grid, IncoherentBundle, ModalField

SIDEREAL_DAY = 86164.0905  # s


def band_context(low_um, high_um, n):
    ctx = Context()
    ctx.observation.band = (low_um * u.um, high_um * u.um)
    ctx.observation.n_wavelengths = n
    return ctx


def tms_uv(east, north, up, latitude, declination, hour_angle):
    """Thompson, Moran & Swenson eqs. 4.15 then 4.1 (angles in radians)."""
    x = -math.sin(latitude) * north + math.cos(latitude) * up
    y = east
    z = math.cos(latitude) * north + math.sin(latitude) * up
    h, d = hour_angle, declination
    uu = math.sin(h) * x + math.cos(h) * y
    vv = -math.sin(d) * math.cos(h) * x + math.sin(d) * math.sin(h) * y + math.cos(d) * z
    return uu, vv


# --- Single telescope ------------------------------------------------------------------------------


def test_telescope_collects_the_flux_of_its_area():
    ctx = band_context(1.0, 1.2, 2)
    source = ctx.add(PointSource(name="src", flux_density=1 * u.Jy))
    scope = Telescope(name="scope", diameter=8 * u.m, obstruction=1 * u.m, n_pixels=256)
    source >> scope
    bundle = ctx.run(["scope.out"])["scope.out"]
    sources = ctx.run(["src.out"])["src.out"]
    assert isinstance(bundle, IncoherentBundle)
    area = math.pi * (8**2 - 1**2) / 4 * u.m**2
    expected = sources.photon_rate(area).to_value(u.photon / u.s)
    np.testing.assert_allclose(bundle.power.to_value(u.photon / u.s), expected, rtol=2e-3)


def test_telescope_keeps_one_mode_per_source():
    ctx = band_context(1.0, 1.2, 3)
    a = ctx.add(PointSource(name="a", flux_density=1 * u.Jy))
    b = ctx.add(PointSource(name="b", flux_density=4 * u.Jy, theta_x=1e-6 * u.rad))
    scope = ctx.add(Telescope(name="scope", diameter=2 * u.m, n_pixels=64))
    ctx.connect(a.port("out"), scope.port("in"))
    ctx.connect(b.port("out"), scope.port("in"))
    bundle = ctx.run(["scope.out"])["scope.out"]
    assert bundle.n_modes == 2
    assert bundle[0].n_wavelengths == 3
    assert bundle.power[1].value == pytest.approx(4 * bundle.power[0].value, rel=1e-9)


def test_off_axis_source_is_imaged_at_f_theta():
    ctx = band_context(1.0, 1.0001, 1)
    theta = 2e-6
    source = ctx.add(PointSource(name="src", flux_density=1 * u.Jy, theta_x=theta * u.rad))
    f = 10.0
    focal = Grid(64, dx=0.25e-6 * f / 1.0 * u.m)  # λ/D / 4 at D = 1 m
    chain = source >> Telescope(name="scope", diameter=1 * u.m, n_pixels=128)
    chain = chain >> ThinLens(name="lens", focal_length=f * u.m)
    chain >> FreeSpace(name="gap", distance=f * u.m, method="fresnel_mft", output_grid=focal)
    image = np.asarray(ctx.run(["gap.out"])["gap.out"].intensity)
    xx, yy = (np.asarray(c) for c in focal.coords(backend="numpy"))
    peak = np.unravel_index(np.argmax(image), image.shape)
    assert xx[peak] == pytest.approx(f * theta, abs=1e-12)  # f θ = 20 µm falls on pixel 8
    assert yy[peak] == pytest.approx(0.0, abs=1e-12)


# --- Projected baselines ---------------------------------------------------------------------------


def test_projected_baselines_at_the_zenith_are_the_ground_baselines():
    lat = -24.6 * u.deg
    positions = [[0, 0, 0], [30, 40, 0], [-10, 25, 0]] * u.m
    array = TelescopeArray(
        positions=positions, latitude=lat, declination=lat, hour_angle=0 * u.hourangle, diameter=1 * u.m
    )
    uv = array.projected_positions().to_value(u.m)
    np.testing.assert_allclose(uv, positions[:, :2].to_value(u.m), atol=1e-9)


@pytest.mark.parametrize("hour_angle", [-3.0, -1.0, 0.0, 2.0, 4.5])
@pytest.mark.parametrize("declination", [-60.0, -20.0, 10.0])
def test_east_west_baseline_traces_an_ellipse(hour_angle, declination):
    array = TelescopeArray(
        positions=[[0, 0, 0], [100, 0, 0]] * u.m,
        latitude=-24.6 * u.deg,
        declination=declination * u.deg,
        hour_angle=hour_angle * u.hourangle,
        diameter=1 * u.m,
    )
    (baseline,) = np.asarray(array.baselines().to_value(u.m))
    h, d = math.radians(15 * hour_angle), math.radians(declination)
    np.testing.assert_allclose(baseline, [100 * math.cos(h), 100 * math.sin(d) * math.sin(h)], atol=1e-9)


def test_projection_follows_the_textbook_formula():
    lat, dec, ha = math.radians(-24.6), math.radians(-40.0), math.radians(25.0)
    positions = np.array([[0.0, 0.0, 0.0], [12.0, -35.0, 3.0], [80.0, 20.0, -1.5]])
    array = TelescopeArray(
        positions=positions * u.m,
        latitude=lat * u.rad,
        declination=dec * u.rad,
        hour_angle=ha * u.rad,
        diameter=1 * u.m,
    )
    expected = [tms_uv(*p, lat, dec, ha) for p in positions]
    np.testing.assert_allclose(array.projected_positions().to_value(u.m), expected, atol=1e-9)


def test_hour_angle_advances_with_the_clock():
    ctx = Context()
    lat, dec = math.radians(-24.6), math.radians(-30.0)
    array = ctx.add(
        TelescopeArray(
            name="array",
            positions=[[0, 0, 0], [50, 50, 0]] * u.m,
            latitude=lat * u.rad,
            declination=dec * u.rad,
            hour_angle=-1 * u.hourangle,
            diameter=1 * u.m,
        )
    )
    ctx.wait(SIDEREAL_DAY / 24 * u.s)
    (baseline,) = np.asarray(array.baselines().to_value(u.m))
    np.testing.assert_allclose(baseline, tms_uv(50, 50, 0, lat, dec, 0.0), atol=1e-6)


def test_sky_frame_array_rotates_in_the_plane_of_the_sky():
    ctx = Context()
    array = ctx.add(
        TelescopeArray(
            name="array",
            frame="sky",
            positions=[[0, 0], [10, 0]] * u.m,
            rotation=0 * u.deg,
            rotation_rate=90 * u.deg / u.hour,
            diameter=1 * u.m,
        )
    )
    np.testing.assert_allclose(array.baselines().to_value(u.m), [[10, 0]], atol=1e-12)
    ctx.wait(1 * u.hour)
    np.testing.assert_allclose(array.baselines().to_value(u.m), [[0, 10]], atol=1e-9)


# --- Presets ---------------------------------------------------------------------------------------


def test_vlti_unit_telescope_baselines_match_eso():
    array = vlti()
    assert array.n_telescopes == 4
    assert array.diameter.to_value(u.m) == pytest.approx(8.2)
    positions = array.ground_positions().to_value(u.m)
    lengths = {
        (i + 1, j + 1): float(np.hypot(*(positions[j, :2] - positions[i, :2])))
        for i, j in itertools.combinations(range(4), 2)
    }
    published = {(1, 2): 56.6, (1, 3): 102.4, (1, 4): 130.2, (2, 3): 46.6, (2, 4): 89.4, (3, 4): 62.5}
    for pair, length in published.items():
        assert lengths[pair] == pytest.approx(length, abs=0.1), pair


def test_life_is_a_six_to_one_x_array_of_two_metre_collectors():
    array = life(nulling_baseline=12 * u.m)
    assert array.n_telescopes == 4
    assert array.diameter.to_value(u.m) == pytest.approx(2.0)
    lengths = sorted(np.hypot(*b) for b in np.asarray(array.baselines().to_value(u.m)))
    nulling, imaging = 12.0, 72.0
    expected = sorted(
        [nulling, nulling, imaging, imaging, math.hypot(nulling, imaging), math.hypot(nulling, imaging)]
    )
    np.testing.assert_allclose(lengths, expected, rtol=1e-12)


# --- Ideal injection -------------------------------------------------------------------------------


def test_array_output_amplitudes_and_phases():
    ctx = band_context(3.0, 4.0, 4)
    tx, ty = 5e-8, -2e-8
    on_axis = ctx.add(PointSource(name="star", flux_density=1 * u.Jy))
    off_axis = ctx.add(
        PointSource(name="planet", flux_density=0.01 * u.Jy, theta_x=tx * u.rad, theta_y=ty * u.rad)
    )
    array = ctx.add(
        TelescopeArray(
            name="array", frame="sky", positions=[[0, 0], [20, 5], [-7, 30]] * u.m, diameter=2 * u.m
        )
    )
    ctx.connect(on_axis.port("out"), array.port("in"))
    ctx.connect(off_axis.port("out"), array.port("in"))
    result = ctx.run(["array.out", "star.out", "planet.out"])
    out = result["array.out"]
    assert isinstance(out, ModalField)
    amplitudes = np.asarray(out.amplitudes)
    assert amplitudes.shape == (2, 4, 3)

    area = math.pi * 2.0**2 / 4
    sources = [result["star.out"], result["planet.out"]]
    for m, source in enumerate(sources):
        flux = source.flux.to_value(u.photon / u.s / u.m**3)[0]
        bandwidth = source.bandwidths.to_value(u.m)
        np.testing.assert_allclose(
            np.abs(amplitudes[m]) ** 2, (area * flux * bandwidth)[:, None] * np.ones(3), rtol=1e-12
        )

    lam = result["star.out"].wavelengths.to_value(u.m)
    uv = np.array([[0, 0], [20, 5], [-7, 30]], dtype=float)
    expected_phase = 2 * math.pi * (uv[:, 0] * tx + uv[:, 1] * ty)[None, :] / lam[:, None]
    relative = amplitudes[1] / amplitudes[1][:, :1] * np.abs(amplitudes[1][:, :1]) / np.abs(amplitudes[1])
    np.testing.assert_allclose(relative, np.exp(1j * (expected_phase - expected_phase[:, :1])), atol=1e-10)
    np.testing.assert_allclose(np.angle(amplitudes[0]), 0.0, atol=1e-12)


def test_array_survives_a_lumos_round_trip():
    from lumos.core.element import Element

    array = vlti()
    loaded = Element.from_lumos(array.to_lumos())
    np.testing.assert_allclose(loaded.baselines().to_value(u.m), array.baselines().to_value(u.m))
