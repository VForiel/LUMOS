r"""Point, astronomical and laboratory sources (REQ-SCN-2, §5.9).

Sources at infinity (:class:`PointSource`, :class:`Star`, :class:`PlanetarySystem`) produce a
:class:`~lumos.fields.SourceList` sampled at the spectral channels of the context
(``ctx.observation.band``); a telescope then samples their plane waves on its pupil. Laboratory sources emit
fields on their own grid: :class:`GaussianLaser` a coherent :class:`~lumos.fields.Wavefront`,
:class:`IncoherentLamp` a :class:`~lumos.fields.RadianceField`.

Physics
-------
- A star of radius :math:`R`, temperature :math:`T` and distance :math:`d` has the flux density
  :math:`F_\lambda = \pi B_\lambda(T) (R/d)^2` (Rybicki & Lightman §1.4); given a magnitude, the blackbody
  shape is scaled so that :math:`F_\nu` at the effective wavelength of the band matches the zero point
  (:mod:`lumos.physics.photometry`).
- A planet at distance :math:`r` from its star reflects :math:`F_\star A_g \Phi(\alpha) (R_p/r)^2` (geometric
  albedo :math:`A_g`, Lambert phase function, phase angle :math:`\alpha`) and emits as a blackbody at its
  equilibrium temperature :math:`T_\mathrm{eq} = T_\star \sqrt{R_\star / 2r}\,(1 - A)^{1/4}` (the albedo is
  used as the Bond albedo).
- A Lambertian lamp of radiance :math:`L` and area :math:`A` emits :math:`P = \pi L A` (Lambert's law).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from astropy import units as u

from lumos.core.element import Element
from lumos.core.param import Param, Setting
from lumos.core.port import Output
from lumos.core.registry import register
from lumos.fields.grid import Grid
from lumos.fields.radiance import RadianceField
from lumos.fields.sources import PHOTON_FLUX_DENSITY, SourceList
from lumos.fields.wavefront import Wavefront
from lumos.physics.blackbody import planck_lambda_si
from lumos.physics.orbits import kepler_position_3d, orbital_period_si
from lumos.physics.photometry import BANDS, lambert_phase, photon_flux_from_fnu_si
from lumos.units import to_si

__all__ = ["GaussianLaser", "IncoherentLamp", "Planet", "PlanetarySystem", "PointSource", "Star"]

_H = 6.62607015e-34
_C = 299_792_458.0


def _channels(values: Any) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(values.wavelengths, dtype=float), np.asarray(values.bandwidths, dtype=float)


def _photon(energy_flux_lambda: Any, lam: np.ndarray) -> np.ndarray:
    """Energy flux density per unit wavelength (W m⁻² m⁻¹) → photon flux density."""
    return np.asarray(energy_flux_lambda) * lam / (_H * _C)


def _sources(tx: Any, ty: Any, lam: np.ndarray, bw: np.ndarray, flux: np.ndarray) -> SourceList:
    return SourceList(
        np.atleast_1d(tx) * u.rad,
        np.atleast_1d(ty) * u.rad,
        lam * u.m,
        bw * u.m,
        np.atleast_2d(flux) * PHOTON_FLUX_DENSITY,
    )


@register("lumos.sources.PointSource")
class PointSource(Element):
    """Unresolved source at infinity with a flat spectral flux density ``F_ν``."""

    observation_dependent = True
    flux_density = Param(unit=u.Jy, default=1 * u.Jy, bounds=(0 * u.Jy, None), doc="flux density F_nu")
    theta_x = Param(unit=u.rad, default=0 * u.rad, doc="direction along x")
    theta_y = Param(unit=u.rad, default=0 * u.rad, doc="direction along y")
    out = Output(SourceList)

    @staticmethod
    def transfer(values: Any, inputs: Any) -> dict[str, Any]:
        """Return the source list of the point source."""
        lam, bw = _channels(values)
        flux = photon_flux_from_fnu_si(values.flux_density, lam)
        return {"out": _sources(values.theta_x, values.theta_y, lam, bw, flux)}


def _stellar_photon_flux(
    temperature: float, radius: float, distance: float, band: str, magnitude: float, lam: np.ndarray
) -> np.ndarray:
    shape = planck_lambda_si(lam, temperature)  # W m-2 sr-1 m-1
    if band:
        reference = BANDS[band]
        lam_eff = float(to_si(reference.effective_wavelength))
        b_nu_eff = float(planck_lambda_si(lam_eff, temperature)) * lam_eff**2 / _C
        fnu = float(to_si(reference.zero_point)) * 10 ** (-0.4 * magnitude)
        return _photon(shape * fnu / b_nu_eff, lam)
    return _photon(math.pi * shape * (radius / distance) ** 2, lam)


@register("lumos.sources.Star")
class Star(Element):
    """Blackbody star, scaled by its radius and distance or by a magnitude in a photometric band."""

    observation_dependent = True
    temperature = Param(unit=u.K, default=5772 * u.K, bounds=(0 * u.K, None), doc="effective temperature")
    radius = Param(unit=u.R_sun, default=1 * u.R_sun, bounds=(0 * u.m, None), doc="stellar radius")
    distance = Param(
        unit=u.pc,
        default=10 * u.pc,
        bounds=(0 * u.m, None),
        inclusive=(False, True),
        doc="distance to the observer",
    )
    theta_x = Param(unit=u.rad, default=0 * u.rad, doc="direction along x")
    theta_y = Param(unit=u.rad, default=0 * u.rad, doc="direction along y")
    band = Setting(str, default="", doc="photometric band of the magnitude ('' to use radius and distance)")
    magnitude = Setting(float, default=0.0, doc="Vega magnitude in the band")
    out = Output(SourceList)

    @staticmethod
    def transfer(values: Any, inputs: Any) -> dict[str, Any]:
        """Return the source list of the star."""
        lam, bw = _channels(values)
        flux = _stellar_photon_flux(
            values.temperature, values.radius, values.distance, values.band, values.magnitude, lam
        )
        return {"out": _sources(values.theta_x, values.theta_y, lam, bw, flux)}

    @classmethod
    def from_catalog(cls, name: str, *, record: Any = None, **values: Any) -> Star:
        """Create the star ``name`` from SIMBAD (or from a star ``record``).

        The temperature is the catalogue effective temperature; the flux is scaled to the V magnitude when it
        is known (else to the radius and distance). The distance comes from the parallax. ``values`` override
        the catalogue.
        """
        from lumos.catalogs.simbad import Simbad

        record = record if record is not None else Simbad().star(name)
        found: dict[str, Any] = {}
        if "teff" in record:
            found["temperature"] = record.teff
        if "radius" in record:
            found["radius"] = record.radius
        if "distance" in record or "parallax" in record:
            found["distance"] = record.distance_estimate().value
        magnitudes = record.magnitudes()
        if "V" in magnitudes and "radius" not in record:
            found["band"] = "V"
            found["magnitude"] = float(magnitudes["V"].value.to_value(u.mag))
        return cls(name=name, **{**found, **values})


def _star_and_planet(values: Any, lam: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    """Photon fluxes of the star and of its planet, and the directions of both, at the current time."""
    star = _stellar_photon_flux(values.star_temperature, values.star_radius, values.distance, "", 0.0, lam)
    period = orbital_period_si(values.semi_major_axis, values.star_mass)
    x, y, z = kepler_position_3d(
        values.time,
        period=period,
        a=values.semi_major_axis,
        e=values.eccentricity,
        inclination=values.inclination,
        argument_periapsis=values.argument_periapsis,
        longitude_node=values.longitude_node,
        time_periapsis=-values.mean_anomaly / (2 * math.pi) * period,
    )
    r = math.sqrt(x * x + y * y + z * z)
    phase = math.acos(max(-1.0, min(1.0, -z / r)))
    reflected = star * values.albedo * float(lambert_phase(phase * u.rad)) * (values.planet_radius / r) ** 2
    t_eq = values.star_temperature * math.sqrt(values.star_radius / (2 * r)) * (1 - values.albedo) ** 0.25
    thermal = _photon(
        math.pi * planck_lambda_si(lam, t_eq) * (values.planet_radius / values.distance) ** 2, lam
    )
    tx = [values.theta_x, values.theta_x + x / values.distance]
    ty = [values.theta_y, values.theta_y + y / values.distance]
    return star, reflected + thermal, tx, ty


class _StarAndOrbit(Element):
    """Fields shared by :class:`PlanetarySystem` and :class:`Planet`: a star and a Keplerian orbit."""

    observation_dependent = True
    time_dependent = True
    star_temperature = Param(unit=u.K, default=5772 * u.K, bounds=(0 * u.K, None))
    star_radius = Param(unit=u.R_sun, default=1 * u.R_sun, bounds=(0 * u.m, None))
    star_mass = Param(unit=u.M_sun, default=1 * u.M_sun, bounds=(0 * u.kg, None), inclusive=(False, True))
    distance = Param(unit=u.pc, default=10 * u.pc, bounds=(0 * u.m, None), inclusive=(False, True))
    planet_radius = Param(unit=u.R_earth, default=1 * u.R_earth, bounds=(0 * u.m, None))
    semi_major_axis = Param(unit=u.au, default=1 * u.au, bounds=(0 * u.m, None), inclusive=(False, True))
    eccentricity = Param(default=0.0, bounds=(0, 1), inclusive=(True, False))
    inclination = Param(unit=u.deg, default=0 * u.deg, doc="0: face-on")
    argument_periapsis = Param(unit=u.deg, default=0 * u.deg)
    longitude_node = Param(unit=u.deg, default=0 * u.deg)
    mean_anomaly = Param(unit=u.deg, default=0 * u.deg, doc="mean anomaly at the start of the clock")
    albedo = Param(default=0.3, bounds=(0, 1), doc="geometric albedo (also used as Bond albedo)")
    theta_x = Param(unit=u.rad, default=0 * u.rad, doc="direction of the star along x")
    theta_y = Param(unit=u.rad, default=0 * u.rad, doc="direction of the star along y")
    out = Output(SourceList)


@register("lumos.sources.PlanetarySystem")
class PlanetarySystem(_StarAndOrbit):
    """A star and one planet on a Keplerian orbit; the planet moves with the context clock."""

    @staticmethod
    def transfer(values: Any, inputs: Any) -> dict[str, Any]:
        """Return the star and the planet at the current time."""
        lam, bw = _channels(values)
        star, planet, tx, ty = _star_and_planet(values, lam)
        return {"out": _sources(tx, ty, lam, bw, np.stack([star, planet]))}


@register("lumos.sources.Planet")
class Planet(_StarAndOrbit):
    """A planet on a Keplerian orbit around its star (the star itself is not emitted).

    Use it with a :class:`Star` for systems of several planets (:meth:`lumos.scenes.Scene.from_catalog`).
    """

    @staticmethod
    def transfer(values: Any, inputs: Any) -> dict[str, Any]:
        """Return the planet at the current time."""
        lam, bw = _channels(values)
        _, planet, tx, ty = _star_and_planet(values, lam)
        return {"out": _sources(tx[1:], ty[1:], lam, bw, planet[None])}

    @classmethod
    def from_catalog(cls, name: str, *, record: Any = None, **values: Any) -> Planet:
        """Create the planet ``name`` from the NASA Exoplanet Archive (or from an exoplanet ``record``).

        Missing parameters keep their defaults; ``values`` override the catalogue.
        """
        from lumos.catalogs.exoplanets import NasaExoplanetArchive

        record = record if record is not None else NasaExoplanetArchive().planet(name)
        return cls(name=record.name, **{**_planet_values(record), **values})


def _planet_values(record: Any) -> dict[str, Any]:
    """Fields of :class:`Planet` from an :class:`~lumos.catalogs.records.ExoplanetRecord`."""
    mapping = {
        "host_teff": "star_temperature",
        "host_radius": "star_radius",
        "host_mass": "star_mass",
        "host_distance": "distance",
        "radius": "planet_radius",
        "semi_major_axis": "semi_major_axis",
        "eccentricity": "eccentricity",
        "inclination": "inclination",
        "argument_periastron": "argument_periapsis",
    }
    found: dict[str, Any] = {}
    for key, field in mapping.items():
        if key in record:
            value = record[key].value
            found[field] = float(value) if field == "eccentricity" else value
    return found


@register("lumos.sources.GaussianLaser")
class GaussianLaser(Element):
    """Laser beam with a Gaussian profile at its waist, sampled on its own grid."""

    wavelength = Param(unit=u.nm, default=633 * u.nm, bounds=(0 * u.m, None), inclusive=(False, True))
    power = Param(unit=u.mW, default=1 * u.mW, bounds=(0 * u.W, None))
    waist = Param(unit=u.mm, default=1 * u.mm, bounds=(0 * u.m, None), inclusive=(False, True))
    pixel_scale = Param(unit=u.um, default=20 * u.um, bounds=(0 * u.m, None), inclusive=(False, True))
    n_pixels = Setting(int, default=256, doc="grid size")
    out = Output(Wavefront)

    @staticmethod
    def transfer(values: Any, inputs: Any) -> dict[str, Any]:
        """Return the beam at its waist."""
        grid = Grid(values.n_pixels, dx=values.pixel_scale * u.m)
        beam = Wavefront.gaussian(
            grid, values.wavelength * u.m, waist=values.waist * u.m, power=values.power * u.W
        )
        return {"out": beam}


@register("lumos.sources.IncoherentLamp")
class IncoherentLamp(Element):
    """Uniform Lambertian disk with a flat spectrum over the band of the context."""

    observation_dependent = True
    power = Param(unit=u.W, default=1 * u.W, bounds=(0 * u.W, None), doc="power emitted in the band")
    diameter = Param(unit=u.mm, default=1 * u.mm, bounds=(0 * u.m, None), inclusive=(False, True))
    pixel_scale = Param(unit=u.um, default=20 * u.um, bounds=(0 * u.m, None), inclusive=(False, True))
    n_pixels = Setting(int, default=128, doc="grid size")
    out = Output(RadianceField)

    @staticmethod
    def transfer(values: Any, inputs: Any) -> dict[str, Any]:
        """Return the radiance of the lamp."""
        lam, bw = _channels(values)
        grid = Grid(values.n_pixels, dx=values.pixel_scale * u.m)
        x, y = (np.asarray(c) for c in grid.axes(backend="numpy"))
        offsets = ((np.arange(4) + 0.5) / 4 - 0.5) * values.pixel_scale
        radius2 = (values.diameter / 2) ** 2
        coverage = (
            sum(
                ((x[None, :] + ox) ** 2 + (y[:, None] + oy) ** 2) <= radius2
                for oy in offsets
                for ox in offsets
            )
            / 16.0
        )
        area = float(np.sum(coverage)) * values.pixel_scale**2
        radiance = values.power / (math.pi * area * float(np.sum(bw)))  # W m-2 sr-1 m-1
        data = np.broadcast_to(radiance * coverage, (len(lam), *grid.shape))
        radiance_unit = u.W / u.m**2 / u.sr / u.m
        return {"out": RadianceField(u.Quantity(data, radiance_unit), grid, u.Quantity(lam, u.m))}
