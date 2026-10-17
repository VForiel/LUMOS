r"""Scenes: what an instrument observes, placed in a frame (step P8.1, REQ-SCN-1, REQ-SCN-2, REQ-SCN-7).

A :class:`Scene` is an :class:`~lumos.core.group.ElementGroup` of **components**. Components producing a
:class:`~lumos.fields.SourceList` (point sources, stars, planetary systems...) are gathered on the output
``sources``; components producing a :class:`~lumos.fields.RadianceField` (diffuse objects: disks, nebulae,
backgrounds, maps...) are summed on the output ``radiance``, sampled on the angular grid of the scene.

Frames
------
- ``"celestial"``: objects at infinity in the ICRS frame; the scene grid is centred on the pointing
  ``(ra, dec)`` and its coordinates are angles on the sky (direction cosines, small-field approximation).
- ``"terrestrial"``: a scene on the Earth seen from a site or a platform at ``(latitude, longitude,
  altitude)``; its grid is angular, as seen from the instrument.
- ``"laboratory"``: objects at a finite ``distance`` from the entrance pupil, with coordinates in metres.
  Radiance is invariant along rays in vacuum (Born & Wolf, *Principles of Optics*, §4.8.1), so a radiance map
  :math:`L(x, y)` at distance :math:`d` is seen as :math:`L(\theta d)` with pixels of :math:`\Delta x / d`
  radians. This keeps the instrument model of objects at infinity: the curvature of the waves from a
  close object is neglected, which holds when the pupil is small compared with :math:`\sqrt{\lambda d}`
  (far field of the pupil); closer objects are propagated explicitly as fields (lasers, lamps).

Diffuse components are evaluated on their own grids and resampled onto the scene grid by area-weighted
averaging, which conserves the flux :math:`\sum L\,\Delta\Omega` of the part inside the scene grid.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from astropy import units as u

from lumos.core.element import Element, Values
from lumos.core.group import ElementGroup, register_group
from lumos.core.param import Param, Setting
from lumos.core.port import Input, Output
from lumos.core.registry import register
from lumos.fields.grid import Grid
from lumos.fields.radiance import RadianceField
from lumos.fields.sources import PHOTON_FLUX_DENSITY, SourceList

__all__ = ["FRAMES", "RADIANCE_UNIT", "Scene", "SceneCombiner", "scene_grid"]

#: Frames of scenes.
FRAMES: tuple[str, ...] = ("celestial", "terrestrial", "laboratory")

#: Unit of the radiance maps of scenes.
RADIANCE_UNIT = u.W / u.m**2 / u.sr / u.m


def _empty_sources(values: Values) -> SourceList:
    lam = np.asarray(values.wavelengths, dtype=float)
    bw = np.asarray(values.bandwidths, dtype=float)
    return SourceList(
        np.zeros(0) * u.rad,
        np.zeros(0) * u.rad,
        lam * u.m,
        bw * u.m,
        np.zeros((0, lam.size)) * PHOTON_FLUX_DENSITY,
    )


def scene_grid(n_pixels: int, pixel_scale: float) -> Grid:
    """Angular grid of a scene: ``n_pixels`` × ``n_pixels`` pixels of ``pixel_scale`` radians."""
    return Grid(int(n_pixels), dx=float(pixel_scale) * u.rad)


def _angular(field: RadianceField, distance: float) -> RadianceField:
    """Return a laboratory map as seen from ``distance``: same radiance, pixels of ``dx / d`` radians."""
    if field.grid.plane == "angular":
        return field
    grid = field.grid
    angular = Grid(
        grid.nx,
        grid.ny,
        dx=grid.dx_si / distance * u.rad,
        dy=grid.dy_si / distance * u.rad,
        x0=grid.x0_si / distance * u.rad,
        y0=grid.y0_si / distance * u.rad,
    )
    return RadianceField(field.radiance, angular, field.wavelengths)


@register("lumos.scenes.SceneCombiner")
class SceneCombiner(Element):
    """Gather the point sources and sum the radiance maps of the components of a scene.

    It holds the frame and the grid of the scene; :class:`Scene` creates it as its child ``"combiner"``.
    """

    observation_dependent = True
    frame = Setting(str, default="celestial", choices=FRAMES, doc="frame of the scene")
    ra = Param(unit=u.deg, default=0 * u.deg, doc="right ascension of the pointing (ICRS, celestial)")
    dec = Param(
        unit=u.deg, default=0 * u.deg, bounds=(-90 * u.deg, 90 * u.deg), doc="declination (celestial)"
    )
    latitude = Param(unit=u.deg, default=0 * u.deg, bounds=(-90 * u.deg, 90 * u.deg), doc="terrestrial")
    longitude = Param(unit=u.deg, default=0 * u.deg, doc="terrestrial (east positive)")
    altitude = Param(unit=u.m, default=0 * u.m, doc="terrestrial (above the WGS84 ellipsoid)")
    distance = Param(
        unit=u.m, default=1 * u.m, bounds=(0 * u.m, None), inclusive=(False, True), doc="laboratory"
    )
    pixel_scale = Param(
        unit=u.mas, default=10 * u.mas, bounds=(0 * u.rad, None), inclusive=(False, True), doc="scene pixel"
    )
    n_pixels = Setting(int, default=64, doc="pixels per side of the scene grid")
    points = Input(SourceList, multiple=True, doc="point components")
    diffuse = Input(RadianceField, multiple=True, doc="diffuse components")
    sources = Output(SourceList, doc="every point source of the scene")
    radiance = Output(RadianceField, doc="total radiance on the scene grid (W m⁻² sr⁻¹ m⁻¹)")

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Concatenate the point sources and resample and sum the radiance maps."""
        points: Sequence[SourceList] = inputs["points"]
        total = _empty_sources(values)
        for sources in points:
            total = total + sources
        grid = scene_grid(values.n_pixels, values.pixel_scale)
        lam = np.asarray(values.wavelengths, dtype=float)
        data = np.zeros((lam.size, *grid.shape))
        for field in cast(Sequence[RadianceField], inputs["diffuse"]):
            if field.has_stokes:
                raise ValueError("scenes combine unpolarised radiance maps (no Stokes axis)")
            if not np.allclose(field.wavelengths_si, lam, rtol=1e-12, atol=0.0):
                raise ValueError("a diffuse component is not sampled at the wavelengths of the observation")
            seen = _angular(field.to(RADIANCE_UNIT), float(values.distance))
            data = data + np.asarray(seen.resample(grid).radiance.to_value(RADIANCE_UNIT))
        return {"sources": total, "radiance": RadianceField(data * RADIANCE_UNIT, grid, lam * u.m)}


@register_group("lumos.scenes.Scene")
class Scene(ElementGroup):
    """Group of scene components in a frame, with the outputs ``sources`` and ``radiance``.

    Components added with :meth:`add` are connected automatically according to the type of their output.

    Parameters
    ----------
    name
        Name of the scene.
    frame
        ``"celestial"`` (default), ``"terrestrial"`` or ``"laboratory"``.
    **settings
        Fields of the :class:`SceneCombiner`: ``ra``, ``dec`` (celestial pointing), ``latitude``,
        ``longitude``, ``altitude`` (terrestrial), ``distance`` (laboratory), ``pixel_scale`` and
        ``n_pixels`` (angular grid of the scene).
    """

    def __init__(self, name: str | None = None, *, frame: str = "celestial", **settings: Any) -> None:
        super().__init__(name=name)
        combiner = ElementGroup.add(
            self, SceneCombiner(name="combiner", frame=frame, **settings), id="combiner"
        )
        self.expose("sources", combiner.port("sources"))
        self.expose("radiance", combiner.port("radiance"))

    @classmethod
    def from_catalog(cls, name: str, *, planets: bool = True, **settings: Any) -> Scene:
        """Create the celestial scene of the star ``name`` and of its known planets (REQ-CAT-6).

        The pointing comes from SIMBAD. When the NASA Exoplanet Archive knows planets around ``name``, the
        star takes the parameters of the archive (temperature, radius, distance) and each planet becomes a
        :class:`~lumos.elements.sources.Planet`; otherwise the star comes from
        :meth:`~lumos.elements.sources.Star.from_catalog`. ``settings`` are passed to the scene (grid...).
        """
        from lumos.catalogs.exoplanets import NasaExoplanetArchive
        from lumos.catalogs.http import CatalogError
        from lumos.catalogs.simbad import Simbad
        from lumos.elements.sources import Planet, Star

        record = Simbad().star(name)
        scene = cls(name=name, frame="celestial", ra=record.ra, dec=record.dec, **settings)
        system = []
        if planets:
            try:
                system = NasaExoplanetArchive().system(name)
            except CatalogError:
                system = []
        if system:
            host = system[0]
            values: dict[str, Any] = {}
            for key, field in (
                ("host_teff", "temperature"),
                ("host_radius", "radius"),
                ("host_distance", "distance"),
            ):
                if key in host:
                    values[field] = host[key].value
            star = Star(name=name, **values) if len(values) == 3 else Star.from_catalog(name, record=record)
        else:
            star = Star.from_catalog(name, record=record)
        scene.add(star)
        for planet in system:
            scene.add(Planet.from_catalog(planet.name, record=planet))
        return scene

    @property
    def combiner(self) -> SceneCombiner:
        """The child holding the frame and grid of the scene."""
        return cast(SceneCombiner, self["combiner"])

    def add(self, *elements: Any, id: str | None = None) -> Any:
        """Add components and connect their output to the scene (by signal type).

        Elements with an output of type :class:`~lumos.fields.SourceList` feed the ``sources`` output, those
        with a :class:`~lumos.fields.RadianceField` output the ``radiance`` output; other elements are added
        without connection (connect them yourself with :meth:`connect`).
        """
        added = ElementGroup.add(self, *elements, id=id)
        combiner = self.combiner
        for element in elements:
            for name, port in element.instance_ports().items():
                if port.direction != "out":
                    continue
                if issubclass(port.signal_type, SourceList):
                    self.connect(element.port(name), combiner.port("points"))
                    break
                if issubclass(port.signal_type, RadianceField):
                    self.connect(element.port(name), combiner.port("diffuse"))
                    break
        return added

    @property
    def components(self) -> dict[str, Any]:
        """The components of the scene by identifier."""
        return {key: element for key, element in self.graph.elements.items() if key != "combiner"}

    @property
    def frame(self) -> str:
        """Frame of the scene."""
        return str(self.combiner.frame)

    @property
    def grid(self) -> Grid:
        """Angular grid of the radiance of the scene."""
        combiner = self.combiner
        return scene_grid(int(combiner.n_pixels), float(combiner.si("pixel_scale")))

    @property
    def pointing(self) -> Any:
        """ICRS pointing of a celestial scene (:class:`astropy.coordinates.SkyCoord`)."""
        from astropy.coordinates import SkyCoord

        if self.frame != "celestial":
            raise ValueError(f"only celestial scenes have a pointing (this scene is {self.frame})")
        combiner = self.combiner
        return SkyCoord(ra=combiner.ra, dec=combiner.dec, frame="icrs")

    @property
    def location(self) -> Any:
        """Site of a terrestrial scene (:class:`astropy.coordinates.EarthLocation`, WGS84)."""
        from astropy.coordinates import EarthLocation

        combiner = self.combiner
        return EarthLocation.from_geodetic(combiner.longitude, combiner.latitude, combiner.altitude)

    def __repr__(self) -> str:
        """Represent the scene by its name, frame and number of components."""
        name = f" {self.name!r}" if self.name else ""
        return f"<Scene{name} frame={self.frame} components={len(self.components)}>"
