r"""Mapped scene components: radiance maps from files and arrays (step P8.4, REQ-SCN-4).

A :class:`MappedComponent` holds a surface-brightness map :math:`\Sigma_\nu` (stored in Jy sr⁻¹) on a
rectangular angular grid, either one image with a spectral shape or a spectral cube, and produces the
spectral radiance :math:`L_\lambda = \Sigma_\nu\,c/\lambda^2` at the channels of the context.

Sources
-------
- :meth:`MappedComponent.from_fits`: FITS images and cubes with a celestial WCS (Calabretta & Greisen 2002,
  A&A 395, 1077). Pixel scales come from the WCS; the brightness unit from ``BUNIT`` or the ``unit``
  argument. Images are oriented with East towards +x and North towards +y, as everywhere in LUMOS: an image
  whose right ascension decreases with the column index (``CDELT1 < 0``, the usual orientation) is flipped
  left-right. A third, spectral axis (``WAVE``, ``FREQ``...) gives the wavelengths of the planes of a cube.
- :meth:`MappedComponent.from_array`: a quantity array and pixel scales.
- :meth:`MappedComponent.from_image`: PNG, JPEG or TIFF images (read by matplotlib) with a calibration, the
  surface brightness of a pixel value of 1; colour images are averaged over their RGB channels.

Units
-----
Accepted brightness units: per solid angle (``Jy/sr``, ``MJy/sr``, ``Jy/arcsec2``, ``W m-2 sr-1 m-1``,
``erg s-1 cm-2 Å-1 arcsec-2``...) or per pixel (``Jy``, ``Jy/pixel``, ``W m-2 m-1``...), divided by the pixel
solid angle. Quantities per unit wavelength are converted to per unit frequency with
:math:`F_\nu = F_\lambda\,\lambda^2 / c` at the wavelength of each plane (images: ``reference_wavelength``).

Spectra
-------
An image is scaled to other wavelengths with a flat :math:`F_\nu` or a blackbody shape, as the parametric
components (:mod:`lumos.scenes.diffuse`). The planes of a cube are interpolated linearly in wavelength;
outside the range of the cube the nearest plane is used and a warning is logged.

The flux of the map is conserved when a :class:`~lumos.scenes.Scene` resamples it onto its grid.
"""

from __future__ import annotations

import logging
import os
from typing import Any, cast

import numpy as np
from astropy import units as u

from lumos.core.element import Element, Values
from lumos.core.param import Param, Setting
from lumos.core.port import Output
from lumos.core.registry import register
from lumos.fields.grid import Grid
from lumos.fields.radiance import RadianceField
from lumos.scenes.diffuse import _planck_nu
from lumos.scenes.scene import RADIANCE_UNIT

__all__ = ["MappedComponent", "to_surface_brightness"]

logger = logging.getLogger("lumos.scenes")

_C = 299_792_458.0
_SB = u.Jy / u.sr


def to_surface_brightness(values: u.Quantity, pixel_solid_angle: u.Quantity, wavelength: Any) -> u.Quantity:
    """Convert a map to a surface brightness per unit frequency (Jy sr⁻¹).

    Parameters
    ----------
    values
        Map in a unit per solid angle, per pixel (``Jy/pixel``) or of flux per pixel (``Jy``), per unit
        frequency or wavelength.
    pixel_solid_angle
        Solid angle of a pixel.
    wavelength
        Wavelength(s) at which per-wavelength quantities are converted, broadcastable against ``values``.

    Raises
    ------
    astropy.units.UnitsError
        If the unit is not a spectral brightness or flux.
    """
    unit = cast(u.UnitBase, values.unit)
    omega = pixel_solid_angle.to(u.sr)
    equivalencies = u.spectral_density(wavelength)
    if (unit * u.pix).is_equivalent(u.Jy, equivalencies=equivalencies) and u.pix in unit.bases:
        per_pixel = cast(Any, values * u.pix).to(u.Jy, equivalencies=equivalencies)
        return cast(u.Quantity, (per_pixel / omega).to(_SB))
    if unit.is_equivalent(u.Jy, equivalencies=equivalencies):
        return cast(u.Quantity, (cast(Any, values).to(u.Jy, equivalencies=equivalencies) / omega).to(_SB))
    if (unit * u.sr).is_equivalent(u.Jy, equivalencies=equivalencies):
        per_sr = cast(Any, values * u.sr).to(u.Jy, equivalencies=equivalencies)
        return cast(u.Quantity, (per_sr / u.sr).to(_SB))
    raise u.UnitsError(
        f"{unit} is not a spectral surface brightness (e.g. Jy/sr) nor a flux per pixel (e.g. Jy/pixel)"
    )


def _wavelengths_m(values: Values) -> np.ndarray:
    return np.atleast_1d(np.asarray(values.map_wavelengths, dtype=float))


def _cube_at(cube: np.ndarray, planes: np.ndarray, lam: np.ndarray, name: str) -> np.ndarray:
    """Interpolate the planes of ``cube`` linearly at the wavelengths ``lam`` (nearest plane outside)."""
    order = np.argsort(planes)
    planes, cube = planes[order], cube[order]
    if np.any(lam < planes[0]) or np.any(lam > planes[-1]):
        logger.warning(
            "%s: channels outside the spectral range of the cube (%.4g-%.4g m) use its nearest plane",
            name,
            planes[0],
            planes[-1],
        )
    out = np.empty((lam.size, *cube.shape[1:]))
    for k, value in enumerate(lam):
        j = int(np.clip(np.searchsorted(planes, value) - 1, 0, max(planes.size - 2, 0)))
        if planes.size == 1:
            out[k] = cube[0]
            continue
        t = float(np.clip((value - planes[j]) / (planes[j + 1] - planes[j]), 0.0, 1.0))
        out[k] = (1 - t) * cube[j] + t * cube[j + 1]
    return out


@register("lumos.scenes.MappedComponent")
class MappedComponent(Element):
    """Surface-brightness map (image or spectral cube) on an angular grid."""

    observation_dependent = True
    data = Param(
        unit=_SB, shape=None, default=np.zeros((1, 1)) * _SB, doc="map (ny, nx) or cube (nλ, ny, nx)"
    )
    map_wavelengths = Param(
        unit=u.um, shape=None, default=[] * u.um, doc="wavelengths of the planes of a cube"
    )
    reference_wavelength = Param(
        unit=u.um, default=1 * u.um, bounds=(0 * u.m, None), inclusive=(False, True), doc="of an image"
    )
    spectrum = Setting(str, default="flat", choices=("flat", "blackbody"), doc="spectral shape of an image")
    temperature = Param(unit=u.K, default=5772 * u.K, bounds=(0 * u.K, None), inclusive=(False, True))
    pixel_scale_x = Param(unit=u.mas, default=1 * u.mas, bounds=(0 * u.rad, None), inclusive=(False, True))
    pixel_scale_y = Param(unit=u.mas, default=1 * u.mas, bounds=(0 * u.rad, None), inclusive=(False, True))
    x0 = Param(unit=u.mas, default=0 * u.mas, doc="position of the central pixel along x (East)")
    y0 = Param(unit=u.mas, default=0 * u.mas, doc="position of the central pixel along y (North)")
    out = Output(RadianceField)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the spectral radiance of the map at the channels of the context."""
        lam = np.asarray(values.wavelengths, dtype=float)
        data = np.asarray(values.data, dtype=float)  # SI: W m-2 Hz-1 sr-1
        if data.ndim == 2:
            if values.spectrum == "flat":
                relative = np.ones_like(lam)
            else:
                reference = np.array([float(values.reference_wavelength)])
                relative = _planck_nu(lam, values.temperature) / _planck_nu(reference, values.temperature)
            sigma = relative[:, None, None] * data[None]
        elif data.ndim == 3:
            planes = _wavelengths_m(values)
            if planes.size != data.shape[0]:
                raise ValueError(f"the cube has {data.shape[0]} planes but {planes.size} map_wavelengths")
            sigma = _cube_at(data, planes, lam, "MappedComponent")
        else:
            raise ValueError(f"data must be a 2D map or a 3D cube, got {data.ndim} dimensions")
        radiance = sigma * (_C / lam**2)[:, None, None]
        ny, nx = data.shape[-2:]
        grid = Grid(
            nx,
            ny,
            dx=float(values.pixel_scale_x) * u.rad,
            dy=float(values.pixel_scale_y) * u.rad,
            x0=float(values.x0) * u.rad,
            y0=float(values.y0) * u.rad,
        )
        return {"out": RadianceField(radiance * RADIANCE_UNIT, grid, lam * u.m)}

    # --- Constructors ---------------------------------------------------------------------------------

    @classmethod
    def from_array(
        cls,
        data: u.Quantity,
        *,
        pixel_scale: u.Quantity,
        pixel_scale_y: u.Quantity | None = None,
        wavelengths: u.Quantity | None = None,
        reference_wavelength: u.Quantity = 1 * u.um,
        name: str | None = None,
        **values: Any,
    ) -> MappedComponent:
        """Create a component from a map ``(ny, nx)`` or a cube ``(nλ, ny, nx)`` with its brightness unit.

        Rows go from South to North and columns from West to East (x points East, y North).
        """
        if not isinstance(data, u.Quantity):
            raise TypeError("data must be a Quantity with a brightness unit")
        sx = pixel_scale.to(u.rad)
        sy = (pixel_scale if pixel_scale_y is None else pixel_scale_y).to(u.rad)
        omega = cast(u.Quantity, cast(Any, sx * sy).value * u.sr)
        array = np.asarray(data.value, dtype=float)
        if array.ndim == 3:
            if wavelengths is None:
                raise ValueError("a cube needs the wavelengths of its planes")
            lam = np.asarray(wavelengths.to_value(u.m, equivalencies=u.spectral()))[:, None, None]
            sigma = to_surface_brightness(data, omega, lam * u.m)
            values.setdefault("map_wavelengths", wavelengths.to(u.um, equivalencies=u.spectral()))
        else:
            sigma = to_surface_brightness(data, omega, reference_wavelength)
        return cls(
            name=name,
            data=sigma,
            reference_wavelength=reference_wavelength,
            pixel_scale_x=sx.to(u.mas),
            pixel_scale_y=sy.to(u.mas),
            **values,
        )

    @classmethod
    def from_fits(
        cls,
        path: str | os.PathLike[str],
        *,
        hdu: int | str = 0,
        unit: Any = None,
        reference_wavelength: u.Quantity = 1 * u.um,
        name: str | None = None,
        **values: Any,
    ) -> MappedComponent:
        """Create a component from a FITS image or cube with a celestial WCS (see the module notes)."""
        from astropy.io import fits
        from astropy.wcs import WCS

        with fits.open(path) as hdul:
            selected = cast(Any, hdul)[hdu]
            header = selected.header
            raw = np.asarray(selected.data, dtype=float)
        bunit = unit if unit is not None else header.get("BUNIT")
        if bunit is None:
            raise u.UnitsError(f"{path}: no BUNIT keyword; give the brightness unit with unit=")
        brightness = (
            cast(Any, u.Unit)(bunit, parse_strict="raise") if isinstance(bunit, str) else u.Unit(bunit)
        )
        wcs = WCS(header)
        matrix = wcs.celestial.pixel_scale_matrix  # degrees per pixel, (world axis, pixel axis)
        if abs(matrix[0, 1]) > 1e-6 * abs(matrix[0, 0]) or abs(matrix[1, 0]) > 1e-6 * abs(matrix[1, 1]):
            logger.warning(
                "%s: the WCS is rotated; LUMOS uses its pixel scales and ignores the rotation", path
            )
        sx, sy = abs(float(matrix[0, 0])) * u.deg, abs(float(matrix[1, 1])) * u.deg
        if matrix[0, 0] < 0:  # right ascension (East) decreasing with the column index
            raw = raw[..., ::-1]
        if matrix[1, 1] < 0:
            raw = raw[..., ::-1, :]
        wavelengths = None
        if raw.ndim == 3:
            spectral = wcs.spectral
            world = np.asarray(spectral.pixel_to_world_values(np.arange(raw.shape[0])), dtype=float)
            axis_unit = u.Unit(spectral.world_axis_units[0])
            wavelengths = (world * axis_unit).to(u.um, equivalencies=u.spectral())
        return cls.from_array(
            raw * brightness,
            pixel_scale=sx,
            pixel_scale_y=sy,
            wavelengths=wavelengths,
            reference_wavelength=reference_wavelength,
            name=name,
            **values,
        )

    @classmethod
    def from_image(
        cls,
        path: str | os.PathLike[str],
        *,
        pixel_scale: u.Quantity,
        calibration: u.Quantity,
        pixel_scale_y: u.Quantity | None = None,
        reference_wavelength: u.Quantity = 1 * u.um,
        name: str | None = None,
        **values: Any,
    ) -> MappedComponent:
        """Create a component from a PNG, JPEG or TIFF image; ``calibration`` is the brightness of a value 1.

        The first row of the file is the top of the image (North); colour images are averaged over RGB.
        """
        import matplotlib.image as mpimg

        raw = np.asarray(mpimg.imread(os.fspath(path)))
        image = raw.astype(float)
        if np.issubdtype(raw.dtype, np.integer):  # 8- or 16-bit TIFF and JPEG files
            image = image / float(np.iinfo(raw.dtype).max)
        if image.ndim == 3:
            image = image[..., :3].mean(axis=-1)
        image = image[::-1]  # bottom row first (origin at the bottom left, §9)
        return cls.from_array(
            cast(u.Quantity, image * calibration),
            pixel_scale=pixel_scale,
            pixel_scale_y=pixel_scale_y,
            reference_wavelength=reference_wavelength,
            name=name,
            **values,
        )
