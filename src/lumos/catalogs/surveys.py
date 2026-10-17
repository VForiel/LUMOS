"""Images of diffuse objects from sky surveys (step P9.6, REQ-CAT-3).

- :class:`SkyView` (NASA GSFC, McGlynn, Scollick & White 1998) resamples dozens of surveys (DSS, 2MASS,
  WISE, GALEX...) around a name or a position.
- :class:`Hips2Fits` (CDS, Fernique et al. 2015, A&A 578, A114) cuts FITS images out of any HiPS survey
  (``CDS/P/DSS2/red``, ``CDS/P/2MASS/K``...).

:func:`survey_component` turns an image into a :class:`~lumos.scenes.MappedComponent`. Survey images are
rarely calibrated in brightness: the brightness of one data unit (``calibration``) or the total flux density
of the image (``flux_density``) must be given when the header has no ``BUNIT``.
"""

from __future__ import annotations

import io
from typing import Any, cast

import numpy as np
from astropy import units as u

from lumos.catalogs.http import CatalogError, fetch
from lumos.catalogs.provider import Provider, register_provider

__all__ = ["Hips2Fits", "SkyView", "survey_component"]


def _image(content: bytes, origin: str) -> Any:
    from astropy.io import fits

    try:
        with fits.open(io.BytesIO(content)) as hdul:
            primary = cast(Any, hdul[0])
            if primary.data is None:
                raise CatalogError(f"{origin}: the FITS answer holds no image")
            return fits.PrimaryHDU(np.array(primary.data), primary.header.copy())
    except OSError as error:
        raise CatalogError(f"{origin}: the answer is not a FITS image ({error})") from error


def _degrees(quantity: u.Quantity) -> float:
    return float(cast(Any, quantity).to_value(u.deg))


@register_provider
class SkyView(Provider):
    """SkyView virtual observatory (NASA GSFC): survey images resampled around a target."""

    name = "skyview"
    description = "SkyView virtual observatory (NASA GSFC)"
    url = "https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl"
    reference = "McGlynn, Scollick & White 1998, IAU Symp. 179, 465"

    def __init__(self, url: str | None = None) -> None:
        self.endpoint = url or type(self).url

    def image(
        self, target: str, *, survey: str = "DSS", pixels: int = 128, size: u.Quantity = 0.25 * u.deg
    ) -> Any:
        """Return a FITS image (``astropy.io.fits.PrimaryHDU``) of ``target`` in ``survey``."""
        params = {
            "Position": target,
            "Survey": survey,
            "Pixels": str(int(pixels)),
            "Size": f"{_degrees(size):g}",
            "Return": "FITS",
        }
        return _image(fetch(self.endpoint, params=params).content, f"SkyView {survey} {target}")


@register_provider
class Hips2Fits(Provider):
    """hips2fits (CDS): FITS cut-outs of HiPS surveys."""

    name = "hips2fits"
    description = "hips2fits cut-out service of HiPS surveys (CDS)"
    url = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
    reference = "Fernique et al. 2015, A&A 578, A114"

    def __init__(self, url: str | None = None) -> None:
        self.endpoint = url or type(self).url

    def image(self, hips: str, *, ra: u.Quantity, dec: u.Quantity, fov: u.Quantity, pixels: int = 128) -> Any:
        """Return a square FITS cut-out of ``hips`` centred on ``(ra, dec)``, of field of view ``fov``."""
        params = {
            "hips": hips,
            "ra": f"{_degrees(ra):.7g}",
            "dec": f"{_degrees(dec):.7g}",
            "fov": f"{_degrees(fov):g}",
            "width": str(int(pixels)),
            "height": str(int(pixels)),
            "projection": "TAN",
            "format": "fits",
        }
        return _image(fetch(self.endpoint, params=params).content, f"hips2fits {hips}")


def survey_component(
    hdu: Any,
    *,
    name: str | None = None,
    calibration: u.Quantity | None = None,
    flux_density: u.Quantity | None = None,
    reference_wavelength: u.Quantity = 1 * u.um,
    **values: Any,
) -> Any:
    """Turn a survey image into a :class:`~lumos.scenes.MappedComponent`.

    Parameters
    ----------
    hdu
        FITS image with a celestial WCS.
    name
        Name of the component.
    calibration
        Brightness of one data unit (e.g. ``1e-3 * u.MJy / u.sr``), for images without ``BUNIT``.
    flux_density
        Total flux density of the image: the data are scaled so that their sum is this flux (negative values,
        background noise, are set to zero first).
    reference_wavelength
        Wavelength at which the calibration applies.
    **values
        Other fields of the component (spectrum, temperature, position...).
    """
    from astropy.io import fits

    from lumos.scenes.mapped import MappedComponent

    data = np.asarray(hdu.data, dtype=float)
    header = hdu.header.copy()
    if flux_density is not None:
        data = np.where(np.isfinite(data), np.maximum(data, 0.0), 0.0)
        total = float(data.sum())
        if total <= 0:
            raise ValueError("the image has no positive pixel to normalise")
        data = data / total * float(cast(Any, flux_density).to_value(u.Jy))
        header["BUNIT"] = "Jy/pixel"
    elif calibration is not None:
        data = data * float(cast(Any, calibration).value)
        header["BUNIT"] = u.Unit(cast(Any, calibration).unit).to_string("fits")
    elif "BUNIT" not in header:
        raise ValueError("the image has no BUNIT: give its calibration or its total flux_density")
    scaled = fits.PrimaryHDU(data, header)
    return MappedComponent.from_hdu(
        scaled, reference_wavelength=reference_wavelength, name=name, origin="survey image", **values
    )
