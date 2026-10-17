"""SVO Filter Profile Service (Rodrigo & Solano 2020, Spanish Virtual Observatory).

:meth:`SvoFilterService.filter` returns the transmission curve of a filter identified as
``<facility>/<instrument>.<band>`` (``"Generic/Johnson.V"``, ``"2MASS/2MASS.Ks"``,
``"JWST/NIRCam.F444W"``...), with the zero point of its photometric system when the service gives one.
"""

from __future__ import annotations

import io
import warnings
from typing import Any

import numpy as np
from astropy import units as u

from lumos.catalogs.http import CatalogError, fetch
from lumos.catalogs.provider import Provider, register_provider
from lumos.catalogs.spectra import FilterCurve

__all__ = ["SvoFilterService"]


@register_provider
class SvoFilterService(Provider):
    """Transmission curves of photometric filters (SVO Filter Profile Service)."""

    name = "svo"
    description = "SVO Filter Profile Service"
    url = "https://svo2.cab.inta-csic.es/theory/fps/fps.php"
    reference = "Rodrigo & Solano 2020, XIV.0 Scientific Meeting of the SEA; Rodrigo et al. 2012"

    def __init__(self, url: str | None = None) -> None:
        self.endpoint = url or type(self).url

    def filter(self, identifier: str) -> FilterCurve:
        """Return the transmission curve of the filter ``identifier``."""
        from astropy.io.votable import parse

        response = fetch(self.endpoint, params={"ID": identifier})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                table: Any = parse(io.BytesIO(response.content), verify="ignore").get_first_table()
            except Exception as error:
                raise CatalogError(f"SVO: unknown filter {identifier!r} ({error})") from error
        params = {param.name: param for param in table.params}
        data = table.to_table()
        unit = u.Unit(str(params["WavelengthUnit"].value)) if "WavelengthUnit" in params else u.AA
        wavelengths = np.asarray(data["Wavelength"], dtype=float) * unit
        transmission = np.asarray(data["Transmission"], dtype=float)
        zero_point = None
        if "ZeroPoint" in params:
            zero_unit = str(params["ZeroPointUnit"].value) if "ZeroPointUnit" in params else "Jy"
            zero_point = float(params["ZeroPoint"].value) * u.Unit(zero_unit)
        return FilterCurve(
            identifier, wavelengths.to(u.AA), transmission, source=self.name, zero_point=zero_point
        )
