"""Libraries of model and empirical spectra (step P9.5).

:class:`PicklesLibrary` reads the stellar spectral flux library of Pickles (1998, PASP 110, 863), 131
spectral types from O5 to M10 (dwarfs to supergiants), 1150 Å to 2.5 µm (``UVKLIB``), from the copy of
the STScI reference atlases (``grid/pickles/dat_uvk``). The index ``pickles_uk.fits`` maps spectral types to
files.
"""

from __future__ import annotations

import io
from typing import Any, cast

from lumos.catalogs.http import CatalogError, fetch
from lumos.catalogs.provider import Provider, register_provider
from lumos.catalogs.spectra import Spectrum, spectrum_from_fits_table

__all__ = ["PicklesLibrary"]


@register_provider
class PicklesLibrary(Provider):
    """Pickles (1998) library of stellar spectra (STScI reference atlases)."""

    name = "pickles"
    description = "Pickles (1998) stellar spectral flux library"
    url = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/pickles/dat_uvk/"
    reference = "Pickles 1998, PASP 110, 863"

    def __init__(self, url: str | None = None) -> None:
        self.endpoint = url or type(self).url

    def index(self) -> dict[str, str]:
        """Return the file of each spectral type (``{"A0V": "pickles_uk_9", ...}``)."""
        from astropy.io import fits

        content = fetch(self.endpoint + "pickles_uk.fits").content
        with fits.open(io.BytesIO(content)) as hdul:
            data = cast(Any, hdul[1]).data
            return {
                str(sptype).strip(): str(name).strip()
                for name, sptype in zip(data["FILENAME"], data["SPTYPE"], strict=True)
            }

    def file_of(self, spectral_type: str) -> str:
        """Return the file name (without extension) of ``spectral_type``."""
        index = self.index()
        key = spectral_type.strip()
        if key not in index:
            raise CatalogError(
                f"Pickles library: no spectrum of type {spectral_type!r}; types: {', '.join(index)}"
            )
        return index[key]

    def spectrum(self, spectral_type: str) -> Spectrum:
        """Return the spectrum of ``spectral_type`` (F_λ, arbitrary normalisation)."""
        filename = f"{self.file_of(spectral_type)}.fits"
        content = fetch(self.endpoint + filename).content
        return spectrum_from_fits_table(
            content, name=spectral_type, source=self.name, reference=f"Pickles 1998, {filename}"
        )
