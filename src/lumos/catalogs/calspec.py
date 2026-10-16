"""CALSPEC: the flux standards of the Hubble Space Telescope (Bohlin, Gordon & Tremblay 2014, PASP 126, 711).

The spectra are FITS tables (``WAVELENGTH`` in Å, ``FLUX`` in erg s⁻¹ cm⁻² Å⁻¹) in the ``current_calspec``
directory of STScI. :meth:`Calspec.spectrum` finds the latest file of a star, preferring the measured STIS
spectrum (``<star>_stis_NNN.fits``) to a model (``<star>_mod_NNN.fits``).
"""

from __future__ import annotations

import re

from lumos.catalogs.http import CatalogError, fetch
from lumos.catalogs.provider import Provider, register_provider
from lumos.catalogs.spectra import Spectrum, spectrum_from_fits_table

__all__ = ["Calspec"]


@register_provider
class Calspec(Provider):
    """CALSPEC spectrophotometric standards (STScI)."""

    name = "calspec"
    description = "CALSPEC flux standards of HST (STScI)"
    url = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/"
    reference = "Bohlin, Gordon & Tremblay 2014, PASP 126, 711"

    def __init__(self, url: str | None = None) -> None:
        self.endpoint = url or type(self).url

    def files(self) -> list[str]:
        """Return the names of the FITS files of the directory."""
        listing = fetch(self.endpoint).text
        return sorted(set(re.findall(r'href="([^"/]+\.fits)"', listing)))

    def file_of(self, star: str) -> str:
        """Return the latest file of ``star`` (a CALSPEC root name such as ``"alpha_lyr"``)."""
        root = star.strip().lower().replace(" ", "_")
        candidates: list[tuple[int, int, str]] = []
        for name in self.files():
            match = re.fullmatch(rf"{re.escape(root)}_(stis|nic|mod|fos)\w*?_(\d+)\.fits", name)
            if match:
                kind = 0 if match.group(1) == "mod" else 1  # measured spectra first
                candidates.append((kind, int(match.group(2)), name))
        if not candidates:
            raise CatalogError(f"CALSPEC: no spectrum of {star!r}")
        return max(candidates)[2]

    def spectrum(self, star: str) -> Spectrum:
        """Return the spectrum of ``star`` (F_λ in erg s⁻¹ cm⁻² Å⁻¹)."""
        filename = self.file_of(star)
        content = fetch(self.endpoint + filename).content
        return spectrum_from_fits_table(content, name=star, source=self.name, reference=f"CALSPEC {filename}")
