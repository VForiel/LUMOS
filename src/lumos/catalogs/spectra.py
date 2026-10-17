r"""Spectra and filter curves read from catalogues (steps P9.3 to P9.5).

:class:`Spectrum` holds a spectral flux density per unit wavelength :math:`F_\lambda` with its provenance;
:class:`FilterCurve` a transmission curve. Both interpolate linearly in wavelength.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from astropy import units as u

__all__ = ["FLAM", "FilterCurve", "Spectrum", "spectrum_from_fits_table"]

#: CALSPEC and synphot unit of F_λ.
FLAM = u.erg / u.s / u.cm**2 / u.AA


@dataclass
class Spectrum:
    """A spectral flux density with its provenance.

    Attributes
    ----------
    wavelengths
        Vacuum wavelengths, increasing.
    flux
        Spectral flux density (per unit wavelength or frequency).
    uncertainty
        Uncertainty of the flux, or ``None``.
    name, source, reference
        Name of the object, provider and reference (file, bibcode).
    """

    wavelengths: u.Quantity
    flux: u.Quantity
    uncertainty: u.Quantity | None = None
    name: str = ""
    source: str = ""
    reference: str = ""

    def at(self, wavelength: u.Quantity) -> u.Quantity:
        """Return the flux interpolated linearly at ``wavelength`` (zero outside the spectrum)."""
        lam = np.asarray(cast(Any, self.wavelengths).to_value(u.m))
        target = np.asarray(cast(Any, wavelength).to_value(u.m, equivalencies=u.spectral()))
        values = np.interp(target, lam, np.asarray(self.flux.value), left=0.0, right=0.0)
        return u.Quantity(values, self.flux.unit)

    def to(self, unit: Any) -> Spectrum:
        """Return the spectrum with the flux in ``unit`` (per wavelength or frequency, energy or photons)."""
        flux = cast(Any, self.flux).to(unit, equivalencies=u.spectral_density(self.wavelengths))
        uncertainty = None
        if self.uncertainty is not None:
            uncertainty = cast(Any, self.uncertainty).to(
                unit, equivalencies=u.spectral_density(self.wavelengths)
            )
        return Spectrum(self.wavelengths, flux, uncertainty, self.name, self.source, self.reference)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.wavelengths)


@dataclass
class FilterCurve:
    """Transmission of a photometric filter.

    Attributes
    ----------
    name
        Identifier of the filter (e.g. ``"Generic/Johnson.V"``).
    wavelengths
        Wavelengths of the curve.
    transmission
        Transmission (dimensionless) at each wavelength.
    source
        Provider of the curve.
    zero_point
        Flux density of magnitude 0 in the photometric system of the filter, if known.
    """

    name: str
    wavelengths: u.Quantity
    transmission: np.ndarray
    source: str = ""
    zero_point: u.Quantity | None = None

    def at(self, wavelength: u.Quantity) -> np.ndarray:
        """Return the transmission interpolated at ``wavelength`` (zero outside the curve)."""
        lam = np.asarray(cast(Any, self.wavelengths).to_value(u.m))
        target = np.asarray(cast(Any, wavelength).to_value(u.m, equivalencies=u.spectral()))
        return np.interp(target, lam, self.transmission, left=0.0, right=0.0)

    def effective_wavelength(self) -> u.Quantity:
        r"""Return the mean wavelength :math:`\int \lambda T\,d\lambda / \int T\,d\lambda`."""
        lam = np.asarray(cast(Any, self.wavelengths).to_value(u.m))
        weight = np.trapezoid(self.transmission, lam)
        return cast(u.Quantity, (np.trapezoid(lam * self.transmission, lam) / weight * u.m).to(u.um))


def _unit(text: str) -> Any:
    aliases = {"ANGSTROMS": u.AA, "ANGSTROM": u.AA, "FLAM": FLAM, "NONE": u.dimensionless_unscaled}
    key = text.strip().upper()
    if key in aliases:
        return aliases[key]
    return u.Unit(text.strip())


def spectrum_from_fits_table(content: bytes, *, name: str, source: str, reference: str) -> Spectrum:
    """Read a synphot-style FITS table (``WAVELENGTH``, ``FLUX`` [, ``STATERROR``]) into a spectrum."""
    import io

    from astropy.io import fits

    with fits.open(io.BytesIO(content)) as hdul:
        table = cast(Any, hdul[1])
        data = table.data
        columns = {column.name.upper(): column for column in table.columns}
        lam_unit = _unit(columns["WAVELENGTH"].unit or "ANGSTROMS")
        flux_unit = _unit(columns["FLUX"].unit or "FLAM")
        wavelengths = np.asarray(data["WAVELENGTH"], dtype=float) * lam_unit
        flux = np.asarray(data["FLUX"], dtype=float) * flux_unit
        uncertainty = None
        if "STATERROR" in columns:
            uncertainty = np.asarray(data["STATERROR"], dtype=float) * flux_unit
    order = np.argsort(wavelengths.value)
    return Spectrum(
        wavelengths[order],
        flux[order],
        None if uncertainty is None else uncertainty[order],
        name=name,
        source=source,
        reference=reference,
    )
