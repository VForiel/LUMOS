"""Units offered by the forms of the web interface (step P14.4, REQ-WEB-3).

For each unit of a parameter, the units of the same dimension a user may choose instead, with the factor that
converts a value in that unit to the unit of the parameter (computed by astropy, without equivalencies: a
wavelength is never offered as a frequency).
"""

from __future__ import annotations

from collections.abc import Iterable

from astropy import units as u

__all__ = ["CANDIDATES", "unit_choices"]

#: Units commonly typed by users, by family; only those of the right dimension are offered.
CANDIDATES: tuple[str, ...] = (
    # lengths
    "pm", "nm", "um", "mm", "cm", "m", "km", "AU", "pc",
    # angles
    "uas", "mas", "arcsec", "arcmin", "deg", "urad", "mrad", "rad",
    # times
    "fs", "ps", "ns", "us", "ms", "s", "min", "h", "d", "yr",
    # powers and energies
    "fW", "pW", "nW", "uW", "mW", "W", "kW", "nJ", "uJ", "mJ", "J",
    # frequencies
    "Hz", "kHz", "MHz", "GHz", "THz",
    # spectral flux densities
    "uJy", "mJy", "Jy",
    # speeds and accelerations
    "nm/s", "um/s", "mm/s", "m/s", "km/s", "mm/s2", "m/s2", "deg/s", "rad/s",
    # electricity, temperature, pressure, mass
    "uV", "mV", "V", "kV", "uA", "mA", "A", "K", "mK", "Pa", "hPa", "bar", "mg", "g", "kg",
    # detectors
    "pix", "electron", "adu",
)  # fmt: skip


def _name(unit: u.UnitBase) -> str:
    return unit.to_string()


def unit_choices(units: Iterable[str]) -> dict[str, list[tuple[str, float]]]:
    """Return, for each unit of ``units``, the ``(unit, factor to it)`` pairs a form may offer.

    The unit itself comes first with the factor 1.
    """
    candidates = []
    for text in CANDIDATES:
        try:
            candidates.append((text, u.Unit(text)))
        except ValueError:  # pragma: no cover - a unit unknown to this version of astropy
            continue
    choices: dict[str, list[tuple[str, float]]] = {}
    for text in sorted(set(units)):
        try:
            target = u.Unit(text)
        except ValueError:
            choices[text] = [(text, 1.0)]
            continue
        found = [(text, 1.0)]
        if target != u.dimensionless_unscaled:
            for name, unit in candidates:
                if name != text and unit.is_equivalent(target):
                    found.append((name, float((1 * unit).to_value(target))))
        choices[text] = found
    return choices
