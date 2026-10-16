"""Typed catalogue records with provenance per field (step P9.2, REQ-CAT-1, REQ-CAT-4).

A record describes one astronomical object: a star (:class:`StarRecord`), an exoplanet
(:class:`ExoplanetRecord`), a Solar-System body (:class:`SolarSystemRecord`) or an extended object
(:class:`ExtendedObjectRecord`). Each field is a :class:`Measurement`: a value (a quantity, or a string for
spectral types and classes), an optional uncertainty, the **provider** it comes from, the bibliographic
**reference**, and whether it was **estimated** by LUMOS (completion from other fields) rather than measured.

Fields are declared per record type with their unit; values are checked on assignment. Magnitudes are fields
named ``mag_<band>`` (unit ``mag``), for any band.

::

    star = StarRecord("Vega", parallax=Measurement(130.23 * u.mas, 0.36 * u.mas, source="simbad"))
    star.parallax                  # <Quantity 130.23 mas>
    star["parallax"].uncertainty   # <Quantity 0.36 mas>
    star.provenance("parallax")    # ("simbad", "")
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, cast

from astropy import units as u

from lumos.units import quantity_from_json, quantity_to_json

__all__ = [
    "ExoplanetRecord",
    "ExtendedObjectRecord",
    "Measurement",
    "Record",
    "SolarSystemRecord",
    "StarRecord",
    "record_from_json",
]

_TEXT = "text"


@dataclass(frozen=True)
class Measurement:
    """A value with its uncertainty and provenance.

    Attributes
    ----------
    value
        Quantity, or string for textual fields (spectral type, object class...).
    uncertainty
        Uncertainty (same dimension as the value), or ``None``.
    source
        Provider of the value (``"simbad"``, ``"gaia"``...), ``"lumos"`` for estimates.
    reference
        Bibliographic reference (bibcode) or catalogue of the value.
    estimated
        Whether LUMOS computed the value from other fields instead of reading it.
    """

    value: Any
    uncertainty: u.Quantity | None = None
    source: str = ""
    reference: str = ""
    estimated: bool = False

    def to_json(self) -> dict[str, Any]:
        """Encode as a JSON-compatible dictionary."""
        value = self.value
        encoded = quantity_to_json(value) if isinstance(value, u.Quantity) else value
        return {
            "value": encoded,
            "uncertainty": None if self.uncertainty is None else quantity_to_json(self.uncertainty),
            "source": self.source,
            "reference": self.reference,
            "estimated": self.estimated,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Measurement:
        """Decode a dictionary produced by :meth:`to_json`."""
        value = data["value"]
        if isinstance(value, Mapping) and "unit" in value:
            value = quantity_from_json(value)
        uncertainty = data.get("uncertainty")
        return cls(
            value,
            None if uncertainty is None else quantity_from_json(uncertainty),
            data.get("source", ""),
            data.get("reference", ""),
            bool(data.get("estimated", False)),
        )


class Record:
    """Base class of records: a name, identifiers and measured fields.

    Parameters
    ----------
    name
        Main name of the object.
    identifiers
        Other names (catalogue designations).
    **fields
        Measurements by field name.
    """

    kind: ClassVar[str] = ""
    #: Declared fields: name → unit (``"text"`` for strings).
    FIELDS: ClassVar[dict[str, Any]] = {}
    _KINDS: ClassVar[dict[str, type[Record]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register record kinds."""
        super().__init_subclass__(**kwargs)
        if cls.kind:
            Record._KINDS[cls.kind] = cls

    def __init__(self, name: str, *, identifiers: list[str] | None = None, **fields: Measurement) -> None:
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "identifiers", list(identifiers or []))
        object.__setattr__(self, "_fields", {})
        for key, measurement in fields.items():
            self.set(key, measurement)

    # --- Fields ------------------------------------------------------------------------------------

    @classmethod
    def unit_of(cls, key: str) -> Any:
        """Return the unit of field ``key`` (``"text"`` for strings)."""
        if key in cls.FIELDS:
            return cls.FIELDS[key]
        if key.startswith("mag_") and len(key) > 4:
            return u.mag
        raise AttributeError(
            f"{cls.__name__} has no field {key!r}; fields: {', '.join(cls.FIELDS)}, mag_<band>"
        )

    def set(self, key: str, measurement: Measurement) -> None:
        """Set field ``key``, checking the unit of the value and of the uncertainty."""
        if not isinstance(measurement, Measurement):
            raise TypeError(f"{key}: expected a Measurement, got {type(measurement).__name__}")
        unit = self.unit_of(key)
        if unit == _TEXT:
            if not isinstance(measurement.value, str):
                raise TypeError(f"{key} is a text field, got {measurement.value!r}")
        else:
            for part in (measurement.value, measurement.uncertainty):
                if part is None:
                    continue
                if not isinstance(part, u.Quantity) or not part.unit.is_equivalent(unit):  # type: ignore[union-attr]
                    raise u.UnitsError(f"{key} must be in units of {unit}, got {part!r}")
        self._fields[key] = measurement

    def __getitem__(self, key: str) -> Measurement:
        """Return the measurement of field ``key``."""
        try:
            return self._fields[key]
        except KeyError:
            raise KeyError(f"{self.name}: no value for {key!r}") from None

    def get(self, key: str) -> Measurement | None:
        """Return the measurement of field ``key``, or ``None``."""
        return self._fields.get(key)

    def __contains__(self, key: object) -> bool:
        """Whether field ``key`` has a value."""
        return key in self._fields

    def __iter__(self) -> Iterator[str]:
        """Iterate over the fields that have a value."""
        return iter(list(self._fields))

    def __getattr__(self, key: str) -> Any:
        """Return the value of field ``key``."""
        fields = self.__dict__.get("_fields", {})
        if key in fields:
            return fields[key].value
        raise AttributeError(f"{type(self).__name__} {self.__dict__.get('name')!r} has no value for {key!r}")

    def __setattr__(self, key: str, value: Any) -> None:
        """Refuse plain attribute assignment (use :meth:`set`)."""
        raise AttributeError("records are changed with set(field, Measurement(...))")

    def provenance(self, key: str) -> tuple[str, str]:
        """Return ``(source, reference)`` of field ``key``."""
        measurement = self[key]
        return measurement.source, measurement.reference

    def magnitudes(self) -> dict[str, Measurement]:
        """Return the magnitudes by band."""
        return {key[4:]: value for key, value in self._fields.items() if key.startswith("mag_")}

    # --- Serialisation ---------------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        """Encode the record as a JSON-compatible dictionary."""
        return {
            "kind": self.kind,
            "name": self.name,
            "identifiers": list(self.identifiers),
            "fields": {key: measurement.to_json() for key, measurement in self._fields.items()},
        }

    def summary(self) -> str:
        """Return a readable list of the fields with their sources."""
        lines = [f"{type(self).__name__} {self.name}"]
        if self.identifiers:
            lines.append(f"  identifiers: {', '.join(self.identifiers)}")
        for key, measurement in self._fields.items():
            value = measurement.value
            text = f"{value}" if measurement.uncertainty is None else f"{value} ± {measurement.uncertainty}"
            origin = measurement.source + (f", {measurement.reference}" if measurement.reference else "")
            flag = " (estimated)" if measurement.estimated else ""
            lines.append(f"  {key} = {text}{flag}  [{origin}]")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Represent the record by its kind, name and number of fields."""
        return f"<{type(self).__name__} {self.name!r} fields={len(self._fields)}>"


def record_from_json(data: Mapping[str, Any]) -> Record:
    """Decode a record produced by :meth:`Record.to_json` (only the record kinds of LUMOS)."""
    kind = data.get("kind")
    cls = Record._KINDS.get(cast(str, kind))
    if cls is None:
        raise ValueError(f"unknown record kind {kind!r}; known: {', '.join(Record._KINDS)}")
    fields = {key: Measurement.from_json(value) for key, value in data.get("fields", {}).items()}
    extra = {"host": data.get("host", "")} if kind == "exoplanet" else {}
    return cls(data["name"], identifiers=list(data.get("identifiers", [])), **extra, **fields)


_VELOCITY = u.km / u.s


class StarRecord(Record):
    """A star: position, motion, distance, physical parameters, magnitudes."""

    kind = "star"
    FIELDS: ClassVar[dict[str, Any]] = {
        "ra": u.deg,
        "dec": u.deg,
        "parallax": u.mas,
        "distance": u.pc,
        "pm_ra": u.mas / u.yr,
        "pm_dec": u.mas / u.yr,
        "radial_velocity": _VELOCITY,
        "spectral_type": _TEXT,
        "object_type": _TEXT,
        "teff": u.K,
        "logg": u.dex(u.cm / u.s**2),
        "metallicity": u.dex,
        "radius": u.R_sun,
        "mass": u.M_sun,
        "luminosity": u.L_sun,
    }

    def distance_estimate(self) -> Measurement:
        r"""Return the distance: the measured one, else :math:`1/\varpi` from the parallax (estimated).

        The uncertainty propagates to first order, :math:`\sigma_d = \sigma_\varpi / \varpi^2`.
        """
        if "distance" in self:
            return self["distance"]
        parallax = self["parallax"]
        p = float(cast(Any, parallax.value).to_value(u.arcsec))
        sigma = None
        if parallax.uncertainty is not None:
            sigma = float(cast(Any, parallax.uncertainty).to_value(u.arcsec)) / p**2 * u.pc
        return Measurement((1 / p) * u.pc, sigma, source="lumos", reference="1/parallax", estimated=True)


class ExoplanetRecord(Record):
    """An exoplanet: orbit, size, mass, temperature, and the name of its host star."""

    kind = "exoplanet"
    FIELDS: ClassVar[dict[str, Any]] = {
        "orbital_period": u.day,
        "semi_major_axis": u.au,
        "eccentricity": u.dimensionless_unscaled,
        "inclination": u.deg,
        "argument_periastron": u.deg,
        "time_periastron": u.day,
        "radius": u.R_earth,
        "mass": u.M_earth,
        "equilibrium_temperature": u.K,
        "insolation": u.dimensionless_unscaled,
        "discovery_method": _TEXT,
        "host_distance": u.pc,
        "host_teff": u.K,
        "host_radius": u.R_sun,
        "host_mass": u.M_sun,
        "ra": u.deg,
        "dec": u.deg,
    }

    def __init__(
        self, name: str, *, host: str = "", identifiers: list[str] | None = None, **fields: Measurement
    ) -> None:
        super().__init__(name, identifiers=identifiers, **fields)
        object.__setattr__(self, "host", host)

    def to_json(self) -> dict[str, Any]:
        """Encode the record, with the name of the host star."""
        return {**super().to_json(), "host": self.host}


class SolarSystemRecord(Record):
    """A Solar-System body at an epoch: apparent position, distances, phase angle, magnitude."""

    kind = "solar_system"
    FIELDS: ClassVar[dict[str, Any]] = {
        "ra": u.deg,
        "dec": u.deg,
        "delta": u.au,
        "heliocentric_distance": u.au,
        "phase_angle": u.deg,
        "angular_diameter": u.arcsec,
        "epoch": u.day,
        "radius": u.km,
        "geometric_albedo": u.dimensionless_unscaled,
    }


class ExtendedObjectRecord(Record):
    """An extended object (nebula, galaxy, cluster): position, size, orientation, class."""

    kind = "extended"
    FIELDS: ClassVar[dict[str, Any]] = {
        "ra": u.deg,
        "dec": u.deg,
        "major_axis": u.arcmin,
        "minor_axis": u.arcmin,
        "position_angle": u.deg,
        "object_type": _TEXT,
        "distance": u.pc,
        "redshift": u.dimensionless_unscaled,
    }
