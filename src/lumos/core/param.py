"""Declarative fields of LUMOS objects: physical parameters, settings and state (REQ-EL-1 to REQ-EL-3).

A class derived from :class:`ParamHost` declares its fields as class attributes::

    class Lens(ParamHost):
        focal_length = Param(u.m, default=1 * u.m, bounds=(0 * u.m, None), inclusive=(False, True))
        modes = Setting(int, default=1)
        position = State(u.mm, default=0 * u.mm)

Three kinds of fields exist:

- :class:`Param`: a physical quantity (always an :class:`astropy.units.Quantity`) describing the
  configuration of an object, such as a focal length;
- :class:`State`: a physical quantity changed by the actions of an equipment, such as a stage position;
- :class:`Setting`: a non-physical value (integer, string, boolean, choice), such as a method name.

For every physical field the host keeps two values:

- the **user value**, exactly as given (same number, same unit), returned when the attribute is read,
  so that ``550 nm`` is never displayed as ``550.0000000001 nm``;
- the **SI value**, a float, complex or read-only array in unscaled SI units, returned by
  :meth:`ParamHost.si` for the numerical kernels.

Arrays are copied on assignment and stored read-only, so they cannot be modified behind the host's back:
every change goes through an assignment, which calls :meth:`ParamHost._on_field_changed` (used to
invalidate cached results).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, Literal

import numpy as np
from astropy import units as u

from lumos.core.priors import Prior
from lumos.units import require_unit, to_si

__all__ = ["MISSING", "Field", "Param", "ParamHost", "Setting", "State"]


class _Missing:
    """Sentinel type for fields without a default value."""

    _instance: ClassVar[_Missing | None] = None

    def __new__(cls) -> _Missing:
        """Return the unique instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        """Represent the sentinel."""
        return "MISSING"


MISSING: Any = _Missing()


def _read_only(value: Any) -> Any:
    """Return a read-only copy of an array (or the value itself for scalars)."""
    if isinstance(value, np.ndarray):
        value = value.copy()
        value.flags.writeable = False
    return value


class Field:
    """Base class of the declarative fields. Subclasses implement :meth:`validate`."""

    kind: ClassVar[str] = "field"

    def __init__(self, *, default: Any = MISSING, doc: str = "") -> None:
        self.name: str = "<unnamed>"
        self.doc = doc
        self.default = default

    def __set_name__(self, owner: type, name: str) -> None:
        """Record the attribute name, used in error messages and serialisation."""
        self.name = name

    def validate(self, value: Any) -> tuple[Any, Any]:
        """Check ``value`` and return ``(user_value, si_value)``."""
        raise NotImplementedError

    def __get__(self, instance: ParamHost | None, owner: type | None = None) -> Any:
        """Return the field declaration on the class, the user value on an instance."""
        if instance is None:
            return self
        return instance._lumos_values[self.name]

    def __set__(self, instance: ParamHost, value: Any) -> None:
        """Validate and store a new value, then notify the host."""
        instance._lumos_assign(self, value, notify=True)


class Param(Field):
    """A physical parameter of a LUMOS object.

    Parameters
    ----------
    unit
        Unit (or any unit of the same dimension) of the parameter; ``None`` for a dimensionless parameter.
    default
        Default value. It is validated when the parameter is declared.
    bounds
        ``(low, high)`` limits, each a quantity or ``None``. They apply to every element of an array.
    inclusive
        Whether ``low`` and ``high`` are allowed values.
    shape
        Required shape: ``()`` for a scalar (default), a tuple for a fixed shape, ``None`` for any shape.
    dtype
        ``"real"`` (default) or ``"complex"``.
    doc
        Description of the parameter (shown in the documentation, the web interface and the MCP server).
    free
        Whether the parameter is free (fitted by calibration or inference) by default.
    prior
        Default prior distribution of the parameter when it is free.
    differentiable
        Whether gradients with respect to this parameter are meaningful.
    equivalencies
        Astropy equivalencies accepted for this parameter, e.g. ``u.spectral()`` for a wavelength.
    """

    kind: ClassVar[str] = "param"

    def __init__(
        self,
        unit: Any = None,
        *,
        default: Any = MISSING,
        bounds: tuple[Any, Any] = (None, None),
        inclusive: tuple[bool, bool] = (True, True),
        shape: tuple[int, ...] | None = (),
        dtype: Literal["real", "complex"] = "real",
        doc: str = "",
        free: bool = False,
        prior: Prior | None = None,
        differentiable: bool = True,
        equivalencies: Any = None,
    ) -> None:
        super().__init__(default=MISSING, doc=doc)
        self.unit: u.UnitBase = u.dimensionless_unscaled if unit is None else u.Unit(unit)  # type: ignore[assignment]
        if dtype not in ("real", "complex"):
            raise ValueError(f"dtype must be 'real' or 'complex', got {dtype!r}")
        self.dtype = dtype
        self.shape = None if shape is None else tuple(shape)
        self.inclusive = inclusive
        self.free = free
        self.differentiable = differentiable
        self.equivalencies = equivalencies or []
        if dtype == "complex" and bounds != (None, None):
            raise ValueError("complex parameters cannot have bounds")
        self.bounds = tuple(None if b is None else require_unit(b, self.unit, name="bounds") for b in bounds)
        self._bounds_si = tuple(None if b is None else to_si(b, self.unit) for b in self.bounds)
        self.check_prior(prior)
        self.prior = prior
        if default is not MISSING:
            self.default, _ = self.validate(default)

    def check_prior(self, prior: Prior | None) -> None:
        """Check that ``prior`` describes values of this parameter's dimension."""
        if prior is not None and not prior.unit.is_equivalent(self.unit, equivalencies=self.equivalencies):
            raise u.UnitConversionError(
                f"{self.name}: prior in '{prior.unit}' is not compatible "
                f"with the parameter unit '{self.unit}'"
            )

    def validate(self, value: Any) -> tuple[u.Quantity, Any]:
        """Check ``value`` and return ``(user_value, si_value)``."""
        quantity = require_unit(value, self.unit, name=self.name, equivalencies=self.equivalencies)
        if self.dtype == "real" and np.iscomplexobj(quantity.value):
            raise TypeError(f"{self.name}: expected a real value, got {quantity}")
        if self.shape is not None and quantity.shape != self.shape:
            raise ValueError(f"{self.name}: expected shape {self.shape}, got {quantity.shape}")

        si = to_si(quantity, self.unit, equivalencies=self.equivalencies)
        if self.dtype == "complex":
            si = np.asarray(si, dtype=complex) if isinstance(si, np.ndarray) else complex(si)
        self._check_bounds(si)

        user = quantity.copy()
        user.flags.writeable = False
        return user, _read_only(si)

    def _check_bounds(self, si: Any) -> None:
        low, high = self._bounds_si
        values = np.asarray(si)
        ok = np.ones(values.shape, dtype=bool)
        if low is not None:
            ok &= values >= low if self.inclusive[0] else values > low
        if high is not None:
            ok &= values <= high if self.inclusive[1] else values < high
        if not np.all(ok):
            brackets = ("[" if self.inclusive[0] else "]", "]" if self.inclusive[1] else "[")
            interval = f"{brackets[0]}{self.bounds[0]}, {self.bounds[1]}{brackets[1]}"
            raise ValueError(f"{self.name}: value outside the allowed interval {interval}")


class State(Param):
    """A physical quantity describing the current state of an equipment (changed by its actions)."""

    kind: ClassVar[str] = "state"


class Setting(Field):
    """A non-physical setting: an integer, a float without unit, a string or a boolean.

    Parameters
    ----------
    type
        Expected Python type. Booleans are not accepted for ``int`` or ``float`` settings.
    default
        Default value, validated when the setting is declared.
    choices
        Optional collection of allowed values.
    doc
        Description of the setting.
    """

    kind: ClassVar[str] = "setting"

    def __init__(
        self,
        type: type,
        *,
        default: Any = MISSING,
        choices: Iterable[Any] | None = None,
        doc: str = "",
    ) -> None:
        super().__init__(default=MISSING, doc=doc)
        self.type = type
        self.choices = None if choices is None else tuple(choices)
        if default is not MISSING:
            self.default, _ = self.validate(default)

    def validate(self, value: Any) -> tuple[Any, Any]:
        """Check ``value`` and return it twice (settings have no SI form)."""
        wrong_bool = isinstance(value, bool) and self.type is not bool
        numeric_ok = self.type is float and isinstance(value, int)
        if wrong_bool or not (isinstance(value, self.type) or numeric_ok):
            raise TypeError(f"{self.name}: expected {self.type.__name__}, got {type(value).__name__}")
        if self.choices is not None and value not in self.choices:
            raise ValueError(f"{self.name}: {value!r} is not one of {list(self.choices)}")
        return value, value


class ParamHost:
    """Base class of objects declaring :class:`Param`, :class:`State` and :class:`Setting` fields.

    Keyword arguments of the constructor set the fields; fields without a default are required.
    """

    _lumos_fields: ClassVar[dict[str, Field]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Collect the declared fields, base classes first, in definition order."""
        super().__init_subclass__(**kwargs)
        fields: dict[str, Field] = {}
        for klass in reversed(cls.__mro__):
            for name, attribute in vars(klass).items():
                if isinstance(attribute, Field):
                    fields[name] = attribute
        cls._lumos_fields = fields

    def __init__(self, **values: Any) -> None:
        self._lumos_values: dict[str, Any] = {}
        self._lumos_si: dict[str, Any] = {}
        self._lumos_free: dict[str, tuple[bool, Prior | None]] = {}
        fields = type(self)._lumos_fields
        unknown = sorted(set(values) - set(fields))
        if unknown:
            raise TypeError(f"{type(self).__name__}() got unexpected argument(s): {', '.join(unknown)}")
        missing = []
        for name, field in fields.items():
            if name in values:
                self._lumos_assign(field, values[name], notify=False)
            elif field.default is not MISSING:
                self._lumos_values[name] = field.default
                self._lumos_si[name] = field.validate(field.default)[1]
            else:
                missing.append(name)
        if missing:
            raise TypeError(f"{type(self).__name__}() missing required argument(s): {', '.join(missing)}")

    @classmethod
    def declared(cls) -> Mapping[str, Field]:
        """Return the declared fields, in definition order (inherited fields first)."""
        return dict(cls._lumos_fields)

    def _lumos_assign(self, field: Field, value: Any, *, notify: bool) -> None:
        user, si = field.validate(value)  # raises before anything is changed
        self._lumos_values[field.name] = user
        self._lumos_si[field.name] = si
        if notify:
            self._on_field_changed(field.name)

    def _on_field_changed(self, name: str) -> None:
        """React to a change of field ``name`` (not called during construction).

        Subclasses override it to invalidate cached results.
        """

    def si(self, name: str) -> Any:
        """Return the value of field ``name`` in unscaled SI units (settings are returned as is)."""
        try:
            return self._lumos_si[name]
        except KeyError:
            raise KeyError(f"{type(self).__name__} has no field {name!r}") from None

    # --- Free parameters -------------------------------------------------------------

    def _physical_field(self, name: str) -> Param:
        field = type(self)._lumos_fields.get(name)
        if field is None:
            raise KeyError(f"{type(self).__name__} has no field {name!r}")
        if not isinstance(field, Param):
            raise TypeError(f"{name}: only physical parameters can be free, not {field.kind}s")
        return field

    def is_free(self, name: str) -> bool:
        """Whether parameter ``name`` is free (fitted by calibration or inference)."""
        field = self._physical_field(name)
        return self._lumos_free.get(name, (field.free, field.prior))[0]

    def prior(self, name: str) -> Prior | None:
        """Return the prior of parameter ``name`` (``None`` if fixed or without prior)."""
        field = self._physical_field(name)
        free, prior = self._lumos_free.get(name, (field.free, field.prior))
        return prior if free else None

    def set_free(self, name: str, free: bool = True, prior: Prior | None = None) -> None:
        """Mark parameter ``name`` as free (with an optional prior) or fixed, for this object only."""
        field = self._physical_field(name)
        field.check_prior(prior)
        self._lumos_free[name] = (free, prior if free else None)
