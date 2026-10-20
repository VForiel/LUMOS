r"""Declarative components: elements made by users without code (step P15.14, ADR-0025).

A *component* is a document of kind ``component``: data only, never code. It names a **template** and fills
it::

    {"id": "acme.Attenuator", "version": 1, "title": "Attenuator", "description": "...", "license": "MIT",
     "authors": ["ACME optics"], "template": "gain_phase",
     "params": {"transmission": {"unit": "", "default": 0.5, "bounds": [0, 1], "doc": "power transmission"}},
     "body": {"gain": "$transmission", "phase": "0 rad"},
     "checks": [{"kind": "power", "wavelength": "1 um", "power": "2 mW", "expect": "1 mW", "rtol": 1e-9}]}

Templates:

- ``transfer_matrix``: a matrix :math:`S_{ij} = A_{ij} e^{i\phi_{ij}}` at a wavelength (``amplitudes``,
  ``phases``, ``wavelength``, ``dispersion``), on a bus of guided lanes, as
  :class:`~lumos.elements.photonics.TransferMatrix`;
- ``gain_phase``: the field multiplied by :math:`\sqrt{g}\,e^{i\varphi}` (``gain``, ``phase``; a gain above 1
  needs ``"amplifier": true``), on any optical signal;
- ``spectral_transmission``: a tabulated power transmission :math:`T(\lambda)` (``wavelengths``,
  ``transmission``), interpolated linearly and 0 outside the table, on any optical signal;
- ``composite``: a group (``group``, the entry of a ``.lumos`` group), copied when the component is created.

A value of the body is a quantity (``"3 dB"``, ``"[1, 2] um"``), a number, a list or ``"$name"``, the value
of a parameter. :func:`register_component` builds and registers the element type
``components.<author>.<Name>``; :func:`check_component` lists what makes a component invalid (structure,
descriptions, license, units and bounds, passivity, and at least one check with an independent reference
value, which it runs). The command ``lumos component check FILE`` does the same.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from astropy import units as u

from lumos.core.element import Element, Values
from lumos.core.param import Param, Setting
from lumos.core.port import Input, Output
from lumos.core.registry import Registry
from lumos.core.signal import OpticalSignal

__all__ = [
    "TEMPLATES",
    "ComponentError",
    "ComponentSpec",
    "check_component",
    "component_from_document",
    "component_of",
    "create",
    "register_component",
]

#: Templates of the components and the keys of their body.
TEMPLATES: dict[str, tuple[str, ...]] = {
    "transfer_matrix": ("amplitudes", "phases", "wavelength", "dispersion"),
    "gain_phase": ("gain", "phase", "amplifier"),
    "spectral_transmission": ("wavelengths", "transmission"),
    "composite": ("group",),
}

_KEYS = (
    "id",
    "version",
    "title",
    "description",
    "license",
    "authors",
    "template",
    "params",
    "body",
    "checks",
)
_ID = re.compile(r"^[a-z][a-z0-9_]*\.[A-Z][A-Za-z0-9_]*$")
_PARAM_KEYS = {"unit", "default", "bounds", "doc"}


class ComponentError(ValueError):
    """A component that cannot be read or registered."""


# --- Specification ----------------------------------------------------------------------------------------


@dataclass
class ComponentSpec:
    """The definition of a component (the content of a document of kind ``component``)."""

    data: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Identifier ``author.Name``."""
        return str(self.data["id"])

    @property
    def type_id(self) -> str:
        """Element type ``components.<author>.<Name>``."""
        return f"components.{self.id}"

    @property
    def template(self) -> str:
        """Name of the template."""
        return str(self.data["template"])

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ComponentSpec:
        """Read a definition, checking its structure (:func:`check_component` checks the rest)."""
        if not isinstance(data, Mapping):
            raise ComponentError("a component is a JSON object")
        unknown = sorted(set(data) - set(_KEYS))
        if unknown:
            raise ComponentError(
                f"unknown key(s) {unknown}: a component holds no code, only {', '.join(_KEYS)}"
            )
        if not isinstance(data.get("id"), str) or not _ID.match(data["id"]):
            raise ComponentError(
                f"the id must be 'author.Name' (e.g. 'acme.Coupler'), got {data.get('id')!r}"
            )
        if data.get("template") not in TEMPLATES:
            raise ComponentError(
                f"unknown template {data.get('template')!r}; templates: {', '.join(TEMPLATES)}"
            )
        return cls(copy.deepcopy(dict(data)))

    def to_json(self) -> dict[str, Any]:
        """Return the definition (a copy)."""
        return copy.deepcopy(self.data)


def component_from_document(document: Mapping[str, Any]) -> ComponentSpec:
    """Read the component of a ``.lumos`` document of kind ``component``."""
    if (
        not isinstance(document, Mapping)
        or document.get("kind") != "component"
        or "component" not in document
    ):
        raise ComponentError("not a component document (kind 'component' with a 'component' entry)")
    return ComponentSpec.from_json(document["component"])


# --- Values of the body -----------------------------------------------------------------------------------

_ARRAY = re.compile(r"^\s*(\[.*\])\s*([^\]]*)$", re.DOTALL)


def _quantity(value: Any) -> u.Quantity:
    """Quantity of a literal of a body: ``"3 dB"``, ``"[1, 2] um"``, a number or a list."""
    if isinstance(value, u.Quantity):
        return value
    if isinstance(value, str):
        array = _ARRAY.match(value)
        if array:
            return u.Quantity(json.loads(array.group(1)), array.group(2).strip())
        return u.Quantity(value)
    return u.Quantity(np.asarray(value, dtype=float))


def _unit(text: Any) -> u.UnitBase:
    return u.dimensionless_unscaled if text in (None, "") else u.Unit(text)


def _resolve(value: Any, values: Mapping[str, Any], params: Mapping[str, Any], target: Any) -> Any:
    """SI value (float or array) of a body value, a ``"$name"`` reading the parameter ``name``."""
    if isinstance(value, str) and value.startswith("$"):
        name = value[1:]
        if name not in params:
            raise ComponentError(f"the parameter {name!r} is missing")
        si = values[name]
        if target is None:
            return si
        return _to(u.Quantity(si, _unit(params[name].get("unit")).si), target)
    quantity = _quantity(value)
    return quantity.value if target is None else _to(quantity, target)


def _to(quantity: Any, target: Any) -> Any:
    quantity = u.Quantity(quantity)
    if target is u.dimensionless_unscaled and quantity.unit.physical_type == "dimensionless":
        return np.asarray(quantity.to_value(u.dimensionless_unscaled))
    return np.asarray(quantity.to_value(target, equivalencies=u.spectral()))


# --- Templates --------------------------------------------------------------------------------------------


def _wavelengths(signal: Any) -> np.ndarray:
    from lumos.fields import IncoherentBundle, Wavefront

    member = signal[0] if isinstance(signal, IncoherentBundle) else signal
    if isinstance(member, Wavefront):
        return np.array([member.wavelength_si])
    return np.asarray(member.wavelengths_si, dtype=float)


def _spectral_scale(signal: Any, factors: np.ndarray) -> Any:
    """Multiply the field of each wavelength of ``signal`` by ``factors`` (one per wavelength)."""
    from lumos.backend._core import get
    from lumos.fields import IncoherentBundle, ModalField, Wavecube, Wavefront

    if isinstance(signal, Wavefront):
        return signal * complex(factors[0])
    if isinstance(signal, ModalField):
        return signal._with(signal.amplitudes * get(signal.backend).asarray(factors[:, None]), signal.labels)
    member = signal[0] if isinstance(signal, IncoherentBundle) else signal
    if not isinstance(member, Wavecube):
        return signal * complex(factors[0])
    extra = 3 if member.polarized else 2  # axes after the wavelength axis
    shape = (-1,) + (1,) * extra
    if isinstance(signal, IncoherentBundle):
        shape = (1, *shape)
    scale = get(signal.backend).asarray(factors.reshape(shape))
    return signal._with_field(signal.field * scale)


def _class(spec: ComponentSpec) -> type[Element]:
    """Build the element class of a templated component (a pure transfer closed over the body)."""
    data = spec.data
    params = dict(data.get("params", {}))
    body = dict(data.get("body", {}))
    namespace: dict[str, Any] = {
        "__doc__": f"{data.get('title', spec.id)}: {data.get('description', '')}",
        "__module__": __name__,
        "_lumos_component": spec.to_json(),
    }
    for name, declared in params.items():
        unit = _unit(declared.get("unit"))
        default = declared.get("default")
        low, high = ([*(declared.get("bounds") or [None, None]), None, None])[:2]
        bounds = (None if low is None else low * unit, None if high is None else high * unit)
        shape = None if isinstance(default, list) else ()
        namespace[name] = Param(
            unit=unit,
            default=np.asarray(default, dtype=float) * unit,
            bounds=bounds,
            shape=shape,
            doc=declared.get("doc", ""),
        )
    template = spec.template

    if template == "gain_phase":
        namespace["in_"] = Input(OpticalSignal, doc="any optical signal (each lane of a bus)")
        namespace["out"] = Output(OpticalSignal, generic=True)

        def transfer(values: Values, inputs: Any) -> dict[str, Any]:
            gain = float(_resolve(body.get("gain", 1.0), values, params, u.dimensionless_unscaled))
            phase = float(_resolve(body.get("phase", 0.0), values, params, u.rad))
            return {
                "out": inputs["in"] * (math.sqrt(max(gain, 0.0)) * complex(math.cos(phase), math.sin(phase)))
            }

    elif template == "spectral_transmission":
        namespace["in_"] = Input(
            OpticalSignal, doc="any optical signal with wavelengths (each lane of a bus)"
        )
        namespace["out"] = Output(OpticalSignal, generic=True)

        def transfer(values: Values, inputs: Any) -> dict[str, Any]:
            table = np.atleast_1d(_resolve(body["wavelengths"], values, params, u.m)).astype(float)
            transmission = np.atleast_1d(
                _resolve(body["transmission"], values, params, u.dimensionless_unscaled)
            )
            signal = inputs["in"]
            curve = np.interp(_wavelengths(signal), table, transmission.astype(float), left=0.0, right=0.0)
            return {"out": _spectral_scale(signal, np.sqrt(np.clip(curve, 0.0, None)))}

    elif template == "transfer_matrix":
        from lumos.fields.modal import ModalField

        namespace["channels"] = Setting(
            list, default=[], doc="indices of the lanes the component acts on ([]: all)"
        )
        namespace["in_"] = Input(ModalField, bus=True, doc="bus of guided lanes (or one lane)")
        namespace["out"] = Output(ModalField, bus=True, doc="bus of guided lanes")

        def transfer(values: Values, inputs: Any) -> dict[str, Any]:
            from lumos.elements.photonics import TransferMatrix

            matrix = Values(
                {
                    "amplitudes": np.asarray(
                        _resolve(body["amplitudes"], values, params, u.dimensionless_unscaled)
                    ),
                    "phases": np.asarray(_resolve(body.get("phases", 0.0), values, params, u.rad)),
                    "wavelength": float(_resolve(body.get("wavelength", "1550 nm"), values, params, u.m)),
                    "dispersion": str(body.get("dispersion", "achromatic")),
                    "channels": values.channels,
                }
            )
            return TransferMatrix._transfer(matrix, inputs)

        def port_lanes(self: Any, name: str) -> int | None:
            matrix = body["amplitudes"]
            source = (
                self.si(matrix[1:])
                if isinstance(matrix, str) and matrix.startswith("$")
                else _quantity(matrix).value
            )
            shape = np.shape(np.asarray(source))
            return None if len(shape) != 2 else int(shape[1] if name == "in" else shape[0])

        namespace["port_lanes"] = port_lanes
    else:
        raise ComponentError(f"the template {template!r} has no element class")

    namespace["transfer"] = staticmethod(transfer)
    name = spec.id.split(".")[1]
    return type(name, (Element,), namespace)


# --- Registration -----------------------------------------------------------------------------------------


def _components(registry: Registry) -> dict[str, ComponentSpec]:
    found = getattr(registry, "_lumos_components", None)
    if found is None:
        found = {}
        registry._lumos_components = found  # type: ignore[attr-defined]
    return found


def _registry(registry: Registry | None) -> Registry:
    from lumos.core import registry as module

    return module.registry if registry is None else registry


def register_component(spec: ComponentSpec | Mapping[str, Any], *, registry: Registry | None = None) -> Any:
    """Register a component and return its element class (``None`` for a composite component).

    Registering the same definition again does nothing; another definition under the same identifier is
    refused.
    """
    reg = _registry(registry)
    parsed = spec if isinstance(spec, ComponentSpec) else ComponentSpec.from_json(spec)
    known = _components(reg)
    existing = known.get(parsed.id)
    if existing is not None:
        if existing.data == parsed.data:
            return None if parsed.template == "composite" else reg.get(parsed.type_id)
        raise ComponentError(f"the component {parsed.id!r} is already registered with another definition")
    if parsed.template == "composite":
        known[parsed.id] = parsed
        return None
    cls = _class(parsed)
    reg.register(parsed.type_id, version=int(parsed.data.get("version", 1)))(cls)
    known[parsed.id] = parsed
    return cls


def component_of(element: Any) -> dict[str, Any] | None:
    """Definition of the component an element was made from, or ``None``."""
    data = getattr(type(element), "_lumos_component", None)
    return copy.deepcopy(data) if isinstance(data, dict) else None


def create(
    component: str, *, name: str | None = None, registry: Registry | None = None, **values: Any
) -> Element:
    """Create an element (or the group of a composite) from the registered component ``component``."""
    reg = _registry(registry)
    spec = _components(reg).get(component)
    if spec is None:
        raise ComponentError(f"no component {component!r} is registered")
    if spec.template == "composite":
        from lumos.io.lumos_format import from_entry

        entry = copy.deepcopy(spec.data["body"]["group"])
        if name is not None:
            entry["name"] = name
        return from_entry(entry)  # its children are elements of the library or registered components
    return reg.get(spec.type_id)(name=name, **values)


# --- Checks -----------------------------------------------------------------------------------------------


def _check_params(params: Any, problems: list[str]) -> None:
    if not isinstance(params, Mapping):
        problems.append("params is an object {name: {unit, default, bounds, doc}}")
        return
    for name, declared in params.items():
        if not isinstance(declared, Mapping):
            problems.append(f"parameter {name}: an object {{unit, default, bounds, doc}}")
            continue
        unknown = set(declared) - _PARAM_KEYS
        if unknown:
            problems.append(f"parameter {name}: unknown key(s) {sorted(unknown)}")
        if not str(declared.get("doc", "")).strip():
            problems.append(f"parameter {name}: give a description (doc)")
        try:
            _unit(declared.get("unit"))
        except ValueError:
            problems.append(f"parameter {name}: unknown unit {declared.get('unit')!r}")
            continue
        low, high = ([*list(declared.get("bounds") or [None, None]), None, None])[:2]
        default = declared.get("default")
        if default is None:
            problems.append(f"parameter {name}: give a default value")
            continue
        values = np.atleast_1d(np.asarray(default, dtype=float))
        if (low is not None and np.any(values < low)) or (high is not None and np.any(values > high)):
            problems.append(f"parameter {name}: the default {default} is outside its bounds [{low}, {high}]")


def _dimension(value: Any, params: Mapping[str, Any], target: Any, what: str, problems: list[str]) -> None:
    """Check that a body value (literal or "$name") has the dimension of ``target``."""
    if isinstance(value, str) and value.startswith("$"):
        name = value[1:]
        if name not in params:
            problems.append(f"body {what}: the parameter {name!r} is missing")
            return
        unit = _unit(params[name].get("unit"))
    else:
        try:
            unit = cast(u.UnitBase, _quantity(value).unit)
        except (TypeError, ValueError):
            problems.append(f"body {what}: cannot read {value!r} as a quantity")
            return
    dimensionless = target is u.dimensionless_unscaled and unit.physical_type == "dimensionless"
    if not (dimensionless or unit.is_equivalent(target, equivalencies=u.spectral())):
        source = f" (parameter {value[1:]})" if isinstance(value, str) and value.startswith("$") else ""
        problems.append(f"body {what}: {unit} is not a {u.Unit(target).physical_type}{source}")


def _check_body(spec: ComponentSpec, problems: list[str]) -> None:
    data = spec.data
    body = data.get("body")
    params = data.get("params", {}) if isinstance(data.get("params"), Mapping) else {}
    if not isinstance(body, Mapping):
        problems.append("body is an object")
        return
    unknown = set(body) - set(TEMPLATES[spec.template])
    if unknown:
        problems.append(f"body: unknown key(s) {sorted(unknown)} for the template {spec.template}")
    template = spec.template
    if template == "gain_phase":
        _dimension(body.get("gain", 1.0), params, u.dimensionless_unscaled, "gain", problems)
        _dimension(body.get("phase", 0.0), params, u.rad, "phase", problems)
        gain = body.get("gain", 1.0)
        if not (isinstance(gain, str) and gain.startswith("$")):
            try:
                if float(_to(_quantity(gain), u.dimensionless_unscaled)) > 1 and not body.get("amplifier"):
                    problems.append(
                        "body gain: a gain above 1 amplifies light (set amplifier to true if it does)"
                    )
            except (TypeError, ValueError, u.UnitsError):
                pass
        elif not body.get("amplifier"):
            declared = params.get(gain[1:], {})
            high = ([*list(declared.get("bounds") or [None, None]), None, None])[1]
            if high is None or high > 1:
                problems.append(
                    f"body gain: bound the parameter {gain[1:]} to 1 at most, or the component amplifies"
                )
    elif template == "spectral_transmission":
        _dimension(body.get("wavelengths"), params, u.m, "wavelengths", problems)
        _dimension(body.get("transmission"), params, u.dimensionless_unscaled, "transmission", problems)
        try:
            lam = np.atleast_1d(_quantity(body["wavelengths"]).value)
            transmission = np.atleast_1d(_quantity(body["transmission"]).value)
            if lam.shape != transmission.shape:
                problems.append("body: one transmission per wavelength")
            elif np.any(np.diff(lam) <= 0):
                problems.append("body wavelengths: give increasing wavelengths")
            if np.any(transmission < 0) or np.any(transmission > 1):
                problems.append("body transmission: a power transmission lies between 0 and 1")
        except (KeyError, TypeError, ValueError):
            pass  # reported above, or given by parameters
    elif template == "transfer_matrix":
        from lumos.elements.photonics import TransferMatrix

        _dimension(body.get("amplitudes"), params, u.dimensionless_unscaled, "amplitudes", problems)
        _dimension(body.get("phases", 0.0), params, u.rad, "phases", problems)
        _dimension(body.get("wavelength", "1550 nm"), params, u.m, "wavelength", problems)
        if body.get("dispersion", "achromatic") not in ("achromatic", "optical_path"):
            problems.append("body dispersion: 'achromatic' or 'optical_path'")
        try:
            amplitudes = np.asarray(_quantity(body["amplitudes"]).value, dtype=float)
            phases = np.asarray(_quantity(body.get("phases", 0.0)).to_value(u.rad), dtype=float)
            if phases.ndim == 0:
                phases = np.full_like(amplitudes, float(phases))
            problems += [f"body: {p}" for p in TransferMatrix._problems(amplitudes, phases)]
        except (KeyError, TypeError, ValueError, u.UnitsError):
            pass
    elif template == "composite":
        from lumos.io.lumos_format import from_entry

        try:
            group = from_entry(copy.deepcopy(body.get("group")))
            if not group.instance_ports():
                problems.append("body group: expose at least one port")
        except Exception as error:  # any reason why the group cannot be read
            problems.append(f"body group: {error}")


def _watts(power: Any) -> float:
    """Total of a power quantity (or of an array of powers), in watts."""
    return float(np.sum(np.asarray(u.Quantity(power).to_value(u.W), dtype=float)))


def _run_check(cls: type[Element], spec: ComponentSpec, check: Mapping[str, Any], index: int) -> list[str]:
    from lumos.fields import Bus, Grid, Wavefront

    where = f"check {index + 1}"
    if check.get("kind") != "power":
        return [f"{where}: the kind of a check is 'power'"]
    try:
        lam = cast(u.Quantity, _quantity(check["wavelength"]).to(u.m, equivalencies=u.spectral()))
        power = cast(u.Quantity, _quantity(check["power"]).to(u.W))
        expect = cast(u.Quantity, _quantity(check["expect"]).to(u.W))
        rtol = float(check.get("rtol", 1e-6))
    except (KeyError, TypeError, ValueError, u.UnitsError) as error:
        return [f"{where}: give wavelength, power and expect (with units): {error}"]
    element = cls()
    if spec.template == "transfer_matrix":
        n = element.port_lanes("in") or 1
        amplitude = math.sqrt(_watts(power))
        signal: Any = Bus.guided(np.full((1, n), amplitude), u.Quantity([lam]))
    else:
        signal = Wavefront.plane(Grid(16, dx=0.1 * u.mm), lam, power=power)
    try:
        out = element.apply(**{"in": signal})["out"]
    except Exception as error:  # the component fails on a plain input
        return [f"{where}: the component fails: {error}"]
    if isinstance(out, Bus):
        watts = sum(float(np.sum(np.abs(np.asarray(lane.amplitudes)) ** 2)) for lane in out)
    else:
        watts = _watts(out.power)
    total = watts * u.W
    if not math.isclose(watts, _watts(expect), rel_tol=rtol, abs_tol=0.0):
        return [f"{where}: the output power is {total:.6g}, the check expects {expect} (rtol {rtol:g})"]
    return []


def check_component(document: Mapping[str, Any]) -> list[str]:
    """Return what makes a component document invalid (empty when it is valid); its checks are run."""
    try:
        spec = component_from_document(document)
    except ComponentError as error:
        return [str(error)]
    data = spec.data
    problems: list[str] = []
    version = data.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        problems.append("version is a whole number from 1")
    for key in ("title", "description", "license"):
        if not str(data.get(key, "")).strip():
            problems.append(f"give a {key}")
    if not isinstance(data.get("authors"), list) or not data["authors"]:
        problems.append("give the authors (a list)")
    _check_params(data.get("params", {}), problems)
    _check_body(spec, problems)
    checks = data.get("checks", [])
    if not isinstance(checks, list):
        problems.append("checks is a list")
        checks = []
    if spec.template != "composite" and not checks:
        problems.append("give at least one check with an independent reference value (kind 'power')")
    if problems:
        return problems
    if ComponentSpec.from_json(json.loads(json.dumps(spec.to_json()))).to_json() != data:
        problems.append("the component does not come back unchanged from its JSON form")
    if spec.template != "composite":
        scratch = Registry()
        try:
            cls = register_component(spec, registry=scratch)
        except (ComponentError, TypeError, ValueError) as error:
            return [f"the component cannot be built: {error}"]
        for index, check in enumerate(checks):
            problems += _run_check(cls, spec, check, index)
    return problems
