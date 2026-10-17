r"""Script blocks: user functions as elements with declared ports (step P11.5, REQ-PIPE-6, ADR-0013).

:func:`script` turns a function into an :class:`~lumos.core.element.Element` class registered under a type
identifier, so that it is placed in a graph, saved in ``.lumos`` files and listed like any element::

    @script(name="mylab.pipeline.Normalise", inputs={"in": DataProduct}, outputs={"out": DataProduct})
    def normalise(frame, *, level: u.Quantity = 1 * u.adu):
        return DataProduct(frame.data / frame.data.max() * level)

    block = normalise(name="normalise", level=10 * u.adu)

- the inputs are given to the function as positional arguments, in the declared order; the function returns a
  mapping from output names to signals, or the signal itself when there is a single output;
- the keyword-only arguments with defaults become the parameters of the element: a quantity default makes a
  :class:`~lumos.core.param.Param` with its unit, a ``bool``, ``int``, ``float`` or ``str`` default a
  :class:`~lumos.core.param.Setting`;
- a ``.lumos`` file holds the type identifier and the parameter values, never the code (NF-7): loading it
  needs the script to be defined first (by importing the module that defines it, or through the
  ``lumos.elements`` entry points of a plugin package);
- defining again a script of the same name and function name (a notebook cell run twice) replaces it.
"""

from __future__ import annotations

import inspect
import keyword
from collections.abc import Callable, Mapping
from typing import Any

from astropy import units as u

from lumos.core.element import Element, Values
from lumos.core.param import Param, Setting
from lumos.core.port import Input, Output
from lumos.core.registry import registry
from lumos.units import si_unit

__all__ = ["script"]


def _field(name: str, default: Any) -> Any:
    if isinstance(default, u.Quantity):
        return Param(unit=default.unit, default=default)
    for kind in (bool, int, float, str):
        if type(default) is kind:
            return Setting(kind, default=default)
    raise TypeError(
        f"parameter {name!r} of a script needs a default that is a quantity, bool, int, float or str, "
        f"got {default!r}"
    )


def _attribute(port: str) -> str:
    return f"{port}_" if keyword.iskeyword(port) else port


def script(
    *,
    name: str,
    inputs: Mapping[str, type],
    outputs: Mapping[str, type],
    version: int = 1,
) -> Callable[[Callable[..., Any]], type[Element]]:
    """Return a decorator turning a function into a registered element class.

    Parameters
    ----------
    name
        Type identifier, e.g. ``"mylab.pipeline.Normalise"`` (the last segment is the class name).
    inputs, outputs
        Port names and signal types (e.g. :class:`~lumos.fields.DataProduct`).
    version
        Version of the serialised form of the block.
    """
    if not outputs:
        raise ValueError("a script needs at least one output")

    def decorator(function: Callable[..., Any]) -> type[Element]:
        signature = inspect.signature(function)
        parameters = {
            p.name: p
            for p in signature.parameters.values()
            if p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is not inspect.Parameter.empty
        }
        fields = {key: _field(key, p.default) for key, p in parameters.items()}
        units = {key: field.unit for key, field in fields.items() if isinstance(field, Param)}
        input_names = list(inputs)
        output_names = list(outputs)

        def transfer(values: Values, signals: Any) -> dict[str, Any]:
            kwargs = {}
            for key in fields:
                value = getattr(values, key)
                kwargs[key] = u.Quantity(value, si_unit(units[key])).to(units[key]) if key in units else value
            result = function(*(signals[n] for n in input_names), **kwargs)
            if not isinstance(result, Mapping):
                if len(output_names) != 1:
                    raise TypeError(
                        f"{name}: the function must return a mapping of the outputs {output_names}"
                    )
                result = {output_names[0]: result}
            return dict(result)

        attributes: dict[str, Any] = {
            "__doc__": function.__doc__ or f"Script block {name}.",
            "__module__": function.__module__,
            "__qualname__": name.rsplit(".", 1)[1],
            "transfer": staticmethod(transfer),
            "_lumos_script": function,
        }
        attributes.update(fields)
        for port, signal in inputs.items():
            attributes[_attribute(port)] = Input(signal)
        for port, signal in outputs.items():
            attributes[_attribute(port)] = Output(signal)
        cls = type(name.rsplit(".", 1)[1], (Element,), attributes)
        if name in registry:
            existing = getattr(registry.get(name), "_lumos_script", None)
            same = existing is not None and (existing.__module__, existing.__qualname__) == (
                function.__module__,
                function.__qualname__,
            )
            if same:
                registry.unregister(name)
        registry.register(name, version)(cls)
        return cls

    return decorator
