"""Generation of a Python script equivalent to a context (REQ-CTX-8, step P5.9).

:func:`to_python` writes a script that rebuilds a context with the public API: imports of the element
classes, creation of the elements and groups with their values, connections, clock, observation settings,
simulation overrides and free parameters. Running the script gives a context with the same ``.lumos``
document and the same results; it is a readable starting point to edit a system built in the web interface
or by an assistant.
"""

from __future__ import annotations

import json
import keyword
import re
from typing import Any

import lumos
from lumos.core.context import Context
from lumos.core.group import ElementGroup
from lumos.core.param import Param, Setting
from lumos.core.priors import prior_to_json
from lumos.fields.grid import Grid
from lumos.units import quantity_to_json

__all__ = ["to_python"]


class _Writer:
    def __init__(self) -> None:
        self.imports: dict[str, set[str]] = {}
        self.lines: list[str] = []
        self.names: set[str] = {"ctx", "u", "np", "Time", "Context", "ElementGroup"}

    def need(self, module: str, name: str) -> None:
        self.imports.setdefault(module, set()).add(name)

    def variable(self, path: str) -> str:
        base = re.sub(r"\W", "_", path.replace(".", "__"))
        if not base or base[0].isdigit() or keyword.iskeyword(base):
            base = f"e_{base}"
        name, n = base, 2
        while name in self.names:
            name, n = f"{base}_{n}", n + 1
        self.names.add(name)
        return name

    def quantity(self, value: Any) -> str:
        encoded = quantity_to_json(value)
        if isinstance(encoded["value"], int | float | str) and not isinstance(encoded["value"], bool):
            return f"u.Quantity({encoded['value']!r}, {encoded['unit']!r})"
        self.need("lumos.units", "quantity_from_json")
        return f"quantity_from_json({json.dumps(encoded)})"

    def setting(self, value: Any) -> str:
        if isinstance(value, Grid):
            self.need("lumos.fields", "Grid")
            return f"Grid.from_json({json.dumps(value.to_json())})"
        return repr(value)

    def element(self, element: Any, path: str) -> str:
        """Write the creation of ``element`` and return its variable name."""
        variable = self.variable(path)
        if isinstance(element, ElementGroup):
            self.need("lumos.core.group", "ElementGroup")
            self.lines.append(f"{variable} = ElementGroup(name={element.name!r})")
            graph = element.graph
            children: dict[str, str] = {}
            for child_id, child in graph.elements.items():
                child_var = self.element(child, f"{path}.{child_id}")
                self.lines.append(f"{variable}.add({child_var}, id={child_id!r})")
                children[child_id] = child_var
            for c in graph.connections:
                source = f"{children[graph.id_of(c.source.owner)]}.port({c.source.name!r})"
                target = f"{children[graph.id_of(c.target.owner)]}.port({c.target.name!r})"
                self.lines.append(f"{variable}.connect({source}, {target})")
            for name, ref in element.exposed().items():
                self.lines.append(
                    f"{variable}.expose({name!r}, {children[graph.id_of(ref.owner)]}.port({ref.name!r}))"
                )
            return variable
        cls = type(element)
        self.need(cls.__module__, cls.__qualname__.split(".")[0])
        arguments = [f"name={element.name!r}"]
        for name, field in cls.declared().items():
            value = getattr(element, name)
            if isinstance(field, Setting):
                arguments.append(f"{name}={self.setting(value)}")
            else:
                arguments.append(f"{name}={self.quantity(value)}")
        self.lines.append(f"{variable} = {cls.__qualname__}({', '.join(arguments)})")
        for name, field in cls.declared().items():
            if not isinstance(field, Param):
                continue
            free, prior = element.is_free(name), element.prior(name)
            if free == field.free and prior is None:
                continue
            if prior is None:
                self.lines.append(f"{variable}.set_free({name!r}, {free!r})")
            else:
                self.need("lumos.core.priors", "prior_from_json")
                encoded = json.dumps(prior_to_json(prior))
                self.lines.append(f"{variable}.set_free({name!r}, {free!r}, prior_from_json({encoded}))")
        return variable


def to_python(ctx: Context) -> str:
    """Return a Python script that rebuilds ``ctx`` (assigned to the variable ``ctx``)."""
    writer = _Writer()
    lines = writer.lines
    lines.append(
        f"ctx = Context(mode={ctx.mode!r}, twin={ctx.twin!r}, backend={ctx.backend!r}, seed={ctx.seed!r}, "
        f"start=Time({str(ctx.clock.start.isot)!r}, scale='utc'), name={ctx.name!r})"
    )
    if ctx.clock.elapsed_si:
        lines.append(f"ctx.clock.reset(elapsed={writer.quantity(ctx.clock.elapsed)})")
    if ctx.meta:
        lines.append(f"ctx.meta = {ctx.meta!r}")
    observation = ctx.observation
    if observation.band is not None:
        low, high = observation.band
        lines.append(f"ctx.observation.band = ({writer.quantity(low)}, {writer.quantity(high)})")
    if observation.n_wavelengths != 1:
        lines.append(f"ctx.observation.n_wavelengths = {int(observation.n_wavelengths)!r}")
    if observation.rule != "midpoint":
        lines.append(f"ctx.observation.rule = {observation.rule!r}")
    if observation.extra:
        lines.append(f"ctx.observation.extra = {observation.extra!r}")
    variables: dict[str, str] = {}
    for identifier, element in ctx.elements.items():
        lines.append("")
        variable = writer.element(element, identifier)
        lines.append(f"ctx.add({variable}, id={identifier!r})")
        variables[identifier] = variable
    if ctx.graph.connections:
        lines.append("")
    for c in ctx.graph.connections:
        source = f"{variables[ctx.id_of(c.source.owner)]}.port({c.source.name!r})"
        target = f"{variables[ctx.id_of(c.target.owner)]}.port({c.target.name!r})"
        lines.append(f"ctx.connect({source}, {target})")
    for identifier, element in ctx.elements.items():
        if ctx.mode == "operation" and ctx.mode_of(element) == "simulation":
            lines.append(f"ctx.simulate({variables[identifier]})")
    if ctx.feedbacks:
        lines.append("")
    for feedback in ctx.feedbacks:
        lines.append(f'ctx.feedback("{feedback.source}", "{feedback.target}", mode="{feedback.mode}")')
    if ctx.ui:
        lines.append(f"ctx.ui = {ctx.ui!r}")

    header = [
        f'"""Context {ctx.name!r} generated by LUMOS {lumos.__version__}."""'
        if ctx.name
        else f'"""Context generated by LUMOS {lumos.__version__}."""',
        "",
        "from astropy import units as u",
        "from astropy.time import Time",
        "",
        "from lumos import Context",
    ]
    for module in sorted(writer.imports):
        header.append(f"from {module} import {', '.join(sorted(writer.imports[module]))}")
    return "\n".join([*header, "", *lines]) + "\n"
