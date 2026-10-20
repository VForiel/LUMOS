r"""Procedures: declarative sequences of actions, measurements and processing (step P15.11, ADR-0024).

A *procedure* is what an experimenter does with a bench: close the shutter, take twenty frames, average them
into a dark, open the shutter, take ten frames and subtract the dark. It is ordered in time and keeps
*named results* between its steps, unlike the graph (a computation) and the studies (sweeps, Monte-Carlo runs,
loops).

A procedure is **data**: a JSON object ``{"name", "description", "steps": [...]}`` whose steps name their kind
in ``do``. It cannot run code: it only calls actions of elements, computes ports and applies the processing
functions of :data:`FUNCTIONS`.

=============  ==========================================================================================
``do``         effect
=============  ==========================================================================================
``action``     calls the action ``action`` of ``element`` with ``args`` and ``kwargs`` (timed, journaled)
``measure``    computes ``outputs`` and stores the value (one port) or a mapping (several) as ``as``;
               ``collect`` appends it to a list
``repeat``     runs ``steps`` ``times`` times
``for_each``   runs ``steps`` with ``as`` set to each of ``values`` (a list, or ``{start, stop, num, log}``)
``process``    applies the function ``function`` to ``args`` and stores the value as ``as``
``set``        sets ``target`` (``element.field`` or ``element.field[i]``) to ``value``
``use``        feeds the input ``port`` with the named result ``result`` in the following measurements
``wait``       lets ``duration`` pass (the clock in simulation)
``if``         runs ``then`` or ``else`` after comparing ``left`` and ``right`` with ``op``
``save``       writes the named ``results`` to the archive ``archive``
``show``       marks the named ``results`` to display
=============  ==========================================================================================

Values are numbers, booleans, text, quantities written as text (``"2 mm"``) or named results (``"$dark"``).
:meth:`Procedure.problems` checks a procedure against its context before it runs (elements, ports, actions,
functions, names defined before use, at most :data:`MAX_ITERATIONS` loop iterations). The context stores its
procedures in :attr:`lumos.Context.procedures` and runs them with :meth:`lumos.Context.run_procedure`.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import time as _time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from astropy import units as u

__all__ = [
    "FUNCTIONS",
    "MAX_DEPTH",
    "MAX_ITERATIONS",
    "Procedure",
    "ProcedureError",
    "ProcedureResult",
    "run_procedure",
]

logger = logging.getLogger("lumos.context")

#: Most loop iterations of a procedure (all loops together, counted before it runs).
MAX_ITERATIONS = 10_000

#: Deepest nesting of loops and conditions.
MAX_DEPTH = 16

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ARRAY = re.compile(r"^\s*(\[.*\])\s*([^\]]*)$", re.DOTALL)
_KINDS = ("action", "measure", "repeat", "for_each", "process", "set", "use", "wait", "if", "save", "show")
_OPERATORS: dict[str, Callable[[Any, Any], Any]] = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


class ProcedureError(ValueError):
    """A procedure that cannot run (invalid, refused, cancelled), or a step that failed."""


class _Callback(Exception):
    """An exception raised by the progress callback, given back unchanged to the caller."""

    def __init__(self, error: BaseException) -> None:
        super().__init__(str(error))
        self.error = error


# --- Processing functions ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Function:
    """A processing function a procedure may call: its implementation, arguments and description."""

    run: Callable[..., Any]
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    doc: str = ""


def _frames(frames: Any) -> list[Any]:
    if not isinstance(frames, list | tuple) or not frames:
        raise ProcedureError("frames: give a list of measurements (a result collected in a loop)")
    return list(frames)


def _stack(frames: Any, method: str = "mean", sigma: float = 3.0) -> Any:
    from lumos.processing.reduction import stack

    return stack(_frames(frames), method=str(method), sigma=float(sigma))


def _subtract(a: Any, b: Any) -> Any:
    return a - b


def _divide(a: Any, b: Any) -> Any:
    from lumos.fields import DataProduct

    if isinstance(a, DataProduct) and isinstance(b, DataProduct):
        return DataProduct(a.data / b.data, axes=a.axes, meta=a.meta)
    return a / b


def _scale(product: Any, factor: Any) -> Any:
    return product * factor


def _subtract_dark(frame: Any, dark: Any) -> Any:
    from lumos.processing.reduction import subtract_dark

    return subtract_dark(frame, dark)


def _flat_field(frame: Any, flat: Any, normalize: bool = True) -> Any:
    from lumos.processing.reduction import flat_field

    return flat_field(frame, flat, normalize=bool(normalize))


def _crop(product: Any, x: Any, y: Any, nx: int, ny: int) -> Any:
    from lumos.processing.reduction import crop

    return crop(product, center=(x, y), size=(int(nx), int(ny)))


def _rebin(product: Any, factor: int) -> Any:
    from lumos.processing.reduction import rebin

    return rebin(product, int(factor))


def _data(product: Any) -> Any:
    """Return the values of a measurement (a quantity array), or the quantity itself."""
    data = getattr(product, "data", product)
    if not isinstance(data, u.Quantity):
        raise ProcedureError(f"expected a measurement or a quantity, got {type(product).__name__}")
    return data


def _total(product: Any) -> u.Quantity:
    return _data(product).sum()


def _maximum(product: Any) -> u.Quantity:
    return _data(product).max()


def _minimum(product: Any) -> u.Quantity:
    return _data(product).min()


def _average(product: Any) -> u.Quantity:
    return _data(product).mean()


def _element(product: Any, index: int) -> u.Quantity:
    return _data(product).ravel()[int(index)]


def _photometry(image: Any, x: Any, y: Any, radius: Any) -> u.Quantity:
    from lumos.processing.photometry import aperture_photometry

    return aperture_photometry(image, center=(x, y), radius=radius).flux


#: The processing functions of the procedures, by name (the only computations a procedure can ask for).
FUNCTIONS: dict[str, Function] = {
    "mean": Function(lambda frames: _stack(frames, "mean"), ("frames",), doc="mean of frames"),
    "median": Function(lambda frames: _stack(frames, "median"), ("frames",), doc="median of frames"),
    "stack": Function(_stack, ("frames",), ("method", "sigma"), "mean, median or sigma_clip of frames"),
    "subtract": Function(_subtract, ("a", "b"), doc="a - b"),
    "divide": Function(_divide, ("a", "b"), doc="a / b"),
    "scale": Function(_scale, ("product", "factor"), doc="product × factor"),
    "subtract_dark": Function(_subtract_dark, ("frame", "dark"), doc="frame minus a dark frame"),
    "flat_field": Function(_flat_field, ("frame", "flat"), ("normalize",), "frame divided by a flat"),
    "crop": Function(_crop, ("product", "x", "y", "nx", "ny"), doc="nx × ny pixels around (x, y)"),
    "rebin": Function(_rebin, ("product", "factor"), doc="sum of factor × factor pixels"),
    "total": Function(_total, ("product",), doc="sum of all the values"),
    "maximum": Function(_maximum, ("product",), doc="largest value"),
    "minimum": Function(_minimum, ("product",), doc="smallest value"),
    "average": Function(_average, ("product",), doc="mean of all the values"),
    "element": Function(_element, ("product", "index"), doc="value number index (flattened)"),
    "photometry": Function(_photometry, ("image", "x", "y", "radius"), doc="flux in a circular aperture"),
}


# --- Values -----------------------------------------------------------------------------------------------


def _reference(value: Any) -> str | None:
    return value[1:] if isinstance(value, str) and value.startswith("$") else None


def _literal(value: Any) -> Any:
    """Return the Python value of a literal: a quantity for text with a unit, else the value itself."""
    if isinstance(value, str):
        text = value.strip()
        array = _ARRAY.match(text)
        try:
            if array:
                return u.Quantity(json.loads(array.group(1)), array.group(2).strip())
            quantity = u.Quantity(text)
        except (TypeError, ValueError):
            return value
        return quantity
    if isinstance(value, Mapping) and "value" in value and "unit" in value:
        return u.Quantity(value["value"], value["unit"])
    return value


def _range(spec: Mapping[str, Any]) -> list[Any]:
    start, stop = _literal(spec["start"]), _literal(spec["stop"])
    num = int(spec["num"])
    unit = start.unit if isinstance(start, u.Quantity) else None
    low = float(np.asarray(u.Quantity(start).to_value(unit))) if unit is not None else float(start)
    high = float(np.asarray(u.Quantity(stop).to_value(unit))) if unit is not None else float(stop)
    values = np.geomspace(low, high, num) if spec.get("log") else np.linspace(low, high, num)
    return [v * unit if unit is not None else float(v) for v in values]


# --- Procedure --------------------------------------------------------------------------------------------


@dataclass
class Procedure:
    """A named list of steps (see the module documentation)."""

    name: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    description: str = ""

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Procedure:
        """Read a procedure from its JSON form (checks its structure; :meth:`problems` checks its content)."""
        if not isinstance(data, Mapping):
            raise ProcedureError("a procedure is a JSON object with a name and steps")
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ProcedureError("a procedure needs a name")
        steps = data.get("steps", [])
        if not isinstance(steps, list):
            raise ProcedureError("the steps of a procedure are a list")
        return cls(name=name, steps=copy.deepcopy(list(steps)), description=str(data.get("description", "")))

    def to_json(self) -> dict[str, Any]:
        """Return the JSON form of the procedure."""
        return {"name": self.name, "description": self.description, "steps": copy.deepcopy(self.steps)}

    # --- Checks -------------------------------------------------------------------------------------

    def problems(self, ctx: Any) -> list[str]:
        """Return what would stop the procedure in the context ``ctx`` (empty when it can run)."""
        checker = _Checker(ctx)
        checker.steps(self.steps, "steps", depth=0)
        if checker.iterations > MAX_ITERATIONS:
            checker.problems.append(
                f"the loops make {checker.iterations} iterations, more than the {MAX_ITERATIONS} allowed"
            )
        return checker.problems

    def count_steps(self) -> int:
        """Return the number of steps run (loops unrolled), used for the progress."""
        return _Checker(None).count(self.steps)

    def devices(self, ctx: Any) -> list[str]:
        """Elements acting on equipment (operation mode) that the procedure calls actions of."""
        found: list[str] = []
        for step in _walk(self.steps):
            if step.get("do") != "action":
                continue
            path = step.get("element")
            try:
                element = ctx[path]
            except (KeyError, TypeError, ValueError):
                continue
            if ctx.mode_of(element) == "operation" and path not in found:
                found.append(str(path))
        return found

    def fed_ports(self) -> list[str]:
        """Return the inputs fed by ``use`` steps."""
        return [str(step["port"]) for step in _walk(self.steps) if step.get("do") == "use" and "port" in step]


def _walk(steps: Sequence[Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        found.append(dict(step))
        for key in ("steps", "then", "else"):
            if isinstance(step.get(key), list):
                found += _walk(step[key])
    return found


class _Checker:
    """Static checks of the steps (with ``ctx`` None: only counts)."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.problems: list[str] = []
        self.defined: set[str] = set()
        self.iterations = 0

    def count(self, steps: Sequence[Any]) -> int:
        total = 0
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            kind = step.get("do")
            inner = _list(step.get("steps"))
            if kind == "repeat":
                total += _int(step.get("times")) * self.count(inner)
            elif kind == "for_each":
                total += _length(step.get("values")) * self.count(inner)
            elif kind == "if":
                then = _list(step.get("then"))
                other = _list(step.get("else"))
                total += 1 + max(self.count(then), self.count(other))
            else:
                total += 1
        return total

    def _loops(self, steps: Sequence[Any]) -> int:
        """Loop iterations of ``steps`` (each pass of each loop, nested ones multiplied)."""
        total = 0
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            kind = step.get("do")
            inner = _list(step.get("steps"))
            if kind in ("repeat", "for_each"):
                n = _int(step.get("times")) if kind == "repeat" else _length(step.get("values"))
                total += n * (1 + self._loops(inner))
            elif kind == "if":
                then = _list(step.get("then"))
                other = _list(step.get("else"))
                total += max(self._loops(then), self._loops(other))
        return total

    def steps(self, steps: Any, where: str, depth: int) -> None:
        if depth == 0:
            self.iterations = self._loops(steps if isinstance(steps, list) else [])
        if depth > MAX_DEPTH:
            self.problems.append(f"{where}: more than {MAX_DEPTH} nested loops or conditions")
            return
        if not isinstance(steps, list):
            self.problems.append(f"{where}: the steps are a list")
            return
        for index, step in enumerate(steps):
            self.step(step, f"{where}[{index}]", depth)

    def need(self, value: Any, where: str) -> None:
        name = _reference(value)
        if name is not None and name not in self.defined:
            self.problems.append(f"{where}: the result {name!r} is used before it is defined")

    def name(self, step: Mapping[str, Any], key: str, where: str) -> None:
        name = step.get(key)
        if not isinstance(name, str) or not _NAME.match(name):
            self.problems.append(f"{where}: {key} must be a name (letters, digits, _), got {name!r}")
        else:
            self.defined.add(name)

    def element(self, path: Any, where: str) -> Any:
        try:
            return self.ctx[path]
        except (KeyError, TypeError, ValueError, LookupError):
            self.problems.append(f"{where}: no element {path!r} in the context")
            return None

    def port(self, path: Any, where: str, direction: str) -> None:
        try:
            ref = self.ctx.port(path)
        except (KeyError, TypeError, ValueError, LookupError, AttributeError):
            self.problems.append(f"{where}: no port {path!r} in the context")
            return
        if ref.direction != direction:
            kind = "an output" if direction == "out" else "an input"
            self.problems.append(f"{where}: {path} is not {kind}")

    def step(self, step: Any, where: str, depth: int) -> None:
        from lumos.core.action import actions_of
        from lumos.core.param import parse_target

        if not isinstance(step, Mapping):
            self.problems.append(f"{where}: a step is a JSON object with 'do'")
            return
        kind = step.get("do")
        where = f"{where} ({kind})"
        if kind not in _KINDS:
            self.problems.append(f"{where}: unknown step {kind!r}; steps: {', '.join(_KINDS)}")
            return
        if kind == "action":
            element = self.element(step.get("element"), where)
            if element is not None and step.get("action") not in actions_of(type(element)):
                known = ", ".join(sorted(actions_of(type(element)))) or "none"
                self.problems.append(
                    f"{where}: {step.get('element')} has no action {step.get('action')!r} ({known})"
                )
            for value in [*step.get("args", []), *dict(step.get("kwargs", {})).values()]:
                self.need(value, where)
        elif kind == "measure":
            outputs = step.get("outputs")
            if not isinstance(outputs, list) or not outputs:
                self.problems.append(f"{where}: outputs is a list of ports such as ['camera.out']")
            else:
                for output in outputs:
                    self.port(output, where, "out")
            self.name(step, "as", where)
        elif kind in ("repeat", "for_each"):
            if kind == "repeat":
                times = step.get("times")
                if isinstance(times, bool) or not isinstance(times, int) or times < 0:
                    self.problems.append(f"{where}: times is a whole number, got {times!r}")
            else:
                values = step.get("values")
                if isinstance(values, Mapping):
                    if not {"start", "stop", "num"} <= set(values):
                        self.problems.append(f"{where}: a range needs start, stop and num")
                elif not isinstance(values, list):
                    self.problems.append(f"{where}: values is a list or a range {{start, stop, num}}")
                self.name(step, "as", where)
            self.steps(step.get("steps", []), f"{where}.steps", depth + 1)
        elif kind == "process":
            function = FUNCTIONS.get(step.get("function"))  # type: ignore[arg-type]
            if function is None:
                self.problems.append(
                    f"{where}: unknown function {step.get('function')!r}; functions: {', '.join(FUNCTIONS)}"
                )
            else:
                args = dict(step.get("args", {}))
                missing = [a for a in function.required if a not in args]
                unknown = [a for a in args if a not in function.required + function.optional]
                if missing:
                    self.problems.append(f"{where}: {step.get('function')} needs {', '.join(missing)}")
                if unknown:
                    self.problems.append(
                        f"{where}: {step.get('function')} has no argument {', '.join(unknown)}"
                    )
                for value in args.values():
                    self.need(value, where)
            self.name(step, "as", where)
        elif kind == "set":
            target = step.get("target")
            owner, _, name = str(target).rpartition(".")
            element = self.element(owner, where) if owner else None
            if not owner:
                self.problems.append(f"{where}: target is 'element.field', got {target!r}")
            elif element is not None:
                try:
                    field_name, _ = parse_target(name)
                except ValueError:
                    field_name = name
                if field_name not in getattr(type(element), "_lumos_fields", {}):
                    self.problems.append(f"{where}: {owner} has no field {field_name!r}")
            self.need(step.get("value"), where)
        elif kind == "use":
            self.port(step.get("port"), where, "in")
            if step.get("result") is not None:
                result = step.get("result")
                self.need(f"${result}", where)
        elif kind == "wait":
            duration = _literal(step.get("duration"))
            if not (
                isinstance(duration, u.Quantity)
                and duration.unit is not None
                and duration.unit.is_equivalent(u.s)
            ):
                self.problems.append(
                    f"{where}: duration is a time such as '2 s', got {step.get('duration')!r}"
                )
        elif kind == "if":
            if step.get("op") not in _OPERATORS:
                self.problems.append(f"{where}: op is one of {', '.join(_OPERATORS)}")
            self.need(step.get("left"), where)
            self.need(step.get("right"), where)
            before = set(self.defined)
            self.steps(step.get("then", []), f"{where}.then", depth + 1)
            after_then = set(self.defined)
            self.defined = set(before)
            self.steps(step.get("else", []), f"{where}.else", depth + 1)
            self.defined |= after_then
        elif kind in ("save", "show"):
            results = step.get("results")
            if not isinstance(results, list):
                self.problems.append(f"{where}: results is a list of names")
            else:
                for result in results:
                    self.need(f"${result}", where)


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _length(values: Any) -> int:
    if isinstance(values, list):
        return len(values)
    if isinstance(values, Mapping):
        return _int(values.get("num"))
    return 0


# --- Execution --------------------------------------------------------------------------------------------


@dataclass
class ProcedureResult:
    """Named results of a run, the results to display and the journal of the steps."""

    name: str
    results: dict[str, Any] = field(default_factory=dict)
    shown: list[str] = field(default_factory=list)
    journal: list[dict[str, Any]] = field(default_factory=list)
    archives: list[Any] = field(default_factory=list)

    def __getitem__(self, name: str) -> Any:
        """Return the named result ``name``."""
        return self.results[name]

    def __contains__(self, name: object) -> bool:
        """Whether a result is named ``name``."""
        return name in self.results


_ABSENT = object()


class _Runner:
    def __init__(
        self,
        ctx: Any,
        procedure: Procedure,
        progress: Callable[[int, int], None] | None,
        cancelled: Callable[[], bool] | None,
        archive_root: Any,
    ) -> None:
        self.ctx = ctx
        self.result = ProcedureResult(procedure.name)
        self.feeds: dict[str, Any] = {}
        self.progress = progress
        self.cancelled = cancelled
        self.archive_root = archive_root
        self.total = max(procedure.count_steps(), 1)
        self.done = 0

    def value(self, value: Any) -> Any:
        name = _reference(value)
        if name is not None:
            return self.result.results[name]
        return _literal(value)

    def tick(self, step: Mapping[str, Any], started: float, **details: Any) -> None:
        self.done += 1
        self.result.journal.append(
            {
                "do": step.get("do"),
                "time_s": self.ctx.elapsed.to_value(u.s),
                "seconds": _time.perf_counter() - started,
            }
            | details
        )
        if self.progress is not None:
            try:
                self.progress(self.done, self.total)
            except Exception as error:  # e.g. the cancellation or the time limit of a job
                raise _Callback(error) from error

    def run(self, steps: Sequence[Mapping[str, Any]]) -> None:
        for step in steps:
            if self.cancelled is not None and self.cancelled():
                raise ProcedureError(f"the procedure {self.result.name!r} was cancelled")
            try:
                self.step(step)
            except (ProcedureError, _Callback):
                raise
            except Exception as error:  # report the step that failed
                raise ProcedureError(f"step {step.get('do')} failed: {error}") from error

    def step(self, step: Mapping[str, Any]) -> None:
        kind = step["do"]
        started = _time.perf_counter()
        results = self.result.results
        if kind == "action":
            element = self.ctx[step["element"]]
            args = [self.value(a) for a in step.get("args", [])]
            kwargs = {k: self.value(v) for k, v in dict(step.get("kwargs", {})).items()}
            getattr(element, step["action"])(*args, **kwargs)
            self.tick(step, started, element=step["element"], action=step["action"])
        elif kind == "measure":
            feeds = {self.ctx.port(port): value for port, value in self.feeds.items()}
            computed = self.ctx.run(step["outputs"], feeds=feeds)
            outputs = step["outputs"]
            value = computed[outputs[0]] if len(outputs) == 1 else {o: computed[o] for o in outputs}
            if step.get("collect"):
                results.setdefault(step["as"], []).append(value)
            else:
                results[step["as"]] = value
            self.tick(step, started, outputs=list(outputs), name=step["as"])
        elif kind == "repeat":
            for _ in range(int(step["times"])):
                self.run(step.get("steps", []))
        elif kind == "for_each":
            spec = step["values"]
            values = _range(spec) if isinstance(spec, Mapping) else [self.value(v) for v in spec]
            name, previous = step["as"], results.get(step["as"], _ABSENT)
            for value in values:
                results[name] = value
                self.run(step.get("steps", []))
            # the loop variable lives in the loop only
            if previous is _ABSENT:
                results.pop(name, None)
            else:
                results[name] = previous
        elif kind == "process":
            function = FUNCTIONS[step["function"]]
            args = {k: self.value(v) for k, v in dict(step.get("args", {})).items()}
            results[step["as"]] = function.run(**args)
            self.tick(step, started, function=step["function"], name=step["as"])
        elif kind == "set":
            owner, _, name = str(step["target"]).rpartition(".")
            self.ctx[owner].set_target(name, self.value(step["value"]))
            self.tick(step, started, target=step["target"])
        elif kind == "use":
            result = step.get("result")
            if result is None:
                self.feeds.pop(step["port"], None)
            else:
                self.feeds[step["port"]] = results[result]
            self.tick(step, started, port=step["port"])
        elif kind == "wait":
            self.ctx.wait(_literal(step["duration"]))
            self.tick(step, started)
        elif kind == "if":
            left, right = self.value(step.get("left")), self.value(step.get("right"))
            try:
                chosen = bool(_OPERATORS[step["op"]](left, right))
            except (TypeError, ValueError, u.UnitsError) as error:
                raise ProcedureError(f"cannot compare {left!r} and {right!r}: {error}") from error
            self.tick(step, started, chosen="then" if chosen else "else")
            self.run(step.get("then", []) if chosen else step.get("else", []))
        elif kind == "save":
            from lumos.results import new_archive

            archive = new_archive(
                self.ctx, str(step.get("archive") or self.result.name), root=self.archive_root
            )
            for name in step["results"]:
                value = results[name]
                items = value if isinstance(value, list) else [value]
                for index, item in enumerate(items):
                    archive.save_data(name if len(items) == 1 else f"{name}_{index}", item)
            self.result.archives.append(archive)
            self.tick(step, started, results=list(step["results"]))
        elif kind == "show":
            for name in step["results"]:
                if name not in self.result.shown:
                    self.result.shown.append(name)
            self.tick(step, started, results=list(step["results"]))


def run_procedure(
    ctx: Any,
    procedure: Procedure,
    *,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    confirm: Callable[[list[str]], bool] | None = None,
    archive_root: Any = None,
) -> ProcedureResult:
    """Run ``procedure`` in the context ``ctx`` and return its named results.

    Parameters
    ----------
    progress
        Called as ``progress(done, total)`` after each step (loops unrolled).
    cancelled
        Checked before each step; when it returns ``True`` the run stops with a :class:`ProcedureError`.
    confirm
        Called with the elements acting on equipment, when there are some; the run starts only if it returns
        ``True`` (a human confirmation, ADR-0014).
    archive_root
        Directory of the archives written by ``save`` steps.

    Raises
    ------
    ProcedureError
        If the procedure has problems, is not confirmed, is cancelled, or a step fails.
    """
    problems = procedure.problems(ctx)
    if problems:
        raise ProcedureError(f"the procedure {procedure.name!r} cannot run: " + "; ".join(problems))
    devices = procedure.devices(ctx)
    if devices and (confirm is None or not confirm(devices)):
        raise ProcedureError(
            f"the procedure {procedure.name!r} acts on equipment ({', '.join(devices)}) and needs a human "
            "confirmation"
        )
    drivers = [ctx[path].driver for path in devices if ctx[path].driver is not None]
    for driver in drivers:
        driver.confirm = lambda command: True  # confirmed by the user for this run
    runner = _Runner(ctx, procedure, progress, cancelled, archive_root)
    entry = ctx._entry("procedure")
    entry.args = procedure.name
    started = _time.perf_counter()
    try:
        runner.run(procedure.steps)
    except _Callback as wrapped:
        raise wrapped.error from None
    finally:
        for driver in drivers:
            driver.confirm = None
        entry.measured = _time.perf_counter() - started
        entry.message = f"procedure {procedure.name!r}: {runner.done} step(s) in {entry.measured:.3g} s"
        ctx.journal.append(entry)
        logger.info("%s", entry.message)
    return runner.result
