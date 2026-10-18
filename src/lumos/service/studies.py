"""Values of sweeps, priors of free parameters and tables of studies (step P14.13, REQ-CTX-10).

A study runs a context several times: a *sweep* over listed values or ranges of some parameters, or a
*Monte-Carlo run* drawing the free parameters from their priors (and the noise of the stochastic elements).
These helpers read the requests of the web interface and of AI assistants (values as ``"5 mm"``) and write
the points back in the same form.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from astropy import units as u

from lumos.core.param import Param
from lumos.core.priors import LogUniform, Normal, Prior, Uniform
from lumos.service.errors import ServiceError
from lumos.service.models import StudyResult
from lumos.service.values import format_value, parse_quantity, parse_value

__all__ = ["count_points", "parse_prior", "prior_summary", "study_table", "sweep_values"]

#: Prior kinds and the names of their two values.
PRIORS: dict[str, tuple[type[Prior], str, str]] = {
    "uniform": (Uniform, "low", "high"),
    "loguniform": (LogUniform, "low", "high"),
    "normal": (Normal, "mean", "std"),
}


def sweep_values(field: Any, spec: Any, *, name: str) -> list[Any]:
    """Return the values of a swept field from a list or a range ``{"start", "stop", "num", "log"}``.

    Quantities are written with their unit (``"1 mm"``); a range is evenly spaced, or geometrically with
    ``log: true`` (``start`` and ``stop`` of the same sign, not zero).
    """
    if isinstance(spec, Mapping):
        missing = {"start", "stop", "num"} - set(spec)
        if missing:
            raise ServiceError(
                f"{name}: the range lacks {', '.join(sorted(missing))}",
                hint="give a list of values, or a range {'start': '1 mm', 'stop': '5 mm', 'num': 5}",
            )
        num = int(spec["num"])
        if num < 1:
            raise ServiceError(f"{name}: num must be at least 1, got {num}", hint="give the number of values")
        unit = getattr(field, "unit", None) if isinstance(field, Param) else None
        start = parse_quantity(spec["start"], unit, name=f"{name}.start")
        stop = parse_quantity(spec["stop"], unit, name=f"{name}.stop")
        stop = stop.to(start.unit, equivalencies=u.spectral())
        a, b = float(start.value), float(stop.value)
        if spec.get("log"):
            if a == 0 or b == 0 or (a > 0) != (b > 0):
                raise ServiceError(
                    f"{name}: a logarithmic range needs start and stop of the same sign, not zero",
                    hint="use a linear range, or bounds such as 1 and 100",
                )
            numbers = np.sign(a) * np.geomspace(abs(a), abs(b), num)
        else:
            numbers = np.linspace(a, b, num)
        values: Sequence[Any] = [u.Quantity(v, start.unit) for v in numbers]
        if not isinstance(field, Param):
            values = [float(v.value) for v in values]
    elif isinstance(spec, list | tuple):
        values = spec
    else:
        raise ServiceError(
            f"{name}: expected a list of values or a range, got {spec!r}",
            hint="e.g. ['1 mm', '2 mm'] or {'start': '1 mm', 'stop': '5 mm', 'num': 5}",
        )
    if not values:
        raise ServiceError(f"{name}: no values to sweep", hint="give at least one value")
    return [
        value.to(field.unit, equivalencies=u.spectral())
        if isinstance(field, Param) and isinstance(value, u.Quantity)
        else parse_value(field, value, name=name)
        for value in values
    ]


def count_points(lengths: Sequence[int], mode: str) -> int:
    """Return the points of a sweep: product of the lengths (``grid``) or their common length (``zip``)."""
    if mode == "grid":
        return math.prod(lengths)
    if mode == "zip":
        if len(set(lengths)) > 1:
            raise ServiceError(
                f"a zip sweep pairs values of the same count, got {sorted(set(lengths))}",
                hint="give as many values to every parameter, or use mode='grid'",
            )
        return lengths[0]
    raise ServiceError(
        f"unknown sweep mode {mode!r}", hint="mode='grid' (every combination) or 'zip' (pairs)"
    )


def parse_prior(field: Param, request: Mapping[str, Any], *, name: str) -> Prior:
    """Return the prior described by ``{"kind": "normal", "mean": "1 mm", "std": "50 um"}``."""
    kind = str(request.get("kind", ""))
    if kind not in PRIORS:
        raise ServiceError(
            f"{name}: unknown prior {kind!r}",
            hint=f"kind is one of {', '.join(PRIORS)} (e.g. {{'kind': 'normal', 'mean': ..., 'std': ...}})",
        )
    cls, first, second = PRIORS[kind]
    if first not in request or second not in request:
        raise ServiceError(
            f"{name}: a {kind} prior needs {first} and {second}", hint=f"give {first} and {second}"
        )
    values = [
        parse_quantity(request[key], field.unit, name=f"{name}.{key}").to(
            field.unit, equivalencies=u.spectral()
        )
        for key in (first, second)
    ]
    try:
        prior = cls(*values)
        field.check_prior(prior)
    except (TypeError, ValueError, u.UnitsError) as error:
        raise ServiceError(f"{name}: {error}", hint=f"check {first} and {second}") from error
    return prior


def prior_summary(prior: Prior | None, unit: Any) -> dict[str, str] | None:
    """Describe a prior, values in the unit of its parameter: ``{"kind": "normal", "mean": "1 mm"}``."""
    if prior is None:
        return None
    kind = next(name for name, (cls, _, _) in PRIORS.items() if type(prior) is cls)
    _, first, second = PRIORS[kind]
    return {"kind": kind, **{key: str(format_value(getattr(prior, key), unit)) for key in (first, second)}}


def _number(text: str) -> tuple[float, str]:
    quantity = u.Quantity(text)
    return float(quantity.value), u.Unit(quantity.unit).to_string()


def study_table(study: StudyResult) -> str:
    """CSV table of a study: one row per point, parameters then outputs (statistics of non-scalar outputs)."""
    first = study.points[0] if study.points else None
    header = ["point", "realization"]
    for path in study.parameters:
        unit = _number(first.params[path])[1] if first else ""
        header.append(f"{path} [{unit}]" if unit else path)
    columns: list[tuple[str, str | None]] = []
    for port in study.outputs:
        summary = first.outputs.get(port) if first else None
        unit = f" [{summary.unit}]" if summary is not None and summary.unit else ""
        if summary is not None and summary.values is not None and len(summary.values) == 1:
            columns.append((port, None))
            header.append(f"{port}{unit}")
        else:
            for statistic in ("min", "max", "mean", "sum"):
                columns.append((port, statistic))
                header.append(f"{port} {statistic}{unit}")
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for index, point in enumerate(study.points, start=1):
        row: list[Any] = [index, point.realization]
        row += [_number(point.params[path])[0] for path in study.parameters]
        for port, statistic in columns:
            summary = point.outputs.get(port)
            if summary is None:
                row.append("")
            elif statistic is None:
                row.append(summary.values[0] if summary.values else "")
            else:
                row.append(summary.statistics.get(statistic, ""))
        writer.writerow(row)
    return buffer.getvalue()
