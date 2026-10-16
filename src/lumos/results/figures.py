"""Export of figures and of the data behind them (step P7.3, REQ-RES-4, REQ-RES-5).

A figure is saved as an image (``.png`` by default, or any matplotlib format) and as an editable ``.plt``
file (:func:`pltedit.save`, VFSciTools), which reopens in the PltEdit editor or with ``pltedit.load``. The
data it shows is saved next to it in a ``.npz`` file (:func:`save_data`).

``.npz`` files hold plain arrays only (no pickled object): a quantity ``name`` is stored as the array of its
values and the string ``__unit__name`` holding its unit, which :func:`load_data` turns back into a quantity.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
from astropy import units as u

__all__ = ["FIGURE_FORMATS", "load_data", "save_data", "save_figure"]

#: Formats accepted by :func:`save_figure`: ``plt`` (PltEdit) and the matplotlib image formats.
FIGURE_FORMATS: tuple[str, ...] = ("png", "plt", "pdf", "svg", "eps", "jpg", "tif")

_UNIT_PREFIX = "__unit__"
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.\-]*$")


def _figure_of(fig: Any) -> Any:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    if isinstance(fig, Axes):
        fig = fig.get_figure()
    if not isinstance(fig, Figure):
        raise TypeError(f"expected a matplotlib Figure or Axes, got {type(fig).__name__}")
    return fig


def save_figure(
    fig: Any,
    path: str | os.PathLike[str],
    *,
    data: Mapping[str, Any] | None = None,
    formats: Iterable[str] = ("png", "plt"),
) -> list[Path]:
    """Save a figure in several formats, and optionally the data it shows.

    Parameters
    ----------
    fig
        Matplotlib ``Figure`` or ``Axes`` (its figure is saved).
    path
        Path without suffix (a suffix is replaced): ``figures/psf`` gives ``figures/psf.png``...
    data
        Arrays or quantities shown by the figure, saved in ``<path>.npz`` (see :func:`save_data`).
    formats
        Formats to write, among :data:`FIGURE_FORMATS`.

    Returns
    -------
    list of Path
        The files written.
    """
    figure = _figure_of(fig)
    wanted = [f.lower().lstrip(".") for f in formats]
    unknown = [f for f in wanted if f not in FIGURE_FORMATS]
    if unknown:
        raise ValueError(f"unknown figure format(s) {unknown}; use {', '.join(FIGURE_FORMATS)}")
    stem = Path(path)
    stem = stem.with_suffix("") if stem.suffix else stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in wanted:
        target = stem.with_name(f"{stem.name}.{fmt}")
        if fmt == "plt":
            import pltedit

            pltedit.save(figure, target)
        else:
            figure.savefig(target, format=fmt)
        written.append(target)
    if data is not None:
        written.append(save_data(stem, data))
    return written


def save_data(path: str | os.PathLike[str], data: Mapping[str, Any]) -> Path:
    """Save named arrays and quantities in a ``.npz`` file (no pickled object) and return its path.

    Names must start with a letter and contain letters, digits, ``_``, ``.`` or ``-``.
    """
    arrays: dict[str, np.ndarray] = {}
    for name, value in data.items():
        if not isinstance(name, str) or not _NAME.match(name):
            raise ValueError(f"invalid data name {name!r}: start with a letter, then letters, digits, _ . -")
        if isinstance(value, u.Quantity):
            arrays[name] = np.asarray(value.value)
            arrays[_UNIT_PREFIX + name] = np.array(u.Unit(cast(Any, value.unit)).to_string())
        else:
            array = np.asarray(value)
            if array.dtype == object:
                raise TypeError(f"{name!r}: only numeric and string arrays can be saved, not Python objects")
            arrays[name] = array
    target = Path(path)
    target = target.with_suffix(".npz") if target.suffix != ".npz" else target
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, allow_pickle=False, **arrays)  # type: ignore[arg-type]
    return target


def load_data(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a file written by :func:`save_data`: arrays, and quantities for the entries with a unit."""
    with np.load(Path(path), allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    result: dict[str, Any] = {}
    for name, array in raw.items():
        if name.startswith(_UNIT_PREFIX):
            continue
        unit = raw.get(_UNIT_PREFIX + name)
        result[name] = array if unit is None else u.Quantity(array, u.Unit(str(unit)))
    return result
