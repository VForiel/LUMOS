"""Archives of runs (step P7.4, REQ-RES-1, REQ-RES-2, REQ-RES-5).

An archive is a dated directory created with ``vfst.archive.new`` (VFSciTools), by default in an
``archives`` directory **next to the executed script or notebook**, which makes every result traceable::

    archives/2026-10-17_1a2b3c4/0001-first-light/
    ├── context.lumos     # the context, reloadable with lumos.io.load
    ├── metadata.yml      # date, environment, versions, git, hardware, backend, seed, clock, origin...
    ├── lumos.log         # the journal of the context, one JSON object per line
    ├── figures/          # .png and .plt (PltEdit) files
    └── data/             # .npz (arrays, data products) and .json (optical signals) files

Archives are created by ``ctx.run(outputs, archive=...)``, which also saves the computed outputs in
``data/``, or explicitly by :func:`new_archive`. :func:`lumos.load_archive` reads them back.
"""

from __future__ import annotations

import copy
import functools
import json
import logging
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from astropy import units as u

from lumos.results.figures import save_data as _save_npz
from lumos.results.figures import save_figure as _save_figure
from lumos.units import quantity_to_json

__all__ = ["DEPENDENCIES", "RunArchive", "default_archive_root", "new_archive"]

logger = logging.getLogger("lumos.results")

#: Distributions whose versions are recorded in the metadata (when installed).
DEPENDENCIES: tuple[str, ...] = (
    "numpy",
    "scipy",
    "astropy",
    "matplotlib",
    "sympy",
    "pyyaml",
    "array-api-compat",
    "vfscitools",
    "jax",
    "jaxlib",
    "cupy-cuda12x",
    "torch",
    "numba",
)

#: Signal classes saved as JSON files in ``data/`` (they define ``to_json`` and ``from_json``).
SIGNAL_CLASSES: tuple[str, ...] = ("Wavefront", "Wavecube", "IncoherentBundle", "RadianceField", "ModalField")


def default_archive_root() -> Path:
    """Return the directory in which archives are created.

    In order of priority: ``lumos.config.archive_root``; an ``archives`` directory next to the running
    notebook (VS Code's ``__vsc_ipynb_file__`` or Jupyter's ``JPY_SESSION_NAME``); next to the executed
    script (``__main__.__file__``); in the current directory otherwise.
    """
    from lumos import config

    if config.archive_root is not None:
        return Path(config.archive_root)
    main = sys.modules.get("__main__")
    notebook = getattr(main, "__vsc_ipynb_file__", None) or os.environ.get("JPY_SESSION_NAME")
    if notebook:
        return Path(notebook).resolve().parent / "archives"
    script = getattr(main, "__file__", None)
    if script:
        return Path(script).resolve().parent / "archives"
    return Path.cwd() / "archives"


def _dependencies() -> dict[str, str]:
    found = {}
    for name in DEPENDENCIES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
    return found


@functools.cache
def _hardware() -> dict[str, Any]:
    """Hardware report of the machine, probed once per process (probing GPUs takes about a second)."""
    from lumos.backend.hardware import hardware_report

    return hardware_report().to_dict()


def _backend_name(ctx: Any) -> str:
    from lumos import backend

    return backend.current().name if ctx.backend == "auto" else ctx.backend


def _metadata(ctx: Any) -> dict[str, Any]:
    import lumos

    return {
        "context": ctx.name,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "lumos_version": lumos.__version__,
        "dependencies": _dependencies(),
        "hardware": copy.deepcopy(_hardware()),
        "backend": _backend_name(ctx),
        "precision": lumos.config.precision,
        "seed": ctx.seed,
        "mode": ctx.mode,
        "twin": ctx.twin,
        "clock": {"start": str(ctx.clock.start.isot), "elapsed": quantity_to_json(ctx.clock.elapsed)},
        "origin": ctx.origin,
        "durations": {},
    }


class RunArchive:
    """An archive directory of a run (see the module documentation).

    Attributes
    ----------
    path
        Directory of the archive.
    metadata
        Content of ``metadata.yml``.
    """

    def __init__(self, path: Path, metadata: dict[str, Any]) -> None:
        self.path = Path(path)
        self.metadata = metadata

    def __fspath__(self) -> str:
        """Let the archive be used as a path."""
        return os.fspath(self.path)

    def __repr__(self) -> str:
        """Represent the archive by its path."""
        return f"<RunArchive {self.path}>"

    @property
    def figures(self) -> Path:
        """Directory of the figures."""
        return self.path / "figures"

    @property
    def data(self) -> Path:
        """Directory of the data files."""
        return self.path / "data"

    def update_metadata(self, **entries: Any) -> None:
        """Add or replace entries of ``metadata.yml``."""
        self.metadata.update(entries)
        (self.path / "metadata.yml").write_text(
            yaml.safe_dump(_plain(self.metadata), sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

    def save_figure(
        self,
        fig: Any,
        name: str,
        *,
        data: Mapping[str, Any] | None = None,
        formats: Sequence[str] = ("png", "plt"),
    ) -> list[Path]:
        """Save a figure in ``figures/<name>.png`` and ``.plt``, and its data in ``data/<name>.npz``."""
        written = _save_figure(fig, self.figures / name, formats=formats)
        if data is not None:
            written.append(_save_npz(self.data / name, data))
        return written

    def save_data(self, name: str, value: Any) -> Path:
        """Save ``value`` in ``data/``.

        - :class:`~lumos.fields.DataProduct`: ``<name>.npz`` (read back with ``DataProduct.from_npz``);
        - optical signals (wavefronts, wavecubes, bundles, radiance and modal fields): ``<name>.json``;
        - a mapping of arrays or quantities: ``<name>.npz`` (see :func:`lumos.results.save_data`);
        - an array or a quantity: ``<name>.npz`` holding the entry ``value``.
        """
        from lumos.fields import DataProduct

        self.data.mkdir(exist_ok=True)
        if isinstance(value, DataProduct):
            target = self.data / f"{name}.npz"
            value.to_npz(target)
            return target
        if type(value).__name__ in SIGNAL_CLASSES and hasattr(value, "to_json"):
            target = self.data / f"{name}.json"
            document = {"signal": type(value).__name__, "data": value.to_json()}
            target.write_text(json.dumps(document), encoding="utf-8")
            return target
        if isinstance(value, Mapping):
            return _save_npz(self.data / f"{name}.npz", value)
        if isinstance(value, u.Quantity | np.ndarray):
            return _save_npz(self.data / f"{name}.npz", {"value": value})
        raise TypeError(f"cannot save a {type(value).__name__} in an archive")

    def write_log(self, ctx: Any) -> Path:
        """Write the journal of ``ctx`` in ``lumos.log`` (one JSON object per line)."""
        target = self.path / "lumos.log"
        lines = [json.dumps(entry, ensure_ascii=False, default=str) for entry in ctx.journal.to_json()]
        target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return target


def _plain(value: Any) -> Any:
    """Convert NumPy scalars and tuples to plain YAML types."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def new_archive(
    ctx: Any,
    name: str | None = None,
    *,
    root: str | os.PathLike[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RunArchive:
    """Create the archive of a context (``context.lumos``, ``metadata.yml``, ``lumos.log``, figures, data).

    Parameters
    ----------
    ctx
        The :class:`~lumos.Context` to archive (its current state).
    name
        Suffix of the archive directory.
    root
        Directory in which the dated archive tree is created (default: :func:`default_archive_root`).
    metadata
        Extra entries of ``metadata.yml``.
    """
    from vfscitools import archive as vfst_archive

    from lumos.io import save

    base = Path(root) if root is not None else default_archive_root()
    entries = {**_metadata(ctx), **dict(metadata or {})}
    created = vfst_archive.new(name, root=base, metadata=_plain(entries))
    archive = RunArchive(Path(created.path), dict(created.metadata))
    archive.update_metadata()  # rewrite with unicode allowed and plain types
    archive.figures.mkdir(exist_ok=True)
    archive.data.mkdir(exist_ok=True)
    save(ctx, archive.path / "context.lumos")
    archive.write_log(ctx)
    logger.info("archive created at %s", archive.path)
    return archive
