"""Reading archives back (step P7.5, REQ-RES-6).

:func:`load_archive` restores what :func:`lumos.results.new_archive` and ``ctx.run(..., archive=...)``
wrote: the context, the metadata, the journal, the data files and the list of figures.

Nothing is unpickled or imported while reading (NF-7): the context goes through the ``.lumos`` reader and its
registry, ``.npz`` files are read with ``allow_pickle=False``, JSON data files are decoded by a fixed list of
signal classes, and ``.plt`` figures (pickled matplotlib figures) are only listed: open them with
``pltedit.load`` if you trust their origin.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from lumos.results.figures import load_data

__all__ = ["LoadedArchive", "load_archive"]


@dataclass
class LoadedArchive:
    """Content of an archive.

    Attributes
    ----------
    path
        Directory of the archive.
    metadata
        Content of ``metadata.yml``.
    context
        The archived :class:`~lumos.Context`.
    journal
        Entries of ``lumos.log`` (dictionaries).
    data
        Data files by name: :class:`~lumos.fields.DataProduct`, optical signals, or dictionaries of arrays and
        quantities.
    figures
        Image files (``.png``) by figure name.
    editable_figures
        PltEdit files (``.plt``) by figure name (not loaded).
    """

    path: Path
    metadata: dict[str, Any]
    context: Any
    journal: list[dict[str, Any]] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    figures: dict[str, Path] = field(default_factory=dict)
    editable_figures: dict[str, Path] = field(default_factory=dict)


def _signal_classes() -> dict[str, Any]:
    from lumos.fields import Bus, IncoherentBundle, ModalField, RadianceField, SourceList, Wavecube, Wavefront

    classes = (Wavefront, Wavecube, IncoherentBundle, RadianceField, ModalField, Bus, SourceList)
    known: dict[str, Any] = {cls.__name__: cls for cls in classes}
    known["SymbolicField"] = _SymbolicFieldReader  # imports SymPy only for archives holding one
    return known


class _SymbolicFieldReader:
    @staticmethod
    def from_json(data: Any) -> Any:
        from lumos.symbolic.serialize import field_from_json

        return field_from_json(data)


def _read_npz(path: Path) -> Any:
    from lumos.fields import DataProduct

    with np.load(path, allow_pickle=False) as archive:
        is_product = "header" in archive.files and "data" in archive.files
    return DataProduct.from_npz(path) if is_product else load_data(path)


def _read_json(path: Path) -> Any:
    document = json.loads(path.read_text(encoding="utf-8"))
    classes = _signal_classes()
    kind = document.get("signal") if isinstance(document, dict) else None
    if kind not in classes:
        raise ValueError(f"{path.name}: unknown signal type {kind!r}; known: {', '.join(classes)}")
    return classes[kind].from_json(document["data"])


def load_archive(path: str | os.PathLike[str]) -> LoadedArchive:
    """Read an archive directory.

    Parameters
    ----------
    path
        Directory of the archive (or a :class:`~lumos.results.RunArchive`).

    Raises
    ------
    FileNotFoundError
        If the directory has no ``metadata.yml``.
    ValueError
        If a data file names an unknown signal type.
    """
    from lumos.io import load

    root = Path(os.fspath(path))
    metadata_file = root / "metadata.yml"
    if not metadata_file.is_file():
        raise FileNotFoundError(f"{root} is not a LUMOS archive: metadata.yml is missing")
    metadata = yaml.safe_load(metadata_file.read_text(encoding="utf-8")) or {}
    context_file = root / "context.lumos"
    context = load(context_file) if context_file.is_file() else None
    journal: list[dict[str, Any]] = []
    log = root / "lumos.log"
    if log.is_file():
        journal = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    data: dict[str, Any] = {}
    data_dir = root / "data"
    if data_dir.is_dir():
        for file in sorted(data_dir.iterdir()):
            name = file.name[: -len(file.suffix)] if file.suffix else file.name
            if file.suffix == ".npz":
                data[name] = _read_npz(file)
            elif file.suffix == ".json":
                data[name] = _read_json(file)
    figures: dict[str, Path] = {}
    editable: dict[str, Path] = {}
    figures_dir = root / "figures"
    if figures_dir.is_dir():
        for file in sorted(figures_dir.iterdir()):
            if file.suffix == ".plt":
                editable[file.stem] = file
            elif file.suffix in (".png", ".svg", ".pdf", ".jpg") and file.stem not in figures:
                figures[file.stem] = file
    return LoadedArchive(root, metadata, context, journal, data, figures, editable)
