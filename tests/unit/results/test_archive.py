"""Tests for run archives (step P7.4, REQ-RES-1, REQ-RES-2, REQ-RES-5).

Requirements:

- REQ-RES-1: every ``ctx.run(...)`` can produce an archive through ``vfst.archive`` (VFSciTools), created
  next to the executed script or notebook (configurable root);
- REQ-RES-2: the archive holds ``context.lumos``, ``metadata.yml`` (UTC date and time, OS, Python version,
  versions of LUMOS and key dependencies, git commit and branch, detected hardware, backend and precision,
  seeds, durations, mode, twin, simulated clock, origin Python/web/MCP), ``lumos.log``, ``figures/`` and
  ``data/``;
- REQ-RES-5: the data behind figures is saved.

Independent references: versions from ``importlib.metadata``, the OS from ``platform.system()``, the Python
version from ``platform.python_version()``, the git commit from ``git rev-parse HEAD``; YAML is read with
``yaml.safe_load`` and JSON lines with ``json.loads``.
"""

import json
import platform
import subprocess
import sys
import textwrap
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml
from astropy import units as u
from astropy.time import Time

import lumos
from lumos import Context
from lumos.elements.detectors import Camera, PowerMeter
from lumos.elements.sources import GaussianLaser
from lumos.fields import DataProduct
from lumos.results import RunArchive, default_archive_root, new_archive

START = Time("2026-10-16T12:00:00", scale="utc")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def bench(seed=7):
    ctx = Context(seed=seed, start=START, name="bench")
    laser = ctx.add(GaussianLaser(name="laser", n_pixels=32))
    laser >> Camera(name="cam", noise=False)
    ctx.connect(laser.port("out"), ctx.add(PowerMeter(name="meter")).port("in"))
    return ctx


def read_metadata(archive):
    return yaml.safe_load((Path(archive.path) / "metadata.yml").read_text(encoding="utf-8"))


# --- Layout -------------------------------------------------------------------------------------------------


def test_archive_layout(tmp_path):
    archive = new_archive(bench(), "demo", root=tmp_path)
    assert isinstance(archive, RunArchive)
    path = Path(archive.path)
    assert path.is_dir()
    assert path.is_relative_to(tmp_path)
    for name in ("context.lumos", "metadata.yml", "lumos.log"):
        assert (path / name).is_file()
    for name in ("figures", "data"):
        assert (path / name).is_dir()
    assert "demo" in path.name


def test_archive_context_reloads(tmp_path):
    ctx = bench()
    archive = new_archive(ctx, root=tmp_path)
    loaded = lumos.io.load(Path(archive.path) / "context.lumos")
    assert set(loaded.elements) == set(ctx.elements)
    assert loaded.seed == ctx.seed


# --- Metadata (REQ-RES-2) -----------------------------------------------------------------------------------


def test_metadata_fields(tmp_path):
    ctx = bench(seed=1234)
    ctx.wait(3 * u.s)
    meta = read_metadata(new_archive(ctx, root=tmp_path))

    created = datetime.fromisoformat(meta["created_at"])
    assert created.utcoffset() == UTC.utcoffset(None)
    assert abs((datetime.now(UTC) - created).total_seconds()) < 600
    assert meta["os"]["system"] == platform.system()
    assert meta["python_version"].startswith(platform.python_version())
    assert meta["lumos_version"] == lumos.__version__
    for package in ("numpy", "scipy", "astropy", "matplotlib"):
        assert meta["dependencies"][package] == version(package)
    assert "git_commit" in meta
    assert "git_branch" in meta
    assert meta["hardware"]["os"] == platform.system()
    assert meta["hardware"]["cpu_logical_cores"] >= 1
    assert meta["backend"] in lumos.backend.known()
    assert meta["precision"] == lumos.config.precision
    assert meta["seed"] == 1234
    assert meta["mode"] == "simulation"
    assert meta["twin"] is True
    assert Time(meta["clock"]["start"], scale="utc") == START
    assert meta["clock"]["elapsed"] == {"value": 3.0, "unit": "s"}
    assert meta["origin"] == "python"
    assert meta["context"] == "bench"
    assert isinstance(meta["durations"], dict)


def test_metadata_git_commit_matches_the_repository(tmp_path):
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git is not available")
    meta = read_metadata(new_archive(bench(), root=tmp_path))
    assert meta["git_commit"] == head


def test_metadata_records_the_origin(tmp_path):
    ctx = bench()
    with ctx.as_origin("mcp"):
        meta = read_metadata(new_archive(ctx, root=tmp_path))
    assert meta["origin"] == "mcp"


def test_metadata_custom_entries(tmp_path):
    meta = read_metadata(new_archive(bench(), root=tmp_path, metadata={"operator": "Ada"}))
    assert meta["operator"] == "Ada"


# --- Log ----------------------------------------------------------------------------------------------------


def test_log_holds_the_journal_as_json_lines(tmp_path):
    ctx = bench()
    ctx["cam"].expose()
    ctx.wait(2 * u.s)
    archive = new_archive(ctx, root=tmp_path)
    lines = (Path(archive.path) / "lumos.log").read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in lines]
    actions = [e for e in entries if e["kind"] == "action"]
    assert [(e["element"], e["action"]) for e in actions] == [("cam", "expose")]
    assert any(e["kind"] == "wait" for e in entries)


# --- Figures and data ---------------------------------------------------------------------------------------


def test_save_figure_in_the_archive(tmp_path):
    archive = new_archive(bench(), root=tmp_path)
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    archive.save_figure(fig, "line", data={"y": np.array([3.0, 4.0])})
    path = Path(archive.path)
    assert (path / "figures" / "line.png").is_file()
    assert (path / "figures" / "line.plt").is_file()
    assert (path / "data" / "line.npz").is_file()


def test_save_data_products_and_signals(tmp_path):
    ctx = bench()
    result = ctx.run(["cam.out", "laser.out"])
    archive = new_archive(ctx, root=tmp_path)
    archive.save_data("image", result["cam.out"])
    archive.save_data("beam", result["laser.out"])
    archive.save_data("values", {"a": np.arange(3)})
    data = Path(archive.path) / "data"
    product = DataProduct.from_npz(data / "image.npz")
    np.testing.assert_array_equal(product.data.value, result["cam.out"].data.value)
    assert (data / "beam.json").is_file()
    assert (data / "values.npz").is_file()


# --- ctx.run(archive=...) ----------------------------------------------------------------------------------


def test_run_with_archive_saves_the_outputs(tmp_path):
    ctx = bench()
    result = ctx.run(["cam.out", "meter.out"], archive="first-light", archive_root=tmp_path)
    archive = result.archive
    assert isinstance(archive, RunArchive)
    data = Path(archive.path) / "data"
    assert (data / "cam.out.npz").is_file()
    assert (data / "meter.out.npz").is_file()
    meta = read_metadata(archive)
    assert meta["durations"]["run"] > 0
    assert meta["outputs"] == ["cam.out", "meter.out"]


def test_run_without_archive_has_none(tmp_path):
    assert bench().run(["meter.out"]).archive is None


# --- Root of the archives ----------------------------------------------------------------------------------


def test_configured_root(tmp_path, monkeypatch):
    monkeypatch.setattr(lumos.config, "archive_root", str(tmp_path / "custom"))
    archive = new_archive(bench())
    assert Path(archive.path).is_relative_to(tmp_path / "custom")


def test_default_root_next_to_a_notebook(tmp_path, monkeypatch):
    monkeypatch.setattr(lumos.config, "archive_root", None)
    notebook = tmp_path / "notebooks" / "demo.ipynb"
    monkeypatch.setenv("JPY_SESSION_NAME", str(notebook))
    assert default_archive_root() == notebook.parent / "archives"


def test_default_root_next_to_the_script(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    script = project / "run.py"
    script.write_text(
        textwrap.dedent(
            """
            import lumos
            from lumos import Context
            from lumos.elements.sources import GaussianLaser
            from lumos.elements.detectors import PowerMeter

            ctx = Context(seed=1)
            laser = ctx.add(GaussianLaser(name="laser", n_pixels=16))
            laser >> PowerMeter(name="meter")
            ctx.run(["meter.out"], archive="script")
            """
        ),
        encoding="utf-8",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    env = {k: v for k, v in __import__("os").environ.items() if k != "JPY_SESSION_NAME"}
    subprocess.run([sys.executable, str(script)], cwd=elsewhere, check=True, timeout=300, env=env)
    archives = list((project / "archives").glob("*/*script*/metadata.yml"))
    assert len(archives) == 1
    assert not (elsewhere / "archives").exists()
