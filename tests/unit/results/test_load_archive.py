"""Tests for reading archives back (step P7.5, REQ-RES-6).

Requirement: ``lumos.load_archive(path)`` restores the context, the results and the metadata of an archive.
Security (NF-7): nothing is unpickled or imported while reading; data files naming an unknown signal type are
refused.

Independent references: the metadata is compared with ``yaml.safe_load`` of ``metadata.yml``; the restored
context must reproduce the archived outputs when run again (noise-free detectors are deterministic).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml
from astropy import units as u

import lumos
from lumos import Context
from lumos.elements.detectors import Camera, PowerMeter
from lumos.elements.sources import GaussianLaser
from lumos.fields import DataProduct, Wavefront
from lumos.results import new_archive


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def bench():
    ctx = Context(seed=3, name="bench")
    laser = ctx.add(GaussianLaser(name="laser", n_pixels=32, power=2 * u.mW))
    laser >> Camera(name="cam", noise=False)
    ctx.connect(laser.port("out"), ctx.add(PowerMeter(name="meter")).port("in"))
    return ctx


@pytest.fixture
def archived(tmp_path):
    ctx = bench()
    ctx["cam"].expose()
    result = ctx.run(["cam.out", "meter.out", "laser.out"], archive="demo", archive_root=tmp_path)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    result.archive.save_figure(fig, "line", data={"y": [1.0, 0.0] * u.W})
    return ctx, result


def test_load_archive_metadata(archived):
    _, result = archived
    loaded = lumos.load_archive(result.archive.path)
    expected = yaml.safe_load((Path(result.archive.path) / "metadata.yml").read_text(encoding="utf-8"))
    assert loaded.metadata == expected
    assert Path(loaded.path) == Path(result.archive.path)


def test_load_archive_restores_the_context(archived):
    ctx, result = archived
    loaded = lumos.load_archive(result.archive.path)
    assert isinstance(loaded.context, Context)
    assert set(loaded.context.elements) == set(ctx.elements)
    assert loaded.context.seed == ctx.seed
    assert loaded.context["laser"].power == 2 * u.mW
    rerun = loaded.context.run(["cam.out"])["cam.out"]
    np.testing.assert_array_equal(rerun.data.value, result["cam.out"].data.value)


def test_load_archive_restores_the_results(archived):
    _, result = archived
    loaded = lumos.load_archive(result.archive.path)
    image = loaded.data["cam.out"]
    assert isinstance(image, DataProduct)
    np.testing.assert_array_equal(image.data.value, result["cam.out"].data.value)
    beam = loaded.data["laser.out"]
    assert isinstance(beam, Wavefront)
    np.testing.assert_allclose(np.asarray(beam.field), np.asarray(result["laser.out"].field))
    assert loaded.data["line"]["y"].unit == u.W


def test_load_archive_lists_figures_without_unpickling(archived):
    _, result = archived
    loaded = lumos.load_archive(result.archive.path)
    assert loaded.figures["line"] == Path(result.archive.path) / "figures" / "line.png"
    assert loaded.editable_figures["line"] == Path(result.archive.path) / "figures" / "line.plt"


def test_load_archive_reads_the_journal(archived):
    _, result = archived
    loaded = lumos.load_archive(result.archive.path)
    actions = [e for e in loaded.journal if e["kind"] == "action"]
    assert [(e["element"], e["action"]) for e in actions] == [("cam", "expose")]


def test_unknown_signal_types_are_refused(archived):
    _, result = archived
    bad = Path(result.archive.path) / "data" / "bad.json"
    bad.write_text(json.dumps({"signal": "os.system", "data": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="signal"):
        lumos.load_archive(result.archive.path)


def test_a_directory_without_metadata_is_not_an_archive(tmp_path):
    with pytest.raises(FileNotFoundError, match="metadata"):
        lumos.load_archive(tmp_path)


def test_archive_can_be_given_as_an_archive_object(tmp_path):
    archive = new_archive(bench(), root=tmp_path)
    assert lumos.load_archive(archive).metadata["seed"] == 3
