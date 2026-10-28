"""Execution of the example notebooks and presets (step P7.6, REQ-DOC-5, acceptance criteria of v0.2).

Requirements:

- REQ-DOC-5: numbered notebooks in ``examples/``, executed in CI; each web preset matches a notebook and a
  shared ``.lumos`` file in ``examples/presets/``;
- acceptance of phase 7: a notebook produces a complete archive, its ``.plt`` files reload with
  ``pltedit.load`` and its metadata holds every field of REQ-RES-2.

The notebooks are executed with ``nbclient`` in a temporary copy, so that their archives are created there.
"""

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nbformat
import pytest
import yaml
from LRFutils import pltedit
from nbclient import NotebookClient

import lumos.io

# On Windows, pyzmq warns that asyncio's default Proactor event loop lacks add_reader and starts a
# selector thread (pyzmq documentation, "asyncio on Windows"): a notice of the kernel client, not of LUMOS.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Proactor event loop does not implement add_reader:RuntimeWarning"
)

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
NOTEBOOKS = sorted(EXAMPLES.glob("[0-9][0-9]_*.ipynb"))
PRESETS = sorted((EXAMPLES / "presets").glob("*.lumos"))

#: Fields of metadata.yml required by REQ-RES-2.
REQUIRED_METADATA = (
    "created_at",
    "os",
    "python_version",
    "lumos_version",
    "dependencies",
    "git_commit",
    "git_branch",
    "hardware",
    "backend",
    "precision",
    "seed",
    "durations",
    "mode",
    "twin",
    "clock",
    "origin",
)

EXPECTED_NOTEBOOKS = (
    "01_first_simulation",
    "02_propagation_methods",
    "03_psf_and_pupils",
    "04_interferometry",
)


def test_the_first_notebooks_exist():
    names = [path.stem for path in NOTEBOOKS]
    for name in EXPECTED_NOTEBOOKS:
        assert name in names


def test_notebooks_are_stored_without_outputs():
    for path in NOTEBOOKS:
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                assert cell.outputs == [], f"{path.name} is stored with outputs"
                assert cell.execution_count is None


CATALOG_CACHE = EXAMPLES / "catalog_cache"


def execute(path, workdir, monkeypatch=None):
    copy = workdir / path.name
    shutil.copy(path, copy)
    for preset in PRESETS:
        (workdir / "presets").mkdir(exist_ok=True)
        shutil.copy(preset, workdir / "presets" / preset.name)
    if monkeypatch is not None and CATALOG_CACHE.is_dir():
        # catalogue answers recorded from the services: the notebooks run offline (no network in CI)
        cache = workdir / "catalog_cache"
        shutil.copytree(CATALOG_CACHE, cache)
        monkeypatch.setenv("LUMOS_CATALOGS_CACHE", str(cache))
        monkeypatch.setenv("LUMOS_CATALOGS_OFFLINE", "1")
    notebook = nbformat.read(copy, as_version=4)
    client = NotebookClient(
        notebook, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(workdir)}}
    )
    client.execute()
    return notebook


@pytest.mark.slow
@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_notebook_runs_and_archives_its_results(path, tmp_path, monkeypatch):
    execute(path, tmp_path, monkeypatch)
    archives = list(tmp_path.glob("archives/*/*/metadata.yml"))
    assert archives, f"{path.name} did not create an archive next to the notebook"
    for metadata_file in archives:
        archive = metadata_file.parent
        metadata = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
        missing = [key for key in REQUIRED_METADATA if key not in metadata]
        assert not missing, f"{archive.name}: missing metadata {missing}"
        assert (archive / "context.lumos").is_file()
        assert (archive / "lumos.log").is_file()
        plts = list((archive / "figures").glob("*.plt"))
        assert plts, f"{archive.name} has no .plt figure"
        for plt_file in plts:
            figure = pltedit.load(plt_file)
            assert figure.axes
            plt.close(figure)
        assert list((archive / "figures").glob("*.png"))
        assert list((archive / "data").iterdir())


def test_every_preset_has_outputs_and_a_notebook():
    assert PRESETS
    notebook_text = "\n".join(path.read_text(encoding="utf-8") for path in NOTEBOOKS)
    for preset in PRESETS:
        ctx = lumos.io.load(preset)
        assert ctx.meta.get("outputs"), f"{preset.name} does not declare its outputs"
        assert ctx.meta.get("description"), f"{preset.name} has no description"
        assert preset.name in notebook_text, f"no notebook uses {preset.name}"


@pytest.mark.slow
@pytest.mark.parametrize("preset", PRESETS, ids=[p.stem for p in PRESETS])
def test_preset_runs(preset):
    ctx = lumos.io.load(preset)
    result = ctx.run(ctx.meta["outputs"])
    for output in ctx.meta["outputs"]:
        assert result[output] is not None


def test_the_extended_scenes_notebook_exists():
    """Step P8.7: notebook of extended scenes (debris disk behind a coronagraph, nebula)."""
    names = [path.stem for path in NOTEBOOKS]
    assert "05_extended_scenes" in names
    assert (EXAMPLES / "presets" / "debris_disk.lumos").is_file()


def test_the_catalogue_notebook_exists():
    """Step P9.9: notebook of targets from catalogues, with its recorded catalogue answers."""
    assert "06_targets_from_catalogs" in [path.stem for path in NOTEBOOKS]
    assert any(CATALOG_CACHE.glob("*.json"))


def test_the_symbolic_notebook_exists():
    """Step P10.7: notebook of symbolic optics, with its preset of a Gaussian beam kept symbolic."""
    assert "07_symbolic_optics" in [path.stem for path in NOTEBOOKS]
    assert (EXAMPLES / "presets" / "symbolic_focus.lumos").is_file()


def test_the_processing_notebooks_exist():
    """Step P11.6 (acceptance of v0.4, UC7): a data pipeline and a detection-capability study."""
    names = [path.stem for path in NOTEBOOKS]
    assert "08_data_pipeline" in names
    assert "09_detection_capability" in names


def test_the_bench_operation_notebook_exists():
    """Step P12.10 (acceptance of v0.5, UC3 and UC4): operation of a bench against the LumosLabs emulators."""
    assert "10_bench_operation" in [path.stem for path in NOTEBOOKS]
