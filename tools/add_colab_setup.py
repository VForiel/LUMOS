"""Insert or refresh the Colab setup cell of the example notebooks.

On Google Colab, LUMOS and the files next to the notebooks (``presets/``, ``catalog_cache/``) are not
available. The cell, tagged ``colab-setup``, installs LUMOS from GitHub and copies these files, only when it
runs on Colab; elsewhere (tests, documentation, Binder, local Jupyter) it does nothing.

Usage::

    python tools/add_colab_setup.py            # every examples/NN_*.ipynb
    python tools/add_colab_setup.py examples/07_symbolic_optics.ipynb
"""

import json
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
TAG = "colab-setup"
SOURCE = """\
# On Google Colab only: install LUMOS and fetch the presets used by the examples.
import sys

if "google.colab" in sys.modules:
    %pip install -q "lumos-maxima[catalogs] @ git+https://github.com/VForiel/LUMOS"
    !git clone -q --depth 1 https://github.com/VForiel/LUMOS /content/LUMOS
    !cp -r /content/LUMOS/examples/presets /content/LUMOS/examples/catalog_cache ."""


def setup_cell() -> dict:
    lines = SOURCE.splitlines(keepends=True)
    return {
        "cell_type": "code",
        "id": "colab-setup",
        "execution_count": None,
        "metadata": {"tags": [TAG]},
        "outputs": [],
        "source": lines,
    }


def update(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"] if TAG not in cell.get("metadata", {}).get("tags", [])]
    position = 1 if cells and cells[0]["cell_type"] == "markdown" else 0
    cells.insert(position, setup_cell())
    notebook["cells"] = cells
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main(arguments: list[str]) -> None:
    paths = [Path(a) for a in arguments] or sorted(EXAMPLES.glob("[0-9][0-9]_*.ipynb"))
    for path in paths:
        update(path)


if __name__ == "__main__":
    main(sys.argv[1:])
