"""Sphinx configuration for the LUMOS documentation (English source, French translation)."""

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import lumos

# -- Project -----------------------------------------------------------------

project = "LUMOS"
author = "Vincent Foriel"
copyright = f"{datetime.now(UTC).year}, {author}"
release = lumos.__version__
version = release

# -- Example notebooks -------------------------------------------------------
# The notebooks live in examples/ (next to their presets); they are copied into the documentation tree so
# that myst-nb executes and renders them under "Examples", with their figures. A first cell links to the
# notebook on Colab and Binder. Catalogue queries are answered from the recorded cache (offline build).

_examples = Path(__file__).resolve().parent.parent / "examples"
_rendered = Path(__file__).resolve().parent / "examples" / "notebooks"
_rendered.mkdir(exist_ok=True)
_github = "VForiel/LUMOS"
_branch = os.environ.get("READTHEDOCS_GIT_IDENTIFIER", "main")


def _launch_cell(name: str) -> dict:
    colab = f"https://colab.research.google.com/github/{_github}/blob/{_branch}/examples/{name}"
    binder = f"https://mybinder.org/v2/gh/{_github}/{_branch}?labpath=examples%2F{name}"
    source = (
        f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab}) "
        f"[![Launch on Binder](https://mybinder.org/badge_logo.svg)]({binder})"
    )
    return {
        "cell_type": "markdown",
        "id": "launch-buttons",
        "metadata": {"tags": ["launch-buttons"]},
        "source": source,
    }


for _notebook in sorted(_examples.glob("[0-9][0-9]_*.ipynb")):
    _content = json.loads(_notebook.read_text(encoding="utf-8"))
    _cells = _content["cells"]
    # after the title cell, so that the page keeps the notebook title
    _cells.insert(1 if _cells and _cells[0]["cell_type"] == "markdown" else 0, _launch_cell(_notebook.name))
    (_rendered / _notebook.name).write_text(json.dumps(_content, indent=1), encoding="utf-8")
shutil.copytree(_examples / "presets", _rendered / "presets", dirs_exist_ok=True)
if (_examples / "catalog_cache").is_dir():
    shutil.copytree(_examples / "catalog_cache", _rendered / "catalog_cache", dirs_exist_ok=True)
    os.environ["LUMOS_CATALOGS_CACHE"] = str(_rendered / "catalog_cache")
    os.environ["LUMOS_CATALOGS_OFFLINE"] = "1"

# -- General -----------------------------------------------------------------

extensions = [
    "myst_nb",  # MyST Markdown and Jupyter notebooks
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
exclude_patterns = [
    "_build",
    "jupyter_execute",
    "locales",
    "specs",  # internal requirements, written in French, not part of the user documentation
    "adr/0000-template.md",
    "examples/notebooks/archives",  # run archives written by the executed notebooks
    "Thumbs.db",
    ".DS_Store",
]

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath", "fieldlist", "tasklist"]
myst_heading_anchors = 3
nb_execution_mode = "cache"  # executed once per change of a notebook (jupyter-cache in _build/)
nb_execution_timeout = 900
nb_execution_raise_on_error = True  # a failing notebook fails the build
nb_execution_excludepatterns = ["examples/notebooks/archives/**"]

autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True  # "Attributes" sections as fields, so properties are not documented twice

# -- Internationalisation ----------------------------------------------------
# English is the source language; translations live in docs/locales/<lang>/LC_MESSAGES/*.po.
# Build French locally with: sphinx-build -b html -D language=fr docs docs/_build/html/fr

language = "en"
locale_dirs = ["locales/"]
gettext_compact = False
gettext_uuid = False

# -- HTML --------------------------------------------------------------------

html_theme = "breeze"
html_title = "LUMOS"
# Publish the JSON Schema of .lumos documents at the root of the site (its $id).
html_extra_path = ["../src/lumos/io/schemas"]

_rtd_version = os.environ.get("READTHEDOCS_VERSION", "latest")
_docs_root = os.environ.get("LUMOS_DOCS_ROOT", "https://lumos-maxima.readthedocs.io")

html_context = {
    "github_user": "VForiel",
    "github_repo": "LUMOS",
    "github_version": "main",
    "doc_path": "docs",
    # (label, URL pattern with %s for the page name, language code) used by Breeze's language switcher.
    "languages": [
        ("English", f"{_docs_root}/en/{_rtd_version}/%s.html", "en"),
        ("Français", f"{_docs_root}/fr/{_rtd_version}/%s.html", "fr"),
    ],
}

_announcements = {
    "en": "LUMOS is in early development: the API is not usable yet.",
    "fr": "LUMOS est en tout début de développement : l'API n'est pas encore utilisable.",
}

html_theme_options = {
    # Theme options are not translated by gettext; Read the Docs exposes the project language.
    "announcement": _announcements.get(os.environ.get("READTHEDOCS_LANGUAGE", "en"), _announcements["en"]),
    "emojis_header_nav": True,
}
