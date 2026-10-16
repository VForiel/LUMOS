"""Sphinx configuration for the LUMOS documentation (English source, French translation)."""

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
# that myst-nb renders them under "Examples".

_examples = Path(__file__).resolve().parent.parent / "examples"
_rendered = Path(__file__).resolve().parent / "examples" / "notebooks"
_rendered.mkdir(exist_ok=True)
for _notebook in sorted(_examples.glob("[0-9][0-9]_*.ipynb")):
    shutil.copy2(_notebook, _rendered / _notebook.name)

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
    "Thumbs.db",
    ".DS_Store",
]

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath", "fieldlist", "tasklist"]
myst_heading_anchors = 3
nb_execution_mode = "off"  # notebooks are executed by the test suite, not by the doc build (for now)

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
