"""Packaging and deployment of a release (step P15.5).

Requirements: the wheel published on PyPI holds the built web interface, the presets and the community
catalogue, also when it is built from the sdist (``python -m build`` makes the wheel from the sdist, as the
Publish workflow does: v1.0.0 first failed this way); the update script of the public instance deploys a
release tag, then the latest commit again (a tag leaves git on a detached HEAD, where ``git pull`` fails).
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGETS = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["hatch"]["build"][
    "targets"
]


def test_the_interface_goes_through_the_sdist_into_the_wheel():
    assert "src/lumos/web/static/" in TARGETS["wheel"]["artifacts"]
    assert "src/lumos/web/static/" in TARGETS["sdist"]["artifacts"]
    for shipped in ("examples/presets", "community"):
        assert shipped in TARGETS["sdist"]["include"]
        assert shipped in TARGETS["wheel"]["force-include"]


def test_the_update_script_comes_back_to_main_after_a_release():
    script = (ROOT / "deploy" / "public" / "update.sh").read_text(encoding="utf-8")
    latest = script[script.index("else") : script.index("fi")]
    assert latest.index("git checkout") < latest.index("git pull")
    assert "main" in latest
