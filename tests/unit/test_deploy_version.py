"""The image of the public instance knows its version (step P15.5).

Requirements: the Docker image is built without the git history, so the version is passed in as a build
argument (``LUMOS_VERSION``) computed from ``git describe --tags --long`` on the host, the same way as the
version of the package (hatch-vcs, ``guess-next-dev`` scheme without local part, ``pyproject.toml``): a tagged
commit has the version of its tag; N commits after tag X.Y.Z, version X.Y.(Z+1).devN. The VPS showed
"v0.0.0" because no version was given.

Reference: setuptools-scm documentation, "Version number construction" (guess-next-dev): ``v1.0.0`` exactly
gives ``1.0.0``; 3 commits after ``v1.0.0`` give ``1.0.1.dev3``.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("release_version", ROOT / "tools" / "release_version.py")
assert spec is not None
assert spec.loader is not None
release_version = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_version)


@pytest.mark.parametrize(
    ("described", "expected"),
    [
        ("v1.0.0-0-g1234abc", "1.0.0"),
        ("v1.0.0-3-g1234abc", "1.0.1.dev3"),
        ("v0.0.1-505-gff0cd57", "0.0.2.dev505"),
        ("v2.1-7-gabcdef0", "2.2.dev7"),
        ("v1.0.0-3-g1234abc-dirty", "1.0.1.dev3"),
        ("1.4.2-0-gdeadbee", "1.4.2"),
    ],
)
def test_the_version_follows_the_package_scheme(described, expected):
    assert release_version.from_describe(described) == expected


def test_a_repository_without_tag_gives_a_development_version():
    assert release_version.from_describe("") == "0.0.0"


def test_the_public_deployment_passes_the_version_to_the_image():
    compose = (ROOT / "deploy" / "public" / "compose.yaml").read_text(encoding="utf-8")
    assert "LUMOS_VERSION: ${LUMOS_VERSION" in compose
    update = (ROOT / "deploy" / "public" / "update.sh").read_text(encoding="utf-8")
    assert "tools/release_version.py" in update
    assert "docker compose -f deploy/public/compose.yaml up -d --build" in update
