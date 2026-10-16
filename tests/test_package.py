"""Packaging tests for step P0.2.

Expected values come from the requirements (docs/specs/cahier_des_charges.md, decisions D1, NF-1, NF-8)
and from ADR-0001, not from the implementation.
"""

from importlib import metadata, resources

from packaging.specifiers import SpecifierSet
from packaging.version import Version

DISTRIBUTION = "lumos-maxima"  # D1: PyPI distribution name


def test_package_is_importable_as_lumos():
    import lumos

    assert lumos.__name__ == "lumos"


def test_version_matches_installed_distribution():
    import lumos

    assert lumos.__version__ == metadata.version(DISTRIBUTION)


def test_version_is_pep440_compliant():
    import lumos

    # Raises InvalidVersion if the string is not PEP 440 compliant.
    Version(lumos.__version__)


def test_requires_python_is_312_or_newer():
    # ADR-0001: SPEC 0 policy gives Python >= 3.12 on 2026-10-16.
    specifier = SpecifierSet(metadata.metadata(DISTRIBUTION)["Requires-Python"])
    assert "3.12" in specifier
    assert "3.11" not in specifier


def test_license_is_mit():
    # NF-8: MIT license.
    meta = metadata.metadata(DISTRIBUTION)
    assert meta["License-Expression"] == "MIT"


def test_package_ships_typing_marker():
    # ADR-0001: typed package (PEP 561).
    assert resources.files("lumos").joinpath("py.typed").is_file()
