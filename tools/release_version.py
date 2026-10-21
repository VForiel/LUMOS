"""Print the version of the checked-out commit, as the package computes it (hatch-vcs, guess-next-dev).

Usage::

    python3 tools/release_version.py        # e.g. 1.0.0 on the tag v1.0.0, 1.0.1.dev3 three commits later

Used to build the Docker image, which has no git history (``deploy/public/update.sh``). Standard library
only: it runs on the host, outside the virtual environment.
"""

import re
import subprocess
import sys

_DESCRIBE = re.compile(r"^v?(?P<tag>\d+(?:\.\d+)*)-(?P<distance>\d+)-g[0-9a-f]+(?:-dirty)?$")


def from_describe(described: str) -> str:
    """Return the version for the output of ``git describe --tags --long``.

    A tagged commit has the version of its tag; ``N`` commits after the tag ``X.Y.Z``, ``X.Y.(Z+1).devN``
    (the last component is raised). Without a tag, ``0.0.0``.
    """
    match = _DESCRIBE.match(described.strip())
    if match is None:
        return "0.0.0"
    parts = [int(part) for part in match["tag"].split(".")]
    distance = int(match["distance"])
    if distance == 0:
        return ".".join(map(str, parts))
    parts[-1] += 1
    return f"{'.'.join(map(str, parts))}.dev{distance}"


def main() -> int:
    """Print the version of the working tree."""
    try:
        described = subprocess.run(
            ["git", "describe", "--tags", "--long", "--match", "v*"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        described = ""
    sys.stdout.write(from_describe(described) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
