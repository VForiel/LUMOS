"""Build the index of the community catalogue, or check it (ADR-0026).

Usage::

    python tools/build_catalogue.py            # write community/index.json from the files
    python tools/build_catalogue.py --check    # list what refuses a contribution; exit 1 if anything does

The CI (workflow ``community.yml``) runs ``--check`` on every change of ``community/``.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from lumos.catalogue import INDEX, build_index, check_catalogue

ROOT = Path(__file__).resolve().parents[1] / "community"


def main() -> int:
    """Write or check the index; return the exit status."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--check", action="store_true", help="check the contributions and the index")
    parser.add_argument("--root", type=Path, default=ROOT, help="the catalogue directory")
    arguments = parser.parse_args()
    if arguments.check:
        problems = check_catalogue(arguments.root)
        for problem in problems:
            logging.error(problem)
        logging.info("%d problem(s)", len(problems))
        return 1 if problems else 0
    index = build_index(arguments.root)
    text = json.dumps(index, indent=2, ensure_ascii=False) + "\n"
    (arguments.root / INDEX).write_text(text, encoding="utf-8", newline="\n")
    logging.info("%d entries written to %s", len(index["entries"]), arguments.root / INDEX)
    return 0


if __name__ == "__main__":
    sys.exit(main())
