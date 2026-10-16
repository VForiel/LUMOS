"""Check that a range of commits respects the LUMOS test-integrity rules.

The rules (docs/specs/cahier_des_charges.md §7.1 and §7.3) protect the "tests first" discipline:
tests are written and committed before the implementation, and are only changed for a documented reason.

R1. A commit must not touch both ``tests/`` and ``src/``.
R2. A commit touching ``tests/`` must have the Conventional Commits type ``test``.
R3. A commit that changes or removes existing test content (deleted lines, deleted or renamed test file,
    modified binary fixture) must carry the breaking marker: ``test(scope)!: …``.
    Pure additions only need ``test(scope): …``.
R4. A ``test(scope)!:`` commit must explain the reason in its body (git trailers such as
    ``Co-Authored-By`` do not count).

Merge commits are not checked.

Usage::

    python tools/check_test_integrity.py origin/main..HEAD
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

TESTS_DIR = "tests/"
SRC_DIR = "src/"

_SUBJECT_RE = re.compile(r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]+)\))?(?P<breaking>!)?: \S")
_TRAILER_RE = re.compile(
    r"^(co-authored-by|signed-off-by|reviewed-by|acked-by|tested-by|helped-by|reported-by|change-id):",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FileChange:
    """One file changed by a commit.

    ``added`` and ``deleted`` are line counts, or ``None`` for binary files.
    ``old_path`` is set for renames and copies.
    """

    path: str
    status: str
    added: int | None
    deleted: int | None
    old_path: str | None = None

    def paths(self) -> tuple[str, ...]:
        """Return the paths touched by this change (old and new path for renames)."""
        return (self.path,) if self.old_path is None else (self.old_path, self.path)


@dataclass(frozen=True)
class Commit:
    """The information about a commit needed to apply the rules."""

    sha: str
    subject: str
    body: str
    is_merge: bool
    changes: list[FileChange] = field(default_factory=list)


@dataclass(frozen=True)
class Violation:
    """A rule broken by a commit."""

    sha: str
    rule: str
    message: str
    subject: str

    def __str__(self) -> str:
        """Format the violation for the CI report."""
        return f"{self.sha[:7]} {self.rule}: {self.message} — {self.subject!r}"


def parse_subject(subject: str) -> tuple[str | None, str | None, bool]:
    """Return ``(type, scope, breaking)`` of a Conventional Commits subject, or ``(None, None, False)``."""
    match = _SUBJECT_RE.match(subject)
    if match is None:
        return None, None, False
    return match["type"], match["scope"], match["breaking"] is not None


def _touches(change: FileChange, directory: str) -> bool:
    return any(path.startswith(directory) for path in change.paths())


def _modifies_existing_content(change: FileChange) -> bool:
    if change.status[0] in {"D", "R"}:
        return True
    if change.status[0] == "M":
        # Binary files report no line counts: any modification counts.
        return change.deleted is None or change.deleted > 0
    return False


def _has_justification(body: str) -> bool:
    lines = [line.strip() for line in body.splitlines()]
    return any(line and not _TRAILER_RE.match(line) for line in lines)


def check_commit(commit: Commit) -> list[Violation]:
    """Apply rules R1 to R4 to one commit."""
    if commit.is_merge:
        return []

    def violation(rule: str, message: str) -> Violation:
        return Violation(commit.sha, rule, message, commit.subject)

    test_changes = [c for c in commit.changes if _touches(c, TESTS_DIR)]
    touches_src = any(_touches(c, SRC_DIR) for c in commit.changes)
    commit_type, _, breaking = parse_subject(commit.subject)
    violations: list[Violation] = []

    if test_changes and touches_src:
        violations.append(violation("R1", "commit mixes tests/ and src/ changes"))
    if test_changes and commit_type != "test":
        violations.append(violation("R2", "commit touching tests/ must have type 'test'"))
    if any(_modifies_existing_content(c) for c in test_changes) and not breaking:
        violations.append(
            violation("R3", "existing test content changed without the breaking marker 'test(scope)!:'")
        )
    if test_changes and commit_type == "test" and breaking and not _has_justification(commit.body):
        violations.append(violation("R4", "breaking test change without a justification in the body"))
    return violations


def _git(args: list[str], cwd: Path | str | None) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout


def _parse_changes(sha: str, cwd: Path | str | None) -> list[FileChange]:
    common = ["diff-tree", "--root", "--no-commit-id", "-r", "-M", "-z"]
    status_fields = _git([*common, "--name-status", sha], cwd).split("\0")
    numstat_fields = _git([*common, "--numstat", sha], cwd).split("\0")

    statuses: list[tuple[str, str, str | None]] = []
    i = 0
    while i < len(status_fields) and status_fields[i]:
        status = status_fields[i]
        if status[0] in {"R", "C"}:
            statuses.append((status, status_fields[i + 2], status_fields[i + 1]))
            i += 3
        else:
            statuses.append((status, status_fields[i + 1], None))
            i += 2

    counts: list[tuple[int | None, int | None]] = []
    i = 0
    while i < len(numstat_fields) and numstat_fields[i]:
        added, deleted, path = numstat_fields[i].split("\t", 2)
        counts.append((None if added == "-" else int(added), None if deleted == "-" else int(deleted)))
        # A rename has an empty path here, followed by the old and new paths.
        i += 3 if path == "" else 1

    return [
        FileChange(path=path, status=status, added=added, deleted=deleted, old_path=old_path)
        for (status, path, old_path), (added, deleted) in zip(statuses, counts, strict=True)
    ]


def collect_commits(rev_range: str, cwd: Path | str | None = None) -> list[Commit]:
    """Read the commits of ``rev_range`` (oldest first) from git."""
    shas = _git(["rev-list", "--reverse", rev_range], cwd).split()
    commits = []
    for sha in shas:
        parents, subject, body = _git(["show", "-s", "--format=%P%x00%s%x00%b", sha], cwd).split("\0", 2)
        commits.append(
            Commit(
                sha=sha,
                subject=subject,
                body=body.strip(),
                is_merge=len(parents.split()) > 1,
                changes=_parse_changes(sha, cwd),
            )
        )
    return commits


def main(argv: list[str] | None = None, cwd: Path | str | None = None) -> int:
    """Check a revision range and return the process exit code (1 if any rule is broken)."""
    parser = argparse.ArgumentParser(description="Check the LUMOS test-integrity rules on a commit range.")
    parser.add_argument("range", help="git revision range, e.g. origin/main..HEAD")
    args = parser.parse_args(argv)

    commits = collect_commits(args.range, cwd=cwd)
    violations = [v for commit in commits for v in check_commit(commit)]
    for v in violations:
        print(v)
    print(f"Checked {len(commits)} commit(s): {len(violations)} violation(s).")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
