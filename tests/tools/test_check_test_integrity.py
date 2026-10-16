"""Tests for tools/check_test_integrity.py (step P0.4).

The rules come from docs/specs/cahier_des_charges.md §7.1 and §7.3:

R1. A commit must not touch both ``tests/`` and ``src/``.
R2. A commit touching ``tests/`` must have the Conventional Commits type ``test``.
R3. A commit that changes or removes existing test content (deleted lines, deleted or renamed test file)
    must carry the breaking marker: ``test(scope)!: …``. Pure additions only need ``test(scope): …``.
R4. A ``test(scope)!:`` commit must explain the reason in its body (non-empty body).

Merge commits are not checked.
"""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "check_test_integrity.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("check_test_integrity", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_test_integrity"] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def make_commit(subject, changes, body="", merge=False):
    return tool.Commit(
        sha="0" * 40,
        subject=subject,
        body=body,
        is_merge=merge,
        changes=[tool.FileChange(path=p, status=s, added=a, deleted=d) for p, s, a, d in changes],
    )


# --- Subject parsing ---------------------------------------------------------


@pytest.mark.parametrize(
    ("subject", "expected"),
    [
        ("feat(P4.3): add MFT", ("feat", "P4.3", False)),
        ("test(P0.2): specify packaging", ("test", "P0.2", False)),
        ("test(P0.2)!: fix wrong Airy zero", ("test", "P0.2", True)),
        ("fix: typo", ("fix", None, False)),
        ("refactor!: drop legacy API", ("refactor", None, True)),
        ("Initial commit", (None, None, False)),
    ],
)
def test_parse_subject(subject, expected):
    assert tool.parse_subject(subject) == expected


# --- Rules on a single commit ------------------------------------------------


def test_new_tests_with_test_type_are_accepted():
    commit = make_commit("test(P1.2): specify Param", [("tests/unit/test_param.py", "A", 40, 0)])
    assert tool.check_commit(commit) == []


def test_appending_to_existing_test_file_is_accepted():
    commit = make_commit("test(P1.3): specify Port", [("tests/unit/test_param.py", "M", 12, 0)])
    assert tool.check_commit(commit) == []


def test_implementation_only_commit_is_accepted():
    commit = make_commit("feat(P1.2): implement Param", [("src/lumos/core/param.py", "A", 80, 0)])
    assert tool.check_commit(commit) == []


def test_r1_mixing_tests_and_src_is_rejected():
    commit = make_commit(
        "feat(P1.2): implement Param",
        [("src/lumos/core/param.py", "A", 80, 0), ("tests/unit/test_param.py", "A", 10, 0)],
    )
    rules = {v.rule for v in tool.check_commit(commit)}
    assert "R1" in rules


def test_r2_touching_tests_without_test_type_is_rejected():
    commit = make_commit("chore: tidy", [("tests/unit/test_param.py", "A", 5, 0)])
    rules = {v.rule for v in tool.check_commit(commit)}
    assert "R2" in rules


def test_r3_deleting_test_lines_without_breaking_marker_is_rejected():
    commit = make_commit("test(P1.2): adjust", [("tests/unit/test_param.py", "M", 1, 1)])
    rules = {v.rule for v in tool.check_commit(commit)}
    assert "R3" in rules


@pytest.mark.parametrize("status", ["D", "R"])
def test_r3_deleting_or_renaming_test_file_requires_breaking_marker(status):
    commit = make_commit("test(P1.2): reorganise", [("tests/unit/test_param.py", status, 0, 30)])
    rules = {v.rule for v in tool.check_commit(commit)}
    assert "R3" in rules


def test_r3_binary_change_counts_as_modification():
    # git numstat reports "-" for binary files; the tool maps it to None.
    commit = make_commit("test(P3.8): update fixture", [("tests/data/ref.fits", "M", None, None)])
    rules = {v.rule for v in tool.check_commit(commit)}
    assert "R3" in rules


def test_justified_breaking_test_change_is_accepted():
    commit = make_commit(
        "test(P4.1)!: correct second Airy zero",
        [("tests/physics/test_airy.py", "M", 1, 1)],
        body="The second zero is at 2.233 λ/D (Born & Wolf, §8.5.2), not 2.23.",
    )
    assert tool.check_commit(commit) == []


def test_r4_breaking_test_change_without_body_is_rejected():
    commit = make_commit("test(P4.1)!: correct value", [("tests/physics/test_airy.py", "M", 1, 1)])
    rules = {v.rule for v in tool.check_commit(commit)}
    assert "R4" in rules


def test_r4_whitespace_body_counts_as_empty():
    commit = make_commit(
        "test(P4.1)!: correct value", [("tests/physics/test_airy.py", "M", 1, 1)], body="  \n\t\n"
    )
    rules = {v.rule for v in tool.check_commit(commit)}
    assert "R4" in rules


def test_co_author_trailer_alone_is_not_a_justification():
    commit = make_commit(
        "test(P4.1)!: correct value",
        [("tests/physics/test_airy.py", "M", 1, 1)],
        body="Co-Authored-By: Someone <someone@example.com>",
    )
    rules = {v.rule for v in tool.check_commit(commit)}
    assert "R4" in rules


def test_merge_commits_are_skipped():
    commit = make_commit(
        "Merge branch 'x'",
        [("src/lumos/a.py", "M", 1, 1), ("tests/test_a.py", "M", 1, 1)],
        merge=True,
    )
    assert tool.check_commit(commit) == []


def test_only_top_level_tests_directory_counts():
    commit = make_commit("feat(P2.1): add backend", [("src/lumos/backend/tests_helpers.py", "A", 10, 0)])
    assert tool.check_commit(commit) == []


def test_violation_mentions_commit_and_rule():
    commit = make_commit("chore: tidy", [("tests/test_a.py", "A", 1, 0)])
    violations = tool.check_commit(commit)
    assert violations
    text = str(violations[0])
    assert "R2" in text
    assert commit.sha[:7] in text


# --- Integration with a real git repository ----------------------------------

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _commit_file(repo, path, content, message):
    file = repo / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content, encoding="utf-8")
    _git(repo, "add", path)
    _git(repo, "commit", "-q", "-m", message)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    _commit_file(tmp_path, "README.md", "x\n", "Initial commit")
    return tmp_path


@requires_git
def test_collect_commits_reads_changes(repo):
    _commit_file(repo, "tests/test_a.py", "a = 1\nb = 2\n", "test(P0.4): add test")
    _commit_file(repo, "tests/test_a.py", "a = 1\n", "test(P0.4): shrink test")
    commits = tool.collect_commits("HEAD~2..HEAD", cwd=repo)
    assert [c.subject for c in commits] == ["test(P0.4): add test", "test(P0.4): shrink test"]
    last = commits[1].changes[0]
    assert (last.path, last.status, last.added, last.deleted) == ("tests/test_a.py", "M", 0, 1)


@requires_git
def test_main_returns_nonzero_on_violation(repo):
    _commit_file(repo, "tests/test_a.py", "a = 1\n", "chore: sneak a test in")
    assert tool.main(["HEAD~1..HEAD"], cwd=repo) == 1


@requires_git
def test_main_returns_zero_on_clean_history(repo):
    _commit_file(repo, "tests/test_a.py", "a = 1\n", "test(P0.4): add test")
    _commit_file(repo, "src/lumos/a.py", "a = 1\n", "feat(P0.4): implement")
    assert tool.main(["HEAD~2..HEAD"], cwd=repo) == 0
