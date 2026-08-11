"""The pre-commit guard must block local-only files from being staged.

`git add -A` swept unrelated files into PRs under review twice in one session:
seven research scripts, then `backups/`, deploy `.env` templates and a lock
artifact. The second sweep put trade-level exit prices and P&L into a public
repository's history, where the commit stays retrievable by SHA even after the
branch is deleted.

A rule that must be remembered at commit time failed both times it mattered, so
the guard makes the unsafe action mechanically difficult instead. These tests
drive the real hook against a real throwaway repository — asserting the shell
by reading it would prove nothing about whether git actually refuses.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

HOOKS_SRC = Path(__file__).resolve().parents[1] / "scripts" / "hooks"
HOOK = HOOKS_SRC / "pre-commit"


def _repo(tmp_path: Path) -> Path:
    """A throwaway git repo with the real hook installed."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    hooks = tmp_path / ".githooks"
    hooks.mkdir()
    for name in ("pre-commit", "commit-msg", "lib-blocked-paths.sh"):
        (hooks / name).write_text((HOOKS_SRC / name).read_text())
        (hooks / name).chmod(0o755)
    subprocess.run(
        ["git", "config", "core.hooksPath", ".githooks"], cwd=tmp_path, check=True
    )
    return tmp_path


def _commit(repo: Path, *files: str, env: dict | None = None, message: str = "test"):
    subprocess.run(["git", "add", "-f", *files], cwd=repo, check=True)
    return subprocess.run(
        ["git", "commit", "-m", message],
        cwd=repo, capture_output=True, text=True, env=env,
    )


def test_the_hook_is_executable() -> None:
    """A non-executable hook is silently ignored by git — worse than none."""
    assert HOOK.exists()
    assert HOOK.stat().st_mode & 0o111, "hook must be executable or git skips it"


@pytest.mark.parametrize(
    "path",
    [
        "deploy/vps/profiles/ibkr-paper.env",
        "backups/phantom_backfill/snapshot.json",
        "skills-lock.json",
        "db.dump",
        "secrets.pem",
        "id_rsa",
        "minute_resolver 2.py",
    ],
)
def test_local_only_files_are_refused(tmp_path, path) -> None:
    repo = _repo(tmp_path)
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x")

    result = _commit(repo, path)

    assert result.returncode != 0, f"{path} was committed; the guard did not fire"
    assert "refusing to commit local-only files" in result.stderr


def test_ordinary_source_still_commits(tmp_path) -> None:
    """The guard must not obstruct normal work, or it will be bypassed."""
    repo = _repo(tmp_path)
    (repo / "src").mkdir()
    (repo / "src" / "thing.py").write_text("x = 1\n")

    result = _commit(repo, "src/thing.py")

    assert result.returncode == 0, result.stderr


def test_a_mixed_stage_is_refused_wholesale(tmp_path) -> None:
    """The realistic case: one good file and one stray from `git add -A`."""
    repo = _repo(tmp_path)
    (repo / "src").mkdir()
    (repo / "src" / "thing.py").write_text("x = 1\n")
    (repo / "backups").mkdir()
    (repo / "backups" / "snap.json").write_text("{}")

    result = _commit(repo, "src/thing.py", "backups/snap.json")

    assert result.returncode != 0
    assert "backups/snap.json" in result.stderr
    assert "src/thing.py" not in result.stderr, "only the offender should be named"


def test_the_override_is_explicit_and_works(tmp_path) -> None:
    """A documented, logged bypass — not a silent --no-verify."""
    import os

    repo = _repo(tmp_path)
    (repo / "outputs").mkdir()
    (repo / "outputs" / "FINDINGS.md").write_text("# result\n")

    blocked = _commit(repo, "outputs/FINDINGS.md")
    assert blocked.returncode != 0, "outputs/ is force-added deliberately, so it prompts"

    allowed = _commit(
        repo, "outputs/FINDINGS.md",
        env={**os.environ, "ALLOW_LOCAL_FILES": "1"},
        message="add findings (ALLOW_LOCAL_FILES: research doc, force-added per convention)",
    )
    assert allowed.returncode == 0, allowed.stderr
    assert "permitting" in allowed.stderr


def test_modifying_an_already_tracked_file_is_not_blocked(tmp_path) -> None:
    """The guard targets newly ADDED paths.

    Once a file is legitimately in the repo, editing it must not require the
    override every time — otherwise the override becomes reflexive and stops
    meaning anything.
    """
    import os

    repo = _repo(tmp_path)
    (repo / "deploy").mkdir()
    (repo / "deploy" / "profile.env").write_text("A=1\n")
    first = _commit(
        repo, "deploy/profile.env",
        env={**os.environ, "ALLOW_LOCAL_FILES": "1"},
        message="seed profile (ALLOW_LOCAL_FILES: intentional fixture for this test)",
    )
    assert first.returncode == 0

    (repo / "deploy" / "profile.env").write_text("A=2\n")
    second = _commit(repo, "deploy/profile.env")
    assert second.returncode == 0, second.stderr


def test_a_rename_into_a_protected_path_is_refused(tmp_path) -> None:
    """The hole: `--diff-filter=A` alone misses renames.

    A tracked, unremarkable file moved to `deploy/profile.env` is recorded by
    git as R, not A — so the guard never saw it and the commit went through
    with no override at all.
    """
    import os

    repo = _repo(tmp_path)
    (repo / "notes.txt").write_text("hello\n")
    assert _commit(repo, "notes.txt").returncode == 0

    (repo / "deploy").mkdir()
    subprocess.run(
        ["git", "mv", "notes.txt", "deploy/profile.env"], cwd=repo, check=True
    )
    result = subprocess.run(
        ["git", "commit", "-m", "move it"],
        cwd=repo, capture_output=True, text=True, env={**os.environ},
    )

    assert result.returncode != 0, "a rename slipped a file into a protected path"
    assert "deploy/profile.env" in result.stderr


def test_the_override_without_a_reason_is_refused(tmp_path) -> None:
    """`ALLOW_LOCAL_FILES=1` alone leaves nothing in git history.

    A reviewer then sees a commit containing a protected file and no
    explanation. The environment variable permits; the message must justify.
    """
    import os

    repo = _repo(tmp_path)
    (repo / "backups").mkdir()
    (repo / "backups" / "snap.json").write_text("{}")

    result = _commit(
        repo, "backups/snap.json",
        env={**os.environ, "ALLOW_LOCAL_FILES": "1"},
        message="add snapshot",           # no marker
    )

    assert result.returncode != 0, "override was accepted with no recorded reason"
    assert "gives no reason" in result.stderr


def test_a_token_reason_is_not_enough(tmp_path) -> None:
    """'ALLOW_LOCAL_FILES: x' is compliance theatre, not a justification."""
    import os

    repo = _repo(tmp_path)
    (repo / "backups").mkdir()
    (repo / "backups" / "snap.json").write_text("{}")

    result = _commit(
        repo, "backups/snap.json",
        env={**os.environ, "ALLOW_LOCAL_FILES": "1"},
        message="add snapshot (ALLOW_LOCAL_FILES: x)",
    )

    assert result.returncode != 0
    assert "too short" in result.stderr


def test_the_reason_survives_into_git_history(tmp_path) -> None:
    """The point of the marker: a reviewer can read why, months later."""
    import os

    repo = _repo(tmp_path)
    (repo / "outputs").mkdir()
    (repo / "outputs" / "FINDINGS.md").write_text("# r\n")

    reason = "research doc under the gitignored outputs tree, per convention"
    ok = _commit(
        repo, "outputs/FINDINGS.md",
        env={**os.environ, "ALLOW_LOCAL_FILES": "1"},
        message=f"add findings (ALLOW_LOCAL_FILES: {reason})",
    )
    assert ok.returncode == 0, ok.stderr

    logged = subprocess.run(
        ["git", "log", "-1", "--pretty=%B"], cwd=repo, capture_output=True, text=True
    ).stdout
    assert reason in logged, "the justification must be readable from history"


def test_both_hooks_share_one_detection_source() -> None:
    """If the two disagree, the override requirement silently stops applying."""
    for name in ("pre-commit", "commit-msg"):
        text = (HOOKS_SRC / name).read_text()
        assert "lib-blocked-paths.sh" in text, f"{name} must use the shared library"
    lib = (HOOKS_SRC / "lib-blocked-paths.sh").read_text()
    assert "BLOCKED_PREFIXES" in lib and "BLOCKED_PATTERNS" in lib
