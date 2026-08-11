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

HOOK = Path(__file__).resolve().parents[1] / "scripts" / "hooks" / "pre-commit"


def _repo(tmp_path: Path) -> Path:
    """A throwaway git repo with the real hook installed."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    hooks = tmp_path / ".githooks"
    hooks.mkdir()
    (hooks / "pre-commit").write_text(HOOK.read_text())
    (hooks / "pre-commit").chmod(0o755)
    subprocess.run(
        ["git", "config", "core.hooksPath", ".githooks"], cwd=tmp_path, check=True
    )
    return tmp_path


def _commit(repo: Path, *files: str, env: dict | None = None):
    subprocess.run(["git", "add", "-f", *files], cwd=repo, check=True)
    return subprocess.run(
        ["git", "commit", "-m", "test"],
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
        repo, "outputs/FINDINGS.md", env={**os.environ, "ALLOW_LOCAL_FILES": "1"}
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
        repo, "deploy/profile.env", env={**os.environ, "ALLOW_LOCAL_FILES": "1"}
    )
    assert first.returncode == 0

    (repo / "deploy" / "profile.env").write_text("A=2\n")
    second = _commit(repo, "deploy/profile.env")
    assert second.returncode == 0, second.stderr
