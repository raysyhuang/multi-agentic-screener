"""Per-trade P&L may only be written where git provably will not pick it up.

The first guard rejected `scripts/` and the repository root by name — a denylist
of two locations, which says nothing about whether a destination is actually
ignored. A changed output directory, a relaxed ignore rule, or a symlink walks
straight past it. This repo has already put trade-level P&L into public history
once; the guard has to ask git rather than pattern-match a path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


@pytest.fixture(scope="module")
def guard():
    """Import the guard without executing the script's module-level analysis."""

    script = REPO / "scripts" / "sniper_forward_returns.py"
    src = script.read_text()
    # Take everything up to the argparse block: the guard is defined above it,
    # and the rest fetches data and runs the study. `__file__` must be supplied
    # because the script derives REPO from it.
    head = src[:src.index("_ap = argparse.ArgumentParser")]
    ns: dict = {"__file__": str(script)}
    exec(compile(head, str(script), "exec"), ns)  # noqa: S102
    return ns["assert_safe_pnl_path"]


def test_a_tracked_destination_is_refused(guard):
    """The failure that already happened here once."""
    tracked = REPO / "scripts" / "sniper_forward_returns.py"
    assert tracked.exists()

    with pytest.raises(SystemExit, match="TRACKED"):
        guard(tracked)


def test_an_unignored_destination_is_refused(guard):
    """Not-tracked is not the same as safe: an untracked file in a tracked
    directory is exactly what `git add -A` sweeps up."""
    candidate = REPO / "scripts" / "_pnl_guard_probe.csv"
    assert not candidate.exists(), "probe path should not exist"

    with pytest.raises(SystemExit, match="not git-ignored"):
        guard(candidate)


def test_an_ignored_outputs_destination_is_allowed(guard):
    """outputs/ is gitignored, so a P&L dump there cannot be committed
    accidentally — the staging guard blocks it deliberately too."""
    allowed = REPO / "outputs" / "research" / "sniper_live_picks_forward_returns.csv"

    resolved = guard(allowed)

    assert resolved == allowed.resolve()


def test_a_destination_outside_the_repo_is_refused(guard, tmp_path):
    """Outside a work tree, `git check-ignore` answers about nothing, so a pass
    there would be meaningless rather than safe."""
    with pytest.raises(SystemExit, match="outside the repo"):
        guard(tmp_path / "leak.csv")


# ── git-failure paths: an unanswerable question must refuse ──────────────────

class _FakeCompleted:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode
        self.stdout = self.stderr = b""


def _patch_git(monkeypatch, codes: dict[str, int]):
    """Force specific exit codes per git subcommand."""
    import subprocess

    real = subprocess.run

    def fake(cmd, *a, **kw):
        if isinstance(cmd, (list, tuple)) and "git" in str(cmd[0]):
            for sub, code in codes.items():
                if sub in cmd:
                    return _FakeCompleted(code)
        return real(cmd, *a, **kw)

    monkeypatch.setattr(subprocess, "run", fake)


def test_a_broken_index_is_refused_not_treated_as_untracked(guard, monkeypatch):
    """The reproduction from review.

    `ls-files --error-unmatch` exits 1 when a path is legitimately not tracked
    and 128 when the index is corrupt or unreadable. The first guard tested only
    `== 0`, so 128 fell through as "not tracked"; with check-ignore still
    answering 0, a P&L destination was ALLOWED on the strength of a git failure.
    A question git could not answer must refuse.
    """
    _patch_git(monkeypatch, {"ls-files": 128, "check-ignore": 0})

    with pytest.raises(SystemExit, match=r"ls-files failed \(exit 128\)"):
        guard(REPO / "outputs" / "research" / "sniper_live_picks_forward_returns.csv")


def test_a_failing_check_ignore_is_refused_not_treated_as_ignored(guard, monkeypatch):
    """The mirror-image hole on the second call."""
    _patch_git(monkeypatch, {"ls-files": 1, "check-ignore": 128})

    with pytest.raises(SystemExit, match=r"check-ignore failed \(exit 128\)"):
        guard(REPO / "outputs" / "research" / "sniper_live_picks_forward_returns.csv")


def test_the_normal_codes_still_mean_what_they_meant(guard, monkeypatch):
    """Tightening must not turn the working case into a refusal."""
    _patch_git(monkeypatch, {"ls-files": 1, "check-ignore": 0})
    allowed = REPO / "outputs" / "research" / "sniper_live_picks_forward_returns.csv"

    assert guard(allowed) == allowed.resolve()

    _patch_git(monkeypatch, {"ls-files": 0, "check-ignore": 0})
    with pytest.raises(SystemExit, match="TRACKED"):
        guard(allowed)


def test_git_being_absent_entirely_is_refused(guard, monkeypatch):
    """If git cannot be executed at all, there is no basis to permit a write."""
    import subprocess

    def boom(*a, **kw):
        raise OSError("git not found")

    monkeypatch.setattr(subprocess, "run", boom)

    with pytest.raises(SystemExit, match="cannot consult git"):
        guard(REPO / "outputs" / "research" / "sniper_live_picks_forward_returns.csv")
