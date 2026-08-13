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
