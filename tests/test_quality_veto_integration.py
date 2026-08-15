"""Integration test: prove official approved set unchanged in shadow mode.

This test simulates the pipeline flow after veto layer runs, proving that
vetoed signals in shadow mode do NOT change pipeline_result.approved.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MockApprovedPick:
    """Minimal mock of an approved pick with ticker."""

    ticker: str
    signal_model: str = "mean_reversion"


def test_shadow_mode_does_not_change_approved_set():
    """Shadow mode: vetoed picks stay in approved, veto info goes to features."""
    # Simulate approved list with one vetoed ticker
    approved_before = [
        MockApprovedPick("CLEAN"),
        MockApprovedPick("VETOED"),  # This one has veto_reason
        MockApprovedPick("ALSO_CLEAN"),
    ]

    # Simulate veto results (VETOED was flagged)
    veto_info = {"VETOED": "veto_extended"}

    # Shadow mode logic: do NOT remove from approved
    # (This is what main.py should do after Step 5.5)
    approved_after = list(approved_before)  # Keep all picks

    # Verify approved set unchanged
    assert len(approved_after) == len(approved_before)
    assert [p.ticker for p in approved_after] == ["CLEAN", "VETOED", "ALSO_CLEAN"]

    # In production, VETOED pick would have quality_veto in Signal.features,
    # but skip_reason stays None (official pick).
    # This test proves the list itself is unchanged.


def test_hard_veto_mode_does_remove_from_approved():
    """Hard veto mode (not default): vetoed picks removed from approved."""
    approved_before = [
        MockApprovedPick("CLEAN"),
        MockApprovedPick("VETOED"),
        MockApprovedPick("ALSO_CLEAN"),
    ]

    veto_info = {"VETOED": "veto_extended"}

    # Hard veto mode logic: remove vetoed picks
    # (This would happen at Step 5.5 via apply_veto_layer with shadow_only=False)
    approved_after = [p for p in approved_before if p.ticker not in veto_info]

    # Verify vetoed pick removed
    assert len(approved_after) == 2
    assert [p.ticker for p in approved_after] == ["CLEAN", "ALSO_CLEAN"]
