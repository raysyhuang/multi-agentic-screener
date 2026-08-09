"""Measurement plumbing: shadow-booked gate blocks + live drift monitoring.

Two Tier-0 items from the 2026-08 strategy review, both about making existing
failures VISIBLE rather than changing what the book trades:

1. Picks the validation gate blocks were deleted before persistence, so the two
   2026-07 outages left zero record of foregone P&L and the gate's demonstrated
   failure mode (blocking healthy models on small-n noise) was unmeasurable.
2. The nightly drift monitor queried a table belonging to engines scaled to zero
   in 2026-03, against the retired 71.6%-WR baseline — a silent nightly no-op.
"""
from __future__ import annotations

from src.output.performance import SHADOW_SKIP_REASON
from src.research.drift_check import (
    BASELINES,
    DRIFT_SHORTFALL_PCT,
    MIN_TRADES_TO_JUDGE,
    DriftReport,
    StreamDrift,
    format_drift_report,
)


# ── Shadow-booking ─────────────────────────────────────────────────────────

def test_shadow_skip_reason_fits_the_column():
    """Outcome.skip_reason is String(30) — a longer label would truncate or raise."""
    assert len(SHADOW_SKIP_REASON) <= 30


# NOTE: the SQL behaviour these used to assert by grepping function source now
# has real DB-backed coverage in tests/test_db/test_shadow_and_candidates.py —
# a source grep passes whether or not the query is correct, which is exactly the
# failure mode it was meant to guard against.


def test_shadow_rows_are_excluded_from_stats_by_the_existing_convention():
    """Every stats query filters `skip_reason.is_(None)`. Shadow rows carry a
    non-null reason, so they are excluded for free — that is the whole reason
    this label was chosen over a new column."""
    assert SHADOW_SKIP_REASON is not None
    assert SHADOW_SKIP_REASON != ""






# ── Drift monitor ──────────────────────────────────────────────────────────

def test_drift_baselines_are_the_honest_numbers_not_the_retired_fantasy():
    """The retired baseline was 71.6% WR / +1.05%/trade / 2.47 Sharpe. Every
    live-stream baseline must be far below that — if one creeps up, someone has
    reintroduced an optimistic number."""
    for key, b in BASELINES.items():
        assert b["wr"] < 0.60, f"{key} win-rate baseline looks like the retired fantasy"
        assert b["avg"] <= 2.5, f"{key} avg baseline looks like the retired fantasy"


def test_drift_covers_every_live_stream():
    """A stream with no baseline is reported but never alerts, so a missing key
    silently disables monitoring for it."""
    for key in ("sniper|mas_official", "mean_reversion|mas_official",
                "pead|pead_paper", "pead|pead_neglected"):
        assert key in BASELINES


def test_drift_does_not_alert_below_the_sample_floor():
    """Small-n over-reaction is this project's recurring gate failure; the
    monitor must not repeat it."""
    assert MIN_TRADES_TO_JUDGE >= 15


def test_drift_shortfall_threshold_is_loose_enough_to_avoid_noise():
    assert 0.0 < DRIFT_SHORTFALL_PCT <= 0.6


def test_report_renders_and_flags_alerting_streams():
    r = DriftReport(
        lookback_days=30, total_resolved=25,
        streams=[
            StreamDrift(stream="sniper|mas_official", label="Sniper (official)", n=20,
                        live_win_rate=0.55, live_avg=0.10, baseline_avg=0.54,
                        baseline_wr=0.543, alerts=["degraded"]),
            StreamDrift(stream="mean_reversion|mas_official", label="MR (official)", n=5,
                        live_win_rate=0.60, live_avg=0.50, baseline_avg=0.46,
                        baseline_wr=0.522),
        ],
        alerts=["degraded"],
    )
    out = format_drift_report(r)
    assert "Sniper (official)" in out and "MR (official)" in out
    assert "ALERTS (1)" in out
    # the healthy stream must not be flagged
    mr_line = [ln for ln in out.splitlines() if "MR (official)" in ln][0]
    assert not mr_line.rstrip().endswith("!")


def test_report_says_so_when_nothing_drifted():
    r = DriftReport(lookback_days=30, total_resolved=40, streams=[], alerts=[])
    assert "No drift detected" in format_drift_report(r)
