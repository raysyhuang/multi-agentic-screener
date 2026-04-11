"""Tests for scripts/backfill_phantom_exits.py — pure logic only, no DB."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from scripts.backfill_phantom_exits import (
    RecomputeResult,
    TradeContext,
    apply_updates,
    build_candidates_query,
    compute_base_stop,
    diff_outcome,
    recompute_exit,
    summarize_diffs,
    write_snapshot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(**overrides) -> TradeContext:
    base = dict(
        outcome_id=1,
        ticker="EQT",
        signal_model="mean_reversion",
        confidence=99.0,
        entry_date=date(2026, 4, 6),
        entry_price=59.6095,
        planned_entry=59.70,
        planned_stop=58.81,
        target=61.85,
        holding_days=3,
        old_exit_reason="trail_stop",
        old_exit_price=59.4904,
        old_exit_date=date(2026, 4, 6),
        old_pnl_pct=-0.1998,
    )
    base.update(overrides)
    return TradeContext(**base)


def _bar(d: date, o: float, h: float, l: float, c: float) -> dict:
    return {"date": d, "open": o, "high": h, "low": l, "close": c}


# ---------------------------------------------------------------------------
# compute_base_stop
# ---------------------------------------------------------------------------


def test_compute_base_stop_tiered_top_bucket():
    ctx = _make_ctx(
        confidence=99.0,
        entry_price=100.0,
        planned_entry=100.0,
        planned_stop=99.0,  # tier_atr = 1.0/0.75 = 1.333
    )
    # score >= 85 => entry - 1.25 * tier_atr = 100 - 1.667 = 98.333 (round 98.33)
    assert compute_base_stop(ctx) == pytest.approx(98.33, abs=0.01)


def test_compute_base_stop_falls_back_when_no_confidence():
    ctx = _make_ctx(confidence=None, planned_stop=95.0)
    assert compute_base_stop(ctx) == 95.0


# ---------------------------------------------------------------------------
# recompute_exit
# ---------------------------------------------------------------------------


def test_phantom_trail_candidate_recomputes_to_different_exit():
    """EQT-shaped case: day 1 high rallies 2.18% above entry, low barely
    dips, close is positive. Under the FIXED logic, this bar must not
    produce a same-bar trail_stop exit. A recompute should differ from the
    recorded -0.20% phantom."""
    ctx = _make_ctx(
        entry_date=date(2026, 4, 6),
        entry_price=59.6095,
        planned_entry=59.70,
        planned_stop=58.81,
        target=61.85,
        holding_days=3,
    )
    bars = [
        _bar(date(2026, 4, 6), 59.550, 60.910, 59.470, 60.400),
        _bar(date(2026, 4, 7), 60.700, 61.550, 60.205, 60.690),
        _bar(date(2026, 4, 8), 58.920, 60.390, 58.090, 60.180),
    ]
    new = recompute_exit(ctx, bars)
    assert new is not None
    # Must not match the phantom exit price (entry * 0.998 ≈ 59.49)
    assert abs(new.exit_price - 59.4904) > 0.01
    # pnl should be materially different from -0.20%
    assert abs(new.pnl_pct - (-0.1998)) > 0.05


def test_recompute_runs_to_expiry_when_nothing_triggers():
    """A flat trade with no arm, no target, no stop → exits at expiry on
    the final holding bar's close."""
    ctx = _make_ctx(
        entry_price=100.0,
        planned_entry=100.0,
        planned_stop=90.0,   # well below any bar
        target=150.0,        # well above any bar
        holding_days=3,
    )
    bars = [
        _bar(date(2026, 3, 10), 100.0, 100.3, 99.8, 100.1),
        _bar(date(2026, 3, 11), 100.1, 100.4, 99.9, 100.2),
        _bar(date(2026, 3, 12), 100.2, 100.5, 100.0, 100.3),
    ]
    new = recompute_exit(ctx, bars)
    assert new is not None
    assert new.exit_reason == "expiry"
    assert new.exit_date == date(2026, 3, 12)


def test_recompute_returns_none_when_no_bars():
    ctx = _make_ctx()
    assert recompute_exit(ctx, []) is None


# ---------------------------------------------------------------------------
# diff_outcome
# ---------------------------------------------------------------------------


def test_unchanged_outcome_returns_no_diff():
    """If recompute matches the recorded values within epsilon, diff is None."""
    ctx = _make_ctx(
        old_exit_reason="expiry",
        old_exit_price=101.2,
        old_exit_date=date(2026, 3, 12),
        old_pnl_pct=1.20,
    )
    new = RecomputeResult(
        exit_reason="expiry",
        exit_price=101.2,
        exit_date=date(2026, 3, 12),
        pnl_pct=1.20,
        mfe=1.5,
        mae=-0.3,
    )
    assert diff_outcome(ctx, new) is None


def test_diff_outcome_detects_exit_reason_change():
    ctx = _make_ctx()  # old_exit_reason="trail_stop", old_pnl_pct=-0.1998
    new = RecomputeResult(
        exit_reason="expiry",
        exit_price=60.18 * 0.999,
        exit_date=date(2026, 4, 8),
        pnl_pct=0.86,
        mfe=2.18,
        mae=-0.23,
    )
    d = diff_outcome(ctx, new)
    assert d is not None
    assert d["old"]["exit_reason"] == "trail_stop"
    assert d["new"]["exit_reason"] == "expiry"
    assert d["old"]["pnl_pct"] == pytest.approx(-0.1998)
    assert d["new"]["pnl_pct"] == pytest.approx(0.86)


def test_diff_outcome_detects_pnl_change_within_same_reason():
    ctx = _make_ctx(old_exit_reason="trail_stop", old_pnl_pct=0.00, old_exit_price=59.6095)
    new = RecomputeResult(
        exit_reason="trail_stop",
        exit_price=60.50,
        exit_date=ctx.old_exit_date,
        pnl_pct=1.49,
        mfe=2.0,
        mae=-0.1,
    )
    d = diff_outcome(ctx, new)
    assert d is not None


# ---------------------------------------------------------------------------
# build_candidates_query
# ---------------------------------------------------------------------------


def test_query_filters_by_date_range_and_exit_reason():
    sql, params = build_candidates_query(
        start_date=date(2026, 3, 26),
        end_date=date(2026, 4, 11),
        ticker=None,
        outcome_id=None,
        exit_reason="trail_stop",
        limit=None,
    )
    assert "s.run_date >= :start_date" in sql
    assert "s.run_date <= :end_date" in sql
    assert "o.exit_reason = :exit_reason" in sql
    assert "o.still_open = false" in sql
    assert "o.skip_reason IS NULL" in sql
    assert params["start_date"] == date(2026, 3, 26)
    assert params["end_date"] == date(2026, 4, 11)
    assert params["exit_reason"] == "trail_stop"
    assert "ticker" not in params
    assert "outcome_id" not in params


def test_query_all_exit_reasons_omits_filter():
    sql, params = build_candidates_query(
        start_date=None, end_date=None, ticker=None, outcome_id=None,
        exit_reason="all", limit=None,
    )
    assert "o.exit_reason = :exit_reason" not in sql
    assert "exit_reason" not in params


def test_query_ticker_and_outcome_id_filters():
    sql, params = build_candidates_query(
        start_date=None, end_date=None, ticker="EQT", outcome_id=42,
        exit_reason="trail_stop", limit=5,
    )
    assert "s.ticker = :ticker" in sql
    assert "o.id = :outcome_id" in sql
    assert "LIMIT 5" in sql
    assert params["ticker"] == "EQT"
    assert params["outcome_id"] == 42


# ---------------------------------------------------------------------------
# Snapshot + apply
# ---------------------------------------------------------------------------


def test_write_snapshot_creates_file_with_payload(tmp_path: Path):
    diffs = [
        {
            "outcome_id": 1,
            "ticker": "EQT",
            "entry_date": "2026-04-06",
            "old": {
                "exit_reason": "trail_stop",
                "exit_price": 59.4904,
                "exit_date": "2026-04-06",
                "pnl_pct": -0.1998,
            },
            "new": {
                "exit_reason": "expiry",
                "exit_price": 60.0,
                "exit_date": "2026-04-08",
                "pnl_pct": 0.65,
            },
        }
    ]
    path = write_snapshot(diffs, tmp_path / "snap")
    assert path.exists()
    import json
    payload = json.loads(path.read_text())
    assert payload["count"] == 1
    assert payload["diffs"][0]["outcome_id"] == 1


@pytest.mark.asyncio
async def test_apply_updates_refuses_without_snapshot(tmp_path: Path):
    """apply_updates must raise if the snapshot artifact doesn't exist,
    independent of what the session would do. This is a belt-and-suspenders
    guard against an accidental apply path that skipped the snapshot."""
    from unittest.mock import AsyncMock

    session = AsyncMock()
    missing = tmp_path / "no_snapshot_here.json"

    diffs = [{"outcome_id": 1, "new": {"exit_reason": "expiry", "exit_price": 1.0, "exit_date": "2026-04-08", "pnl_pct": 0.0}}]
    with pytest.raises(RuntimeError, match="snapshot artifact missing"):
        await apply_updates(session, diffs, snapshot_path=missing)
    # The session must never have been asked to execute anything.
    session.execute.assert_not_called()
    session.commit.assert_not_called()


@pytest.mark.asyncio
async def test_apply_updates_issues_one_update_per_diff(tmp_path: Path):
    """With a valid snapshot, apply_updates issues one UPDATE per diff and commits once."""
    from unittest.mock import AsyncMock

    session = AsyncMock()
    snap = tmp_path / "snap.json"
    snap.write_text("{}")

    diffs = [
        {
            "outcome_id": 1,
            "new": {"exit_reason": "expiry", "exit_price": 60.0, "exit_date": "2026-04-08", "pnl_pct": 0.65},
        },
        {
            "outcome_id": 2,
            "new": {"exit_reason": "trail_stop", "exit_price": 47.5, "exit_date": "2026-04-01", "pnl_pct": 0.70},
        },
    ]
    n = await apply_updates(session, diffs, snapshot_path=snap)
    assert n == 2
    assert session.execute.await_count == 2
    session.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# summarize_diffs
# ---------------------------------------------------------------------------


def test_summarize_aggregates_pnl_and_transitions():
    diffs = [
        {"old": {"exit_reason": "trail_stop", "pnl_pct": -0.20}, "new": {"exit_reason": "trail_stop", "pnl_pct": 1.73}},
        {"old": {"exit_reason": "trail_stop", "pnl_pct": -0.20}, "new": {"exit_reason": "target", "pnl_pct": 0.72}},
        {"old": {"exit_reason": "trail_stop", "pnl_pct": -0.09}, "new": {"exit_reason": "expiry", "pnl_pct": 1.50}},
    ]
    summary = summarize_diffs(diffs)
    assert summary["changed"] == 3
    assert summary["old_pnl_sum"] == pytest.approx(-0.49, abs=0.01)
    assert summary["new_pnl_sum"] == pytest.approx(3.95, abs=0.01)
    assert summary["pnl_delta"] == pytest.approx(4.44, abs=0.01)
    assert summary["transitions"] == {
        "trail_stop -> trail_stop": 1,
        "trail_stop -> target": 1,
        "trail_stop -> expiry": 1,
    }
