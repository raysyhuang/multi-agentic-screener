"""Unit tests for PEAD slot sweep — synthetic trades lock the mechanism.

These tests pass WITHOUT needing the 3Y OHLCV parquet or real PEAD trade cache.
They use synthetic 20-day overlapping trades to verify:
  - uncapped replay measures peak concurrent correctly
  - open-slot caps skip excess entries
  - weekly entry limits bind as expected
  - sector caps bind when sector data is present

Full cohort sweep is for a checkout that has the real PEAD trade list."""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from scripts.pead_slot_sweep import (
    compute_metrics,
    filter_sector_cap,
    filter_weekly_entry_limit,
    to_book_trades,
)


def synthetic_overlapping_trades(n: int, start: date, hold_days: int = 20) -> list[dict]:
    """Generate n trades entering on consecutive days with `hold_days` hold.
    This creates `hold_days` peak concurrent open positions if uncapped."""
    out = []
    for i in range(n):
        entry = start + timedelta(days=i)
        exit_d = entry + timedelta(days=hold_days)
        out.append({
            "ticker": f"SYM{i % 10}",
            "entry_date": entry.isoformat(),
            "exit_date": exit_d.isoformat(),
            "pnl_pct": 1.5,
            "sector": f"SECTOR_{i % 3}",
        })
    return out


def test_uncapped_measures_peak_concurrent():
    """Uncapped replay measures peak concurrent = hold_days when trades overlap."""
    trades = synthetic_overlapping_trades(n=25, start=date(2023, 1, 1), hold_days=20)
    m = compute_metrics(to_book_trades(trades), max_concurrent=None)
    assert m["taken"] == 25
    assert m["skipped"] == 0
    # Peak concurrent = 20 (the hold period, since we have consecutive daily entries).
    assert m["peak_concurrent"] == 20


def test_open_slot_cap_skips_excess():
    """Open-slot cap skips entries when slots are full."""
    trades = synthetic_overlapping_trades(n=25, start=date(2023, 1, 1), hold_days=20)
    m = compute_metrics(to_book_trades(trades), max_concurrent=5)
    # Only 5 concurrent slots → some trades are skipped.
    assert m["taken"] < 25
    assert m["skipped"] > 0
    assert m["peak_concurrent"] <= 5


def test_weekly_entry_limit_binds():
    """Weekly entry limit filters the trade list before replay."""
    trades = synthetic_overlapping_trades(n=25, start=date(2023, 1, 1), hold_days=20)
    # Limit to 3 entries per 7-day window.
    filtered = filter_weekly_entry_limit(trades, max_per_week=3)
    # With consecutive daily entries, we should get at most 3 entries per week.
    # Over 25 days (~3.5 weeks), expect ~10-11 entries.
    assert len(filtered) < len(trades)
    assert len(filtered) <= 3 * 4  # ~4 weeks, 3 per week


def test_weekly_entry_limit_none():
    """Weekly limit None or <=0 returns all trades."""
    trades = synthetic_overlapping_trades(n=10, start=date(2023, 1, 1), hold_days=20)
    assert len(filter_weekly_entry_limit(trades, max_per_week=None)) == 10
    assert len(filter_weekly_entry_limit(trades, max_per_week=0)) == 10
    assert len(filter_weekly_entry_limit(trades, max_per_week=-1)) == 10


def test_sector_cap_binds():
    """Sector cap limits concurrent open positions per sector."""
    trades = synthetic_overlapping_trades(n=25, start=date(2023, 1, 1), hold_days=20)
    # With i % 3 sector assignment, we have 3 sectors.
    # Max 2 per sector concurrently.
    filtered, sector_ok = filter_sector_cap(trades, max_per_sector=2)
    assert sector_ok is True  # sector column is present and valid
    assert len(filtered) < len(trades)  # some trades are skipped


def test_sector_cap_missing_column():
    """Sector cap returns (trades, False) if sector column is missing."""
    trades = [
        {"ticker": "A", "entry_date": "2023-01-01", "exit_date": "2023-01-21", "pnl_pct": 1.0},
        {"ticker": "B", "entry_date": "2023-01-02", "exit_date": "2023-01-22", "pnl_pct": 1.0},
    ]
    filtered, sector_ok = filter_sector_cap(trades, max_per_sector=2)
    assert sector_ok is False
    assert len(filtered) == len(trades)


def test_sector_cap_mostly_null():
    """Sector cap returns (trades, False) if >50% of sectors are null."""
    trades = synthetic_overlapping_trades(n=10, start=date(2023, 1, 1), hold_days=20)
    # Nullify sector for 8 of 10 trades (80%).
    for i in range(8):
        trades[i]["sector"] = None
    filtered, sector_ok = filter_sector_cap(trades, max_per_sector=2)
    assert sector_ok is False
    assert len(filtered) == len(trades)


def test_sector_cap_none():
    """Sector cap None or <=0 returns all trades."""
    trades = synthetic_overlapping_trades(n=10, start=date(2023, 1, 1), hold_days=20)
    filtered, _ = filter_sector_cap(trades, max_per_sector=None)
    assert len(filtered) == 10
    filtered, _ = filter_sector_cap(trades, max_per_sector=0)
    assert len(filtered) == 10


def test_to_book_trades_handles_malformed():
    """to_book_trades skips malformed records gracefully."""
    trades = [
        {"ticker": "A", "entry_date": "2023-01-01", "exit_date": "2023-01-21", "pnl_pct": 1.5},
        {"ticker": "B", "entry_date": "bad-date", "exit_date": "2023-01-22", "pnl_pct": 1.0},
        {"ticker": "C", "entry_date": "2023-01-03", "exit_date": "2023-01-23", "pnl_pct": None},
    ]
    book_trades = to_book_trades(trades)
    # Only the first trade is valid.
    assert len(book_trades) == 1
    assert book_trades[0].entry == date(2023, 1, 1)


def test_compute_metrics_empty():
    """compute_metrics handles empty trade list."""
    m = compute_metrics([], max_concurrent=5)
    assert m["taken"] == 0
    assert m["skipped"] == 0
    assert m["peak_concurrent"] == 0
    assert m["return_pct"] == 0.0
    assert m["sharpe"] is None


def test_compute_metrics_all_winners():
    """compute_metrics computes positive return for all-winner cohort."""
    trades = synthetic_overlapping_trades(n=10, start=date(2023, 1, 1), hold_days=20)
    # All trades have +1.5% P&L.
    m = compute_metrics(to_book_trades(trades), max_concurrent=None)
    assert m["taken"] == 10
    assert m["return_pct"] > 0  # compounded positive return


def test_open_slot_cap_binds_at_low_cap():
    """Open-slot cap=3 on a stacked earnings week skips many trades."""
    # Simulate an earnings week: 10 trades entering on consecutive days, 20d hold.
    trades = synthetic_overlapping_trades(n=10, start=date(2023, 1, 1), hold_days=20)
    m_uncap = compute_metrics(to_book_trades(trades), max_concurrent=None)
    m_cap3 = compute_metrics(to_book_trades(trades), max_concurrent=3)
    # Uncapped takes all 10, cap=3 takes fewer.
    assert m_uncap["taken"] == 10
    assert m_cap3["taken"] < 10
    assert m_cap3["skipped"] > 0
    assert m_cap3["peak_concurrent"] <= 3


def test_weekly_limit_with_gaps():
    """Weekly entry limit correctly handles gaps between trades."""
    # 3 trades in week 1, then 10-day gap, then 5 trades in week 3.
    trades = []
    start = date(2023, 1, 1)
    for i in range(3):
        trades.append({
            "ticker": f"W1_{i}", "entry_date": (start + timedelta(days=i)).isoformat(),
            "exit_date": (start + timedelta(days=i + 20)).isoformat(), "pnl_pct": 1.0,
        })
    start2 = start + timedelta(days=14)  # 2-week gap
    for i in range(5):
        trades.append({
            "ticker": f"W3_{i}", "entry_date": (start2 + timedelta(days=i)).isoformat(),
            "exit_date": (start2 + timedelta(days=i + 20)).isoformat(), "pnl_pct": 1.0,
        })
    # Limit to 3 entries per 7-day window.
    filtered = filter_weekly_entry_limit(trades, max_per_week=3)
    # Week 1: 3 entries (all admitted). Gap. Week 3: 3 of 5 admitted.
    assert len(filtered) == 6


def test_exclusion_threshold_check():
    """Verify we can compute exclusion % for the 30% kill threshold."""
    trades = synthetic_overlapping_trades(n=100, start=date(2023, 1, 1), hold_days=20)
    m_cap3 = compute_metrics(to_book_trades(trades), max_concurrent=3)
    exclusion_pct = (m_cap3["skipped"] / len(trades)) * 100.0
    # With 100 consecutive daily 20-day holds, cap=3 should skip many (>30%).
    assert exclusion_pct > 30.0  # This would trigger the kill note.
