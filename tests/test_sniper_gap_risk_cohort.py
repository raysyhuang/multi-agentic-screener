"""Unit tests for the --cohort path of scripts/sniper_gap_risk.py.

No network: a synthetic price frame and a synthetic frozen bundle. Pins the
ex-ante property of gap_vol (bars on/after the signal date are never used),
the one-stream loader, and the exit-reason split of a threshold.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd

from scripts.sniper_gap_risk import (
    attach_gap_vol,
    backtest_thresholds,
    gap_vol_asof,
    load_live_cohort,
    threshold_effect,
)


def _frame(n: int = 80, gap_after: float | None = None, gap_bars: int = 8) -> pd.DataFrame:
    """n calm bars (0.5% overnight gaps); optionally big gaps on the LAST gap_bars
    bars — enough of the trailing-60 window that the 90th percentile sees them."""
    dates = pd.bdate_range("2026-01-01", periods=n)
    close = [100.0] * n
    opens = [100.5] * n
    if gap_after is not None:
        for i in range(n - gap_bars, n):
            opens[i] = 100.0 * (1 + gap_after)
    return pd.DataFrame({"date": dates, "open": opens, "high": 101.0,
                         "low": 99.0, "close": close, "volume": 1_000_000})


def test_gap_vol_is_strictly_ex_ante():
    df = _frame(gap_after=0.30)          # 30% gaps on the final 8 bars
    first_gap = df["date"].iloc[-8].date()
    # asof == the first gap bar: that bar and everything after are excluded
    # (strictly before asof), so the feature is still the calm ~0.5%.
    assert abs(gap_vol_asof(df, first_gap) - 0.5) < 1e-6
    # asof after the last bar: the gap bars are history and dominate the p90.
    later = (df["date"].iloc[-1] + pd.Timedelta(days=1)).date()
    assert gap_vol_asof(df, later) > 5.0


def test_gap_vol_needs_history():
    assert gap_vol_asof(_frame(n=10), date(2026, 3, 1)) is None


def test_load_live_cohort_one_stream_only(tmp_path):
    bundle = {"trades": {
        "sniper|mas_official": [
            {"ticker": "AAA", "signal_date": "2026-07-01", "entry_date": "2026-07-02",
             "exit_reason": "time_stop", "pnl_pct": -6.0, "mfe": 0.1, "mae": -7.0},
            {"ticker": "BBB", "signal_date": "2026-07-03", "entry_date": "2026-07-06",
             "exit_reason": "trail_stop", "pnl_pct": 1.5, "mfe": 2.0, "mae": -1.0},
            {"ticker": "CCC", "signal_date": "2026-07-03", "pnl_pct": None},  # open → skipped
        ],
        "mean_reversion|mas_official": [
            {"ticker": "ZZZ", "signal_date": "2026-07-01", "exit_reason": "stop", "pnl_pct": -1.0},
        ],
    }}
    p = tmp_path / "bundle.json"
    p.write_text(json.dumps(bundle))
    rows, sha = load_live_cohort(p)
    assert [r["ticker"] for r in rows] == ["AAA", "BBB"]      # never blended, open row dropped
    assert rows[0]["signal_date"] == date(2026, 7, 1)
    assert len(sha) == 64


def test_threshold_effect_splits_by_exit_reason():
    price = {"AAA": _frame(gap_after=0.30), "BBB": _frame()}
    rows = [
        {"ticker": "AAA", "signal_date": date(2026, 6, 1), "exit_reason": "time_stop", "pnl": -6.0},
        {"ticker": "BBB", "signal_date": date(2026, 6, 1), "exit_reason": "trail_stop", "pnl": 1.5},
        {"ticker": "NOPE", "signal_date": date(2026, 6, 1), "exit_reason": "trail_stop", "pnl": 0.0},
    ]
    scored = attach_gap_vol(rows, price)
    assert scored[2]["gap_vol"] is None                        # unknown ticker → no feature
    scored = [r for r in scored if r["gap_vol"] is not None]
    assert scored[0]["gap_vol"] > scored[1]["gap_vol"]
    e = threshold_effect(scored, thr=5.0)
    assert e["kept"] == 1 and e["dropped"] == 1
    assert e["dropped_by_reason"] == {"time_stop": 1}
    assert e["kept_avg"] == 1.5


def test_backtest_thresholds_are_percentiles():
    scored = [{"gap_vol": float(i)} for i in range(100)]
    thr = backtest_thresholds(scored)
    assert thr == {"p95": 95.0, "p90": 90.0, "p80": 80.0, "p70": 70.0}
