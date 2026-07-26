"""PEAD E1 quality filters (2026-07-26): revenue beat + day-1 reaction band.

Backtest: E1 lifts PEAD from +1.76%/Sharpe 1.43/20.8% DD to +2.25%/1.69/6.6% DD
and the recent-period edge from +1.18% to +1.50%. These lock the gate logic.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.signals.post_earnings_drift import _report_day_reaction, score_post_earnings_drift


def _df(reaction_pct: float, bars: int = 21):
    """Flat history, then a final (report-day) bar with the given % reaction."""
    d0 = date(2026, 6, 1)
    dates = [d0 + timedelta(days=i) for i in range(bars)]
    close = [100.0] * (bars - 1) + [100.0 * (1 + reaction_pct / 100)]
    return pd.DataFrame({
        "date": dates, "open": close, "high": [c * 1.02 for c in close],
        "low": [c * 0.98 for c in close], "close": close, "volume": [1e6] * bars,
    }), dates[-1]


FEAT = {"close": None, "atr_14": 3.0}


def _feat(df):
    return {"close": float(df["close"].iloc[-1]), "atr_14": 3.0}


def test_report_day_reaction_and_freshness():
    df, rd = _df(6.0)
    assert _report_day_reaction(df, rd) == pytest.approx(6.0)   # last bar = report day
    assert _report_day_reaction(df, None) is None              # no date
    # A report 5 sessions ago is stale (beyond max_age) → no reaction returned.
    old = df["date"].iloc[-6]
    assert _report_day_reaction(df, old) is None


def test_e1_fires_on_quality_beat():
    df, rd = _df(6.0)  # reaction 6% is inside [2,12]
    sig = score_post_earnings_drift("AAA", df, _feat(df), earnings_surprise_pct=12.0,
                                    revenue_surprise_pct=3.0, report_date=rd)
    assert sig is not None
    assert sig.components["revenue_surprise_pct"] == 3.0
    assert sig.components["day1_reaction_pct"] == 6.0


def test_e1_rejects_no_revenue_beat():
    df, rd = _df(6.0)
    assert score_post_earnings_drift("AAA", df, _feat(df), 12.0, revenue_surprise_pct=1.0, report_date=rd) is None
    assert score_post_earnings_drift("AAA", df, _feat(df), 12.0, revenue_surprise_pct=None, report_date=rd) is None


def test_e1_rejects_reaction_out_of_band():
    df_lo, rd_lo = _df(1.0)   # <2%: market rejected
    df_hi, rd_hi = _df(15.0)  # >12%: consumed
    assert score_post_earnings_drift("AAA", df_lo, _feat(df_lo), 12.0, revenue_surprise_pct=3.0, report_date=rd_lo) is None
    assert score_post_earnings_drift("AAA", df_hi, _feat(df_hi), 12.0, revenue_surprise_pct=3.0, report_date=rd_hi) is None


def test_e1_off_fires_on_eps_only():
    df, rd = _df(15.0)  # reaction out of band, no revenue — but E1 off
    sig = score_post_earnings_drift("AAA", df, _feat(df), 12.0, revenue_surprise_pct=None,
                                    report_date=rd, e1_filters=False)
    assert sig is not None  # raw EPS-only signal still fires
