"""H2 tagging — synthetic, no network. Pins that an MR trade only ever sees the
most recent report at or BEFORE its own signal date, and that the POST_MISS
window counts sessions, not calendar days."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from scripts.h2_mr_post_miss import split, tag_trades
from src.research import event_study as es


def _panel(n=60):
    days, d = [], date(2026, 1, 5)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    df = pd.DataFrame({"date": days, "open": 100.0, "high": 100.0, "low": 100.0,
                       "close": 100.0, "volume": 1_000_000})
    return es.build_panel({"AAA": df, "BBB": df.copy()})


def _trade(panel, ticker, i):
    return {"ticker": ticker, "signal_date": panel.dates[i], "entry_date": panel.dates[i + 1],
            "pnl_pct": -1.0}


def test_a_future_report_is_invisible_to_an_earlier_trade():
    p = _panel()
    events = [{"ticker": "AAA", "signal_date": p.dates[30], "surprise": -25.0}]
    tagged = tag_trades(pd.DataFrame([_trade(p, "AAA", 29), _trade(p, "AAA", 30), _trade(p, "AAA", 33)]),
                        events, p)
    assert pd.isna(tagged.loc[0, "last_surprise"])              # day before the report: knows nothing
    assert tagged.loc[1, "last_surprise"] == -25.0 and tagged.loc[1, "sessions_since_report"] == 0
    assert tagged.loc[2, "sessions_since_report"] == 3


def test_most_recent_report_wins_and_tickers_do_not_leak():
    p = _panel()
    events = [{"ticker": "AAA", "signal_date": p.dates[10], "surprise": -30.0},
              {"ticker": "AAA", "signal_date": p.dates[40], "surprise": +15.0},
              {"ticker": "BBB", "signal_date": p.dates[41], "surprise": -50.0}]
    tagged = tag_trades(pd.DataFrame([_trade(p, "AAA", 42)]), events, p)
    assert tagged.loc[0, "last_surprise"] == 15.0 and tagged.loc[0, "sessions_since_report"] == 2


def test_window_is_in_sessions_and_threshold_is_inclusive():
    p = _panel()
    events = [{"ticker": "AAA", "signal_date": p.dates[20], "surprise": -10.0}]
    tagged = tag_trades(pd.DataFrame([_trade(p, "AAA", 30), _trade(p, "AAA", 31)]), events, p)
    flag, rest = split(tagged, miss=-10.0, window=10)
    assert len(flag) == 1 and len(rest) == 1                     # 10 sessions in, 11 out
    beat_flag, _ = split(tagged, miss=-10.0, window=10, beat=True)
    assert len(beat_flag) == 0                                   # a miss is never a beat
