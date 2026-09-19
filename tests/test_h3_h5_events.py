"""H3/H5 event construction — synthetic, no network.

Pins the look-ahead-sensitive pieces: a dividend is classified only from
declarations dated before it; dividend entries sit two sessions after the
declaration; a report's fundamentals become usable only at the second session
on/after its date; and an episode counts once until it has been off for the
re-arm period.
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd

import scripts.h3_dividends_splits as h3
import scripts.h5_quality_drawdown as h5
from src.research import event_study as es


def _panel(n=40, start=date(2025, 1, 6)):
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    df = pd.DataFrame({"date": days, "open": 100.0, "high": 100.0, "low": 100.0,
                       "close": 100.0, "volume": 1_000_000})
    return es.build_panel({"AAA": df})


def _div(t, decl, amt, freq=4):
    return {"ticker": t, "declaration_date": decl, "cash_amount": amt, "frequency": freq,
            "dividend_type": "CD", "currency": "USD"}


def test_dividend_classification_uses_only_earlier_declarations():
    divs = [_div("AAA", "2024-01-10", 0.10), _div("AAA", "2024-04-10", 0.10),
            _div("AAA", "2024-07-10", 0.13),   # +30% vs previous same-frequency -> increase
            _div("AAA", "2024-10-10", 0.14),   # +7.7% -> nothing
            _div("BBB", "2024-05-01", 0.50),   # first ever, lookback covered -> initiation
            _div("CCC", "2022-03-01", 0.20)]   # lookback NOT covered by fetched history -> skipped
    ev = {(e["ticker"], e["event_date"]): e["kind"]
          for e in h3.dividend_events(divs, {"AAA", "BBB", "CCC"})}
    assert ev[("AAA", "2024-07-10")] == "increase"
    assert ("AAA", "2024-10-10") not in ev
    assert ev[("BBB", "2024-05-01")] == "initiation"
    assert ("AAA", "2024-01-10") in ev and ev[("AAA", "2024-01-10")] == "initiation"
    assert not any(t == "CCC" for t, _ in ev)


def test_dividend_entry_is_the_second_session_split_entry_the_first():
    p = _panel()
    ev = [{"ticker": "AAA", "event_date": str(p.dates[10].date())}]
    assert h3.place(ev, p, offset=1)[0]["entry_date"] == p.dates[11]
    assert h3.place(ev, p, offset=0)[0]["entry_date"] == p.dates[10]
    # an event on a non-session day snaps to the next session before the offset
    sat = p.dates[10] + pd.Timedelta(days=(5 - p.dates[10].weekday()) % 7 or 7)
    placed = h3.place([{"ticker": "AAA", "event_date": str(sat.date())}], p, offset=1)
    assert placed and placed[0]["entry_date"] > sat


def test_fundamentals_become_usable_at_the_second_session(tmp_path, monkeypatch):
    p = _panel(n=60)
    idx = p.dates
    reports = []
    for k in range(6):                          # six quarterly reports, revenue growing 20%/yr
        reports.append({"date": str((idx[5 + 8 * k]).date()), "epsActual": 1.0,
                        "epsEstimated": 0.9, "revenueActual": 100.0 * (1.2 ** (k / 4))})
    monkeypatch.setattr("scripts.h1_pead_wide.EARNINGS_CACHE_DIR", tmp_path)
    (tmp_path / "AAA.json").write_text(json.dumps(reports))
    f = h5.fundamentals_by_session("AAA", idx)
    i3 = 5 + 8 * 3                              # 4th report: first with four EPS actuals
    assert pd.isna(f["ttm_eps"].iloc[i3]) and f["ttm_eps"].iloc[i3 + 1] == 4.0
    # the 5th report (k=4) is the first with a YoY figure; usable only at i+1
    i4 = 5 + 8 * 4
    assert pd.isna(f["rev_yoy"].iloc[i4]) and pd.notna(f["rev_yoy"].iloc[i4 + 1])
    assert abs(f["rev_yoy"].iloc[i4 + 1] - 0.2) < 1e-9


def test_onsets_count_an_episode_once_until_rearmed():
    flag = pd.Series([False] * 25 + [True] * 5 + [False] * 5 + [True] * 3 + [False] * 25 + [True] * 2)
    assert h5.onsets(flag, rearm=20) == [25, 63]
