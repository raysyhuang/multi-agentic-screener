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
    p = _panel(n=400)
    idx = p.dates
    reports = []
    for k in range(6):                          # six quarterly reports (63 sessions apart), revenue +20%/yr
        reports.append({"date": str((idx[5 + 63 * k]).date()), "epsActual": 1.0,
                        "epsEstimated": 0.9, "revenueActual": 100.0 * (1.2 ** (k / 4))})
    monkeypatch.setattr("scripts.h1_pead_wide.EARNINGS_CACHE_DIR", tmp_path)
    (tmp_path / "AAA.json").write_text(json.dumps(reports))
    f = h5.fundamentals_by_session("AAA", idx)
    i3 = 5 + 63 * 3                             # 4th report: first with four EPS actuals
    assert pd.isna(f["ttm_eps"].iloc[i3]) and f["ttm_eps"].iloc[i3 + 1] == 4.0
    # the 5th report (k=4) is the first with a YoY figure; usable only at i+1
    i4 = 5 + 63 * 4
    assert pd.isna(f["rev_yoy"].iloc[i4]) and pd.notna(f["rev_yoy"].iloc[i4 + 1])
    assert abs(f["rev_yoy"].iloc[i4 + 1] - 0.2) < 1e-9


def test_onsets_count_an_episode_once_until_rearmed():
    flag = pd.Series([False] * 25 + [True] * 5 + [False] * 5 + [True] * 3 + [False] * 25 + [True] * 2)
    assert h5.onsets(flag, rearm=20) == [25, 63]


def test_a_prior_dividend_without_a_declaration_date_still_blocks_an_initiation():
    divs = [{"ticker": "AAA", "ex_dividend_date": "2024-02-15", "cash_amount": 0.10, "frequency": 4,
             "dividend_type": "CD", "currency": "USD"},                      # no declaration_date
            _div("AAA", "2024-05-01", 0.10)]
    ev = h3.dividend_events(divs, {"AAA"})
    assert ev == []                     # not an initiation: a dividend existed 76 days earlier


def test_dividends_are_compared_on_one_share_basis():
    """NVDA-style: $0.04 before a 10:1 split, $0.01 after it = +150%, an increase.
    CIM-style: $0.11 before a 1:3 reverse split, $0.35 after = +6%, not one."""
    divs = [_div("NVD", "2024-02-21", 0.04) | {"ex_dividend_date": "2024-03-05"},
            _div("NVD", "2024-05-22", 0.01) | {"ex_dividend_date": "2024-06-11"},
            _div("CIM", "2024-02-01", 0.11) | {"ex_dividend_date": "2024-03-01"},
            _div("CIM", "2024-06-01", 0.35) | {"ex_dividend_date": "2024-06-20"}]
    splits = [{"ticker": "NVD", "execution_date": "2024-06-10", "split_from": 1, "split_to": 10},
              {"ticker": "CIM", "execution_date": "2024-05-24", "split_from": 3, "split_to": 1}]
    ev = {(e["ticker"], e["event_date"]): e["kind"] for e in h3.dividend_events(divs, {"NVD", "CIM"}, splits)}
    assert ev.get(("NVD", "2024-05-22")) == "increase"
    assert ("CIM", "2024-06-01") not in ev


def test_dividend_classification_does_not_depend_on_row_order():
    base = [_div("CSW", "2024-01-10", 0.50) | {"ex_dividend_date": "2024-01-20"},
            _div("CSW", "2024-01-10", 0.06) | {"ex_dividend_date": "2024-01-20"},   # supplemental component
            _div("CSW", "2024-04-10", 0.57) | {"ex_dividend_date": "2024-04-20"},
            _div("CSW", "2024-04-10", 0.06) | {"ex_dividend_date": "2024-04-20"}]
    a = h3.dividend_events(base, {"CSW"})
    b = h3.dividend_events(list(reversed(base)), {"CSW"})
    assert sorted((e["event_date"], e["kind"]) for e in a) == sorted((e["event_date"], e["kind"]) for e in b)
    assert ("2024-04-10", "increase") not in [(e["event_date"], e["kind"]) for e in a]   # 0.56 -> 0.63 = +12.5%


def test_h5_requires_consecutive_quarters_and_five_positive_revenues(tmp_path, monkeypatch):
    p = _panel(n=260)
    idx = p.dates
    # quarterly reports every 63 sessions except one skipped quarter
    days = [5, 68, 131, 194, 257]
    rows = [{"date": str(idx[i].date()), "epsActual": 1.0, "epsEstimated": 0.9,
             "revenueActual": rv} for i, rv in zip(days, [100, 105, -30, 115, 120])]
    monkeypatch.setattr("scripts.h1_pead_wide.EARNINGS_CACHE_DIR", tmp_path)
    (tmp_path / "AAA.json").write_text(json.dumps(rows))
    f = h5.fundamentals_by_session("AAA", idx)
    assert f["rev_yoy"].isna().all()               # an intermediate revenue is negative: no YoY
    rows[2]["revenueActual"] = 110
    rows.pop(3)                                    # drop a quarter: k-4 would be ~1.25y back
    (tmp_path / "AAA.json").write_text(json.dumps(rows))
    f = h5.fundamentals_by_session("AAA", idx)
    assert f["rev_yoy"].isna().all() and f["ttm_eps"].isna().all()


def test_conflicting_same_day_splits_are_skipped_not_compounded():
    cal = pd.to_datetime(pd.bdate_range("2025-01-06", periods=10))
    splits = [{"ticker": "ZZZ", "execution_date": str(cal[5].date()), "split_from": 1, "split_to": 10000},
              {"ticker": "ZZZ", "execution_date": str(cal[5].date()), "split_from": 10000, "split_to": 1},
              {"ticker": "YYY", "execution_date": str(cal[5].date()), "split_from": 1, "split_to": 2},
              {"ticker": "YYY", "execution_date": str(cal[5].date()), "split_from": 1, "split_to": 2}]
    m = es.split_price_multiplier(splits, cal, ["ZZZ", "YYY"])
    assert (m["ZZZ"] == 1.0).all()                 # conflicting: skipped
    assert m["YYY"].iloc[0] == 2.0                 # identical duplicate: applied once
