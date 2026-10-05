"""RECLAIM research watch: today's triggers into the alert, nothing persisted."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.output.telegram import format_daily_alert
from src.reclaim_watch import WatchItem, WatchResult, run_reclaim_watch, scan
from src.signals import reclaim as rc
from src.utils.trading_calendar import trading_sessions

FIXTURES = Path(__file__).parent / "fixtures"


def _lfus_item() -> WatchItem:
    return WatchItem(lane=rc.DRIFT, symbol="LFUS", trigger_date=date(2026, 10, 1), close_k=435.55,
                     sma50_k=424.9641, stop=420.86, max_entry=457.4565, risk_at_close=0.0337,
                     earnings_soon=False, overlap=False)


def test_section_renders_triggers_with_the_no_edge_label_and_buy_zone():
    watch = WatchResult(date(2026, 10, 1), [_lfus_item()], {rc.RANGE: 4, rc.DRIFT: 9}).to_alert()
    msg = format_daily_alert([], "bear", "2026-10-02", execution_mode="quant_only", reclaim_watch=watch)
    assert "Reclaim — Research Watch</b> (1 trigger)" in msg
    assert "NOT a pick, not tracked" in msg
    assert "LFUS" in msg and "Stop <b>$420.86</b>" in msg
    # Both edges of the PASS band: an open at or below the stop is a gap-through skip.
    assert "Valid only if <b>$420.86</b> &lt; open ≤ <b>$457.46</b>" in msg
    assert "gap-through skip" in msg


def test_empty_watch_says_so_and_shows_alive_episodes():
    watch = WatchResult(date(2026, 10, 1), [], {rc.RANGE: 4, rc.DRIFT: 9}).to_alert()
    msg = format_daily_alert([], "bear", "2026-10-02", reclaim_watch=watch)
    assert "No triggers today." in msg and "Range-tech 4" in msg and "Drift-G3 9" in msg


def test_flags_risk_and_earnings_and_section_present_in_every_branch():
    item = _lfus_item()
    item.risk_at_close, item.earnings_soon = 0.11, True
    watch = WatchResult(date(2026, 10, 1), [item], {}).to_alert()
    pick = {"ticker": "AAA", "direction": "LONG", "entry_price": 10.0, "stop_loss": 9.5,
            "target_1": 11.0, "confidence": 80, "signal_model": "mean_reversion", "holding_period": 3}
    for kwargs in ({"picks": []}, {"picks": [pick]}, {"picks": [], "validation_failed": True}):
        msg = format_daily_alert(regime="bear", run_date="2026-10-02", reclaim_watch=watch, **kwargs)
        assert "Research Watch" in msg and "likely SKIP" in msg and "earnings" in msg
    assert "Research Watch" not in format_daily_alert([], "bear", "2026-10-02")


def test_scan_finds_only_triggers_confirming_on_the_signal_date():
    """No-trigger universe: the scan runs end to end and reports nothing."""
    df = pd.read_csv(FIXTURES / "reclaim_qnt_2026-10-01.csv")
    sessions = trading_sessions(date(2025, 2, 10), date(2026, 10, 1))
    res = scan({"QNT": df}, {"QNT": {"mcap": 5e9, "sector": "Technology"}}, sessions)
    assert res.signal_date == date(2026, 10, 1)
    assert all(i.trigger_date == date(2026, 10, 1) for i in res.items)
    assert set(res.alive) == set(rc.LANES)


@pytest.mark.asyncio
async def test_watch_refuses_stale_data_instead_of_reporting_no_triggers():
    from src.config import Settings

    async def fetch(tickers, start, end):
        return {t: pd.DataFrame({"date": [date(2026, 9, 1)], "open": [1.0], "high": [1.0],
                                 "low": [1.0], "close": [1.0], "volume": [1e6]}) for t in tickers}

    rows = [{"symbol": "AAA", "marketCap": 5e9, "price": 50.0, "volume": 1e6, "sector": "Technology"}]
    with pytest.raises(RuntimeError, match="have a bar"):
        await run_reclaim_watch(date(2026, 10, 2), Settings(), rows, fetch=fetch)


@pytest.mark.asyncio
async def test_watch_refuses_a_universe_without_market_caps():
    """The Polygon fallback universe has no marketCap: 'could not scan', never 'no triggers'."""
    from src.config import Settings

    async def fetch(tickers, start, end):  # pragma: no cover - must not be reached
        raise AssertionError("should not fetch")

    rows = [{"symbol": "AAA", "price": 50.0, "volume": 1e6}]
    with pytest.raises(RuntimeError, match="market cap"):
        await run_reclaim_watch(date(2026, 10, 2), Settings(), rows, fetch=fetch)


def test_no_trigger_line_discloses_partial_coverage():
    watch = WatchResult(date(2026, 10, 1), [], {rc.RANGE: 1}, symbols=1000, current=930).to_alert()
    msg = format_daily_alert([], "bear", "2026-10-02", reclaim_watch=watch)
    assert "No triggers today (930/1,000 symbols current)." in msg
    full = WatchResult(date(2026, 10, 1), [], {}, symbols=1000, current=1000).to_alert()
    assert "No triggers today." in format_daily_alert([], "bear", "2026-10-02", reclaim_watch=full)


def test_scan_accepts_a_date_indexed_frame_for_coverage():
    df = pd.read_csv(FIXTURES / "reclaim_qnt_2026-10-01.csv").set_index("date")
    sessions = trading_sessions(date(2025, 2, 10), date(2026, 10, 1))
    res = scan({"QNT": df}, {"QNT": {"mcap": 5e9}}, sessions)
    assert (res.symbols, res.current) == (1, 1)
