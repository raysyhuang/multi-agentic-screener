"""Delisted names must not enter the universe.

The FMP screener returned 519 rows flagged `isActivelyTrading=False` out of
2,570 (VMW, PXD, DFS, TWTR, ...). A stale last price and volume let them clear
every gate, and on 2026-10-02 65 of them took OHLCV slots, each walking the whole
Polygon -> FMP -> yfinance fallback before failing. Two layers, like the ETF
gate: the screener asks FMP for active names only, and `filter_universe` drops
any row explicitly flagged inactive in case the endpoint ignores the parameter.
"""

from __future__ import annotations

import pytest

from src.data.fmp_client import FMPClient
from src.signals.filter import FilterFunnel, filter_universe


def _row(symbol: str, **extra) -> dict:
    row = {
        "symbol": symbol, "price": 50.0, "volume": 2_000_000, "marketCap": 5e9,
        "exchangeShortName": "NYSE", "isEtf": False, "isFund": False,
    }
    row.update(extra)
    return row


@pytest.mark.parametrize("flag", [False, "false", "False", 0])
def test_explicitly_inactive_rows_are_dropped_and_counted(flag):
    funnel = FilterFunnel()
    out = filter_universe([_row("PXD", isActivelyTrading=flag), _row("AAPL", isActivelyTrading=True)],
                          funnel=funnel)
    assert [r["symbol"] for r in out] == ["AAPL"]
    assert funnel.failed_inactive == 1
    assert funnel.to_dict()["failed_inactive"] == 1


def test_missing_or_unrecognised_flag_is_admitted():
    """Polygon-shaped rows never carry the field; absence is not 'inactive'."""
    rows = [_row("AAA"), _row("BBB", isActivelyTrading=None), _row("CCC", isActivelyTrading="maybe"),
            _row("DDD", type="", isActivelyTrading=None)]
    funnel = FilterFunnel()
    assert len(filter_universe(rows, funnel=funnel)) == 4
    assert funnel.failed_inactive == 0


class _FakeResponse:
    @staticmethod
    def json() -> list[dict]:
        return [{"symbol": "AAPL"}]


@pytest.mark.asyncio
async def test_screener_requests_actively_trading_names_only(monkeypatch):
    captured: dict = {}

    async def fake_request(url, params):
        captured.update(params)
        return _FakeResponse()

    client = FMPClient()
    monkeypatch.setattr(client, "_request", fake_request)
    await client.get_stock_screener()
    assert captured["isActivelyTrading"] == "true"


def test_daily_alert_shows_the_universe_line_in_every_branch():
    from src.output.telegram import format_daily_alert

    stats = {"total_input": 2570, "passed": 2051, "failed_inactive": 519}
    pick = {"ticker": "AAA", "direction": "LONG", "entry_price": 10.0, "stop_loss": 9.5,
            "target_1": 11.0, "confidence": 80, "signal_model": "mean_reversion", "holding_period": 3}
    for kwargs in ({"picks": []}, {"picks": [pick]}, {"picks": [], "validation_failed": True}):
        msg = format_daily_alert(regime="bear", run_date="2026-10-05", execution_mode="quant_only",
                                 universe_stats=stats, **kwargs)
        assert "Universe: 2,051 screened (519 delisted dropped)" in msg
    msg = format_daily_alert(picks=[], regime="bear", run_date="2026-10-05")
    assert "Universe:" not in msg


def test_inactive_is_counted_even_when_another_gate_would_also_reject():
    funnel = FilterFunnel()
    rows = [_row("PENY", price=1.0, isActivelyTrading=False), _row("OK", isActivelyTrading=True)]
    filter_universe(rows, funnel=funnel)
    assert funnel.failed_inactive == 1 and funnel.failed_price == 0
    assert funnel.passed + funnel.failed_inactive + funnel.failed_price == funnel.total_input
