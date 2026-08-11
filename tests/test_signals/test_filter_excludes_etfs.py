"""The universe gate must actually exclude ETFs — it did not for six months.

`filter_universe` documents "exclude ETFs/ETNs (we trade individual stocks)" and
implemented it by reading `stock["type"]`. That is the Polygon-shaped field. The
FMP screener is the PRIMARY universe source (`AggregatorClient.get_universe`
tries FMP first and only falls back to Polygon), and it does not return `type`
at all — it reports `isEtf` / `isFund` booleans. So the gate read None on every
FMP row and excluded nothing.

The consequence was live: TQQQ, a 3x leveraged ETF, reached the official MAS
candidate pool with a sniper score of 97.5. Leveraged products clear the
sniper's `ATR% >= 5` floor structurally rather than because of a setup, and
their relative-strength-vs-SPY component measures leveraged beta rather than the
idiosyncratic strength the signal is built on.

The record below is FMP's real `profile-symbol` response for TQQQ, trimmed to
the fields the filter reads.
"""

from __future__ import annotations

import pytest

from src.signals.filter import FilterFunnel, filter_universe

# Real FMP shape: no `type` key, `isEtf` true, beta 3.7.
TQQQ_FMP = {
    "symbol": "TQQQ",
    "companyName": "ProShares UltraPro QQQ",
    "price": 73.9563,
    "volume": 26_992_343,
    "marketCap": 45_410_727_071,
    "exchange": "NASDAQ",
    "exchangeShortName": "NASDAQ",
    "beta": 3.714,
    "isEtf": True,
    "isFund": False,
    "isActivelyTrading": True,
}

AAPL_FMP = {
    "symbol": "AAPL",
    "companyName": "Apple Inc.",
    "price": 220.0,
    "volume": 50_000_000,
    "marketCap": 3_300_000_000_000,
    "exchange": "NASDAQ",
    "exchangeShortName": "NASDAQ",
    "isEtf": False,
    "isFund": False,
    "isActivelyTrading": True,
}


def test_leveraged_etf_is_excluded_from_the_universe() -> None:
    funnel = FilterFunnel()
    passed = filter_universe([TQQQ_FMP], funnel=funnel)

    assert [s["symbol"] for s in passed] == []
    assert funnel.failed_type == 1, "the ETF must be dropped by the type gate"


def test_ordinary_stock_still_passes() -> None:
    passed = filter_universe([AAPL_FMP])
    assert [s["symbol"] for s in passed] == ["AAPL"]


def test_mutual_fund_flag_is_also_excluded() -> None:
    fund = {**AAPL_FMP, "symbol": "VFIAX", "isEtf": False, "isFund": True}
    assert filter_universe([fund]) == []


@pytest.mark.parametrize("stock_type", ["ETF", "etf", "ETN", "FUND", "REIT"])
def test_polygon_style_type_field_still_works(stock_type: str) -> None:
    """The Polygon fallback path supplies `type` and no boolean flags."""
    row = {
        "symbol": "XXXX",
        "price": 50.0,
        "volume": 1_000_000,
        "exchangeShortName": "NYSE",
        "type": stock_type,
    }
    assert filter_universe([row]) == []


@pytest.mark.parametrize(
    "encoding",
    [True, "true", "True", "TRUE", " true ", 1, "1", "yes"],
)
def test_truthy_encodings_all_exclude(encoding) -> None:
    assert filter_universe([{**AAPL_FMP, "isEtf": encoding}]) == []


@pytest.mark.parametrize(
    "encoding",
    [False, "false", "False", "FALSE", " false ", 0, "0", "no", "", None],
)
def test_falsey_encodings_never_exclude(encoding) -> None:
    """The fail-closed risk: `bool("false")` is True.

    Reading these flags for raw truthiness would drop EVERY FMP row the day the
    provider switched from JSON booleans to strings — a silent zero-universe
    run, which is far worse than admitting an ETF. Caught in review by Hawk.
    """
    funnel = FilterFunnel()
    passed = filter_universe([{**AAPL_FMP, "isEtf": encoding}], funnel=funnel)

    assert [s["symbol"] for s in passed] == ["AAPL"]
    assert funnel.failed_type == 0
    assert funnel.unrecognized_type_flags == 0, f"{encoding!r} should be recognised"


def test_unrecognised_encoding_admits_the_row_and_is_counted() -> None:
    """Unknown shape must not silently decide either way — it must be visible."""
    funnel = FilterFunnel()
    passed = filter_universe([{**AAPL_FMP, "isEtf": "maybe"}], funnel=funnel)

    assert [s["symbol"] for s in passed] == ["AAPL"], "must not fail closed"
    assert funnel.unrecognized_type_flags == 1, "but the operator must be told"


def test_a_string_false_universe_is_not_wiped_out() -> None:
    """End-to-end shape of the regression: a whole screener page of "false"."""
    # Tickers must be alphabetic — `_is_valid_ticker` rejects digits, so a
    # "TIC0"-style symbol would fail the format gate and mask what this asserts.
    symbols = [f"{a}{b}" for a in "ABCDEFGHIJ" for b in "ABCDE"]
    rows = [
        {**AAPL_FMP, "symbol": s, "isEtf": "false", "isFund": "false"}
        for s in symbols
    ]
    assert len(filter_universe(rows)) == 50


def test_rows_with_neither_field_are_not_dropped() -> None:
    """Absent flags must not become an accidental reject-everything gate.

    The Polygon universe builder sets `type` to "" deliberately (its query is
    already restricted to common stock), so a row carrying no type information
    at all is a legitimate stock, not an unknown.
    """
    row = {
        "symbol": "MSFT",
        "price": 400.0,
        "volume": 20_000_000,
        "exchangeShortName": "NASDAQ",
        "type": "",
    }
    assert [s["symbol"] for s in filter_universe([row])] == ["MSFT"]
