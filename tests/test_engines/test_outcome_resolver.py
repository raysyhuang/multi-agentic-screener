import pandas as pd

from src.engines import outcome_resolver


def test_fetch_prices_batch_uses_single_threaded_yfinance(monkeypatch):
    captured = {}

    def fake_download(*args, **kwargs):
        captured["threads"] = kwargs.get("threads")
        return pd.DataFrame()

    monkeypatch.setattr(outcome_resolver.yf, "download", fake_download)

    outcome_resolver._fetch_prices_batch(["AAPL", "MSFT"])

    assert captured["threads"] is False


def test_fetch_prices_batch_recovers_missing_ticker_with_single_fallback(monkeypatch):
    calls: list[tuple[object, bool]] = []
    idx = pd.to_datetime(["2026-02-10", "2026-02-11"])

    batch_df = pd.DataFrame(
        {
            ("Close", "AAPL"): [100.0, 101.0],
            ("High", "AAPL"): [101.0, 102.0],
            ("Low", "AAPL"): [99.0, 100.0],
        },
        index=idx,
    )

    msft_df = pd.DataFrame(
        {
            "Close": [200.0, 201.0],
            "High": [201.0, 202.0],
            "Low": [199.0, 200.0],
        },
        index=idx,
    )

    def fake_download(tickers, *args, **kwargs):
        calls.append((tickers, kwargs.get("threads")))
        if isinstance(tickers, list):
            return batch_df
        if tickers == "MSFT":
            return msft_df
        return pd.DataFrame()

    monkeypatch.setattr(outcome_resolver.yf, "download", fake_download)

    result = outcome_resolver._fetch_prices_batch(["AAPL", "MSFT"])

    assert "AAPL" in result
    assert "MSFT" in result
    assert len(result["MSFT"]) == 2
    assert all(t is False for _, t in calls)
