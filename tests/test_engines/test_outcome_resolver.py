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
