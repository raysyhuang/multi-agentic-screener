"""Polygon.io data client — OHLCV, options flow, and news."""

from __future__ import annotations

from datetime import date, timedelta

import httpx
import pandas as pd

from src.config import get_settings

BASE_URL = "https://api.polygon.io"


class PolygonClient:
    def __init__(self):
        self._api_key = get_settings().polygon_api_key

    def _params(self, **kwargs) -> dict:
        return {"apiKey": self._api_key, **kwargs}

    async def get_ohlcv(
        self,
        ticker: str,
        from_date: date,
        to_date: date,
        timespan: str = "day",
    ) -> pd.DataFrame:
        """Fetch daily OHLCV bars."""
        url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/{timespan}/{from_date}/{to_date}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=self._params(adjusted="true", limit=5000))
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.rename(columns={
            "o": "open", "h": "high", "l": "low", "c": "close",
            "v": "volume", "vw": "vwap", "t": "timestamp", "n": "trades",
        })
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        df = df.sort_values("date").reset_index(drop=True)
        return df

    async def get_options_flow(self, ticker: str, on_date: date) -> list[dict]:
        """Fetch options contracts for unusual activity detection."""
        url = f"{BASE_URL}/v3/reference/options/contracts"
        params = self._params(
            underlying_ticker=ticker,
            expiration_date_gte=str(on_date),
            expiration_date_lte=str(on_date + timedelta(days=45)),
            limit=100,
        )
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        return data.get("results", [])

    async def get_news(self, ticker: str, limit: int = 20) -> list[dict]:
        """Fetch recent news articles for a ticker."""
        url = f"{BASE_URL}/v2/reference/news"
        params = self._params(ticker=ticker, limit=limit, order="desc")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        return data.get("results", [])

    async def get_all_tickers(self, market: str = "stocks") -> list[dict]:
        """Fetch full ticker list for universe construction."""
        url = f"{BASE_URL}/v3/reference/tickers"
        params = self._params(market=market, active="true", limit=1000)
        all_tickers = []
        async with httpx.AsyncClient(timeout=30) as client:
            while url:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                all_tickers.extend(data.get("results", []))
                url = data.get("next_url")
                params = {"apiKey": self._api_key} if url else {}
        return all_tickers

    async def get_snapshot(self, ticker: str) -> dict:
        """Get current-day snapshot (prev close, today's OHLCV, etc.)."""
        url = f"{BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=self._params())
            resp.raise_for_status()
            data = resp.json()
        return data.get("ticker", {})
