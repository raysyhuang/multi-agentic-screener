"""Polygon.io data client — OHLCV, options flow, and news."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta

import httpx
import pandas as pd

from src.config import get_settings
from src.data.circuit_breaker import RateLimitError

logger = logging.getLogger(__name__)

BASE_URL = "https://api.polygon.io"

MAX_RETRIES = 3
BACKOFF_BASE = 0.5  # seconds: 0.5, 1.0, 2.0


async def _request_with_backoff(
    client: httpx.AsyncClient, url: str, params: dict,
) -> httpx.Response:
    """Make an HTTP GET with exponential backoff on 429 responses."""
    for attempt in range(MAX_RETRIES):
        resp = await client.get(url, params=params)
        if resp.status_code == 429:
            if attempt < MAX_RETRIES - 1:
                delay = BACKOFF_BASE * (2 ** attempt)
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                logger.warning(
                    "Polygon 429 rate limited (attempt %d/%d) — retrying in %.1fs",
                    attempt + 1, MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)
                continue
            raise RateLimitError("polygon", retry_after=None)
        resp.raise_for_status()
        return resp
    raise RateLimitError("polygon")


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
            resp = await _request_with_backoff(
                client, url, self._params(adjusted="true", limit=5000),
            )
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
            resp = await _request_with_backoff(client, url, params)
            data = resp.json()
        return data.get("results", [])

    async def get_news(self, ticker: str, limit: int = 20) -> list[dict]:
        """Fetch recent news articles for a ticker."""
        url = f"{BASE_URL}/v2/reference/news"
        params = self._params(ticker=ticker, limit=limit, order="desc")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await _request_with_backoff(client, url, params)
            data = resp.json()
        return data.get("results", [])

    async def get_all_tickers(
        self,
        market: str = "stocks",
        ticker_type: str | None = None,
        max_pages: int = 20,
        ticker_gte: str | None = None,
        ticker_lt: str | None = None,
    ) -> list[dict]:
        """Fetch ticker list for universe construction.

        Args:
            market: Market type (default "stocks")
            ticker_type: Filter by type (e.g. "CS" for Common Stock)
            max_pages: Safety limit on pagination (default 20 = 20,000 tickers)
            ticker_gte: Only return tickers >= this value (alphabetical range start)
            ticker_lt: Only return tickers < this value (alphabetical range end)
        """
        url = f"{BASE_URL}/v3/reference/tickers"
        params = self._params(market=market, active="true", limit=1000)
        if ticker_type:
            params["type"] = ticker_type
        if ticker_gte:
            params["ticker.gte"] = ticker_gte
        if ticker_lt:
            params["ticker.lt"] = ticker_lt
        all_tickers = []
        page = 0
        async with httpx.AsyncClient(timeout=30) as client:
            while url and page < max_pages:
                resp = await _request_with_backoff(client, url, params)
                data = resp.json()
                all_tickers.extend(data.get("results", []))
                url = data.get("next_url")
                params = {"apiKey": self._api_key} if url else {}
                page += 1
        return all_tickers

    async def get_grouped_daily(self, on_date: date) -> list[dict]:
        """Fetch grouped daily bars for all US stocks on a given date."""
        url = f"{BASE_URL}/v2/aggs/grouped/locale/us/market/stocks/{on_date}"
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await _request_with_backoff(
                client, url, self._params(adjusted="true"),
            )
            data = resp.json()
        return data.get("results", [])

    async def get_snapshot(self, ticker: str) -> dict:
        """Get current-day snapshot (prev close, today's OHLCV, etc.)."""
        url = f"{BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await _request_with_backoff(client, url, self._params())
            data = resp.json()
        return data.get("ticker", {})
