"""Financial Modeling Prep client — fundamentals, earnings, insider transactions."""

from __future__ import annotations

import asyncio
import logging
from datetime import date

import httpx
import pandas as pd

from src.config import get_settings
from src.data.circuit_breaker import RateLimitError

logger = logging.getLogger(__name__)

BASE_URL = "https://financialmodelingprep.com/stable"

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
                    "FMP 429 rate limited (attempt %d/%d) — retrying in %.1fs",
                    attempt + 1, MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)
                continue
            raise RateLimitError("fmp", retry_after=None)
        resp.raise_for_status()
        return resp
    raise RateLimitError("fmp")


class FMPClient:
    def __init__(self):
        self._api_key = get_settings().fmp_api_key

    def _params(self, **kwargs) -> dict:
        return {"apikey": self._api_key, **kwargs}

    async def get_earnings_calendar(
        self, from_date: date, to_date: date
    ) -> list[dict]:
        """Upcoming earnings dates."""
        url = f"{BASE_URL}/earnings-calendar"
        params = self._params(**{"from": str(from_date), "to": str(to_date)})
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await _request_with_backoff(client, url, params)
        return resp.json()

    async def get_earnings_surprise(self, ticker: str) -> list[dict]:
        """Historical earnings data (actual vs estimate)."""
        url = f"{BASE_URL}/earnings"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await _request_with_backoff(client, url, self._params(symbol=ticker))
        return resp.json()

    async def get_insider_trading(self, ticker: str, limit: int = 50) -> list[dict]:
        """Recent insider transactions."""
        url = f"{BASE_URL}/insider-trading/search"
        params = self._params(symbol=ticker, limit=limit, page=0)
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await _request_with_backoff(client, url, params)
        return resp.json()

    async def get_institutional_holders(self, ticker: str) -> list[dict]:
        """Institutional ownership data."""
        url = f"{BASE_URL}/institutional-ownership/symbol-positions-summary"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await _request_with_backoff(client, url, self._params(symbol=ticker))
        return resp.json()

    async def get_company_profile(self, ticker: str) -> dict:
        """Company profile with sector, market cap, etc."""
        url = f"{BASE_URL}/profile"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await _request_with_backoff(client, url, self._params(symbol=ticker))
            data = resp.json()
        return data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else {})

    async def get_stock_screener(
        self,
        market_cap_more_than: int = 300_000_000,
        volume_more_than: int = 500_000,
        price_more_than: float = 5.0,
        exchange: str = "NYSE,NASDAQ",
        limit: int = 5000,
    ) -> list[dict]:
        """Screen stocks by basic criteria — used for universe construction."""
        url = f"{BASE_URL}/company-screener"
        params = self._params(
            marketCapMoreThan=market_cap_more_than,
            volumeMoreThan=volume_more_than,
            priceMoreThan=price_more_than,
            exchange=exchange,
            limit=limit,
        )
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await _request_with_backoff(client, url, params)
        return resp.json()

    async def get_key_metrics(self, ticker: str, period: str = "annual") -> list[dict]:
        """Key financial metrics (P/E, EV/EBITDA, etc.)."""
        url = f"{BASE_URL}/key-metrics"
        params = self._params(symbol=ticker, period=period)
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await _request_with_backoff(client, url, params)
        return resp.json()

    async def get_daily_prices(
        self, ticker: str, from_date: date, to_date: date
    ) -> pd.DataFrame:
        """Historical daily prices as DataFrame."""
        url = f"{BASE_URL}/historical-price-eod/full"
        params = self._params(symbol=ticker, **{"from": str(from_date), "to": str(to_date)})
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await _request_with_backoff(client, url, params)
            data = resp.json()

        historical = data.get("historical", [])
        if not historical:
            return pd.DataFrame()

        df = pd.DataFrame(historical)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.sort_values("date").reset_index(drop=True)
        return df
