"""Financial Modeling Prep client — fundamentals, earnings, insider transactions."""

from __future__ import annotations

from datetime import date

import httpx
import pandas as pd

from src.config import get_settings

BASE_URL = "https://financialmodelingprep.com/api/v3"


class FMPClient:
    def __init__(self):
        self._api_key = get_settings().fmp_api_key

    def _params(self, **kwargs) -> dict:
        return {"apikey": self._api_key, **kwargs}

    async def get_earnings_calendar(
        self, from_date: date, to_date: date
    ) -> list[dict]:
        """Upcoming earnings dates."""
        url = f"{BASE_URL}/earning_calendar"
        params = self._params(**{"from": str(from_date), "to": str(to_date)})
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
        return resp.json()

    async def get_earnings_surprise(self, ticker: str) -> list[dict]:
        """Historical earnings surprises."""
        url = f"{BASE_URL}/earnings-surprises/{ticker}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=self._params())
            resp.raise_for_status()
        return resp.json()

    async def get_insider_trading(self, ticker: str, limit: int = 50) -> list[dict]:
        """Recent insider transactions."""
        url = f"{BASE_URL}/insider-trading"
        params = self._params(symbol=ticker, limit=limit)
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
        return resp.json()

    async def get_institutional_holders(self, ticker: str) -> list[dict]:
        """Institutional ownership data."""
        url = f"{BASE_URL}/institutional-holder/{ticker}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=self._params())
            resp.raise_for_status()
        return resp.json()

    async def get_company_profile(self, ticker: str) -> dict:
        """Company profile with sector, market cap, etc."""
        url = f"{BASE_URL}/profile/{ticker}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=self._params())
            resp.raise_for_status()
            data = resp.json()
        return data[0] if data else {}

    async def get_stock_screener(
        self,
        market_cap_more_than: int = 300_000_000,
        volume_more_than: int = 500_000,
        price_more_than: float = 5.0,
        exchange: str = "NYSE,NASDAQ",
        limit: int = 5000,
    ) -> list[dict]:
        """Screen stocks by basic criteria — used for universe construction."""
        url = f"{BASE_URL}/stock-screener"
        params = self._params(
            marketCapMoreThan=market_cap_more_than,
            volumeMoreThan=volume_more_than,
            priceMoreThan=price_more_than,
            exchange=exchange,
            limit=limit,
        )
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
        return resp.json()

    async def get_key_metrics(self, ticker: str, period: str = "annual") -> list[dict]:
        """Key financial metrics (P/E, EV/EBITDA, etc.)."""
        url = f"{BASE_URL}/key-metrics/{ticker}"
        params = self._params(period=period)
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
        return resp.json()

    async def get_daily_prices(
        self, ticker: str, from_date: date, to_date: date
    ) -> pd.DataFrame:
        """Historical daily prices as DataFrame."""
        url = f"{BASE_URL}/historical-price-full/{ticker}"
        params = self._params(**{"from": str(from_date), "to": str(to_date)})
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        historical = data.get("historical", [])
        if not historical:
            return pd.DataFrame()

        df = pd.DataFrame(historical)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.sort_values("date").reset_index(drop=True)
        return df
