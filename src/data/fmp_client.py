"""Financial Modeling Prep client — fundamentals, earnings, insider transactions."""

from __future__ import annotations

from datetime import date

import httpx
import pandas as pd

from src.config import get_settings

BASE_URL = "https://financialmodelingprep.com/stable"


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
            resp = await client.get(url, params=params)
            resp.raise_for_status()
        return resp.json()

    async def get_earnings_surprise(self, ticker: str) -> list[dict]:
        """Historical earnings data (actual vs estimate)."""
        url = f"{BASE_URL}/earnings"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=self._params(symbol=ticker))
            resp.raise_for_status()
        return resp.json()

    async def get_insider_trading(self, ticker: str, limit: int = 50) -> list[dict]:
        """Recent insider transactions."""
        url = f"{BASE_URL}/insider-trading/search"
        params = self._params(symbol=ticker, limit=limit, page=0)
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
        return resp.json()

    async def get_institutional_holders(self, ticker: str) -> list[dict]:
        """Institutional ownership data."""
        url = f"{BASE_URL}/institutional-ownership/symbol-positions-summary"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=self._params(symbol=ticker))
            resp.raise_for_status()
        return resp.json()

    async def get_company_profile(self, ticker: str) -> dict:
        """Company profile with sector, market cap, etc."""
        url = f"{BASE_URL}/profile"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=self._params(symbol=ticker))
            resp.raise_for_status()
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
            resp = await client.get(url, params=params)
            resp.raise_for_status()
        return resp.json()

    async def get_key_metrics(self, ticker: str, period: str = "annual") -> list[dict]:
        """Key financial metrics (P/E, EV/EBITDA, etc.)."""
        url = f"{BASE_URL}/key-metrics"
        params = self._params(symbol=ticker, period=period)
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
        return resp.json()

    async def get_daily_prices(
        self, ticker: str, from_date: date, to_date: date
    ) -> pd.DataFrame:
        """Historical daily prices as DataFrame."""
        url = f"{BASE_URL}/historical-price-eod/full"
        params = self._params(symbol=ticker, **{"from": str(from_date), "to": str(to_date)})
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
