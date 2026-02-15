"""FRED / macro data client — VIX, yield curve, breadth indicators."""

from __future__ import annotations

from datetime import date, timedelta

import httpx
import pandas as pd

# FRED data is free, no key needed for basic series via yfinance / public endpoints.
# We use yfinance for VIX (^VIX) and treasury yields since FRED API requires registration.
# For broader macro data we hit the FRED public JSON endpoint.

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Key series IDs
SERIES = {
    "vix": "VIXCLS",
    "yield_10y": "DGS10",
    "yield_2y": "DGS2",
    "yield_spread": "T10Y2Y",  # 10Y-2Y spread
    "fed_funds": "FEDFUNDS",
    "initial_claims": "ICSA",
}


class FREDClient:
    def __init__(self, api_key: str | None = None):
        self._api_key = api_key  # Optional — we also use yfinance fallback

    async def get_series(
        self,
        series_id: str,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> pd.DataFrame:
        """Fetch a FRED series. Falls back to empty if no API key."""
        if not self._api_key:
            return await self._yfinance_fallback(series_id, from_date, to_date)

        if from_date is None:
            from_date = date.today() - timedelta(days=365)
        if to_date is None:
            to_date = date.today()

        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "observation_start": str(from_date),
            "observation_end": str(to_date),
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(FRED_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()

        observations = data.get("observations", [])
        if not observations:
            return pd.DataFrame()

        df = pd.DataFrame(observations)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["date", "value"]].dropna().reset_index(drop=True)
        return df

    async def _yfinance_fallback(
        self, series_id: str, from_date: date | None, to_date: date | None
    ) -> pd.DataFrame:
        """Use yfinance for VIX when no FRED key available."""
        import yfinance as yf
        from concurrent.futures import ThreadPoolExecutor
        import asyncio

        yf_map = {
            "VIXCLS": "^VIX",
            "DGS10": "^TNX",
            "DGS2": "^IRX",
        }
        yf_ticker = yf_map.get(series_id)
        if not yf_ticker:
            return pd.DataFrame()

        start = str(from_date or date.today() - timedelta(days=365))
        end = str(to_date or date.today())

        def _fetch():
            tk = yf.Ticker(yf_ticker)
            df = tk.history(start=start, end=end)
            if df.empty:
                return pd.DataFrame()
            df = df.reset_index()
            df["date"] = pd.to_datetime(df["Date"]).dt.date
            df["value"] = df["Close"]
            return df[["date", "value"]].reset_index(drop=True)

        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        return await loop.run_in_executor(executor, _fetch)

    async def get_vix(self, lookback_days: int = 90) -> pd.DataFrame:
        from_date = date.today() - timedelta(days=lookback_days)
        return await self.get_series(SERIES["vix"], from_date)

    async def get_yield_curve(self, lookback_days: int = 90) -> pd.DataFrame:
        """Return 10Y-2Y spread."""
        from_date = date.today() - timedelta(days=lookback_days)
        return await self.get_series(SERIES["yield_spread"], from_date)

    async def get_macro_snapshot(self) -> dict:
        """Fetch latest values for key macro indicators."""
        vix = await self.get_vix(lookback_days=5)
        yield_curve = await self.get_yield_curve(lookback_days=5)
        return {
            "vix": float(vix["value"].iloc[-1]) if not vix.empty else None,
            "yield_spread_10y2y": float(yield_curve["value"].iloc[-1]) if not yield_curve.empty else None,
        }
