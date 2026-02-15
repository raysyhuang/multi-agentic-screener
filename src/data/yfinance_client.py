"""yfinance fallback client — price data when primary sources fail."""

from __future__ import annotations

from datetime import date
from concurrent.futures import ThreadPoolExecutor
import asyncio

import pandas as pd
import yfinance as yf


class YFinanceClient:
    """Synchronous yfinance wrapped for async usage via executor."""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _fetch_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        tk = yf.Ticker(ticker)
        df = tk.history(start=start, end=end, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    async def get_ohlcv(
        self, ticker: str, from_date: date, to_date: date
    ) -> pd.DataFrame:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._fetch_history,
            ticker,
            str(from_date),
            str(to_date),
        )

    def _fetch_info(self, ticker: str) -> dict:
        tk = yf.Ticker(ticker)
        return dict(tk.info)

    async def get_info(self, ticker: str) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._fetch_info, ticker)

    def _fetch_bulk(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        df = yf.download(tickers, start=start, end=end, auto_adjust=True, threads=True)
        return df

    async def get_bulk_ohlcv(
        self, tickers: list[str], from_date: date, to_date: date
    ) -> pd.DataFrame:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._fetch_bulk,
            tickers,
            str(from_date),
            str(to_date),
        )
