"""Unified data interface — calls all clients in parallel, merges results."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta

import pandas as pd

from src.config import get_settings
from src.data.polygon_client import PolygonClient
from src.data.fmp_client import FMPClient
from src.data.yfinance_client import YFinanceClient
from src.data.fred_client import FREDClient

logger = logging.getLogger(__name__)


class DataAggregator:
    """Orchestrates data fetching across all providers with fallback logic."""

    def __init__(self):
        settings = get_settings()
        self.polygon = PolygonClient()
        self.fmp = FMPClient()
        self.yfinance = YFinanceClient()
        self.fred = FREDClient(api_key=settings.fred_api_key or None)

    async def get_ohlcv(
        self,
        ticker: str,
        from_date: date,
        to_date: date,
    ) -> pd.DataFrame:
        """Fetch OHLCV with fallback chain: Polygon → FMP → yfinance."""
        try:
            df = await self.polygon.get_ohlcv(ticker, from_date, to_date)
            if not df.empty:
                return df
        except Exception as e:
            logger.warning("Polygon OHLCV failed for %s: %s", ticker, e)

        try:
            df = await self.fmp.get_daily_prices(ticker, from_date, to_date)
            if not df.empty:
                return df
        except Exception as e:
            logger.warning("FMP OHLCV failed for %s: %s", ticker, e)

        try:
            df = await self.yfinance.get_ohlcv(ticker, from_date, to_date)
            return df
        except Exception as e:
            logger.error("All OHLCV sources failed for %s: %s", ticker, e)
            return pd.DataFrame()

    async def get_bulk_ohlcv(
        self,
        tickers: list[str],
        from_date: date,
        to_date: date,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV for many tickers concurrently."""
        tasks = [self.get_ohlcv(t, from_date, to_date) for t in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out = {}
        for ticker, result in zip(tickers, results):
            if isinstance(result, Exception):
                logger.error("Failed to fetch %s: %s", ticker, result)
                out[ticker] = pd.DataFrame()
            else:
                out[ticker] = result
        return out

    async def get_universe(self) -> list[dict]:
        """Build initial universe from FMP screener."""
        try:
            return await self.fmp.get_stock_screener()
        except Exception as e:
            logger.error("FMP screener failed: %s", e)
            return []

    async def get_ticker_fundamentals(self, ticker: str) -> dict:
        """Aggregate fundamental data for a single ticker."""
        earnings_task = self.fmp.get_earnings_surprise(ticker)
        insider_task = self.fmp.get_insider_trading(ticker)
        profile_task = self.fmp.get_company_profile(ticker)

        results = await asyncio.gather(
            earnings_task, insider_task, profile_task,
            return_exceptions=True,
        )

        earnings = results[0] if not isinstance(results[0], Exception) else []
        insiders = results[1] if not isinstance(results[1], Exception) else []
        profile = results[2] if not isinstance(results[2], Exception) else {}

        return {
            "earnings_surprises": earnings[:4] if earnings else [],
            "insider_transactions": insiders[:20] if insiders else [],
            "profile": profile,
        }

    async def get_ticker_news(self, ticker: str) -> list[dict]:
        """Fetch recent news for sentiment scoring."""
        try:
            return await self.polygon.get_news(ticker, limit=20)
        except Exception as e:
            logger.warning("News fetch failed for %s: %s", ticker, e)
            return []

    async def get_macro_context(self) -> dict:
        """Fetch macro indicators for regime detection."""
        # VIX and yield curve
        macro = await self.fred.get_macro_snapshot()

        # SPY and QQQ recent prices for regime detection
        to_date = date.today()
        from_date = to_date - timedelta(days=60)

        spy_task = self.get_ohlcv("SPY", from_date, to_date)
        qqq_task = self.get_ohlcv("QQQ", from_date, to_date)

        spy_df, qqq_df = await asyncio.gather(spy_task, qqq_task)

        macro["spy_prices"] = spy_df
        macro["qqq_prices"] = qqq_df
        return macro

    async def get_upcoming_earnings(self, days_ahead: int = 14) -> list[dict]:
        """Earnings calendar for catalyst detection."""
        from_date = date.today()
        to_date = from_date + timedelta(days=days_ahead)
        try:
            return await self.fmp.get_earnings_calendar(from_date, to_date)
        except Exception as e:
            logger.warning("Earnings calendar failed: %s", e)
            return []
