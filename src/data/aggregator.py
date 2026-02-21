"""Unified data interface — calls all clients in parallel, merges results.

Memory-safe patterns (ported from gemini_STST):
  - Semaphore-controlled concurrency prevents memory explosion from 200+ concurrent requests
  - Batch processing with explicit gc.collect() between batches
  - Configurable batch sizes for different Heroku dyno tiers

Caching:
  - SQLite response cache checked BEFORE acquiring the semaphore (no slot needed for a hit)
  - TTL-based expiry with separate constants for different data types
  - Toggled via _cache_enabled flag
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.config import get_settings
from src.data.polygon_client import PolygonClient
from src.data.fmp_client import FMPClient
from src.data.yfinance_client import YFinanceClient
from src.data.fred_client import FREDClient
from src.data.circuit_breaker import APICircuitBreaker
from src.data.cache import (
    DataCache,
    TTL_FUNDAMENTALS,
    TTL_NEWS,
    TTL_UNIVERSE,
    TTL_MACRO,
    TTL_EARNINGS_CALENDAR,
    classify_ohlcv_ttl,
    df_to_json,
    json_to_df,
)

logger = logging.getLogger(__name__)

# Memory-safe concurrency limits (tuned for Heroku 512 MB)
MAX_CONCURRENCY = 20   # Max simultaneous API requests
OHLCV_BATCH_SIZE = 50  # Tickers per batch in bulk fetch


class DataAggregator:
    """Orchestrates data fetching across all providers with fallback logic."""

    def __init__(self):
        settings = get_settings()
        self.polygon = PolygonClient()
        self.fmp = FMPClient()
        self.yfinance = YFinanceClient()
        self.fred = FREDClient(api_key=settings.fred_api_key or None)
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        self._cache = DataCache()
        self._cache_enabled = True
        self._circuit_breaker = APICircuitBreaker()

    async def get_ohlcv(
        self,
        ticker: str,
        from_date: date,
        to_date: date,
    ) -> pd.DataFrame:
        """Fetch OHLCV with fallback chain: Polygon -> FMP -> yfinance.

        Cache is checked before acquiring the semaphore.
        """
        if self._cache_enabled:
            key = DataCache.build_key(
                "ohlcv", ticker, "daily",
                from_date=str(from_date), to_date=str(to_date),
            )
            cached = self._cache.get(key)
            if cached is not None:
                try:
                    return json_to_df(cached)
                except Exception as e:
                    logger.warning("Cache deserialization failed for %s, treating as miss: %s", ticker, e)

        async with self._semaphore:
            if not self._circuit_breaker.is_open("polygon"):
                try:
                    df = await self.polygon.get_ohlcv(ticker, from_date, to_date)
                    if not df.empty:
                        self._circuit_breaker.record_success("polygon")
                        if self._cache_enabled:
                            ttl = classify_ohlcv_ttl(to_date)
                            self._cache.put(key, df_to_json(df), ttl, source="polygon", ticker=ticker, endpoint="ohlcv")
                        return df
                except Exception as e:
                    self._circuit_breaker.record_failure("polygon")
                    logger.warning("Polygon OHLCV failed for %s: %s", ticker, e)
            else:
                logger.debug("Polygon circuit open, skipping for %s", ticker)

            if not self._circuit_breaker.is_open("fmp"):
                try:
                    df = await self.fmp.get_daily_prices(ticker, from_date, to_date)
                    if not df.empty:
                        self._circuit_breaker.record_success("fmp")
                        if self._cache_enabled:
                            ttl = classify_ohlcv_ttl(to_date)
                            self._cache.put(key, df_to_json(df), ttl, source="fmp", ticker=ticker, endpoint="ohlcv")
                        return df
                except Exception as e:
                    self._circuit_breaker.record_failure("fmp")
                    logger.warning("FMP OHLCV failed for %s: %s", ticker, e)
            else:
                logger.debug("FMP circuit open, skipping for %s", ticker)

            try:
                df = await self.yfinance.get_ohlcv(ticker, from_date, to_date)
                if self._cache_enabled and not df.empty:
                    ttl = classify_ohlcv_ttl(to_date)
                    self._cache.put(key, df_to_json(df), ttl, source="yfinance", ticker=ticker, endpoint="ohlcv")
                return df
            except Exception as e:
                logger.error("All OHLCV sources failed for %s: %s", ticker, e)
                return pd.DataFrame()

    async def get_bulk_ohlcv(
        self,
        tickers: list[str],
        from_date: date,
        to_date: date,
        batch_size: int = OHLCV_BATCH_SIZE,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV for many tickers in memory-safe batches.

        Processes tickers in batches with explicit gc.collect() between batches
        to prevent memory buildup on constrained environments (Heroku 512 MB).
        """
        out: dict[str, pd.DataFrame] = {}
        total_batches = (len(tickers) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(tickers), batch_size):
            batch = tickers[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            logger.info(
                "OHLCV batch %d/%d: fetching %d tickers...",
                batch_num, total_batches, len(batch),
            )

            tasks = [self.get_ohlcv(t, from_date, to_date) for t in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for ticker, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.error("Failed to fetch %s: %s", ticker, result)
                    out[ticker] = pd.DataFrame()
                else:
                    out[ticker] = result

            # Memory cleanup between batches
            if total_batches > 1:
                gc.collect()

        logger.info("Bulk OHLCV complete: %d/%d tickers fetched", len(out), len(tickers))
        return out

    async def get_universe(self) -> list[dict]:
        """Build initial universe from FMP screener, falling back to Polygon."""
        if self._cache_enabled:
            key = DataCache.build_key("universe", "", "screener")
            cached = self._cache.get(key)
            if cached is not None:
                return json.loads(cached)

        # Try FMP first
        try:
            result = await self.fmp.get_stock_screener()
            if result:
                if self._cache_enabled:
                    self._cache.put(key, json.dumps(result), TTL_UNIVERSE, source="fmp", endpoint="universe")
                return result
        except Exception as e:
            logger.warning("FMP screener failed: %s — falling back to Polygon", e)

        # Fallback: Polygon tickers reference + grouped daily bars
        try:
            result = await self._build_polygon_universe()
            if self._cache_enabled and result:
                self._cache.put(key, json.dumps(result), TTL_UNIVERSE, source="polygon", endpoint="universe")
            return result
        except Exception as e:
            logger.error("Polygon universe fallback also failed: %s", e)
            return []

    async def _build_polygon_universe(self) -> list[dict]:
        """Build universe from Polygon reference tickers + grouped daily bars."""
        # MIC code → exchange short name
        # XNGS = NASDAQ Global Select (AAPL, MSFT, NVDA, GOOGL, META, TSLA, etc.)
        # XNCM = NASDAQ Capital Market, XNMS = NASDAQ Global Market
        _EXCHANGE_MAP = {
            "XNYS": "NYSE", "XNAS": "NASDAQ", "XASE": "AMEX",
            "ARCX": "NYSE", "BATS": "NASDAQ",
            "XNGS": "NASDAQ", "XNCM": "NASDAQ", "XNMS": "NASDAQ",
        }

        # Run reference tickers (CS only) and grouped daily in parallel
        async def _fetch_grouped() -> list[dict]:
            today = date.today()
            for offset in range(0, 6):
                try_date = today - timedelta(days=offset)
                try:
                    bars = await self.polygon.get_grouped_daily(try_date)
                    if bars:
                        logger.info("Polygon grouped daily: %d bars for %s", len(bars), try_date)
                        return bars
                except Exception:
                    continue
            return []

        # Alphabet ranges to stay under Starter plan's ~1000 result cap per query
        _RANGES = [
            ("A", "D"), ("D", "G"), ("G", "J"), ("J", "M"),
            ("M", "P"), ("P", "S"), ("S", "V"), ("V", None),
        ]

        ref_chunks, grouped = await asyncio.gather(
            asyncio.gather(*(
                self.polygon.get_all_tickers(
                    market="stocks", ticker_type="CS",
                    ticker_gte=gte, ticker_lt=lt,
                ) for gte, lt in _RANGES
            )),
            _fetch_grouped(),
        )
        ref_tickers = [t for chunk in ref_chunks for t in chunk]

        # Build lookup: ticker → {exchange, market_cap} (only NYSE/NASDAQ)
        ref_map: dict[str, dict] = {}
        for t in ref_tickers:
            ticker = t.get("ticker", "")
            exchange = _EXCHANGE_MAP.get(t.get("primary_exchange", ""), "")
            if exchange in ("NYSE", "NASDAQ"):
                ref_map[ticker] = {
                    "exchange": exchange,
                    "market_cap": t.get("market_cap") or 0,
                    "sic_description": t.get("sic_description") or "",
                }

        mcap_count = sum(1 for v in ref_map.values() if v["market_cap"] > 0)
        # Log exchange breakdown and unmapped exchanges for debugging
        exchange_counts: dict[str, int] = {}
        unmapped: dict[str, int] = {}
        for t in ref_tickers:
            pe = t.get("primary_exchange", "")
            mapped = _EXCHANGE_MAP.get(pe)
            if mapped:
                exchange_counts[mapped] = exchange_counts.get(mapped, 0) + 1
            else:
                unmapped[pe] = unmapped.get(pe, 0) + 1
        logger.info(
            "Polygon reference: %d common stocks on NYSE/NASDAQ (%d with market cap) | breakdown: %s",
            len(ref_map), mcap_count, exchange_counts,
        )
        if unmapped:
            logger.debug("Polygon unmapped exchanges: %s", unmapped)

        if not grouped:
            logger.error("No grouped daily data found in the last 6 days")
            return []

        # Merge: only include tickers that are CS on NYSE/NASDAQ
        universe = []
        for bar in grouped:
            ticker = bar.get("T", "")
            if ticker in ref_map:
                ref = ref_map[ticker]
                universe.append({
                    "symbol": ticker,
                    "price": bar.get("c", 0),
                    "volume": bar.get("v", 0),
                    "marketCap": ref["market_cap"],
                    "exchangeShortName": ref["exchange"],
                    "sector": ref.get("sic_description", ""),
                    "type": "",  # CS → empty (passes the ETF/ETN filter)
                })

        logger.info("Polygon universe fallback: %d tickers built", len(universe))
        return universe

    async def get_ticker_fundamentals(self, ticker: str) -> dict:
        """Aggregate fundamental data for a single ticker."""
        if self._cache_enabled:
            key = DataCache.build_key("fmp", ticker, "fundamentals")
            cached = self._cache.get(key)
            if cached is not None:
                return json.loads(cached)

        earnings_task = self.fmp.get_earnings_surprise(ticker)
        insider_task = self.fmp.get_insider_trading(ticker)
        profile_task = self.fmp.get_company_profile(ticker)

        results = await asyncio.gather(
            earnings_task, insider_task, profile_task,
            return_exceptions=True,
        )
        had_failures = any(isinstance(result, Exception) for result in results)

        earnings = results[0] if not isinstance(results[0], Exception) else []
        insiders = results[1] if not isinstance(results[1], Exception) else []
        profile = results[2] if not isinstance(results[2], Exception) else {}

        data = {
            "earnings_surprises": earnings[:4] if earnings else [],
            "insider_transactions": insiders[:20] if insiders else [],
            "profile": profile,
        }

        all_empty = not earnings and not insiders and not profile
        should_cache = not (had_failures and all_empty)

        if self._cache_enabled and should_cache:
            self._cache.put(key, json.dumps(data), TTL_FUNDAMENTALS, source="fmp", ticker=ticker, endpoint="fundamentals")
        elif had_failures and all_empty:
            logger.info("Skipping empty fundamentals cache for %s after FMP failures", ticker)

        return data

    async def get_ticker_news(self, ticker: str) -> list[dict]:
        """Fetch recent news for sentiment scoring."""
        if self._cache_enabled:
            key = DataCache.build_key("polygon", ticker, "news")
            cached = self._cache.get(key)
            if cached is not None:
                return json.loads(cached)

        try:
            result = await self.polygon.get_news(ticker, limit=20)
            if self._cache_enabled and result:
                self._cache.put(key, json.dumps(result), TTL_NEWS, source="polygon", ticker=ticker, endpoint="news")
            return result
        except Exception as e:
            logger.warning("News fetch failed for %s: %s", ticker, e)
            return []

    async def get_macro_context(self) -> dict:
        """Fetch macro indicators for regime detection."""
        if self._cache_enabled:
            key = DataCache.build_key("macro", "", "snapshot")
            cached = self._cache.get(key)
            if cached is not None:
                try:
                    payload = json.loads(cached)
                    # Restore DataFrames from serialized form
                    payload["spy_prices"] = json_to_df(payload["spy_prices"])
                    payload["qqq_prices"] = json_to_df(payload["qqq_prices"])
                    return payload
                except Exception as e:
                    logger.warning("Macro cache deserialization failed, treating as miss: %s", e)

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

        if self._cache_enabled:
            # Serialize DataFrames for JSON storage
            cache_payload = {**macro}
            cache_payload["spy_prices"] = df_to_json(spy_df)
            cache_payload["qqq_prices"] = df_to_json(qqq_df)
            self._cache.put(key, json.dumps(cache_payload), TTL_MACRO, source="macro", endpoint="snapshot")

        return macro

    async def get_upcoming_earnings(self, days_ahead: int = 14) -> list[dict]:
        """Earnings calendar for catalyst detection."""
        from_date = date.today()
        to_date = from_date + timedelta(days=days_ahead)

        if self._cache_enabled:
            key = DataCache.build_key(
                "fmp", "", "earnings_calendar",
                from_date=str(from_date), to_date=str(to_date),
            )
            cached = self._cache.get(key)
            if cached is not None:
                return json.loads(cached)

        try:
            result = await self.fmp.get_earnings_calendar(from_date, to_date)
            if self._cache_enabled and result:
                self._cache.put(key, json.dumps(result), TTL_EARNINGS_CALENDAR, source="fmp", endpoint="earnings_calendar")
            return result
        except Exception as e:
            logger.warning("Earnings calendar failed: %s", e)
            return []

    def get_cache_stats(self) -> dict:
        """Return cache performance statistics."""
        return self._cache.get_stats()

    async def snapshot_ohlcv(
        self,
        tickers: list[str],
        from_date: date,
        to_date: date,
        output_dir: str | Path = "data/snapshots",
    ) -> Path:
        """Fetch OHLCV for tickers and save as a parquet snapshot for reproducibility.

        Returns the path to the saved snapshot directory.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        tag = f"{from_date}_{to_date}"
        snapshot_dir = output_path / tag
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        data = await self.get_bulk_ohlcv(tickers, from_date, to_date)

        for ticker, df in data.items():
            if not df.empty:
                parquet_path = snapshot_dir / f"{ticker}.parquet"
                df.to_parquet(parquet_path, index=False)

        manifest = {
            "tickers": list(data.keys()),
            "from_date": str(from_date),
            "to_date": str(to_date),
            "count": sum(1 for df in data.values() if not df.empty),
        }
        (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        logger.info(
            "Snapshot saved: %d tickers to %s", manifest["count"], snapshot_dir,
        )
        return snapshot_dir

    @staticmethod
    def load_snapshot(snapshot_dir: str | Path) -> dict[str, pd.DataFrame]:
        """Load a previously-saved OHLCV snapshot from parquet files."""
        snapshot_path = Path(snapshot_dir)
        if not snapshot_path.is_dir():
            raise FileNotFoundError(f"Snapshot directory not found: {snapshot_path}")

        data: dict[str, pd.DataFrame] = {}
        for parquet_file in sorted(snapshot_path.glob("*.parquet")):
            ticker = parquet_file.stem
            data[ticker] = pd.read_parquet(parquet_file)

        logger.info("Loaded snapshot: %d tickers from %s", len(data), snapshot_path)
        return data
