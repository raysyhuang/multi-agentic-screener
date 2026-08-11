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
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.config import get_settings
from src.data.polygon_client import PolygonClient
from src.data.fmp_client import FMPClient
from src.data.yfinance_client import YFinanceClient
from src.data.fred_client import FREDClient
from src.data.mcp_client import MCPClient
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

# Benchmarks fetched by the macro snapshot; their provenance is stored with
# the cached snapshot so a hit can replay it (see get_macro_context).
_BENCHMARK_TICKERS = ("SPY", "QQQ")

# Cap on failed tickers listed in the provenance record. The count is reported
# separately and is never truncated.
_MAX_REPORTED_FAILURES = 50


class DataAggregator:
    """Orchestrates data fetching across all providers with fallback logic."""

    def __init__(self):
        settings = get_settings()
        self.polygon = PolygonClient()
        self.fmp = FMPClient()
        self.yfinance = YFinanceClient()
        self.fred = FREDClient(api_key=settings.fred_api_key or None)
        self.mcp = MCPClient() if settings.mcp_enabled else None
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        self._cache = DataCache()
        self._cache_enabled = True
        self._circuit_breaker = APICircuitBreaker()
        # Which provider actually served each bar this run. The fallback chain
        # is Polygon -> FMP -> yfinance and it is silent: a Polygon outage
        # degrades the whole run to yfinance data with nothing in the record to
        # say so. `get_last_ohlcv_provenance()` covers the research path only
        # (src/research/signal_backtest.py); this is its live-path counterpart.
        self._ohlcv_sources: Counter[str] = Counter()
        self._ohlcv_failures: list[str] = []
        self._ohlcv_cache_hits: int = 0
        # ticker -> provider that served it ("" = failed). Lets the macro
        # snapshot attribute SPY/QQQ without diffing a shared counter.
        self._ohlcv_ticker_source: dict[str, str] = {}
        self._universe_source: str = ""
        self._universe_errors: list[str] = []
        self._universe_cache_hit: bool = False
        self._macro_source: str = ""
        self._macro_cache_hit: bool = False
        # Latched: a provider whose breaker opened at ANY point in the run stays
        # here. The breaker's own cooldown is 5 minutes, and scoring runs longer
        # than that, so reading current state when the record is written would
        # report an empty set for a run that spent its universe fetch bypassing
        # Polygon entirely — a disclosure field denying the outage it exists to
        # disclose.
        self._circuits_opened: set[str] = set()

    async def get_ohlcv(
        self,
        ticker: str,
        from_date: date,
        to_date: date,
    ) -> pd.DataFrame:
        """Fetch OHLCV with fallback chain: Polygon -> FMP -> yfinance.

        Cache is checked before acquiring the semaphore.
        """
        key = None
        if self._cache_enabled:
            key = DataCache.build_key(
                "ohlcv", ticker, "daily",
                from_date=str(from_date), to_date=str(to_date),
            )
            cached, cached_source = self._cache.get_with_source(key)
            if cached is not None:
                try:
                    df = json_to_df(cached)
                    # Attribute to the provider that ORIGINALLY served the row —
                    # a hit is a delivery mechanism, not a data source. Hits are
                    # counted separately so cache reliance stays visible too.
                    self._ohlcv_sources[cached_source or "unknown"] += 1
                    self._ohlcv_ticker_source[ticker] = cached_source or "unknown"
                    self._ohlcv_cache_hits += 1
                    return df
                except Exception as e:
                    logger.warning("Cache deserialization failed for %s, treating as miss: %s", ticker, e)

        async with self._semaphore:
            if not self._circuit_breaker.is_open("polygon"):
                try:
                    df = await self.polygon.get_ohlcv(ticker, from_date, to_date)
                    if not df.empty:
                        self._circuit_breaker.record_success("polygon")
                        self._ohlcv_sources["polygon"] += 1
                        self._ohlcv_ticker_source[ticker] = "polygon"
                        self._store_ohlcv(key, df, to_date, source="polygon", ticker=ticker)
                        return df
                except Exception as e:
                    self._circuit_breaker.record_failure("polygon")
                    self._latch_circuit("polygon")
                    logger.warning("Polygon OHLCV failed for %s: %s", ticker, e)
            else:
                self._circuits_opened.add("polygon")
                logger.debug("Polygon circuit open, skipping for %s", ticker)

            if not self._circuit_breaker.is_open("fmp"):
                try:
                    df = await self.fmp.get_daily_prices(ticker, from_date, to_date)
                    if not df.empty:
                        self._circuit_breaker.record_success("fmp")
                        self._ohlcv_sources["fmp"] += 1
                        self._ohlcv_ticker_source[ticker] = "fmp"
                        self._store_ohlcv(key, df, to_date, source="fmp", ticker=ticker)
                        return df
                except Exception as e:
                    self._circuit_breaker.record_failure("fmp")
                    self._latch_circuit("fmp")
                    logger.warning("FMP OHLCV failed for %s: %s", ticker, e)
            else:
                self._circuits_opened.add("fmp")
                logger.debug("FMP circuit open, skipping for %s", ticker)

            try:
                df = await self.yfinance.get_ohlcv(ticker, from_date, to_date)
                if df.empty:
                    self._ohlcv_failures.append(ticker)
                    self._ohlcv_ticker_source[ticker] = ""
                else:
                    self._ohlcv_sources["yfinance"] += 1
                    self._ohlcv_ticker_source[ticker] = "yfinance"
                    self._store_ohlcv(key, df, to_date, source="yfinance", ticker=ticker)
                return df
            except Exception as e:
                logger.error("All OHLCV sources failed for %s: %s", ticker, e)
                self._ohlcv_failures.append(ticker)
                self._ohlcv_ticker_source[ticker] = ""
                return pd.DataFrame()

    def _latch_circuit(self, provider: str) -> None:
        """Record that this failure tripped the breaker, at the moment it did.

        Checked immediately after `record_failure` because that is the only
        instant the transition is observable — the cooldown expires well inside
        a single run, so asking later gets "closed" for a provider that was cut
        out of most of it.
        """
        if self._circuit_breaker.is_open(provider):
            self._circuits_opened.add(provider)

    def _store_ohlcv(
        self,
        key: str | None,
        df: pd.DataFrame,
        to_date: date,
        *,
        source: str,
        ticker: str,
    ) -> None:
        """Persist a fetched frame. Never raises.

        A cache write is bookkeeping about a fetch that has already succeeded.
        Performing it inside the provider's `try` made a failed serialization or
        a locked SQLite file indistinguishable from the provider being down: the
        handler recorded a circuit failure, execution fell through to the next
        provider, and that one incremented the counter too. One ticker ended up
        attributed to two providers and the totals exceeded the ticker count —
        corrupting precisely the record this is meant to make trustworthy.
        """
        if not self._cache_enabled or key is None:
            return
        try:
            self._cache.put(
                key, df_to_json(df), classify_ohlcv_ttl(to_date),
                source=source, ticker=ticker, endpoint="ohlcv",
            )
        except Exception as e:
            logger.warning("OHLCV cache write failed for %s (%s): %s", ticker, source, e)

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
                    # get_ohlcv swallows provider errors itself, so reaching here
                    # means the task died outside that handling (cancellation,
                    # semaphore teardown, a bug). Without this the ticker is
                    # dropped to an empty frame and the provenance record claims
                    # nothing failed.
                    self._ohlcv_failures.append(ticker)
                    self._ohlcv_ticker_source[ticker] = ""
                    out[ticker] = pd.DataFrame()
                else:
                    out[ticker] = result

            # Memory cleanup between batches
            if total_batches > 1:
                gc.collect()

        logger.info("Bulk OHLCV complete: %d/%d tickers fetched", len(out), len(tickers))
        return out

    def get_data_provenance(self) -> dict:
        """Which providers actually served this run's data.

        The OHLCV fallback chain (Polygon -> FMP -> yfinance) is silent, so a
        provider outage changes the data underneath a run without changing
        anything visible in its output. Recording the tally makes a degraded run
        distinguishable from a healthy one after the fact, which is the whole
        point of the provenance rule in CLAUDE.md.
        """
        return {
            # Keyed by the provider that ORIGINALLY served each bar, cache hits
            # included — a hit is a delivery mechanism, not a data source.
            "ohlcv_by_source": dict(self._ohlcv_sources),
            "ohlcv_cache_hits": self._ohlcv_cache_hits,
            # Bounded: a total provider outage fails every ticker, and
            # max_ohlcv_tickers is 1000, so the unbounded list could put a
            # multi-kilobyte array into every governance artifact — largest on
            # exactly the runs already in trouble. The count is always exact;
            # the sample is enough to recognise a pattern.
            "ohlcv_failed_tickers": sorted(set(self._ohlcv_failures))[:_MAX_REPORTED_FAILURES],
            "ohlcv_failed_count": len(set(self._ohlcv_failures)),
            "ohlcv_failures_truncated": len(set(self._ohlcv_failures)) > _MAX_REPORTED_FAILURES,
            # "" only before the universe step runs; "unavailable" means every
            # provider failed, which is a different fact and must not read as
            # "not yet attempted".
            "universe_source": self._universe_source,
            "universe_cache_hit": self._universe_cache_hit,
            "universe_errors": list(self._universe_errors),
            # "live" or "cache" — a cached snapshot still drives regime and
            # eligibility, and its SPY/QQQ bars are folded into ohlcv_by_source.
            "macro_source": self._macro_source,
            "macro_cache_hit": self._macro_cache_hit,
            # Latched across the run, not sampled at report time — see
            # _circuits_opened. Named for what it actually asserts.
            "circuits_opened_during_run": sorted(self._circuits_opened),
        }

    def reset_data_provenance(self) -> None:
        """Clear counters so a long-lived aggregator reports per-run figures.

        Call this at the START of a run, before any fetching. Calling it partway
        through erases the evidence for whatever was already fetched — the
        benchmark bars behind the regime and eligibility decisions, for one.
        """
        self._ohlcv_sources.clear()
        self._ohlcv_failures.clear()
        self._ohlcv_cache_hits = 0
        self._ohlcv_ticker_source.clear()
        self._universe_source = ""
        self._universe_errors.clear()
        self._universe_cache_hit = False
        self._macro_source = ""
        self._macro_cache_hit = False
        self._circuits_opened.clear()

    async def get_universe(self) -> list[dict]:
        """Build initial universe from FMP screener, falling back to Polygon."""
        if self._cache_enabled:
            key = DataCache.build_key("universe", "", "screener")
            cached, cached_source = self._cache.get_with_source(key)
            if cached is not None:
                try:
                    universe = json.loads(cached)
                except Exception as e:
                    # Corrupt cached JSON is a miss, not a pipeline failure —
                    # the same tolerance the OHLCV path already has.
                    logger.warning("Universe cache deserialization failed, treating as miss: %s", e)
                else:
                    # "cache" is not an answer to "where did the universe come
                    # from" — the stored row knows whether FMP or Polygon built it.
                    self._universe_source = cached_source or "unknown"
                    self._universe_cache_hit = True
                    return universe

        # Try FMP first
        try:
            result = await self.fmp.get_stock_screener()
            if result:
                self._universe_source = "fmp"
                if self._cache_enabled:
                    # A failed cache WRITE is not a failed fetch. Letting it
                    # reach the handler below would label a perfectly good FMP
                    # universe a provider failure and throw it away in favour of
                    # the Polygon fallback.
                    try:
                        self._cache.put(key, json.dumps(result), TTL_UNIVERSE, source="fmp", endpoint="universe")
                    except Exception as cache_err:
                        logger.warning("Universe cache write failed: %s", cache_err)
                return result
        except Exception as e:
            logger.warning("FMP screener failed: %s — falling back to Polygon", e)
            self._universe_errors.append(f"fmp: {e}")
        else:
            if not result:
                self._universe_errors.append("fmp: returned no rows")

        # Fallback: Polygon tickers reference + grouped daily bars
        try:
            result = await self._build_polygon_universe()
            if result:
                self._universe_source = "polygon"
                if self._cache_enabled:
                    # Same isolation as the FMP branch above: a failed cache
                    # write here would be caught as a provider failure, flip the
                    # source to "unavailable" and return [] — a false
                    # zero-universe run off a universe that was fetched fine.
                    try:
                        self._cache.put(key, json.dumps(result), TTL_UNIVERSE, source="polygon", endpoint="universe")
                    except Exception as cache_err:
                        logger.warning("Universe cache write failed: %s", cache_err)
                return result
            self._universe_errors.append("polygon: returned no rows")
        except Exception as e:
            logger.error("Polygon universe fallback also failed: %s", e)
            self._universe_errors.append(f"polygon: {e}")

        # Both providers are gone. An empty list with a blank source reads
        # identically to "the universe step has not run yet"; say which it is.
        self._universe_source = "unavailable"
        logger.error("Universe unavailable — every provider failed: %s", self._universe_errors)
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

        # Insider transactions are deliberately NOT fetched (2026-07-27). The only
        # consumer was the disabled catalyst model, and the full-scale IC study
        # (scripts/insider_ic_study.py, 1453 point-in-time observations) found NO
        # predictive edge — every net-ratio and cluster bucket was flat-to-negative
        # vs base rate (>=3 distinct buyers: -116bp/20d), and 336/503 tickers had no
        # usable filings at all. It cost ~150 of the 750/day FMP budget per run for
        # data that reached no scorer. score_insider_activity() is retained for the
        # catalyst path should it ever be revived.
        earnings_task = self.fmp.get_earnings_surprise(ticker)
        profile_task = self.fmp.get_company_profile(ticker)
        analyst_task = self.fmp.get_analyst_estimates(ticker)
        ratios_task = self.fmp.get_ratios(ticker)

        results = await asyncio.gather(
            earnings_task, profile_task, analyst_task, ratios_task,
            return_exceptions=True,
        )
        had_failures = any(isinstance(result, Exception) for result in results)

        earnings = results[0] if not isinstance(results[0], Exception) else []
        profile = results[1] if not isinstance(results[1], Exception) else {}
        analyst_estimates = results[2] if not isinstance(results[2], Exception) else []
        ratios = results[3] if not isinstance(results[3], Exception) else {}

        data = {
            "earnings_surprises": earnings[:4] if earnings else [],
            "insider_transactions": [],  # not fetched — see note above
            "profile": profile,
            "analyst_estimates": analyst_estimates[:8] if analyst_estimates else [],
            "ratios": ratios if isinstance(ratios, dict) else {},
        }

        profile_ok = (
            isinstance(profile, dict)
            and bool(profile.get("symbol") or profile.get("companyName"))
        )
        ratios_ok = isinstance(ratios, dict) and any(v is not None for v in ratios.values())
        # Insider is no longer fetched, so it is not evidence of payload health.
        all_empty = not earnings and not profile_ok and not analyst_estimates and not ratios_ok
        should_cache = not all_empty

        if self._cache_enabled and should_cache:
            self._cache.put(key, json.dumps(data), TTL_FUNDAMENTALS, source="fmp", ticker=ticker, endpoint="fundamentals")
        elif all_empty:
            if had_failures:
                logger.info("Skipping empty fundamentals cache for %s after FMP failures", ticker)
            else:
                logger.info("Skipping empty fundamentals cache for %s (provider returned empty payload)", ticker)

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

    async def get_bulk_news(self, tickers: list[str], per_ticker_limit: int = 20) -> dict[str, list[dict]]:
        """Fetch news for many tickers using FMP bulk endpoint with Polygon fallback."""
        if not tickers:
            return {}

        out: dict[str, list[dict]] = {t: [] for t in tickers}
        uncached: list[str] = []

        if self._cache_enabled:
            for ticker in tickers:
                key = DataCache.build_key("polygon", ticker, "news")
                cached = self._cache.get(key)
                if cached is not None:
                    try:
                        out[ticker] = json.loads(cached)[:per_ticker_limit]
                    except Exception:
                        uncached.append(ticker)
                else:
                    uncached.append(ticker)
        else:
            uncached = list(tickers)

        if not uncached:
            return out

        # Chunk to keep query strings reasonable.
        chunk_size = 75
        for i in range(0, len(uncached), chunk_size):
            chunk = uncached[i:i + chunk_size]
            try:
                bulk = await self.fmp.get_stock_news_bulk(chunk, limit=min(1000, len(chunk) * per_ticker_limit))
                for article in bulk:
                    symbols: list[str] = []
                    symbol = article.get("symbol") or article.get("ticker")
                    if symbol:
                        symbols.append(str(symbol).upper())
                    tickers_field = article.get("tickers")
                    if isinstance(tickers_field, list):
                        symbols.extend(str(t).upper() for t in tickers_field if t)

                    for sym in symbols:
                        if sym in out and len(out[sym]) < per_ticker_limit:
                            out[sym].append(article)
            except Exception as e:
                logger.warning("FMP bulk news failed for %d tickers: %s", len(chunk), e)

        # Fallback to Polygon where bulk returned nothing.
        fallback = [t for t in uncached if not out.get(t)]
        if fallback:
            tasks = [self.get_ticker_news(t) for t in fallback]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for ticker, result in zip(fallback, results):
                if isinstance(result, Exception):
                    continue
                out[ticker] = (result or [])[:per_ticker_limit]

        if self._cache_enabled:
            for ticker, articles in out.items():
                if articles:
                    key = DataCache.build_key("polygon", ticker, "news")
                    self._cache.put(key, json.dumps(articles), TTL_NEWS, source="fmp_bulk", ticker=ticker, endpoint="news")

        return out

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
                    # A cached macro snapshot returns without touching the
                    # provenance-aware OHLCV path, so the SPY/QQQ bars that
                    # drive the regime and eligibility decisions would leave no
                    # trace — the record would claim no benchmark data was used
                    # at all. Rehydrate the provenance stored alongside them.
                    self._macro_source = "cache"
                    self._rehydrate_benchmark_provenance(
                        payload.pop("_benchmark_provenance", None)
                    )
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

        # Read per-ticker attribution rather than diffing the shared counter
        # before and after. A diff silently assumes nothing else fetches during
        # the gather; that holds today only because macro runs first, and it
        # would start attributing other tickers' providers to the benchmarks the
        # moment anything ran alongside it. Per-ticker is correct regardless of
        # ordering. "" means the fetch failed, which is itself worth replaying.
        benchmark_provenance = {
            t: self._ohlcv_ticker_source.get(t, "") for t in _BENCHMARK_TICKERS
        }

        # Explicitly clear the cache flag: a malformed cached payload can set it
        # partway through before falling back here, and a run that fetched live
        # must not report itself as cache-served.
        self._macro_source = "live"
        self._macro_cache_hit = False
        macro["spy_prices"] = spy_df
        macro["qqq_prices"] = qqq_df

        if self._cache_enabled:
            # Serialize DataFrames for JSON storage
            cache_payload = {**macro}
            cache_payload["spy_prices"] = df_to_json(spy_df)
            cache_payload["qqq_prices"] = df_to_json(qqq_df)
            cache_payload["_benchmark_provenance"] = benchmark_provenance
            try:
                self._cache.put(key, json.dumps(cache_payload), TTL_MACRO, source="macro", endpoint="snapshot")
            except Exception as e:
                # Last of the "cache bookkeeping decides the pipeline outcome"
                # shapes: an unserializable macro payload would have thrown out
                # of a macro fetch that had already succeeded.
                logger.warning("Macro cache write failed: %s", e)

        return macro

    def _rehydrate_benchmark_provenance(self, stored: dict | None) -> None:
        """Restore SPY/QQQ attribution from a cached macro snapshot.

        ``None`` means the snapshot predates this field and has no attribution
        to restore: record the two benchmarks as "unknown" rather than omitting
        them, since the run genuinely did use benchmark data and silence would
        read as "none was used".

        Anything else is driven by the known benchmark set, never by iterating
        whatever the cache happens to hold. Cached JSON is untrusted input here:
        it can predate a shape change — an earlier revision stored
        ``{"polygon": 2}`` rather than ``{"SPY": "polygon", ...}`` — and
        iterating its items would turn the count ``2`` into a provider label.
        A benchmark with no readable string source is recorded as failed, which
        also gives ``{}`` the meaning the shape implies rather than silently
        recording nothing.

        A benchmark that failed when the snapshot was built is still failed on
        replay: the cached empty frame goes on feeding the regime calculation
        every time, so the failure has to persist with it rather than
        disappearing after the first run.
        """
        if stored is None:
            self._ohlcv_sources["unknown"] += len(_BENCHMARK_TICKERS)
            self._ohlcv_cache_hits += len(_BENCHMARK_TICKERS)
            self._macro_cache_hit = True
            return

        # Not a mapping at all — same treatment as an unreadable source.
        if not isinstance(stored, dict):
            stored = {}

        for ticker in _BENCHMARK_TICKERS:
            source = stored.get(ticker)
            source = source if isinstance(source, str) else ""
            if source:
                self._ohlcv_sources[source] += 1
                self._ohlcv_ticker_source[ticker] = source
                self._ohlcv_cache_hits += 1
            else:
                self._ohlcv_failures.append(ticker)
                self._ohlcv_ticker_source[ticker] = ""

        # Set last: if anything above raised, the caller falls through to a live
        # fetch, and a flag claiming the macro came from cache would contradict
        # the run that actually happened.
        self._macro_cache_hit = True

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

    async def enrich_candidates_via_mcp(
        self,
        tickers: list[str],
        top_n: int | None = None,
    ) -> dict[str, dict]:
        """Enrich top candidates with institutional-grade MCP data.

        Called after the initial screen to add fundamentals, earnings context,
        news, and credit risk from MCP connectors. Only enriches the top N
        candidates to control cost.

        Returns {ticker: enrichment_dict} — empty dict for tickers with no data.
        """
        if not self.mcp or not self.mcp.available:
            return {}

        settings = get_settings()
        limit = top_n or settings.mcp_enrich_top_n
        subset = tickers[:limit]

        if not subset:
            return {}

        logger.info("MCP enrichment: fetching data for %d candidates", len(subset))

        tasks = [self.mcp.enrich_candidate(t) for t in subset]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        enrichments: dict[str, dict] = {}
        for ticker, result in zip(subset, results):
            if isinstance(result, Exception):
                logger.warning("MCP enrichment failed for %s: %s", ticker, result)
                enrichments[ticker] = {}
            else:
                enrichments[ticker] = result

        enriched_count = sum(1 for v in enrichments.values() if v)
        logger.info(
            "MCP enrichment complete: %d/%d candidates enriched",
            enriched_count, len(subset),
        )

        # Cache enriched data
        if self._cache_enabled:
            for ticker, data in enrichments.items():
                if data:
                    key = DataCache.build_key("mcp", ticker, "enrichment")
                    self._cache.put(
                        key, json.dumps(data), TTL_FUNDAMENTALS,
                        source="mcp", ticker=ticker, endpoint="enrichment",
                    )

        return enrichments

    async def get_mcp_macro_enrichment(self) -> dict:
        """Fetch macro enrichment from LSEG via MCP (yield curves, credit spreads).

        Supplements FRED macro data for richer regime detection.
        """
        if not self.mcp or not self.mcp.available:
            return {}

        if self._cache_enabled:
            key = DataCache.build_key("mcp", "", "macro_enrichment")
            cached = self._cache.get(key)
            if cached is not None:
                return json.loads(cached)

        result = await self.mcp.get_macro_enrichment()

        if self._cache_enabled and result:
            key = DataCache.build_key("mcp", "", "macro_enrichment")
            self._cache.put(
                key, json.dumps(result), TTL_MACRO,
                source="mcp_lseg", endpoint="macro_enrichment",
            )

        return result

    def get_mcp_stats(self) -> dict:
        """Return MCP connector stats for monitoring."""
        if not self.mcp:
            return {"enabled": False}
        return {"enabled": True, **self.mcp.get_stats()}

    def get_cache_stats(self) -> dict:
        """Return cache performance statistics."""
        return self._cache.get_stats()

    def close(self) -> None:
        """Release resources held by the aggregator (cache connection, executors)."""
        try:
            self._cache.close()
        except Exception:
            pass
        try:
            self.yfinance.close()
        except Exception:
            pass

    def get_fmp_budget_status(self) -> dict:
        """Expose FMP call-budget usage for runtime monitoring."""
        return self.fmp.get_budget_status()

    def get_fmp_endpoint_status(self) -> dict:
        """Expose FMP endpoint availability state for diagnostics/UI."""
        return self.fmp.get_endpoint_status()

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
