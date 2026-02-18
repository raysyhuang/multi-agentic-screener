"""KooCore-D adapter — transforms /api/picks into EngineResultPayload.

KooCore exposes ticker lists by strategy (weekly, pro30, movers) but no
trade parameters.  This adapter enriches each ticker with a current price
from Polygon and computes default stop/target/confidence per strategy.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, datetime

import aiohttp

from src.config import get_settings
from src.contracts import EnginePick, EngineResultPayload

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy defaults
# ---------------------------------------------------------------------------

_STRATEGY_DEFAULTS: dict[str, dict] = {
    "weekly": {
        "strategy": "swing",
        "holding_period_days": 7,
        "base_confidence": 60,
        "risk_pct": 0.05,       # 5% stop
        "reward_pct": 0.10,     # 10% target → 2:1 R:R
    },
    "pro30": {
        "strategy": "momentum",
        "holding_period_days": 30,
        "base_confidence": 55,
        "risk_pct": 0.08,       # 8% stop
        "reward_pct": 0.18,     # 18% target → ~2.25:1 R:R
    },
    "movers": {
        "strategy": "breakout",
        "holding_period_days": 5,
        "base_confidence": 50,
        "risk_pct": 0.04,       # 4% stop
        "reward_pct": 0.08,     # 8% target → 2:1 R:R
    },
}

# Confidence bonus for being ranked earlier in a strategy list
_RANK_BONUS_PER_SLOT = 3  # e.g., 1st pick gets +6, 2nd gets +3, 3rd gets 0


# ---------------------------------------------------------------------------
# Price enrichment via Polygon snapshot
# ---------------------------------------------------------------------------

async def _get_price_polygon(
    session: aiohttp.ClientSession,
    ticker: str,
    api_key: str,
) -> float | None:
    """Fetch the previous close from Polygon snapshot. Returns None on failure."""
    url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
    try:
        async with session.get(
            url,
            params={"apiKey": api_key},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                logger.debug("Polygon snapshot %s HTTP %d", ticker, resp.status)
                return None
            data = await resp.json()
            ticker_data = data.get("ticker", {})
            # prevDay.c = previous close; day.c = current day close (may be 0 pre-market)
            price = (
                ticker_data.get("day", {}).get("c")
                or ticker_data.get("prevDay", {}).get("c")
                or ticker_data.get("lastTrade", {}).get("p")
            )
            return float(price) if price and float(price) > 0 else None
    except Exception as e:
        logger.debug("Polygon snapshot failed for %s: %s", ticker, e)
        return None


async def _get_prices_batch(
    session: aiohttp.ClientSession,
    tickers: list[str],
    api_key: str,
) -> dict[str, float]:
    """Fetch prices for multiple tickers. Returns {ticker: price} for successes."""
    import asyncio

    tasks = {
        t: asyncio.create_task(_get_price_polygon(session, t, api_key))
        for t in tickers
    }
    results: dict[str, float] = {}
    for ticker, task in tasks.items():
        price = await task
        if price is not None:
            results[ticker] = price
        else:
            logger.warning("No price for %s — excluding from KooCore picks", ticker)
    return results


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

async def fetch_koocore(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    timeout_s: float = 45.0,
    max_attempts: int = 2,
) -> EngineResultPayload | None:
    """Fetch KooCore /api/picks and transform into EngineResultPayload."""
    picks_url = f"{base_url.rstrip('/')}/api/picks"
    start = time.monotonic()

    # 1. Fetch raw picks from KooCore
    raw: dict | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            async with session.get(
                picks_url,
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(
                        "KooCore /api/picks HTTP %d (attempt %d/%d): %s",
                        resp.status, attempt, max_attempts, body[:200],
                    )
                    if attempt < max_attempts:
                        await asyncio.sleep(1.0 * attempt)
                        continue
                    return None
                raw = await resp.json()
                break
        except asyncio.TimeoutError:
            logger.warning(
                "KooCore /api/picks timed out after %.0fs (attempt %d/%d)",
                timeout_s, attempt, max_attempts,
            )
            if attempt < max_attempts:
                await asyncio.sleep(1.0 * attempt)
                continue
            return None
        except Exception as e:
            logger.warning(
                "KooCore fetch failed (attempt %d/%d): %s: %s",
                attempt, max_attempts, type(e).__name__, e,
            )
            if attempt < max_attempts:
                await asyncio.sleep(1.0 * attempt)
                continue
            return None
    if raw is None:
        return None

    # 2. Parse picks_data — format: {"picks_data": {"2026-02-17": {"movers": [...], ...}}}
    picks_data = raw.get("picks_data", {})
    if not picks_data:
        logger.warning("KooCore returned empty picks_data")
        return None

    # Use most recent date's picks
    latest_date = max(picks_data.keys())
    day_picks = picks_data[latest_date]

    # Collect all unique tickers across strategies
    strategy_tickers: dict[str, list[str]] = {}
    all_tickers: set[str] = set()
    for strategy_name in ("weekly", "pro30", "movers"):
        tickers = day_picks.get(strategy_name, [])
        if tickers:
            strategy_tickers[strategy_name] = tickers
            all_tickers.update(tickers)

    if not all_tickers:
        logger.info("KooCore has no picks for %s", latest_date)
        elapsed = time.monotonic() - start
        return EngineResultPayload(
            engine_name="koocore_d",
            engine_version="adapter-v1",
            run_date=latest_date,
            run_timestamp=datetime.utcnow().isoformat(),
            picks=[],
            candidates_screened=0,
            pipeline_duration_s=elapsed,
            status="success",
        )

    # 3. Enrich with prices
    polygon_key = get_settings().polygon_api_key
    prices = await _get_prices_batch(session, list(all_tickers), polygon_key)

    # 4. Build EnginePick list
    engine_picks: list[EnginePick] = []
    seen: set[str] = set()  # avoid duplicates if ticker appears in multiple strategies

    for strategy_name, tickers in strategy_tickers.items():
        defaults = _STRATEGY_DEFAULTS.get(strategy_name)
        if not defaults:
            continue

        for rank, ticker in enumerate(tickers):
            if ticker in seen:
                continue
            seen.add(ticker)

            price = prices.get(ticker)
            if price is None:
                continue

            risk_pct = defaults["risk_pct"]
            reward_pct = defaults["reward_pct"]
            stop_loss = round(price * (1 - risk_pct), 2)
            target_price = round(price * (1 + reward_pct), 2)

            # Confidence: base + rank bonus (earlier = higher)
            rank_bonus = max(0, (2 - rank)) * _RANK_BONUS_PER_SLOT
            confidence = min(100, defaults["base_confidence"] + rank_bonus)

            engine_picks.append(EnginePick(
                ticker=ticker,
                strategy=defaults["strategy"],
                entry_price=round(price, 2),
                stop_loss=stop_loss,
                target_price=target_price,
                confidence=confidence,
                holding_period_days=defaults["holding_period_days"],
                thesis=f"KooCore {strategy_name} pick (rank #{rank + 1})",
                risk_factors=[],
                raw_score=None,
                metadata={
                    "koocore_strategy": strategy_name,
                    "rank_in_strategy": rank + 1,
                },
            ))

    elapsed = time.monotonic() - start
    logger.info(
        "KooCore adapter: %d picks from %d strategies (%.1fs)",
        len(engine_picks), len(strategy_tickers), elapsed,
    )

    return EngineResultPayload(
        engine_name="koocore_d",
        engine_version="adapter-v1",
        run_date=latest_date,
        run_timestamp=datetime.utcnow().isoformat(),
        picks=engine_picks,
        candidates_screened=len(all_tickers),
        pipeline_duration_s=elapsed,
        status="success",
    )
