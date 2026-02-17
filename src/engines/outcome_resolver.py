"""Engine Outcome Resolver — resolves past engine picks and updates credibility.

Runs during afternoon check (4:30 PM ET) and weekly meta-review.
For each engine's unresolved picks:
  1. Fetches current price data
  2. Computes actual return vs. predicted target
  3. Updates the engine_pick_outcomes table
  4. Generates learning feedback per engine
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import yfinance as yf
from sqlalchemy import select

from src.db.models import EnginePickOutcome
from src.db.session import get_session

logger = logging.getLogger(__name__)


async def resolve_engine_outcomes() -> list[dict]:
    """Resolve all unresolved engine pick outcomes.

    Uses a single DB session to avoid detached-instance errors when updating
    objects fetched in a prior session.

    Returns a list of feedback dicts per engine summarizing performance.
    """
    async with get_session() as session:
        result = await session.execute(
            select(EnginePickOutcome).where(
                EnginePickOutcome.outcome_resolved == False  # noqa: E712
            )
        )
        unresolved = result.scalars().all()

        if not unresolved:
            logger.info("No unresolved engine picks to process")
            return []

        logger.info("Resolving %d engine pick outcomes", len(unresolved))

        # Group by ticker for batch price fetching
        tickers = list(set(o.ticker for o in unresolved))
        price_data = _fetch_prices_batch(tickers)

        resolved_count = 0
        updates: list[dict] = []

        for outcome in unresolved:
            today = date.today()
            entry_date = outcome.run_date
            days_since_entry = (today - entry_date).days
            hold_days = outcome.holding_period_days

            # Only resolve if holding period has elapsed
            if days_since_entry < hold_days:
                continue

            prices = price_data.get(outcome.ticker)
            if not prices:
                logger.warning("No price data for %s, skipping", outcome.ticker)
                continue

            # Compute outcome using actual stop_loss from the pick
            resolution = _compute_pick_outcome(
                ticker=outcome.ticker,
                entry_price=outcome.entry_price,
                target_price=outcome.target_price,
                stop_loss=outcome.stop_loss,
                entry_date=entry_date,
                hold_days=hold_days,
                prices=prices,
            )

            # Update in-session — no need to re-fetch since we're in the same session
            outcome.outcome_resolved = True
            outcome.actual_return_pct = resolution["actual_return_pct"]
            outcome.hit_target = resolution["hit_target"]
            outcome.exit_reason = resolution["exit_reason"]
            outcome.days_held = resolution["days_held"]
            outcome.max_favorable_pct = resolution["max_favorable_pct"]
            outcome.max_adverse_pct = resolution["max_adverse_pct"]

            updates.append({
                "engine_name": outcome.engine_name,
                "ticker": outcome.ticker,
                "strategy": outcome.strategy,
                "confidence": outcome.confidence,
                **resolution,
            })
            resolved_count += 1

        await session.commit()

    logger.info("Resolved %d engine pick outcomes", resolved_count)

    # Generate per-engine feedback
    feedback = _generate_feedback(updates)
    return feedback


def _extract_ticker_df(data, ticker: str, single_ticker: bool):
    """Extract a single ticker's DataFrame from yfinance download result.

    Handles both MultiIndex column orderings:
      - (Price, Ticker) — newer yfinance versions
      - (Ticker, Price) — older versions
    Also handles case-insensitive column names.
    """
    if single_ticker:
        return data

    columns = data.columns
    if not isinstance(columns, pd.MultiIndex):
        return None

    level_0_vals = columns.get_level_values(0).unique().tolist()
    level_1_vals = columns.get_level_values(1).unique().tolist()

    # Determine which level contains the ticker
    if ticker in level_1_vals:
        try:
            return data.xs(ticker, level=1, axis=1)
        except KeyError:
            return None
    elif ticker in level_0_vals:
        try:
            return data.xs(ticker, level=0, axis=1)
        except KeyError:
            return None

    # Ticker not found in either level
    return None


def _get_col(row, name: str) -> float:
    """Get column value case-insensitively (handles 'Close' vs 'close')."""
    if name in row.index:
        return float(row[name])
    lower = name.lower()
    for col in row.index:
        if col.lower() == lower:
            return float(row[col])
    raise KeyError(f"Column '{name}' not found in {list(row.index)}")


def _fetch_prices_batch(tickers: list[str]) -> dict[str, list[dict]]:
    """Fetch price history for multiple tickers using yfinance.

    Returns {ticker: [{date, close, high, low}, ...]}.
    """
    if not tickers:
        return {}

    result: dict[str, list[dict]] = {}
    single_ticker = len(tickers) == 1

    try:
        # Fetch 60 days of data to cover any holding period
        data = yf.download(
            tickers,
            period="60d",
            progress=False,
            auto_adjust=True,
            threads=True,
        )
        if data.empty:
            return result

        for ticker in tickers:
            try:
                df = _extract_ticker_df(data, ticker, single_ticker)
                if df is None or df.empty:
                    logger.warning("No data extracted for %s", ticker)
                    continue

                prices = []
                for idx, row in df.iterrows():
                    try:
                        prices.append({
                            "date": idx.date() if hasattr(idx, "date") else idx,
                            "close": _get_col(row, "Close"),
                            "high": _get_col(row, "High"),
                            "low": _get_col(row, "Low"),
                        })
                    except KeyError as e:
                        logger.warning("Missing column for %s on %s: %s", ticker, idx, e)
                        continue
                result[ticker] = prices
            except Exception as e:
                logger.warning("Failed to extract prices for %s: %s", ticker, e)

    except Exception as e:
        logger.warning("yfinance batch download failed: %s", e)

    return result


def _compute_pick_outcome(
    ticker: str,
    entry_price: float,
    target_price: float | None,
    stop_loss: float | None,
    entry_date: date,
    hold_days: int,
    prices: list[dict],
) -> dict:
    """Compute outcome for a single pick."""
    # Guard against zero/invalid entry price
    if not entry_price or entry_price <= 0:
        return {
            "actual_return_pct": 0.0,
            "hit_target": False,
            "exit_reason": "invalid_entry",
            "days_held": 0,
            "max_favorable_pct": 0.0,
            "max_adverse_pct": 0.0,
        }

    # Filter prices to the holding period
    hold_prices = [p for p in prices if p["date"] > entry_date]
    hold_prices = hold_prices[:hold_days]  # limit to holding period

    if not hold_prices:
        return {
            "actual_return_pct": 0.0,
            "hit_target": False,
            "exit_reason": "no_data",
            "days_held": 0,
            "max_favorable_pct": 0.0,
            "max_adverse_pct": 0.0,
        }

    # Compute returns
    exit_price = hold_prices[-1]["close"]
    actual_return_pct = ((exit_price - entry_price) / entry_price) * 100

    # Compute MFE/MAE
    max_high = max(p["high"] for p in hold_prices)
    min_low = min(p["low"] for p in hold_prices)
    max_favorable_pct = ((max_high - entry_price) / entry_price) * 100
    max_adverse_pct = ((min_low - entry_price) / entry_price) * 100

    # Did it hit target?
    hit_target = False
    if target_price and target_price > entry_price:
        hit_target = max_high >= target_price
    elif target_price and target_price < entry_price:
        hit_target = min_low <= target_price

    # Determine exit reason using actual stop_loss from the pick
    stop_return_threshold = -10.0  # default fallback
    if stop_loss and stop_loss > 0:
        stop_return_threshold = ((stop_loss - entry_price) / entry_price) * 100

    if hit_target:
        exit_reason = "target"
    elif max_adverse_pct <= stop_return_threshold:
        exit_reason = "stop"
    else:
        exit_reason = "expiry"

    return {
        "actual_return_pct": round(actual_return_pct, 2),
        "hit_target": hit_target,
        "exit_reason": exit_reason,
        "days_held": len(hold_prices),
        "max_favorable_pct": round(max_favorable_pct, 2),
        "max_adverse_pct": round(max_adverse_pct, 2),
    }


def _generate_feedback(updates: list[dict]) -> list[dict]:
    """Generate per-engine learning feedback from resolved outcomes."""
    from collections import defaultdict

    by_engine: dict[str, list[dict]] = defaultdict(list)
    for u in updates:
        by_engine[u["engine_name"]].append(u)

    feedback: list[dict] = []
    for engine_name, picks in by_engine.items():
        total = len(picks)
        hits = sum(1 for p in picks if p["hit_target"])
        hit_rate = hits / total if total > 0 else 0.0
        avg_return = sum(p["actual_return_pct"] for p in picks) / total if total > 0 else 0.0

        # Per-strategy analysis
        by_strategy: dict[str, list[dict]] = defaultdict(list)
        for p in picks:
            by_strategy[p["strategy"]].append(p)

        strategy_insights: list[str] = []
        for strat, strat_picks in by_strategy.items():
            strat_hits = sum(1 for p in strat_picks if p["hit_target"])
            strat_rate = strat_hits / len(strat_picks) if strat_picks else 0.0
            strat_avg = sum(p["actual_return_pct"] for p in strat_picks) / len(strat_picks)
            strategy_insights.append(
                f"{strat}: {strat_hits}/{len(strat_picks)} hits ({strat_rate:.0%}), "
                f"avg return {strat_avg:+.1f}%"
            )

        notes: list[str] = []
        if hit_rate < 0.3:
            notes.append(f"Low hit rate ({hit_rate:.0%}) — consider tightening filters")
        if avg_return < 0:
            notes.append(f"Negative avg return ({avg_return:+.1f}%) — review strategy")

        feedback.append({
            "engine_name": engine_name,
            "resolved_count": total,
            "hit_rate": round(hit_rate, 3),
            "avg_return_pct": round(avg_return, 2),
            "strategy_breakdown": strategy_insights,
            "notes": notes,
        })

    return feedback
