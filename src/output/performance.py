"""Outcome tracking — records prediction vs reality, daily P&L checks."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import select, update

from src.db.models import Signal, Outcome
from src.db.session import get_session
from src.data.aggregator import DataAggregator

logger = logging.getLogger(__name__)


async def check_open_positions() -> list[dict]:
    """Check all open positions against current prices.

    Called daily at 4:30 PM ET to update outcomes.
    """
    aggregator = DataAggregator()
    updates = []

    async with get_session() as session:
        # Get all open outcomes
        result = await session.execute(
            select(Outcome).where(Outcome.still_open == True)
        )
        open_outcomes = result.scalars().all()

        if not open_outcomes:
            logger.info("No open positions to check")
            return []

        for outcome in open_outcomes:
            try:
                update_data = await _evaluate_position(outcome, aggregator)
                if update_data:
                    for key, value in update_data.items():
                        setattr(outcome, key, value)
                    updates.append({
                        "ticker": outcome.ticker,
                        "pnl_pct": update_data.get("pnl_pct", 0),
                        "exit_reason": update_data.get("exit_reason", "open"),
                        "still_open": update_data.get("still_open", True),
                    })
            except Exception as e:
                logger.error("Failed to evaluate %s: %s", outcome.ticker, e)

    logger.info("Updated %d positions", len(updates))
    return updates


async def _evaluate_position(outcome: Outcome, aggregator: DataAggregator) -> dict | None:
    """Evaluate a single open position against current market data."""
    # Get the original signal for stop/target
    async with get_session() as session:
        result = await session.execute(
            select(Signal).where(Signal.id == outcome.signal_id)
        )
        signal = result.scalar_one_or_none()

    if not signal:
        return None

    # Fetch latest price data
    to_date = date.today()
    from_date = outcome.entry_date
    df = await aggregator.get_ohlcv(outcome.ticker, from_date, to_date)

    if df.empty:
        return None

    latest = df.iloc[-1]
    current_price = float(latest["close"])
    entry_price = outcome.entry_price

    # Track high/low since entry
    since_entry = df[df["date"] >= outcome.entry_date]
    max_price = float(since_entry["high"].max())
    min_price = float(since_entry["low"].min())

    pnl_pct = (current_price - entry_price) / entry_price * 100
    max_favorable = (max_price - entry_price) / entry_price * 100
    max_adverse = (min_price - entry_price) / entry_price * 100

    days_held = (to_date - outcome.entry_date).days

    update = {
        "pnl_pct": round(pnl_pct, 4),
        "max_favorable": round(max_favorable, 4),
        "max_adverse": round(max_adverse, 4),
    }

    # Check exit conditions
    if current_price <= signal.stop_loss:
        update.update({
            "exit_price": signal.stop_loss,
            "exit_date": to_date,
            "exit_reason": "stop",
            "still_open": False,
            "pnl_pct": round((signal.stop_loss - entry_price) / entry_price * 100, 4),
        })
    elif current_price >= signal.target_1:
        update.update({
            "exit_price": signal.target_1,
            "exit_date": to_date,
            "exit_reason": "target",
            "still_open": False,
            "pnl_pct": round((signal.target_1 - entry_price) / entry_price * 100, 4),
        })
    elif days_held >= signal.holding_period_days:
        update.update({
            "exit_price": current_price,
            "exit_date": to_date,
            "exit_reason": "expiry",
            "still_open": False,
        })

    # Store daily prices for tracking
    daily_prices = {}
    for _, row in since_entry.iterrows():
        d = row["date"]
        key = str(d) if isinstance(d, date) else str(d)[:10]
        daily_prices[key] = {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
    update["daily_prices"] = daily_prices

    return update


async def record_new_signals(pipeline_results: list[dict]) -> None:
    """Record new approved signals as outcomes for tracking."""
    async with get_session() as session:
        for result in pipeline_results:
            signal_id = result.get("signal_id")
            if not signal_id:
                continue

            outcome = Outcome(
                signal_id=signal_id,
                ticker=result["ticker"],
                entry_date=result.get("entry_date", date.today() + timedelta(days=1)),
                entry_price=result["entry_price"],
                still_open=True,
            )
            session.add(outcome)


async def build_validation_card_from_history(days: int = 90):
    """Build a ValidationCard from recent closed outcomes for live validation.

    Returns None if insufficient data (< 10 closed trades).
    """
    from src.backtest.validation_card import generate_validation_card

    async with get_session() as session:
        cutoff = date.today() - timedelta(days=days)
        result = await session.execute(
            select(Outcome, Signal)
            .join(Signal, Outcome.signal_id == Signal.id)
            .where(Outcome.still_open == False, Outcome.entry_date >= cutoff)
        )
        rows = result.all()

    if len(rows) < 10:
        return None

    trade_returns: list[float] = []
    by_regime: dict[str, list[float]] = {}
    signal_models: set[str] = set()

    for outcome, signal in rows:
        pnl = outcome.pnl_pct or 0.0
        trade_returns.append(pnl)
        by_regime.setdefault(signal.regime.lower(), []).append(pnl)
        signal_models.add(signal.signal_model)

    # Approximate slippage returns: reduce each return by 0.10% (10 bps round-trip)
    slippage_returns = [r - 0.10 for r in trade_returns]

    return generate_validation_card(
        signal_model="aggregate",
        trade_returns=trade_returns,
        trade_returns_by_regime=by_regime,
        slippage_returns=slippage_returns,
        variants_tested=max(1, len(signal_models)),
    )


async def get_performance_summary(days: int = 30) -> dict:
    """Get aggregate performance data for the meta-analyst."""
    async with get_session() as session:
        cutoff = date.today() - timedelta(days=days)

        # Get closed outcomes
        result = await session.execute(
            select(Outcome).where(
                Outcome.still_open == False,
                Outcome.entry_date >= cutoff,
            )
        )
        closed = result.scalars().all()

        if not closed:
            return {"total_signals": 0, "message": "No closed trades in period"}

        # Get associated signals for model info
        signal_ids = [o.signal_id for o in closed]
        sig_result = await session.execute(
            select(Signal).where(Signal.id.in_(signal_ids))
        )
        signals = {s.id: s for s in sig_result.scalars().all()}

        # Aggregate by model
        by_model = {}
        by_regime = {}
        by_confidence_bucket = {}
        all_pnls = []

        for outcome in closed:
            pnl = outcome.pnl_pct or 0
            all_pnls.append(pnl)

            signal = signals.get(outcome.signal_id)
            if signal:
                model = signal.signal_model
                regime = signal.regime
                conf = signal.confidence

                by_model.setdefault(model, []).append(pnl)
                by_regime.setdefault(regime, []).append(pnl)

                bucket = "high" if conf >= 70 else "medium" if conf >= 50 else "low"
                by_confidence_bucket.setdefault(bucket, []).append(pnl)

        def _summary(pnls: list[float]) -> dict:
            if not pnls:
                return {"trades": 0, "win_rate": 0, "avg_pnl": 0}
            wins = sum(1 for p in pnls if p > 0)
            return {
                "trades": len(pnls),
                "win_rate": round(wins / len(pnls), 4),
                "avg_pnl": round(sum(pnls) / len(pnls), 4),
            }

        # Compute full metrics for overall
        from src.backtest.metrics import compute_metrics
        overall_metrics = compute_metrics(all_pnls)

        return {
            "period_days": days,
            "total_signals": len(closed),
            "overall": _summary(all_pnls),
            "risk_metrics": {
                "sharpe_ratio": overall_metrics.sharpe_ratio,
                "sortino_ratio": overall_metrics.sortino_ratio,
                "max_drawdown_pct": overall_metrics.max_drawdown_pct,
                "profit_factor": overall_metrics.profit_factor,
                "calmar_ratio": overall_metrics.calmar_ratio,
                "expectancy": overall_metrics.expectancy,
                "payoff_ratio": overall_metrics.payoff_ratio,
                "avg_win_pct": overall_metrics.avg_win_pct,
                "avg_loss_pct": overall_metrics.avg_loss_pct,
                "max_consecutive_wins": overall_metrics.max_consecutive_wins,
                "max_consecutive_losses": overall_metrics.max_consecutive_losses,
            },
            "by_model": {k: _summary(v) for k, v in by_model.items()},
            "by_regime": {k: _summary(v) for k, v in by_regime.items()},
            "by_confidence": {k: _summary(v) for k, v in by_confidence_bucket.items()},
            "confidence_calibration": _build_calibration(closed, signals),
        }


def _build_calibration(
    outcomes: list, signals_map: dict
) -> list[dict]:
    """Build confidence calibration: actual vs expected win rate by bucket.

    Buckets: 0-40, 40-55, 55-70, 70-85, 85-100.
    Expected win rate is the average confidence in that bucket / 100.
    """
    buckets = [
        (0, 40, "0-40"),
        (40, 55, "40-55"),
        (55, 70, "55-70"),
        (70, 85, "70-85"),
        (85, 101, "85-100"),
    ]
    result = []

    for low, high, label in buckets:
        bucket_pnls = []
        bucket_confs = []
        for outcome in outcomes:
            signal = signals_map.get(outcome.signal_id)
            if not signal:
                continue
            conf = signal.confidence or 0
            if low <= conf < high:
                bucket_pnls.append(outcome.pnl_pct or 0)
                bucket_confs.append(conf)

        if not bucket_pnls:
            continue

        actual_wr = sum(1 for p in bucket_pnls if p > 0) / len(bucket_pnls)
        expected_wr = (sum(bucket_confs) / len(bucket_confs)) / 100.0

        result.append({
            "bucket": label,
            "trades": len(bucket_pnls),
            "actual_win_rate": round(actual_wr, 4),
            "expected_win_rate": round(expected_wr, 4),
            "calibration_error": round(abs(actual_wr - expected_wr), 4),
            "avg_pnl": round(sum(bucket_pnls) / len(bucket_pnls), 4),
        })

    return result
