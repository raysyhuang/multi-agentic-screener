"""Outcome tracking — records prediction vs reality, daily P&L checks."""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta

from sqlalchemy import select

import pandas as pd

from src.contracts import HealthState, PositionHealthCard
from src.db.models import Signal, Outcome, PositionDailyMetric, SignalExitEvent
from src.db.session import get_session
from src.data.aggregator import DataAggregator

logger = logging.getLogger(__name__)


async def check_open_positions() -> tuple[
    list[dict], list[PositionHealthCard], list[PositionHealthCard], list[dict]
]:
    """Check all open positions against current prices and compute health cards.

    Called daily at 4:30 PM ET to update outcomes.

    Returns:
        (updates, all_health_cards, state_changes, resolved_near_misses)
    """
    from src.output.health import compute_health_card
    from src.contracts import HealthCardConfig
    from src.features.regime import classify_regime

    aggregator = DataAggregator()
    updates = []
    all_health_cards: list[PositionHealthCard] = []
    state_changes: list[PositionHealthCard] = []
    config = HealthCardConfig()

    # Fetch current regime once for all positions
    current_regime: str | None = None
    try:
        macro = await aggregator.get_macro_context()
        regime_result = classify_regime(
            spy_df=macro.get("spy_prices"),
            qqq_df=macro.get("qqq_prices"),
            vix=macro.get("vix"),
            yield_spread=macro.get("yield_spread_10y2y"),
        )
        current_regime = regime_result.regime.value
    except Exception as e:
        logger.warning("Regime fetch failed (non-fatal): %s", e)

    async with get_session() as session:
        # Get all open outcomes
        result = await session.execute(
            select(Outcome).where(Outcome.still_open == True)
        )
        open_outcomes = result.scalars().all()

        if not open_outcomes:
            logger.info("No open positions to check")
            return [], [], [], []

        # Evaluate P&L and collect OHLCV DataFrames for reuse
        position_dfs: dict[int, pd.DataFrame] = {}
        for outcome in open_outcomes:
            try:
                update_data, df = await _evaluate_position(outcome, aggregator)
                if df is not None:
                    position_dfs[outcome.signal_id] = df
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

    # Health card loop — each position gets its own session to prevent
    # one failure from cascading and poisoning the transaction for others.
    for outcome in open_outcomes:
        if not outcome.still_open:
            continue
        try:
            async with get_session() as session:
                # Fetch signal
                sig_result = await session.execute(
                    select(Signal).where(Signal.id == outcome.signal_id)
                )
                signal = sig_result.scalar_one_or_none()
                if not signal:
                    continue

                # Get previous metrics (up to 3) for state + velocity
                prev_result = await session.execute(
                    select(PositionDailyMetric)
                    .where(PositionDailyMetric.signal_id == outcome.signal_id)
                    .order_by(PositionDailyMetric.metric_date.desc())
                    .limit(3)
                )
                prev_metrics = prev_result.scalars().all()
                prev_metric = prev_metrics[0] if prev_metrics else None
                previous_state = (
                    HealthState(prev_metric.health_state) if prev_metric else None
                )
                # Chronological order (oldest first) for velocity
                previous_scores = (
                    [m.promising_score for m in reversed(prev_metrics)]
                    if prev_metrics else None
                )

                # Reuse OHLCV from P&L evaluation
                df = position_dfs.get(outcome.signal_id)
                if df is None or df.empty:
                    continue

                card = await compute_health_card(
                    outcome=outcome,
                    signal=signal,
                    df=df,
                    config=config,
                    previous_state=previous_state,
                    current_regime=current_regime,
                    previous_scores=previous_scores,
                )
                if card is None:
                    continue

                all_health_cards.append(card)
                if card.state_changed:
                    state_changes.append(card)

                # Persist daily metric (upsert — idempotent on reruns)
                await _persist_daily_metric(session, card)

                # If EXIT and not already closed, persist exit event (guarded)
                if card.state == HealthState.EXIT:
                    await _persist_exit_event(session, card, outcome)

        except Exception as e:
            logger.error("Health card failed for %s (non-fatal): %s", outcome.ticker, e)

    logger.info(
        "Updated %d positions, %d health cards, %d state changes",
        len(updates), len(all_health_cards), len(state_changes),
    )

    # Resolve divergence outcomes for closed positions
    try:
        from src.governance.divergence_ledger import update_divergence_outcomes
        div_results = await update_divergence_outcomes()
        if div_results:
            logger.info("Resolved %d divergence outcomes", len(div_results))
    except Exception as e:
        logger.error("Divergence outcome update failed (non-fatal): %s", e)

    # Resolve near-miss counterfactuals
    resolved_near_misses: list[dict] = []
    try:
        from src.governance.divergence_ledger import resolve_near_miss_counterfactuals
        resolved_near_misses = await resolve_near_miss_counterfactuals()
        if resolved_near_misses:
            logger.info("Resolved %d near-miss counterfactuals", len(resolved_near_misses))
    except Exception as e:
        logger.error("Near-miss resolution failed (non-fatal): %s", e)

    return updates, all_health_cards, state_changes, resolved_near_misses


async def _persist_daily_metric(session, card: PositionHealthCard) -> None:
    """Upsert a PositionDailyMetric row from a health card.

    Uses delete-then-insert for idempotent reruns (unique on signal_id + metric_date).
    """
    from sqlalchemy import delete

    details = {
        "trend": card.trend_health.details,
        "momentum": card.momentum_health.details,
        "volume": card.volume_confirmation.details,
        "risk": card.risk_integrity.details,
        "regime": card.regime_alignment.details,
    }

    # Idempotent upsert: remove any existing row for this signal+date
    await session.execute(
        delete(PositionDailyMetric).where(
            PositionDailyMetric.signal_id == card.signal_id,
            PositionDailyMetric.metric_date == card.as_of_date,
        )
    )

    # Extract regime context from the card's regime component details
    regime_details = card.regime_alignment.details or {}

    session.add(PositionDailyMetric(
        signal_id=card.signal_id,
        ticker=card.ticker,
        metric_date=card.as_of_date,
        promising_score=card.promising_score,
        health_state=card.state.value,
        trend_score=card.trend_health.score,
        momentum_score=card.momentum_health.score,
        volume_score=card.volume_confirmation.score,
        risk_score=card.risk_integrity.score,
        regime_score=card.regime_alignment.score,
        current_price=card.current_price,
        pnl_pct=card.pnl_pct,
        rsi_14=card.momentum_health.details.get("rsi_14"),
        atr_14=card.atr_14,
        atr_stop_distance=card.atr_stop_distance,
        rvol=card.volume_confirmation.details.get("rvol"),
        days_held=card.days_held,
        score_velocity=card.score_velocity,
        signal_regime=regime_details.get("signal_regime"),
        current_regime=regime_details.get("current_regime"),
        details=details,
        hard_invalidation=card.hard_invalidation,
        invalidation_reason=card.invalidation_reason,
    ))


async def _persist_exit_event(session, card: PositionHealthCard, outcome: Outcome) -> None:
    """Write a SignalExitEvent when health card triggers EXIT.

    Guarded by unique(signal_id, exit_reason) — skips if an exit event
    for this signal+reason already exists (prevents duplicates on reruns).
    """
    exit_reason = "invalidation" if card.hard_invalidation else "health_exit"

    # Check if this exit event already exists
    existing = await session.execute(
        select(SignalExitEvent).where(
            SignalExitEvent.signal_id == card.signal_id,
            SignalExitEvent.exit_reason == exit_reason,
        )
    )
    if existing.scalar_one_or_none() is not None:
        logger.debug(
            "Exit event already exists for signal_id=%d reason=%s — skipping",
            card.signal_id, exit_reason,
        )
        return

    session.add(SignalExitEvent(
        signal_id=card.signal_id,
        outcome_id=outcome.id,
        ticker=card.ticker,
        exit_date=card.as_of_date,
        exit_price=card.current_price,
        entry_price=outcome.entry_price,
        pnl_pct=card.pnl_pct,
        exit_reason=exit_reason,
        signal_model=card.signal_model,
        health_state_at_exit=card.state.value,
        promising_score_at_exit=card.promising_score,
        invalidation_reason=card.invalidation_reason,
        days_held=card.days_held,
    ))


async def _evaluate_position(
    outcome: Outcome, aggregator: DataAggregator
) -> tuple[dict | None, pd.DataFrame | None]:
    """Evaluate a single open position against current market data.

    Returns (update_dict, ohlcv_dataframe) so the caller can reuse the
    DataFrame for health card computation without a second API call.
    """
    # Get the original signal for stop/target
    async with get_session() as session:
        result = await session.execute(
            select(Signal).where(Signal.id == outcome.signal_id)
        )
        signal = result.scalar_one_or_none()

    if not signal:
        return None, None

    # Fetch latest price data — use centralized lookback to cover all indicator windows
    from src.output.health import HISTORY_LOOKBACK_DAYS

    to_date = date.today()
    from_date = outcome.entry_date - timedelta(days=HISTORY_LOOKBACK_DAYS)
    df = await aggregator.get_ohlcv(outcome.ticker, from_date, to_date)

    if df.empty:
        return None, None

    latest = df.iloc[-1]
    current_price = float(latest["close"])
    entry_price = outcome.entry_price

    # Track high/low since entry (normalize date type to Timestamp for comparison)
    if "date" in df.columns:
        entry_ts = pd.Timestamp(outcome.entry_date)
        date_series = pd.to_datetime(df["date"])
        since_entry = df[date_series >= entry_ts]
    else:
        since_entry = df
    max_price = float(since_entry["high"].max())
    min_price = float(since_entry["low"].min())

    pnl_pct = (current_price - entry_price) / entry_price * 100
    max_favorable = (max_price - entry_price) / entry_price * 100
    max_adverse = (min_price - entry_price) / entry_price * 100

    from src.utils.trading_calendar import trading_days_between
    days_held = trading_days_between(outcome.entry_date, to_date)

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

    return update, df


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

        def _finite_or_none(value):
            if value is None:
                return None
            if isinstance(value, (int, float)) and not math.isfinite(value):
                return None
            return value

        return {
            "period_days": days,
            "total_signals": len(closed),
            "overall": _summary(all_pnls),
            "risk_metrics": {
                "sharpe_ratio": _finite_or_none(overall_metrics.sharpe_ratio),
                "sortino_ratio": _finite_or_none(overall_metrics.sortino_ratio),
                "max_drawdown_pct": _finite_or_none(overall_metrics.max_drawdown_pct),
                "profit_factor": _finite_or_none(overall_metrics.profit_factor),
                "calmar_ratio": _finite_or_none(overall_metrics.calmar_ratio),
                "expectancy": _finite_or_none(overall_metrics.expectancy),
                "payoff_ratio": _finite_or_none(overall_metrics.payoff_ratio),
                "avg_win_pct": _finite_or_none(overall_metrics.avg_win_pct),
                "avg_loss_pct": _finite_or_none(overall_metrics.avg_loss_pct),
                "max_consecutive_wins": _finite_or_none(overall_metrics.max_consecutive_wins),
                "max_consecutive_losses": _finite_or_none(overall_metrics.max_consecutive_losses),
            },
            "by_model": {k: _summary(v) for k, v in by_model.items()},
            "by_regime": {k: _summary(v) for k, v in by_regime.items()},
            "by_confidence": {k: _summary(v) for k, v in by_confidence_bucket.items()},
            "confidence_calibration": _build_calibration(closed, signals),
        }


async def get_near_miss_stats(days: int = 30) -> dict | None:
    """Near-miss statistics for the meta-analyst.

    Returns None if no near-miss data exists.
    """
    from src.db.models import NearMiss

    async with get_session() as session:
        cutoff = date.today() - timedelta(days=days)
        result = await session.execute(
            select(NearMiss).where(NearMiss.run_date >= cutoff)
        )
        rows = result.scalars().all()

    if not rows:
        return None

    total = len(rows)

    # --- By stage ---
    by_stage: dict[str, dict] = {}
    for nm in rows:
        bucket = by_stage.setdefault(nm.stage, {"count": 0, "convictions": []})
        bucket["count"] += 1
        bucket["convictions"].append(nm.net_conviction)

    by_stage_out = {}
    for stage, b in by_stage.items():
        convs = b["convictions"]
        by_stage_out[stage] = {
            "count": b["count"],
            "avg_conviction": round(sum(convs) / len(convs), 2),
            "conviction_range": [round(min(convs), 2), round(max(convs), 2)],
        }

    # --- Conviction distribution ---
    conviction_distribution = {"0_25": 0, "25_50": 0, "50_75": 0, "75_100": 0}
    for nm in rows:
        c = nm.net_conviction
        if c < 25:
            conviction_distribution["0_25"] += 1
        elif c < 50:
            conviction_distribution["25_50"] += 1
        elif c < 75:
            conviction_distribution["50_75"] += 1
        else:
            conviction_distribution["75_100"] += 1

    # --- By regime ---
    by_regime: dict[str, dict] = {}
    for nm in rows:
        regime = (nm.regime or "unknown").lower()
        bucket = by_regime.setdefault(regime, {"count": 0, "convictions": []})
        bucket["count"] += 1
        bucket["convictions"].append(nm.net_conviction)

    by_regime_out = {}
    for regime, b in by_regime.items():
        convs = b["convictions"]
        by_regime_out[regime] = {
            "count": b["count"],
            "avg_conviction": round(sum(convs) / len(convs), 2),
        }

    # --- By signal model ---
    by_signal_model: dict[str, dict] = {}
    for nm in rows:
        model = nm.signal_model or "unknown"
        bucket = by_signal_model.setdefault(model, {"count": 0, "convictions": []})
        bucket["count"] += 1
        bucket["convictions"].append(nm.net_conviction)

    by_signal_model_out = {}
    for model, b in by_signal_model.items():
        convs = b["convictions"]
        by_signal_model_out[model] = {
            "count": b["count"],
            "avg_conviction": round(sum(convs) / len(convs), 2),
        }

    # --- Closest misses (top 3 by conviction) ---
    sorted_by_conviction = sorted(rows, key=lambda nm: nm.net_conviction, reverse=True)
    closest_misses = [
        {
            "ticker": nm.ticker,
            "conviction": round(nm.net_conviction, 2),
            "stage": nm.stage,
            "key_risk": nm.key_risk,
            "run_date": str(nm.run_date),
        }
        for nm in sorted_by_conviction[:3]
    ]

    # --- Counterfactual section (resolved near-misses) ---
    resolved = [nm for nm in rows if nm.outcome_resolved and nm.counterfactual_return is not None]
    counterfactual: dict | None = None
    if resolved:
        cf_returns = [nm.counterfactual_return for nm in resolved]
        cf_wins = sum(1 for r in cf_returns if r > 0)
        cf_total = len(cf_returns)

        # By stage
        cf_by_stage: dict[str, list[float]] = {}
        cf_by_regime: dict[str, list[float]] = {}
        cf_by_model: dict[str, list[float]] = {}
        cf_by_exit: dict[str, int] = {}
        for nm in resolved:
            cf_by_stage.setdefault(nm.stage, []).append(nm.counterfactual_return)
            regime = (nm.regime or "unknown").lower()
            cf_by_regime.setdefault(regime, []).append(nm.counterfactual_return)
            model = nm.signal_model or "unknown"
            cf_by_model.setdefault(model, []).append(nm.counterfactual_return)
            exit_reason = nm.counterfactual_exit_reason or "unknown"
            cf_by_exit[exit_reason] = cf_by_exit.get(exit_reason, 0) + 1

        def _cf_stats(rets: list[float]) -> dict:
            wins = sum(1 for r in rets if r > 0)
            return {
                "count": len(rets),
                "win_rate": round(wins / len(rets), 4) if rets else 0,
                "avg_return": round(sum(rets) / len(rets), 4) if rets else 0,
            }

        counterfactual = {
            "total_resolved": cf_total,
            "win_rate": round(cf_wins / cf_total, 4),
            "avg_return": round(sum(cf_returns) / cf_total, 4),
            "by_stage": {s: _cf_stats(r) for s, r in cf_by_stage.items()},
            "by_regime": {r: _cf_stats(v) for r, v in cf_by_regime.items()},
            "by_signal_model": {m: _cf_stats(v) for m, v in cf_by_model.items()},
            "by_exit_reason": cf_by_exit,
        }

    return {
        "period_days": days,
        "total_near_misses": total,
        "by_stage": by_stage_out,
        "conviction_distribution": conviction_distribution,
        "by_regime": by_regime_out,
        "by_signal_model": by_signal_model_out,
        "closest_misses": closest_misses,
        "counterfactual": counterfactual,
    }


async def get_divergence_stats(days: int = 30) -> dict | None:
    """Portfolio-level divergence statistics for the meta-analyst.

    Returns None if no divergence data exists (fail-closed).
    """
    from src.db.models import DivergenceEvent, DivergenceOutcome

    async with get_session() as session:
        cutoff = date.today() - timedelta(days=days)

        # LEFT JOIN so unresolved events are included
        result = await session.execute(
            select(DivergenceEvent, DivergenceOutcome)
            .outerjoin(
                DivergenceOutcome,
                DivergenceEvent.id == DivergenceOutcome.divergence_id,
            )
            .where(DivergenceEvent.run_date >= cutoff)
        )
        rows = result.all()

    if not rows:
        return None

    total_events = len(rows)
    resolved = [(ev, out) for ev, out in rows if out is not None]
    total_resolved = len(resolved)

    # Overall improvement rate
    improved_count = sum(1 for _, out in resolved if out.improved_vs_quant is True)
    overall_improvement_rate = (
        round(improved_count / total_resolved, 4) if total_resolved > 0 else None
    )

    # Net portfolio delta — sum of ALL return_deltas
    all_deltas = [out.return_delta for _, out in resolved if out.return_delta is not None]
    net_portfolio_delta = round(sum(all_deltas), 4) if all_deltas else None

    # --- By event type ---
    by_event_type: dict[str, dict] = {}
    for ev, out in rows:
        et = ev.event_type
        bucket = by_event_type.setdefault(et, {
            "events": 0, "resolved": 0, "wins": 0,
            "deltas": [], "costs": [],
        })
        bucket["events"] += 1
        if ev.llm_cost_usd:
            bucket["costs"].append(ev.llm_cost_usd)
        if out is not None:
            bucket["resolved"] += 1
            if out.improved_vs_quant is True:
                bucket["wins"] += 1
            if out.return_delta is not None:
                bucket["deltas"].append(out.return_delta)

    by_event_type_out = {}
    for et, b in by_event_type.items():
        by_event_type_out[et] = {
            "events": b["events"],
            "resolved": b["resolved"],
            "win_rate": round(b["wins"] / b["resolved"], 4) if b["resolved"] > 0 else None,
            "avg_return_delta": round(sum(b["deltas"]) / len(b["deltas"]), 4) if b["deltas"] else None,
            "total_cost": round(sum(b["costs"]), 4),
        }

    # --- By reason code ---
    by_reason_code: dict[str, dict] = {}
    for ev, out in rows:
        codes = ev.reason_codes or []
        for code in codes:
            bucket = by_reason_code.setdefault(code, {
                "events": 0, "with_outcome": 0, "wins": 0, "deltas": [],
            })
            bucket["events"] += 1
            if out is not None:
                bucket["with_outcome"] += 1
                if out.improved_vs_quant is True:
                    bucket["wins"] += 1
                if out.return_delta is not None:
                    bucket["deltas"].append(out.return_delta)

    by_reason_code_out = {}
    for code, b in by_reason_code.items():
        by_reason_code_out[code] = {
            "events": b["events"],
            "with_outcome": b["with_outcome"],
            "win_rate": round(b["wins"] / b["with_outcome"], 4) if b["with_outcome"] > 0 else None,
            "avg_return_delta": round(sum(b["deltas"]) / len(b["deltas"]), 4) if b["deltas"] else None,
        }

    # --- By regime ---
    by_regime: dict[str, dict] = {}
    for ev, out in rows:
        regime = (ev.regime or "unknown").lower()
        bucket = by_regime.setdefault(regime, {
            "events": 0, "resolved": 0, "wins": 0, "deltas": [],
        })
        bucket["events"] += 1
        if out is not None:
            bucket["resolved"] += 1
            if out.improved_vs_quant is True:
                bucket["wins"] += 1
            if out.return_delta is not None:
                bucket["deltas"].append(out.return_delta)

    by_regime_out = {}
    for regime, b in by_regime.items():
        by_regime_out[regime] = {
            "events": b["events"],
            "resolved": b["resolved"],
            "win_rate": round(b["wins"] / b["resolved"], 4) if b["resolved"] > 0 else None,
            "avg_return_delta": round(sum(b["deltas"]) / len(b["deltas"]), 4) if b["deltas"] else None,
        }

    # --- Run-level deltas (portfolio-level metric) ---
    run_deltas: dict[str, list[float]] = {}
    for ev, out in resolved:
        if out.return_delta is not None:
            run_key = str(ev.run_date)
            run_deltas.setdefault(run_key, []).append(out.return_delta)

    run_level_deltas = sorted(
        [
            {
                "run_date": rd,
                "divergence_count": len(deltas),
                "net_delta": round(sum(deltas), 4),
                "positive": sum(deltas) > 0,
            }
            for rd, deltas in run_deltas.items()
        ],
        key=lambda x: x["run_date"],
    )

    # Run-level trend (null if < 4 runs)
    run_level_trend = None
    if len(run_level_deltas) >= 4:
        mid = len(run_level_deltas) // 2
        early = run_level_deltas[:mid]
        recent = run_level_deltas[mid:]
        early_avg = sum(r["net_delta"] for r in early) / len(early)
        recent_avg = sum(r["net_delta"] for r in recent) / len(recent)
        run_level_trend = {
            "recent_4_avg_delta": round(recent_avg, 4),
            "early_4_avg_delta": round(early_avg, 4),
            "improving": recent_avg > early_avg,
        }

    # --- Cost efficiency ---
    total_llm_cost = sum(ev.llm_cost_usd or 0.0 for ev, _ in rows)
    positive_deltas = [d for d in all_deltas if d > 0]
    total_positive_delta = sum(positive_deltas)
    positive_divergence_count = len(positive_deltas)

    cost_efficiency = {
        "total_llm_cost": round(total_llm_cost, 4),
        "total_positive_delta_generated": round(total_positive_delta, 4),
        "cost_per_positive_divergence": (
            round(total_llm_cost / positive_divergence_count, 4)
            if positive_divergence_count > 0 and total_llm_cost > 0
            else None
        ),
        "net_delta_per_dollar": (
            round((net_portfolio_delta or 0) / total_llm_cost, 4)
            if total_llm_cost > 0
            else None
        ),
    }

    return {
        "period_days": days,
        "total_events": total_events,
        "total_resolved": total_resolved,
        "overall_improvement_rate": overall_improvement_rate,
        "net_portfolio_delta": net_portfolio_delta,
        "by_event_type": by_event_type_out,
        "by_reason_code": by_reason_code_out,
        "by_regime": by_regime_out,
        "run_level_deltas": run_level_deltas,
        "run_level_trend": run_level_trend,
        "cost_efficiency": cost_efficiency,
    }


async def get_equity_curve(days: int = 90) -> list[dict]:
    """Build equity curve from closed outcomes (cumulative returns)."""
    async with get_session() as session:
        cutoff = date.today() - timedelta(days=days)
        result = await session.execute(
            select(Outcome).where(
                Outcome.still_open == False,
                Outcome.entry_date >= cutoff,
            ).order_by(Outcome.exit_date.asc())
        )
        closed = result.scalars().all()

    curve = []
    cumulative = 0.0
    for o in closed:
        pnl = o.pnl_pct or 0.0
        cumulative += pnl
        curve.append({
            "time": str(o.exit_date) if o.exit_date else str(o.entry_date),
            "value": round(cumulative, 4),
            "pnl": round(pnl, 4),
        })
    return curve


async def get_drawdown_curve(days: int = 90) -> list[dict]:
    """Build drawdown series from equity curve."""
    curve = await get_equity_curve(days)
    if not curve:
        return []

    peak = 0.0
    drawdown = []
    for point in curve:
        peak = max(peak, point["value"])
        dd = point["value"] - peak
        drawdown.append({
            "time": point["time"],
            "value": round(dd, 4),
        })
    return drawdown


async def get_return_distribution(days: int = 90) -> dict:
    """Return distribution of trade P&L by signal model."""
    async with get_session() as session:
        cutoff = date.today() - timedelta(days=days)
        result = await session.execute(
            select(Outcome, Signal)
            .join(Signal, Outcome.signal_id == Signal.id)
            .where(Outcome.still_open == False, Outcome.entry_date >= cutoff)
        )
        rows = result.all()

    by_model: dict[str, list[float]] = {}
    for outcome, signal in rows:
        pnl = outcome.pnl_pct or 0.0
        by_model.setdefault(signal.signal_model, []).append(pnl)

    distribution = {}
    for model, pnls in by_model.items():
        import numpy as np
        arr = np.array(pnls)
        distribution[model] = {
            "returns": [round(p, 4) for p in pnls],
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4),
            "count": len(pnls),
        }
    return distribution


async def get_regime_matrix(days: int = 180) -> list[dict]:
    """Win rate matrix: model x regime, color-coded."""
    async with get_session() as session:
        cutoff = date.today() - timedelta(days=days)
        result = await session.execute(
            select(Outcome, Signal)
            .join(Signal, Outcome.signal_id == Signal.id)
            .where(Outcome.still_open == False, Outcome.entry_date >= cutoff)
        )
        rows = result.all()

    # Build nested dict: model -> regime -> [pnls]
    matrix: dict[str, dict[str, list[float]]] = {}
    for outcome, signal in rows:
        pnl = outcome.pnl_pct or 0.0
        matrix.setdefault(signal.signal_model, {}).setdefault(signal.regime, []).append(pnl)

    result_list = []
    for model, regimes in matrix.items():
        for regime, pnls in regimes.items():
            wins = sum(1 for p in pnls if p > 0)
            result_list.append({
                "model": model,
                "regime": regime,
                "trades": len(pnls),
                "win_rate": round(wins / len(pnls), 4) if pnls else 0,
                "avg_pnl": round(sum(pnls) / len(pnls), 4) if pnls else 0,
            })
    return result_list


async def get_mode_comparison() -> list[dict]:
    """Compare metrics across execution modes (quant_only vs hybrid vs agentic_full)."""
    from src.db.models import DailyRun

    async with get_session() as session:
        # Get runs grouped by execution mode
        result = await session.execute(
            select(DailyRun).order_by(DailyRun.run_date.desc()).limit(90)
        )
        runs = result.scalars().all()

        # Get outcomes for each mode's signals
        all_outcomes_result = await session.execute(
            select(Outcome, Signal)
            .join(Signal, Outcome.signal_id == Signal.id)
            .where(Outcome.still_open == False)
        )
        all_rows = all_outcomes_result.all()

    # Map run_date -> execution_mode
    date_to_mode = {r.run_date: (r.execution_mode or "agentic_full") for r in runs}

    # Group outcomes by mode
    by_mode: dict[str, list[float]] = {}
    for outcome, signal in all_rows:
        mode = date_to_mode.get(signal.run_date, "agentic_full")
        by_mode.setdefault(mode, []).append(outcome.pnl_pct or 0.0)

    comparison = []
    for mode, pnls in by_mode.items():
        wins = sum(1 for p in pnls if p > 0)
        comparison.append({
            "mode": mode,
            "trades": len(pnls),
            "win_rate": round(wins / len(pnls), 4) if pnls else 0,
            "avg_pnl": round(sum(pnls) / len(pnls), 4) if pnls else 0,
            "total_return": round(sum(pnls), 4),
        })
    return comparison


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
