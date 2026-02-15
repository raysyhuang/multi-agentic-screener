"""Main entry point — daily pipeline orchestration and scheduling."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict
from datetime import date, datetime, timedelta

from src.config import get_settings
from src.contracts import (
    StageEnvelope,
    StageName,
    StageStatus,
    StageError,
    DataIngestPayload,
    TickerSnapshot,
    FeaturePayload,
    TickerFeatures,
    SignalPrefilterPayload,
    CandidateScores,
    RegimePayload,
    RegimeInfo,
    AgentReviewPayload,
    TickerReview,
    FinalOutputPayload,
    FinalPick,
)
from src.data.aggregator import DataAggregator
from sqlalchemy import select, func
from src.db.models import DailyRun, Signal, Candidate, AgentLog, Outcome, PipelineArtifact
from src.db.session import get_session, init_db
from src.features.technical import compute_all_technical_features, compute_rsi2_features, latest_features
from src.features.fundamental import score_earnings_surprise, score_insider_activity, days_to_next_earnings
from src.features.sentiment import score_news_batch
from src.features.regime import classify_regime, RegimeAssessment, get_regime_allowed_models, compute_breadth_score
from src.signals.filter import filter_universe, filter_by_ohlcv, FilterFunnel, OHLCVFunnel
from src.signals.breakout import score_breakout
from src.signals.mean_reversion import score_mean_reversion
from src.signals.catalyst import score_catalyst
from src.signals.ranker import (
    rank_candidates,
    deduplicate_signals,
    filter_correlated_picks,
    detect_confluence,
    apply_confluence_bonus,
    apply_cooldown,
    MODEL_MAP,
)
from src.agents.orchestrator import run_agent_pipeline, PipelineRun
from src.backtest.validation_card import run_validation_checks
from src.output.telegram import send_alert, format_daily_alert
from src.output.performance import check_open_positions, record_new_signals
from src.governance.artifacts import GovernanceContext
from src.governance.performance_monitor import (
    compute_rolling_metrics,
    check_decay,
    RollingMetrics,
)
from src.portfolio.construct import build_trade_plan, PortfolioConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _json_safe(obj):
    """Recursively convert non-JSON-serializable types in a dict/list structure."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    return obj


async def run_morning_pipeline() -> None:
    """Main daily pipeline — runs at 6:00 AM ET.

    Fail-closed: any unhandled exception guarantees a NoTrade DB record
    and Telegram alert rather than a silent abort.
    """
    import uuid

    start_time = time.monotonic()
    today = date.today()
    settings = get_settings()
    run_id = uuid.uuid4().hex[:12]

    logger.info("=" * 60)
    logger.info("Starting morning pipeline for %s (run_id=%s)", today, run_id)
    logger.info("Trading mode: %s", settings.trading_mode)
    logger.info("=" * 60)

    try:
        await _run_pipeline_core(today, settings, run_id, start_time)
    except Exception as exc:
        elapsed = time.monotonic() - start_time
        logger.exception("PIPELINE FAILED — fail-closed with NoTrade (run_id=%s): %s", run_id, exc)

        # Guarantee a DailyRun + NoTrade artifact in DB
        try:
            async with get_session() as session:
                session.add(DailyRun(
                    run_date=today,
                    regime="unknown",
                    regime_details={"error": str(exc)},
                    universe_size=0,
                    candidates_scored=0,
                    pipeline_duration_s=round(elapsed, 2),
                ))
                no_trade = FinalOutputPayload(
                    decision="NoTrade",
                    no_trade_reason=f"Pipeline error: {type(exc).__name__}: {exc}",
                )
                session.add(PipelineArtifact(
                    run_id=run_id,
                    run_date=today,
                    stage=StageName.FINAL_OUTPUT.value,
                    status=StageStatus.FAILED.value,
                    payload=_json_safe(no_trade.model_dump()),
                    errors=[{"code": "PIPELINE_CRASH", "message": str(exc)}],
                ))
        except Exception as db_exc:
            logger.error("Failed to write fail-closed record: %s", db_exc)

        # Attempt Telegram alert about failure
        try:
            await send_alert(
                f"PIPELINE FAILED ({today}, run_id={run_id})\n"
                f"Error: {type(exc).__name__}: {exc}\n"
                f"Decision: NoTrade (fail-closed)"
            )
        except Exception:
            logger.error("Failed to send fail-closed Telegram alert")


async def _check_paper_gate(settings) -> bool:
    """Enforce 30-day paper trading gate before allowing LIVE mode.

    Returns True if pipeline should proceed, False if blocked.
    Logs a warning and forces PAPER mode if gate fails.
    """
    if settings.trading_mode != "LIVE":
        return True  # PAPER mode always allowed

    from src.backtest.metrics import compute_metrics

    async with get_session() as session:
        result = await session.execute(
            select(
                func.count(Outcome.id),
                func.min(Outcome.entry_date),
                func.max(Outcome.entry_date),
            ).where(Outcome.still_open == False)
        )
        row = result.one()
        total_closed, earliest, latest = row

        if total_closed < 10 or earliest is None or latest is None:
            logger.warning(
                "PAPER GATE: Only %d closed trades — need 30+ days of history. "
                "Forcing PAPER mode.", total_closed
            )
            settings.trading_mode = "PAPER"
            return True

        span_days = (latest - earliest).days
        if span_days < 30:
            logger.warning(
                "PAPER GATE: Only %d days of paper trading history (need 30). "
                "Forcing PAPER mode.", span_days
            )
            settings.trading_mode = "PAPER"
            return True

        # Check profit factor
        pnl_result = await session.execute(
            select(Outcome.pnl_pct).where(Outcome.still_open == False)
        )
        pnls = [r[0] or 0.0 for r in pnl_result.all()]

    metrics = compute_metrics(pnls)
    if metrics.profit_factor < 1.0:
        logger.warning(
            "PAPER GATE: Profit factor %.2f < 1.0 — not profitable yet. "
            "Forcing PAPER mode.", metrics.profit_factor
        )
        settings.trading_mode = "PAPER"
        return True

    logger.info(
        "PAPER GATE PASSED: %d days history, %d trades, PF=%.2f — LIVE mode allowed",
        span_days, len(pnls), metrics.profit_factor,
    )
    return True


async def _run_pipeline_core(
    today: date, settings, run_id: str, start_time: float
) -> None:
    """Inner pipeline logic — extracted so fail-closed wrapper can catch errors."""
    stage_envelopes: list[StageEnvelope] = []

    # Enforce 30-day paper trading gate
    await _check_paper_gate(settings)

    if settings.trading_mode == "PAPER":
        logger.info("PAPER MODE — all picks are recommendations only, not live trades")

    aggregator = DataAggregator()

    # --- Governance context (audit trail for this run) ---
    gov = GovernanceContext(run_id=run_id, run_date=str(today))
    gov.__enter__()
    gov.set_trading_mode(settings.trading_mode)
    gov.set_config_hash({
        "min_price": settings.min_price,
        "min_adv": settings.min_avg_daily_volume,
        "top_n": settings.top_n_for_interpretation,
        "slippage": settings.slippage_pct,
    })

    # --- Step 1: Macro context and regime detection (preliminary, breadth added after OHLCV) ---
    logger.info("Step 1: Fetching macro context...")
    macro = await aggregator.get_macro_context()
    regime_assessment = classify_regime(
        spy_df=macro.get("spy_prices"),
        qqq_df=macro.get("qqq_prices"),
        vix=macro.get("vix"),
        yield_spread=macro.get("yield_spread_10y2y"),
    )
    logger.info("Regime: %s (confidence: %.2f)", regime_assessment.regime.value, regime_assessment.confidence)

    regime_context = {
        "regime": regime_assessment.regime.value,
        "confidence": regime_assessment.confidence,
        "vix": regime_assessment.vix_level,
        "spy_trend": regime_assessment.spy_trend,
        "qqq_trend": regime_assessment.qqq_trend,
        "yield_spread": regime_assessment.yield_spread,
    }

    # Regime envelope
    allowed_models = get_regime_allowed_models(regime_assessment.regime)
    regime_envelope = StageEnvelope(
        run_id=run_id,
        stage=StageName.REGIME,
        payload=RegimePayload(
            asof_date=today,
            regime=RegimeInfo(
                label=regime_assessment.regime.value,
                confidence=regime_assessment.confidence,
                signals_allowed=allowed_models,
            ),
            gated_candidates=[],  # populated after signal ranking
        ),
    )
    stage_envelopes.append(regime_envelope)

    # --- Step 2: Build universe ---
    logger.info("Step 2: Building universe...")
    raw_universe = await aggregator.get_universe()
    universe_funnel = FilterFunnel()
    filtered = filter_universe(raw_universe, funnel=universe_funnel)
    logger.info("Universe: %d raw → %d filtered", len(raw_universe), len(filtered))

    # Data ingest envelope
    ingest_envelope = StageEnvelope(
        run_id=run_id,
        stage=StageName.DATA_INGEST,
        payload=DataIngestPayload(
            asof_date=today,
            universe=[
                TickerSnapshot(
                    ticker=s.get("symbol", s.get("ticker", "")),
                    last_price=s.get("lastSale", s.get("price", 0.0)),
                    volume=int(s.get("volume", 0)),
                    market_cap=s.get("marketCap"),
                    source_provenance="polygon",
                )
                for s in filtered[:50]  # log first 50 for envelope size
            ],
        ),
    )
    stage_envelopes.append(ingest_envelope)

    # --- Step 3: Fetch OHLCV for filtered universe ---
    logger.info("Step 3: Fetching OHLCV for %d tickers...", len(filtered))
    tickers = [s["symbol"] for s in filtered[:200]]  # Cap at 200 for API limits
    from_date = today - timedelta(days=300)  # 1 year of data for indicators
    price_data = await aggregator.get_bulk_ohlcv(tickers, from_date, today)

    # Further filter by OHLCV quality
    ohlcv_funnel = OHLCVFunnel(total_input=len(tickers))
    qualified_tickers = [t for t in tickers if filter_by_ohlcv(t, price_data.get(t), funnel=ohlcv_funnel)]
    ohlcv_funnel.passed = len(qualified_tickers)
    ohlcv_funnel.log_summary()

    # --- Step 3b: Recompute regime with breadth from actual OHLCV data ---
    breadth = compute_breadth_score(price_data)
    if breadth is not None:
        logger.info("Breadth score: %.2f (%d/%d above 20d SMA)", breadth, int(breadth * len(price_data)), len(price_data))
        regime_assessment = classify_regime(
            spy_df=macro.get("spy_prices"),
            qqq_df=macro.get("qqq_prices"),
            vix=macro.get("vix"),
            yield_spread=macro.get("yield_spread_10y2y"),
            breadth_score=breadth,
        )
        logger.info("Regime (with breadth): %s (confidence: %.2f)", regime_assessment.regime.value, regime_assessment.confidence)
        regime_context["breadth_score"] = breadth
        regime_context["regime"] = regime_assessment.regime.value
        regime_context["confidence"] = regime_assessment.confidence

    # --- Step 4: Feature engineering ---
    logger.info("Step 4: Computing features for %d tickers...", len(qualified_tickers))
    features_by_ticker = {}
    earnings_calendar = await aggregator.get_upcoming_earnings()

    for ticker in qualified_tickers:
        df = price_data[ticker]
        if df.empty:
            continue

        # Technical features
        df = compute_all_technical_features(df)
        df = compute_rsi2_features(df)
        feat = latest_features(df)
        feat["ticker"] = ticker

        # Fundamental + sentiment (parallel per ticker would be ideal, but rate-limited)
        try:
            fund_data = await aggregator.get_ticker_fundamentals(ticker)
            fund_data["earnings_surprises"] = score_earnings_surprise(
                fund_data.get("earnings_surprises", [])
            )
            fund_data["insider_activity"] = score_insider_activity(
                fund_data.get("insider_transactions", [])
            )
            feat["fundamental"] = fund_data
        except Exception as e:
            logger.debug("Fundamental fetch failed for %s: %s", ticker, e)
            feat["fundamental"] = {}

        try:
            news = await aggregator.get_ticker_news(ticker)
            feat["sentiment"] = score_news_batch(news)
        except Exception:
            feat["sentiment"] = {}

        feat["days_to_earnings"] = days_to_next_earnings(earnings_calendar, ticker)
        features_by_ticker[ticker] = feat

        # Store price data for signal models
        price_data[ticker] = df

    # Feature envelope
    feature_envelope = StageEnvelope(
        run_id=run_id,
        stage=StageName.FEATURE,
        payload=FeaturePayload(
            asof_date=today,
            ticker_features=[
                TickerFeatures(
                    ticker=t,
                    returns_5d=f.get("roc_5"),
                    returns_10d=f.get("roc_10"),
                    rsi_14=f.get("rsi_14"),
                    atr_pct=f.get("atr_pct"),
                    rvol_20d=f.get("rvol_20d"),
                    distance_from_sma20=f.get("dist_sma_20"),
                    distance_from_sma50=f.get("dist_sma_50"),
                )
                for t, f in features_by_ticker.items()
            ],
        ),
    )
    stage_envelopes.append(feature_envelope)

    # --- Step 5: Signal generation ---
    logger.info("Step 5: Generating signals...")
    all_signals = []

    for ticker in qualified_tickers:
        feat = features_by_ticker.get(ticker, {})
        df = price_data.get(ticker)
        if df is None or df.empty:
            continue

        # Breakout model
        breakout = score_breakout(ticker, df, feat)
        if breakout:
            all_signals.append(breakout)

        # Mean reversion model
        mean_rev = score_mean_reversion(ticker, df, feat)
        if mean_rev:
            all_signals.append(mean_rev)

        # Catalyst model
        catalyst = score_catalyst(
            ticker, feat,
            fundamental_data=feat.get("fundamental", {}),
            days_to_earnings=feat.get("days_to_earnings"),
            sentiment=feat.get("sentiment"),
        )
        if catalyst:
            all_signals.append(catalyst)

    logger.info("Generated %d raw signals", len(all_signals))

    # Confluence detection: identify tickers flagged by multiple models
    confluence_map = detect_confluence(all_signals)
    confluence_tickers = [t for t, cr in confluence_map.items() if cr.is_confluence]
    if confluence_tickers:
        logger.info("Confluence tickers (%d): %s", len(confluence_tickers), ", ".join(confluence_tickers))

    # Signal cooldown: suppress recently-fired tickers
    recent_signals = await _get_recent_signals(days=7)
    all_signals = apply_cooldown(all_signals, recent_signals)

    # Deduplicate (keep best per ticker)
    all_signals = deduplicate_signals(all_signals)

    # Signal prefilter envelope
    signal_envelope = StageEnvelope(
        run_id=run_id,
        stage=StageName.SIGNAL_PREFILTER,
        payload=SignalPrefilterPayload(
            asof_date=today,
            candidates=[
                CandidateScores(
                    ticker=s.ticker,
                    model_scores={MODEL_MAP.get(type(s), "unknown"): s.score},
                    aggregate_score=s.score,
                )
                for s in all_signals
            ],
        ),
    )
    stage_envelopes.append(signal_envelope)

    # --- Step 6: Rank and select top candidates ---
    logger.info("Step 6: Ranking candidates...")
    ranked = rank_candidates(
        all_signals,
        regime=regime_assessment.regime,
        features_by_ticker=features_by_ticker,
        top_n=settings.top_n_for_interpretation,
    )
    logger.info("Top %d candidates selected (pre-correlation)", len(ranked))

    # Confluence bonus: boost scores for multi-model agreement
    ranked = apply_confluence_bonus(ranked, confluence_map)

    # Correlation filter: drop highly-correlated picks
    ranked = filter_correlated_picks(ranked, price_data)
    logger.info("Top %d candidates after correlation filter", len(ranked))

    # Update regime envelope with gated candidates
    regime_envelope.payload.gated_candidates = [c.ticker for c in ranked]

    # --- Step 7: LLM Agent Pipeline ---
    logger.info("Step 7: Running LLM agent pipeline...")
    pipeline_result = await run_agent_pipeline(
        candidates=ranked,
        regime_context=regime_context,
        run_id=run_id,
    )

    # Agent review envelope
    agent_review_envelope = StageEnvelope(
        run_id=run_id,
        stage=StageName.AGENT_REVIEW,
        payload=AgentReviewPayload(
            ticker_reviews=[
                TickerReview(
                    ticker=pick.ticker,
                    signal_thesis=pick.interpretation.thesis,
                    signal_confidence=pick.confidence,
                    counter_thesis=(
                        pick.debate.bear_case.argument if pick.debate else None
                    ),
                    risk_decision=pick.risk_gate.decision.value.lower(),
                    risk_notes=pick.risk_gate.reasoning,
                )
                for pick in pipeline_result.approved
            ]
            + [
                TickerReview(
                    ticker=t,
                    signal_thesis="",
                    signal_confidence=0,
                    risk_decision="veto",
                    risk_notes="Vetoed during debate or risk gate",
                )
                for t in pipeline_result.vetoed
            ],
        ),
    )
    stage_envelopes.append(agent_review_envelope)

    # --- Step 7b: Validation Gate (NoSilentPass) ---
    logger.info("Step 7b: Running validation gate...")
    feature_cols = list(features_by_ticker[next(iter(features_by_ticker))].keys()) if features_by_ticker else []
    next_business_day = today + timedelta(days=1)

    # Build validation card from historical outcomes (None if < 10 closed trades)
    from src.output.performance import build_validation_card_from_history
    hist_card = await build_validation_card_from_history(days=90)
    if hist_card:
        logger.info(
            "Validation card from history: %d trades, fragility=%.1f, robust=%s",
            hist_card.total_trades, hist_card.fragility_score, hist_card.is_robust,
        )
    else:
        logger.info("No historical validation card (< 10 closed trades) — structural checks only")

    validation_result = run_validation_checks(
        run_date=today,
        signal_dates=[today] * len(pipeline_result.approved),
        execution_dates=[next_business_day] * len(pipeline_result.approved),
        feature_columns=feature_cols,
        validation_card=hist_card,
    )

    validation_envelope = StageEnvelope(
        run_id=run_id,
        stage=StageName.VALIDATION,
        payload=validation_result,
    )
    stage_envelopes.append(validation_envelope)

    if validation_result.validation_status == "fail":
        logger.warning(
            "VALIDATION GATE FAILED — NoSilentPass blocks all picks. "
            "Failed checks: %s",
            [k for k, v in validation_result.checks.items() if v == "fail"],
        )
        # Override approved picks — emit NoTrade
        pipeline_result.approved.clear()

    # Final output envelope
    if pipeline_result.approved:
        final_envelope = StageEnvelope(
            run_id=run_id,
            stage=StageName.FINAL_OUTPUT,
            payload=FinalOutputPayload(
                decision="Top1To2",
                picks=[
                    FinalPick(
                        ticker=p.ticker,
                        entry_zone=p.entry_price,
                        stop_loss=p.stop_loss,
                        targets=[p.target_1] + ([p.target_2] if p.target_2 else []),
                        confidence=p.confidence,
                        regime_context=regime_assessment.regime.value,
                    )
                    for p in pipeline_result.approved
                ],
            ),
        )
    else:
        no_trade_reason = "Validation gate failed" if validation_result.validation_status == "fail" else "No candidates survived pipeline"
        final_envelope = StageEnvelope(
            run_id=run_id,
            stage=StageName.FINAL_OUTPUT,
            payload=FinalOutputPayload(
                decision="NoTrade",
                no_trade_reason=no_trade_reason,
            ),
        )
    stage_envelopes.append(final_envelope)

    # --- Step 8: Save to database ---
    logger.info("Step 8: Saving results to database...")
    elapsed = time.monotonic() - start_time

    async with get_session() as session:
        # Save daily run
        daily_run = DailyRun(
            run_date=today,
            regime=regime_assessment.regime.value,
            regime_details=regime_context,
            universe_size=len(filtered),
            candidates_scored=len(ranked),
            pipeline_duration_s=round(elapsed, 2),
        )
        session.add(daily_run)

        # Save candidates
        for c in ranked:
            session.add(Candidate(
                run_date=today,
                ticker=c.ticker,
                close_price=c.entry_price,
                avg_daily_volume=c.features.get("vol_sma_20", 0) or 0,
                composite_score=c.regime_adjusted_score,
                signal_model=c.signal_model,
                features=c.features,
            ))

        # Save approved signals and create outcome records for tracking
        new_signals: list[Signal] = []
        for pick in pipeline_result.approved:
            signal = Signal(
                run_date=today,
                ticker=pick.ticker,
                direction=pick.direction,
                signal_model=pick.signal_model,
                entry_price=pick.entry_price,
                stop_loss=pick.stop_loss,
                target_1=pick.target_1,
                target_2=pick.target_2,
                holding_period_days=pick.holding_period,
                confidence=pick.confidence,
                interpreter_thesis=pick.interpretation.thesis,
                debate_summary=pick.debate.rebuttal_summary if pick.debate else None,
                risk_gate_decision=pick.risk_gate.decision.value,
                risk_gate_reasoning=pick.risk_gate.reasoning,
                regime=regime_assessment.regime.value,
                features=pick.features,
            )
            session.add(signal)
            new_signals.append(signal)

        # Flush to get signal IDs before creating outcomes
        await session.flush()

        # Create Outcome records so afternoon check can track them
        next_trading_day = today + timedelta(days=1)
        for signal in new_signals:
            outcome = Outcome(
                signal_id=signal.id,
                ticker=signal.ticker,
                entry_date=next_trading_day,  # T+1 open execution
                entry_price=signal.entry_price,
                still_open=True,
            )
            session.add(outcome)

        # Save agent logs
        for log in pipeline_result.agent_logs:
            session.add(AgentLog(
                run_date=today,
                agent_name=log.get("agent", "unknown"),
                model_used=log.get("model", "unknown"),
                ticker=log.get("ticker"),
                output_data=log,
                tokens_in=log.get("tokens_in"),
                tokens_out=log.get("tokens_out"),
                latency_ms=log.get("latency_ms"),
                cost_usd=log.get("cost_usd"),
            ))

        # Persist stage envelopes for full run traceability
        for envelope in stage_envelopes:
            try:
                payload_dict = (
                    envelope.payload.model_dump()
                    if hasattr(envelope.payload, "model_dump")
                    else envelope.payload
                )
                payload_dict = _json_safe(payload_dict)
            except Exception:
                payload_dict = str(envelope.payload)
            session.add(PipelineArtifact(
                run_id=run_id,
                run_date=today,
                stage=envelope.stage.value,
                status=envelope.status.value,
                payload=payload_dict,
                errors=[e.model_dump() for e in envelope.errors] if envelope.errors else [],
            ))

    # --- Step 8b: Governance record ---
    elapsed = time.monotonic() - start_time
    gov.set_regime(regime_assessment.regime.value)
    gov.set_models_active(get_regime_allowed_models(regime_assessment.regime))
    gov.set_pipeline_stats(
        universe_size=len(filtered),
        candidates_scored=len(ranked),
        picks_approved=len(pipeline_result.approved),
        duration_s=elapsed,
    )

    # Decay detection: compare recent live performance against baseline
    await _check_and_record_decay(gov)

    gov.__exit__(None, None, None)

    # Persist governance record as a pipeline artifact
    async with get_session() as session:
        session.add(PipelineArtifact(
            run_id=run_id,
            run_date=today,
            stage="governance",
            status="success",
            payload=_json_safe(gov.record.to_dict()),
        ))

    # --- Step 8c: Portfolio construction ---
    if pipeline_result.approved:
        trade_plan = build_trade_plan(
            candidates=[{
                "ticker": p.ticker,
                "direction": p.direction,
                "entry_price": p.entry_price,
                "stop_loss": p.stop_loss,
                "target_1": p.target_1,
                "confidence": p.confidence,
                "signal_model": p.signal_model,
                "holding_period": p.holding_period,
                "atr_pct": p.features.get("atr_pct", 0),
                "avg_daily_volume": p.features.get("vol_sma_20", 0),
            } for p in pipeline_result.approved],
            regime=regime_assessment.regime.value,
        )
        for plan in trade_plan:
            logger.info(
                "Trade plan: %s %s — %.1f%% of portfolio ($%.0f, %d shares, R:R %.1f:1)",
                plan.direction, plan.ticker, plan.weight_pct,
                plan.notional_usd, plan.shares, plan.reward_risk_ratio,
            )

    # --- Step 9: Send Telegram alert ---
    logger.info("Step 9: Sending Telegram alert...")
    picks_for_alert = []
    for pick in pipeline_result.approved:
        picks_for_alert.append({
            "ticker": pick.ticker,
            "direction": pick.direction,
            "entry_price": pick.entry_price,
            "stop_loss": pick.stop_loss,
            "target_1": pick.target_1,
            "confidence": pick.confidence,
            "signal_model": pick.signal_model,
            "thesis": pick.interpretation.thesis,
            "holding_period": pick.holding_period,
        })

    validation_failed = validation_result.validation_status == "fail"
    failed_checks = [k for k, v in validation_result.checks.items() if v == "fail"] if validation_failed else None
    alert_msg = format_daily_alert(
        picks_for_alert,
        regime_assessment.regime.value,
        str(today),
        validation_failed=validation_failed,
        failed_checks=failed_checks,
        key_risks=validation_result.key_risks or None,
    )
    await send_alert(alert_msg)

    logger.info(
        "Pipeline complete in %.1fs: %d picks, %d stage envelopes, run_id=%s",
        elapsed, len(pipeline_result.approved), len(stage_envelopes), run_id,
    )


async def _get_recent_signals(days: int = 7) -> list[dict]:
    """Fetch recent signals from DB for cooldown filtering."""
    try:
        async with get_session() as session:
            cutoff = date.today() - timedelta(days=days)
            result = await session.execute(
                select(Signal.ticker, Signal.run_date).where(Signal.run_date >= cutoff)
            )
            return [{"ticker": r[0], "run_date": r[1]} for r in result.all()]
    except Exception as e:
        logger.warning("Failed to fetch recent signals for cooldown: %s", e)
        return []


async def _check_and_record_decay(gov: GovernanceContext) -> None:
    """Check for model decay using recent vs baseline outcomes."""
    try:
        async with get_session() as session:
            # Recent outcomes (last 30 days)
            cutoff_recent = date.today() - timedelta(days=30)
            result = await session.execute(
                select(Outcome).where(
                    Outcome.still_open == False,
                    Outcome.entry_date >= cutoff_recent,
                )
            )
            recent = result.scalars().all()

            # Baseline outcomes (30-90 days ago)
            cutoff_baseline = date.today() - timedelta(days=90)
            result = await session.execute(
                select(Outcome).where(
                    Outcome.still_open == False,
                    Outcome.entry_date >= cutoff_baseline,
                    Outcome.entry_date < cutoff_recent,
                )
            )
            baseline = result.scalars().all()

        if len(recent) < 5 or len(baseline) < 5:
            logger.info("Insufficient data for decay check (recent=%d, baseline=%d)", len(recent), len(baseline))
            return

        recent_trades = [
            {"pnl_pct": o.pnl_pct, "max_adverse": o.max_adverse, "max_favorable": o.max_favorable}
            for o in recent
        ]
        baseline_trades = [
            {"pnl_pct": o.pnl_pct, "max_adverse": o.max_adverse, "max_favorable": o.max_favorable}
            for o in baseline
        ]

        live_metrics = compute_rolling_metrics(recent_trades)
        baseline_metrics = compute_rolling_metrics(baseline_trades)
        decay_result = check_decay(live_metrics, baseline_metrics)

        gov.set_decay(decay_result.is_decaying, decay_result.triggers)
        if decay_result.is_decaying:
            gov.add_flag("decay_detected")
            for trigger in decay_result.triggers:
                gov.add_flag(f"decay: {trigger}")

    except Exception as e:
        logger.warning("Decay check failed: %s", e)


async def run_weekly_meta_review() -> None:
    """Weekly meta-analyst review — runs Sunday at 7:00 PM ET."""
    from src.agents.meta_analyst import MetaAnalystAgent
    from src.output.performance import get_performance_summary

    logger.info("Starting weekly meta-review...")
    today = date.today()

    performance_data = await get_performance_summary(days=30)

    if performance_data.get("total_signals", 0) == 0:
        logger.info("No closed trades in past 30 days — skipping meta-review")
        return

    analyst = MetaAnalystAgent()
    result = await analyst.analyze(performance_data)

    if result:
        async with get_session() as session:
            session.add(AgentLog(
                run_date=today,
                agent_name="meta_analyst",
                model_used=analyst.model,
                ticker=None,
                output_data=result.model_dump(),
            ))

        logger.info(
            "Meta-review complete: win_rate=%.2f, biases=%s, adjustments=%d",
            result.win_rate,
            result.biases_detected,
            len(result.threshold_adjustments),
        )
    else:
        logger.warning("Meta-review returned no result")


async def run_afternoon_check() -> None:
    """Afternoon position check — runs at 4:30 PM ET."""
    logger.info("Running afternoon position check...")
    updates = await check_open_positions()

    if updates:
        from src.output.telegram import format_outcome_alert, send_alert
        msg = format_outcome_alert(updates)
        if msg:
            await send_alert(msg)

    logger.info("Afternoon check complete: %d positions updated", len(updates))


def start_scheduler() -> None:
    """Start the APScheduler with morning and afternoon jobs."""
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger

    settings = get_settings()
    scheduler = AsyncIOScheduler(timezone="US/Eastern")

    # Morning pipeline
    scheduler.add_job(
        run_morning_pipeline,
        CronTrigger(
            hour=settings.morning_run_hour,
            minute=settings.morning_run_minute,
            timezone="US/Eastern",
        ),
        id="morning_pipeline",
        name="Daily Morning Pipeline",
    )

    # Afternoon check
    scheduler.add_job(
        run_afternoon_check,
        CronTrigger(
            hour=settings.afternoon_check_hour,
            minute=settings.afternoon_check_minute,
            timezone="US/Eastern",
        ),
        id="afternoon_check",
        name="Afternoon Position Check",
    )

    # Weekly meta-analyst review (Sunday 7 PM ET)
    scheduler.add_job(
        run_weekly_meta_review,
        CronTrigger(
            day_of_week="sun",
            hour=19,
            minute=0,
            timezone="US/Eastern",
        ),
        id="weekly_meta_review",
        name="Weekly Meta-Analyst Review",
    )

    scheduler.start()
    logger.info(
        "Scheduler started: morning=%02d:%02d, afternoon=%02d:%02d, weekly=Sun 19:00 ET",
        settings.morning_run_hour, settings.morning_run_minute,
        settings.afternoon_check_hour, settings.afternoon_check_minute,
    )
    return scheduler


async def main():
    """Entry point — init DB, optionally run pipeline now, start scheduler."""
    import sys
    import uvicorn

    await init_db()

    if "--run-now" in sys.argv:
        await run_morning_pipeline()
        return

    if "--check-now" in sys.argv:
        await run_afternoon_check()
        return

    if "--meta-now" in sys.argv:
        await run_weekly_meta_review()
        return

    # Start scheduler + API server
    scheduler = start_scheduler()

    port = int(os.environ.get("PORT", 8000))
    config = uvicorn.Config(
        "api.app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    try:
        await server.serve()
    finally:
        scheduler.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
