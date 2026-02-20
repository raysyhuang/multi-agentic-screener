"""Main entry point — daily pipeline orchestration and scheduling."""

from __future__ import annotations

import asyncio
from collections import defaultdict
import hashlib
import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import get_settings, ExecutionMode
from src.contracts import (
    StageEnvelope,
    StageName,
    StageStatus,
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
from src.data.universe_selection import select_ohlcv_tickers
from sqlalchemy import delete, select, func
from src.db.models import DailyRun, Signal, Candidate, AgentLog, Outcome, PipelineArtifact, DivergenceEvent, NearMiss, PositionDailyMetric, SignalExitEvent
from src.db.session import get_session, init_db
from src.features.technical import compute_all_technical_features, compute_rsi2_features, latest_features
from src.features.fundamental import score_earnings_surprise, score_insider_activity, days_to_next_earnings
from src.features.sentiment import score_news_batch
from src.features.regime import classify_regime, get_regime_allowed_models, compute_breadth_score
from src.signals.filter import filter_universe, filter_by_ohlcv, FilterFunnel, OHLCVFunnel
from src.validation.stage_validator import (
    PipelineHealthReport,
    validate_macro_regime,
    validate_universe,
    validate_ohlcv,
    validate_features,
    validate_signals,
    validate_ranking,
    validate_agent_pipeline,
    validate_final_output,
    cross_validate_ohlcv,
)
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
from src.output.telegram import send_alert, format_daily_alert, format_cross_engine_alert
from src.output.performance import check_open_positions
from src.governance.artifacts import GovernanceContext
from src.governance.performance_monitor import (
    compute_rolling_metrics,
    check_decay,
)
from src.portfolio.construct import build_trade_plan
from src.governance.divergence_ledger import (
    freeze_quant_baseline,
    compute_divergences,
)

def _setup_logging() -> None:
    """Configure logging — text or JSON format based on LOG_FORMAT env var."""
    settings = get_settings()
    if settings.log_format == "json":
        import json as _json

        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": self.formatTime(record),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                if hasattr(record, "run_id"):
                    log_entry["run_id"] = record.run_id
                if record.exc_info and record.exc_info[0]:
                    log_entry["exception"] = self.formatException(record.exc_info)
                return _json.dumps(log_entry)

        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logging.root.handlers = [handler]
        logging.root.setLevel(logging.INFO)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )


_setup_logging()
logger = logging.getLogger(__name__)
_EASTERN_TZ = ZoneInfo("America/New_York")


class RunIDFilter(logging.Filter):
    """Attach run_id to all log records within a pipeline execution."""

    def __init__(self, run_id: str):
        super().__init__()
        self.run_id = run_id

    def filter(self, record):
        record.run_id = self.run_id
        return True


def _json_safe(obj):
    """Recursively convert non-JSON-serializable types in a dict/list structure."""
    import math
    import numpy as np
    import pandas as pd
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) else v
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _portfolio_ticker_set(portfolio: list[dict] | None) -> set[str]:
    """Ticker set used for same-day cross-engine alert dedupe."""
    out: set[str] = set()
    for p in portfolio or []:
        t = str(p.get("ticker") or "").upper().strip()
        if t:
            out.add(t)
    return out


def _stable_payload_hash(payload: dict) -> str:
    """Deterministic SHA-256 hash for payload revision tracking."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_engine_run_timestamp(run_timestamp: str | None) -> datetime | None:
    """Parse engine ISO timestamp into timezone-aware UTC datetime."""
    if not run_timestamp:
        return None
    raw = str(run_timestamp).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _trading_date_et(now: datetime | None = None) -> date:
    """Return trading date in US/Eastern.

    Prevents UTC date rollovers from mislabeling evening jobs as "tomorrow".
    """
    current = now.astimezone(_EASTERN_TZ) if now else datetime.now(_EASTERN_TZ)
    return current.date()


async def run_morning_pipeline() -> None:
    """Main daily pipeline — runs at 6:00 AM ET.

    Fail-closed: any unhandled exception guarantees a NoTrade DB record
    and Telegram alert rather than a silent abort.
    """
    import uuid

    start_time = time.monotonic()
    today = _trading_date_et()
    settings = get_settings()
    run_id = uuid.uuid4().hex[:12]

    # Attach run_id to all log records for this pipeline execution
    run_id_filter = RunIDFilter(run_id)
    logging.getLogger().addFilter(run_id_filter)

    logger.info("=" * 60)
    logger.info("Starting morning pipeline for %s (run_id=%s)", today, run_id)
    logger.info("Trading mode: %s", settings.trading_mode)
    logger.info("=" * 60)

    _state: dict = {}
    try:
        await _run_pipeline_core(today, settings, run_id, start_time, _state=_state)
    except Exception as exc:
        elapsed = time.monotonic() - start_time
        logger.exception("PIPELINE FAILED — fail-closed with NoTrade (run_id=%s): %s", run_id, exc)

        # Finalize governance context if it was created (captures exception)
        gov = _state.get("gov")
        if gov:
            try:
                gov.__exit__(type(exc), exc, exc.__traceback__)
            except Exception:
                logger.error("Failed to finalize governance context on failure")

        # Guarantee a DailyRun + NoTrade artifact in DB
        try:
            async with get_session() as session:
                existing_run = await session.execute(
                    select(DailyRun).where(DailyRun.run_date == today)
                )
                daily_run = existing_run.scalar_one_or_none()
                if daily_run:
                    daily_run.regime = "unknown"
                    daily_run.regime_details = {"error": str(exc)}
                    daily_run.pipeline_duration_s = round(elapsed, 2)
                else:
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
    finally:
        logging.getLogger().removeFilter(run_id_filter)


async def _check_paper_gate(settings) -> bool:
    """Enforce 30-day paper trading gate before allowing LIVE mode.

    Returns True if pipeline should proceed, False if blocked.
    Logs a warning and forces PAPER mode if gate fails.
    Override with FORCE_LIVE=true for emergency use (logged as warning).
    """
    if settings.trading_mode != "LIVE":
        return True  # PAPER mode always allowed

    # FORCE_LIVE override for emergency use
    if settings.force_live:
        logger.warning(
            "FORCE_LIVE=true — bypassing 30-day paper trading gate. "
            "This should only be used in emergencies."
        )
        return True

    from src.backtest.metrics import compute_metrics

    # Check run count in last 30 days (non-LIVE runs)
    async with get_session() as session:
        cutoff_30d = date.today() - timedelta(days=30)
        run_count_result = await session.execute(
            select(func.count(DailyRun.id)).where(
                DailyRun.run_date >= cutoff_30d,
            )
        )
        run_count = run_count_result.scalar() or 0

        if run_count < 30:
            msg = (
                f"LIVE GATE BLOCKED: Only {run_count} pipeline runs in last 30 days "
                f"(need 30). Forcing PAPER mode."
            )
            logger.error(msg)
            try:
                await send_alert(f"LIVE GATE BLOCKED\n{msg}")
            except Exception:
                pass
            settings.trading_mode = "PAPER"
            return True

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


def _build_quant_only_result(
    ranked: list,
    regime_context: dict,
    max_picks: int = 2,
) -> PipelineRun:
    """Build a PipelineResult from ranked candidates without any LLM calls.

    Uses deterministic stubs for interpretation/debate/risk_gate fields
    so all downstream consumers (DB, Telegram, envelopes) work unchanged.
    """
    from src.agents.base import (
        SignalInterpretation,
        DebateResult,
        DebatePosition,
        RiskGateOutput,
        GateDecision,
    )
    from src.agents.orchestrator import PipelineResult, PipelineRun

    approved: list[PipelineResult] = []
    for candidate in ranked[:max_picks]:
        stub_interp = SignalInterpretation(
            ticker=candidate.ticker,
            thesis=f"Quant-only pick: {candidate.signal_model} score={candidate.raw_score:.2f}",
            confidence=min(candidate.regime_adjusted_score * 100, 99.0),
            key_drivers=list(candidate.components.keys())[:3] if candidate.components else ["score"],
            risk_flags=[],
            suggested_entry=candidate.entry_price,
            suggested_stop=candidate.stop_loss,
            suggested_target=candidate.target_1,
            timeframe_days=candidate.holding_period,
        )
        stub_debate = DebateResult(
            ticker=candidate.ticker,
            bull_case=DebatePosition(
                position="BULL", argument="Quant signal above threshold.",
                evidence=["composite_score"], weakness="No LLM validation", conviction=60,
            ),
            bear_case=DebatePosition(
                position="BEAR", argument="Skipped in quant_only mode.",
                evidence=[], weakness="N/A", conviction=0,
            ),
            rebuttal_summary="Debate skipped in quant_only mode.",
            final_verdict="PROCEED",
            net_conviction=60,
            key_risk="No LLM adversarial review",
        )
        stub_gate = RiskGateOutput(
            ticker=candidate.ticker,
            decision=GateDecision.APPROVE,
            reasoning="Auto-approved in quant_only mode based on composite score.",
            position_size_pct=5.0,
        )
        approved.append(PipelineResult(
            ticker=candidate.ticker,
            signal_model=candidate.signal_model,
            direction=candidate.direction,
            entry_price=candidate.entry_price,
            stop_loss=candidate.stop_loss,
            target_1=candidate.target_1,
            target_2=candidate.target_2,
            holding_period=candidate.holding_period,
            confidence=stub_interp.confidence,
            interpretation=stub_interp,
            debate=stub_debate,
            risk_gate=stub_gate,
            features=candidate.features,
        ))

    return PipelineRun(
        run_date=date.today(),
        regime=regime_context.get("regime", "unknown"),
        regime_details=regime_context,
        candidates_scored=len(ranked),
        interpreted=0,
        debated=0,
        approved=approved,
        vetoed=[],
        agent_logs=[],
    )


async def _run_hybrid_pipeline(
    ranked: list,
    regime_context: dict,
    run_id: str,
    max_picks: int = 2,
) -> PipelineRun:
    """Run interpreter only, then auto-approve top picks by confidence.

    No debate or risk gate — lower cost than full agentic pipeline.
    """
    from src.agents.base import (
        DebateResult,
        DebatePosition,
        RiskGateOutput,
        GateDecision,
    )
    from src.agents.signal_interpreter import SignalInterpreterAgent
    from src.agents.orchestrator import PipelineResult, PipelineRun

    settings = get_settings()
    interpreter = SignalInterpreterAgent()
    agent_logs: list[dict] = []
    interpretations = []

    for candidate in ranked[:settings.top_n_for_interpretation]:
        retry_result = await interpreter.interpret(candidate, regime_context)
        if retry_result.value:
            interpretations.append((candidate, retry_result.value))
            agent_logs.append({
                "agent": "signal_interpreter",
                "ticker": candidate.ticker,
                "confidence": retry_result.value.confidence,
                "attempts": retry_result.attempt_count,
                **interpreter.last_call_meta,
            })

    # Sort by confidence, take top picks
    interpretations.sort(key=lambda x: x[1].confidence, reverse=True)

    approved: list[PipelineResult] = []
    for candidate, interp in interpretations[:max_picks]:
        stub_debate = DebateResult(
            ticker=candidate.ticker,
            bull_case=DebatePosition(
                position="BULL", argument=interp.thesis,
                evidence=interp.key_drivers, weakness="No adversarial review", conviction=interp.confidence,
            ),
            bear_case=DebatePosition(
                position="BEAR", argument="Skipped in hybrid mode.",
                evidence=[], weakness="N/A", conviction=0,
            ),
            rebuttal_summary="Debate skipped in hybrid mode.",
            final_verdict="PROCEED",
            net_conviction=interp.confidence,
            key_risk="No adversarial debate or risk gate review",
        )
        stub_gate = RiskGateOutput(
            ticker=candidate.ticker,
            decision=GateDecision.APPROVE,
            reasoning="Auto-approved in hybrid mode based on interpreter confidence.",
            position_size_pct=5.0,
        )
        approved.append(PipelineResult(
            ticker=candidate.ticker,
            signal_model=candidate.signal_model,
            direction=candidate.direction,
            entry_price=candidate.entry_price,
            stop_loss=interp.suggested_stop or candidate.stop_loss,
            target_1=interp.suggested_target or candidate.target_1,
            target_2=candidate.target_2,
            holding_period=interp.timeframe_days or candidate.holding_period,
            confidence=interp.confidence,
            interpretation=interp,
            debate=stub_debate,
            risk_gate=stub_gate,
            features=candidate.features,
        ))

    return PipelineRun(
        run_date=date.today(),
        regime=regime_context.get("regime", "unknown"),
        regime_details=regime_context,
        candidates_scored=len(ranked),
        interpreted=len(interpretations),
        debated=0,
        approved=approved,
        vetoed=[],
        agent_logs=agent_logs,
    )


async def _run_pipeline_core(
    today: date, settings, run_id: str, start_time: float,
    _state: dict | None = None,
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
    if _state is not None:
        _state["gov"] = gov
    gov.set_trading_mode(settings.trading_mode)
    gov.set_execution_mode(settings.execution_mode)
    gov.set_config_hash({
        "min_price": settings.min_price,
        "min_adv": settings.min_avg_daily_volume,
        "top_n": settings.top_n_for_interpretation,
        "slippage": settings.slippage_pct,
    })

    # --- Pipeline health report (accumulated throughout the run) ---
    pipeline_health = PipelineHealthReport(run_date=today)

    # --- Step 1: Macro context and regime detection (preliminary, breadth added after OHLCV) ---
    logger.info("Step 1: Fetching macro context...")
    macro = await aggregator.get_macro_context()
    spy_df = macro.get("spy_prices")
    qqq_df = macro.get("qqq_prices")
    # Validate macro DataFrames — empty/None means regime detection will be unreliable
    if spy_df is None or (isinstance(spy_df, pd.DataFrame) and spy_df.empty):
        logger.warning("SPY price data missing — regime detection may be inaccurate")
        spy_df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    if qqq_df is None or (isinstance(qqq_df, pd.DataFrame) and qqq_df.empty):
        logger.warning("QQQ price data missing — regime detection may be inaccurate")
        qqq_df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    regime_assessment = classify_regime(
        spy_df=spy_df,
        qqq_df=qqq_df,
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

    # Validate: macro/regime
    pipeline_health.add_stage(validate_macro_regime(
        macro=macro, spy_df=spy_df, qqq_df=qqq_df,
        regime=regime_assessment.regime.value,
        confidence=regime_assessment.confidence,
        vix=macro.get("vix"),
    ))

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

    # Validate: universe
    pipeline_health.add_stage(validate_universe(
        raw_count=len(raw_universe),
        filtered_count=len(filtered),
        filtered=filtered,
    ))

    # --- Step 3: Fetch OHLCV for filtered universe ---
    logger.info("Step 3: Fetching OHLCV for %d tickers...", len(filtered))
    tickers = select_ohlcv_tickers(
        filtered_universe=filtered,
        max_tickers=settings.max_ohlcv_tickers,
    )
    logger.info(
        "OHLCV selection: %d selected (cap=%d) from %d filtered",
        len(tickers), settings.max_ohlcv_tickers, len(filtered),
    )
    from_date = today - timedelta(days=300)  # 1 year of data for indicators
    price_data = await aggregator.get_bulk_ohlcv(tickers, from_date, today)

    # Further filter by OHLCV quality
    ohlcv_funnel = OHLCVFunnel(total_input=len(tickers))
    qualified_tickers = [t for t in tickers if filter_by_ohlcv(t, price_data.get(t), funnel=ohlcv_funnel)]
    ohlcv_funnel.passed = len(qualified_tickers)
    ohlcv_funnel.log_summary()

    # --- Step 3c: Dataset verification ---
    from src.data.dataset_verification import verify_dataset
    health_report = verify_dataset(filtered, tickers, price_data, qualified_tickers, today)
    logger.info(
        "Dataset health: %s (%d/%d checks passed)",
        "PASS" if health_report.passed else "WARN",
        health_report.passed_count, health_report.total_checks,
    )
    if not health_report.passed:
        for w in health_report.warnings:
            logger.warning("Dataset: %s", w)

    # Validate: OHLCV data quality (includes split artifact detection)
    pipeline_health.add_stage(validate_ohlcv(
        tickers_requested=len(tickers),
        price_data=price_data,
        qualified_count=len(qualified_tickers),
        split_check_tickers=qualified_tickers,
    ))

    # Cross-validate OHLCV for top tickers against second source
    try:
        cross_val = await cross_validate_ohlcv(
            aggregator=aggregator,
            top_tickers=qualified_tickers[:10],
            primary_data=price_data,
            from_date=from_date,
            to_date=today,
        )
        pipeline_health.add_stage(cross_val)
    except Exception as e:
        logger.warning("OHLCV cross-validation failed (non-fatal): %s", e)

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
        degraded_components: list[str] = []
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
            degraded_components.append("fundamentals")

        try:
            news = await aggregator.get_ticker_news(ticker)
            feat["sentiment"] = score_news_batch(news)
        except Exception:
            feat["sentiment"] = {}
            degraded_components.append("sentiment")

        if degraded_components:
            feat["_degraded"] = True
            feat["_degraded_components"] = degraded_components
            logger.warning(
                "Degraded data for %s: missing %s",
                ticker, ", ".join(degraded_components),
            )

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

    # Validate: features
    pipeline_health.add_stage(validate_features(
        features_by_ticker=features_by_ticker,
        qualified_count=len(qualified_tickers),
    ))

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

    # Propagate data quality flags to signals
    for sig in all_signals:
        feat = features_by_ticker.get(sig.ticker, {})
        if feat.get("_degraded"):
            sig.data_quality = "degraded"
            sig.degraded_components = feat.get("_degraded_components", [])
            logger.warning(
                "Signal %s/%s fired on degraded data (missing: %s)",
                sig.ticker, getattr(sig, 'signal_model', 'unknown'),
                ", ".join(feat.get("_degraded_components", [])),
            )
        else:
            sig.data_quality = "full"
            sig.degraded_components = []

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

    # Validate: signals
    pipeline_health.add_stage(validate_signals(
        all_signals=all_signals,
        qualified_count=len(qualified_tickers),
        allowed_models=allowed_models,
    ))

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

    # Validate: ranking
    pipeline_health.add_stage(validate_ranking(
        ranked_count=len(ranked),
        post_correlation_count=len(ranked),
        total_signals=len(all_signals),
    ))

    # --- Step 7: Agent Pipeline (mode-dependent) ---
    execution_mode = ExecutionMode(settings.execution_mode)
    logger.info("Step 7: Running pipeline in %s mode...", execution_mode.value)

    if execution_mode == ExecutionMode.QUANT_ONLY:
        pipeline_result = _build_quant_only_result(
            ranked, regime_context, max_picks=settings.max_final_picks,
        )
    elif execution_mode == ExecutionMode.HYBRID:
        pipeline_result = await _run_hybrid_pipeline(
            ranked, regime_context, run_id=run_id,
            max_picks=settings.max_final_picks,
        )
    else:
        pipeline_result = await run_agent_pipeline(
            candidates=ranked,
            regime_context=regime_context,
            run_id=run_id,
        )

    # --- Step 7a: Divergence Ledger — freeze quant baseline and compute divergences ---
    divergence_records = []
    if execution_mode != ExecutionMode.QUANT_ONLY:
        try:
            quant_baseline_result = _build_quant_only_result(
                ranked, regime_context, max_picks=settings.max_final_picks,
            )
            config_hash_str = json.dumps(
                _json_safe({"min_price": settings.min_price,
                            "min_adv": settings.min_avg_daily_volume,
                            "top_n": settings.top_n_for_interpretation,
                            "max_picks": settings.max_final_picks}),
                sort_keys=True,
            )
            quant_baseline = freeze_quant_baseline(
                ranked=ranked,
                max_picks=settings.max_final_picks,
                regime=regime_assessment.regime.value,
                config_hash=hashlib.md5(config_hash_str.encode()).hexdigest(),
            )
            divergence_records = compute_divergences(
                quant_baseline=quant_baseline,
                agentic_result=pipeline_result,
                agent_logs=pipeline_result.agent_logs,
            )
            if divergence_records:
                logger.info(
                    "Divergence ledger: %d events — %s",
                    len(divergence_records),
                    ", ".join(f"{d.event_type.value}({d.ticker})" for d in divergence_records),
                )
        except Exception as e:
            logger.error("Divergence ledger failed (non-fatal): %s", e)
            divergence_records = []

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

    # Current-run per-pick reward:risk ratios (used by NoSilentPass gate).
    rr_values: list[float] = []
    for p in pipeline_result.approved:
        ep = float(getattr(p, "entry_price", 0) or 0)
        sl = float(getattr(p, "stop_loss", 0) or 0)
        t1 = float(getattr(p, "target_1", 0) or 0)
        direction = str(getattr(p, "direction", "LONG") or "LONG").upper()

        if ep <= 0 or sl <= 0 or t1 <= 0:
            rr_values.append(0.0)
            continue

        if direction == "SHORT":
            risk = sl - ep
            reward = ep - t1
        else:
            risk = ep - sl
            reward = t1 - ep

        rr_values.append((reward / risk) if risk > 0 and reward > 0 else 0.0)

    validation_result = run_validation_checks(
        run_date=today,
        signal_dates=[today] * len(pipeline_result.approved),
        execution_dates=[next_business_day] * len(pipeline_result.approved),
        feature_columns=feature_cols,
        validation_card=hist_card,
        risk_reward_ratios=rr_values,
        min_risk_reward=1.0,
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

    # Validate: agent pipeline
    pipeline_health.add_stage(validate_agent_pipeline(
        execution_mode=execution_mode.value,
        approved_count=len(pipeline_result.approved),
        vetoed_count=len(pipeline_result.vetoed),
        agent_logs=pipeline_result.agent_logs,
        ranked_count=len(ranked),
    ))

    # Validate: final output
    pipeline_health.add_stage(validate_final_output(
        approved_picks=pipeline_result.approved,
        validation_status=validation_result.validation_status,
        regime=regime_assessment.regime.value,
    ))

    # Log pipeline health summary
    logger.info(
        "Pipeline health: %s (%d/%d stages OK) — %d warnings",
        pipeline_health.overall_severity.value.upper(),
        sum(1 for s in pipeline_health.stages if s.passed),
        pipeline_health.total_stages,
        len(pipeline_health.all_warnings),
    )
    for w in pipeline_health.all_warnings:
        logger.warning("Pipeline: %s", w)

    # --- Step 8: Save to database ---
    logger.info("Step 8: Saving results to database...")
    elapsed = time.monotonic() - start_time

    async with get_session() as session:
        # Save daily run (upsert: update if re-run on same day)
        existing_run = await session.execute(
            select(DailyRun).where(DailyRun.run_date == today)
        )
        daily_run = existing_run.scalar_one_or_none()
        if daily_run:
            logger.info("Re-run detected for %s — purging child tables", today)

            # Delete in dependency order (outcomes depend on signals)
            # 1. Find signals from this run_date
            stale_signals = await session.execute(
                select(Signal.id).where(Signal.run_date == today)
            )
            stale_signal_ids = [row[0] for row in stale_signals.fetchall()]
            if stale_signal_ids:
                # Only delete unresolved outcomes — preserve closed ones with real exit data
                await session.execute(
                    delete(Outcome).where(
                        Outcome.signal_id.in_(stale_signal_ids),
                        Outcome.still_open == True,  # noqa: E712
                    )
                )
                # Warn about resolved outcomes we're preserving
                resolved_count_result = await session.execute(
                    select(func.count()).select_from(Outcome).where(
                        Outcome.signal_id.in_(stale_signal_ids),
                        Outcome.still_open == False,  # noqa: E712
                    )
                )
                resolved_count = resolved_count_result.scalar() or 0
                if resolved_count:
                    logger.warning(
                        "Preserving %d resolved outcome(s) from previous run", resolved_count
                    )
                await session.execute(
                    delete(PositionDailyMetric).where(
                        PositionDailyMetric.signal_id.in_(stale_signal_ids)
                    )
                )
                await session.execute(
                    delete(SignalExitEvent).where(
                        SignalExitEvent.signal_id.in_(stale_signal_ids)
                    )
                )

            # 2. Delete flat child tables keyed on run_date
            for model in (Candidate, Signal, AgentLog, PipelineArtifact):
                await session.execute(delete(model).where(model.run_date == today))

            # 3. Delete divergence/near-miss keyed on run_date
            await session.execute(delete(DivergenceEvent).where(DivergenceEvent.run_date == today))
            await session.execute(delete(NearMiss).where(NearMiss.run_date == today))

            # Update the existing DailyRun row
            logger.info("Updating existing DailyRun for %s", today)
            daily_run.regime = regime_assessment.regime.value
            daily_run.regime_details = _json_safe(regime_context)
            daily_run.universe_size = len(filtered)
            daily_run.candidates_scored = len(ranked)
            daily_run.pipeline_duration_s = round(elapsed, 2)
            daily_run.execution_mode = execution_mode.value
            daily_run.dataset_health = _json_safe(health_report.to_dict())
            daily_run.pipeline_health = _json_safe(pipeline_health.to_dict())
        else:
            daily_run = DailyRun(
                run_date=today,
                regime=regime_assessment.regime.value,
                regime_details=_json_safe(regime_context),
                universe_size=len(filtered),
                candidates_scored=len(ranked),
                pipeline_duration_s=round(elapsed, 2),
                execution_mode=execution_mode.value,
                dataset_health=_json_safe(health_report.to_dict()),
                pipeline_health=_json_safe(pipeline_health.to_dict()),
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
                features=_json_safe(c.features),
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
                features=_json_safe(pick.features),
            )
            session.add(signal)
            new_signals.append(signal)

        # Flush to get signal IDs before creating outcomes
        await session.flush()

        # Create Outcome records so afternoon check can track them
        from src.utils.trading_calendar import next_trading_day as _next_td
        next_entry_date = _next_td(today)
        for signal in new_signals:
            outcome = Outcome(
                signal_id=signal.id,
                ticker=signal.ticker,
                entry_date=next_entry_date,  # T+1 open execution (trading day)
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
                output_data=_json_safe(log),
                tokens_in=log.get("tokens_in"),
                tokens_out=log.get("tokens_out"),
                latency_ms=log.get("latency_ms"),
                cost_usd=log.get("cost_usd"),
            ))

        # Persist divergence events
        for drec in divergence_records:
            session.add(DivergenceEvent(
                run_id=run_id,
                run_date=today,
                ticker=drec.ticker,
                event_type=drec.event_type.value,
                execution_mode=execution_mode.value,
                quant_rank=drec.quant_rank,
                agentic_rank=drec.agentic_rank,
                quant_size=drec.quant_size,
                agentic_size=drec.agentic_size,
                quant_score=drec.quant_score,
                agentic_score=drec.agentic_score,
                reason_codes=drec.reason_codes,
                llm_cost_usd=drec.llm_cost_usd,
                confidence=drec.confidence,
                regime=regime_assessment.regime.value,
                quant_baseline_snapshot=_json_safe(quant_baseline) if divergence_records else None,
                quant_entry_price=drec.quant_entry_price,
                quant_stop_loss=drec.quant_stop_loss,
                quant_target_1=drec.quant_target_1,
                quant_holding_period=drec.quant_holding_period,
                quant_direction=drec.quant_direction,
                outcome_resolved=False,
            ))

        # Persist near-miss records
        if hasattr(pipeline_result, 'near_misses'):
            for nm in pipeline_result.near_misses:
                session.add(NearMiss(
                    run_id=run_id,
                    run_date=today,
                    ticker=nm.ticker,
                    stage=nm.stage,
                    debate_verdict=nm.debate_verdict,
                    net_conviction=nm.net_conviction,
                    bull_conviction=nm.bull_conviction,
                    bear_conviction=nm.bear_conviction,
                    key_risk=nm.key_risk,
                    risk_gate_decision=nm.risk_gate_decision,
                    risk_gate_reasoning=nm.risk_gate_reasoning,
                    interpreter_confidence=nm.interpreter_confidence,
                    signal_model=nm.signal_model,
                    regime=regime_assessment.regime.value,
                    entry_price=nm.entry_price,
                    stop_loss=nm.stop_loss,
                    target_price=nm.target_price,
                    timeframe_days=nm.timeframe_days,
                ))
            if pipeline_result.near_misses:
                logger.info(
                    "Near-misses logged: %d — %s",
                    len(pipeline_result.near_misses),
                    ", ".join(f"{nm.stage}({nm.ticker}, conv={nm.net_conviction:.0f})" for nm in pipeline_result.near_misses),
                )

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
        execution_mode=execution_mode.value,
    )
    try:
        await send_alert(alert_msg)
    except Exception as alert_exc:
        logger.error("Step 9 Telegram alert failed (non-fatal): %s", alert_exc)

    # --- Steps 10-14: Cross-Engine Integration ---
    await _run_cross_engine_steps(
        today=today,
        run_id=run_id,
        regime_context={
            "regime": regime_assessment.regime.value,
            "confidence": regime_assessment.confidence,
            "vix": regime_context.get("vix"),
        },
        screener_picks=picks_for_alert,
        settings=settings,
    )

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


def _validate_cross_engine_pick_risk(pick: dict) -> tuple[bool, str]:
    """Require complete long-side risk parameters for cross-engine picks."""
    ticker = str(pick.get("ticker", "")).upper().strip()
    if not ticker:
        return False, "missing ticker"
    if ticker in {"CASH", "USD", "USDT", "USDC", "NONE", "N/A", "NA"}:
        return False, f"{ticker}: non-tradable placeholder ticker"
    if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", ticker):
        return False, f"{ticker}: invalid ticker format"

    entry = pick.get("entry_price")
    stop = pick.get("stop_loss")
    target = pick.get("target_price")

    if entry is None or entry <= 0:
        return False, f"{ticker}: invalid entry_price={entry}"
    if stop is None or stop <= 0:
        return False, f"{ticker}: missing/invalid stop_loss={stop}"
    if target is None or target <= 0:
        return False, f"{ticker}: missing/invalid target_price={target}"
    if not (stop < entry < target):
        return False, (
            f"{ticker}: invalid ordering stop/entry/target "
            f"({stop}, {entry}, {target})"
        )
    return True, ""


async def _run_cross_engine_steps(
    today: date,
    run_id: str,
    regime_context: dict,
    screener_picks: list[dict],
    settings,
) -> None:
    """Steps 10-14: Cross-engine integration pipeline.

    Step 10: Collect external engine results (3 parallel HTTP calls)
    Step 11: Resolve previous day's engine pick outcomes → update credibility weights
    Step 12: Cross-Engine Verifier Agent (audit credibility, detect anomalies)
    Step 13: Cross-Engine Synthesizer Agent (final portfolio, executive summary)
    Step 14: Save to DB + send Telegram cross-engine alert
    """
    if not settings.cross_engine_enabled:
        logger.info("Cross-engine integration disabled, skipping steps 10-14")
        return

    from src.engines.collector import collect_engine_results
    from src.engines.credibility import (
        compute_credibility_snapshot,
        compute_weighted_picks,
    )
    from src.engines.regime_gate import apply_regime_strategy_gate
    from src.engines.outcome_resolver import resolve_engine_outcomes
    from src.agents.cross_engine_verifier import CrossEngineVerifierAgent
    from src.agents.cross_engine_synthesizer import CrossEngineSynthesizerAgent
    from src.db.models import (
        ExternalEngineResult,
        ExternalEngineResultRevision,
        EnginePickOutcome,
        CrossEngineSynthesis,
    )

    try:
        # --- Step 10: Collect external engine results ---
        logger.info("Step 10: Collecting external engine results...")
        engine_results = await collect_engine_results()

        if not engine_results:
            logger.info("No external engine results available, recording degraded synthesis state")
            from src.data.dataset_verification import verify_cross_engine
            cross_health = verify_cross_engine(engine_results=[], regime_context=regime_context)

            async with get_session() as session:
                existing_synth = await session.execute(
                    select(CrossEngineSynthesis).where(CrossEngineSynthesis.run_date == today)
                )
                row = existing_synth.scalar_one_or_none()
                summary = (
                    "No external engine results passed validation this cycle. "
                    "Cross-engine synthesis skipped and marked degraded."
                )
                if row:
                    row.convergent_tickers = []
                    row.portfolio_recommendation = []
                    row.regime_consensus = summary
                    row.engines_reporting = 0
                    row.executive_summary = summary
                    row.verifier_notes = {"skipped": "no_engine_results"}
                    row.credibility_weights = {}
                    row.cross_engine_health = _json_safe(cross_health.to_dict())
                else:
                    session.add(CrossEngineSynthesis(
                        run_date=today,
                        convergent_tickers=[],
                        portfolio_recommendation=[],
                        regime_consensus=summary,
                        engines_reporting=0,
                        executive_summary=summary,
                        verifier_notes={"skipped": "no_engine_results"},
                        credibility_weights={},
                        cross_engine_health=_json_safe(cross_health.to_dict()),
                    ))
            return

        # Store raw results in DB (true upsert + immutable revision trail)
        async with get_session() as session:
            for er in engine_results:
                try:
                    engine_run_date = date.fromisoformat(er.run_date)
                except ValueError:
                    logger.warning(
                        "Engine %s returned invalid run_date '%s'; falling back to %s",
                        er.engine_name, er.run_date, today,
                    )
                    engine_run_date = today

                payload_dict = _json_safe(er.model_dump())
                payload_hash = _stable_payload_hash(payload_dict)
                source_run_ts = _parse_engine_run_timestamp(er.run_timestamp)
                if source_run_ts is None:
                    logger.warning(
                        "Engine %s returned unparseable run_timestamp '%s'",
                        er.engine_name,
                        er.run_timestamp,
                    )

                # Check for existing result row for this engine/date.
                existing = await session.execute(
                    select(ExternalEngineResult).where(
                        ExternalEngineResult.engine_name == er.engine_name,
                        ExternalEngineResult.run_date == engine_run_date,
                    )
                )
                existing_row = existing.scalar_one_or_none()

                # Upsert metadata row.
                if existing_row:
                    new_revision = (existing_row.ingest_revision or 1) + 1
                    existing_row.ingest_revision = new_revision
                    existing_row.status = er.status
                    existing_row.regime = er.regime
                    existing_row.picks_count = len(er.picks)
                    existing_row.payload = payload_dict
                    existing_row.source_run_timestamp = source_run_ts
                    existing_row.source_payload_hash = payload_hash
                    existing_row.last_ingested_at = datetime.now(timezone.utc)
                    engine_result_id = existing_row.id
                else:
                    new_revision = 1
                    row = ExternalEngineResult(
                        engine_name=er.engine_name,
                        run_date=engine_run_date,
                        status=er.status,
                        regime=er.regime,
                        picks_count=len(er.picks),
                        ingest_revision=new_revision,
                        source_run_timestamp=source_run_ts,
                        source_payload_hash=payload_hash,
                        last_ingested_at=datetime.now(timezone.utc),
                        payload=payload_dict,
                    )
                    session.add(row)
                    await session.flush()
                    engine_result_id = row.id

                # Append immutable revision snapshot for rerun auditability.
                session.add(ExternalEngineResultRevision(
                    engine_result_id=engine_result_id,
                    engine_name=er.engine_name,
                    run_date=engine_run_date,
                    revision=new_revision,
                    status=er.status,
                    regime=er.regime,
                    picks_count=len(er.picks),
                    source_run_timestamp=source_run_ts,
                    source_payload_hash=payload_hash,
                    payload=payload_dict,
                ))
                logger.info(
                    "Engine %s upserted for %s at revision=%d (hash=%s)",
                    er.engine_name,
                    engine_run_date,
                    new_revision,
                    payload_hash[:12],
                )

                # Upsert per-pick outcomes for this engine/date.
                # If any outcomes are already resolved, preserve them for historical integrity.
                existing_pick_rows = (
                    await session.execute(
                        select(EnginePickOutcome).where(
                            EnginePickOutcome.engine_name == er.engine_name,
                            EnginePickOutcome.run_date == engine_run_date,
                        )
                    )
                ).scalars().all()
                resolved_rows = sum(1 for row in existing_pick_rows if row.outcome_resolved)
                if resolved_rows > 0:
                    logger.warning(
                        "Engine %s rerun for %s has %d resolved pick outcomes; "
                        "preserving existing outcomes and skipping pick refresh",
                        er.engine_name,
                        engine_run_date,
                        resolved_rows,
                    )
                    continue

                if existing_pick_rows:
                    await session.execute(
                        delete(EnginePickOutcome).where(
                            EnginePickOutcome.engine_name == er.engine_name,
                            EnginePickOutcome.run_date == engine_run_date,
                        )
                    )

                seen_tickers: set[str] = set()
                for pick in er.picks:
                    ticker_key = str(pick.ticker).upper()
                    if ticker_key in seen_tickers:
                        logger.warning(
                            "Engine %s returned duplicate ticker %s for %s; "
                            "keeping first occurrence only",
                            er.engine_name,
                            ticker_key,
                            engine_run_date,
                        )
                        continue
                    seen_tickers.add(ticker_key)
                    session.add(EnginePickOutcome(
                        engine_name=er.engine_name,
                        run_date=engine_run_date,
                        ticker=pick.ticker,
                        strategy=pick.strategy,
                        entry_price=pick.entry_price,
                        target_price=pick.target_price,
                        stop_loss=pick.stop_loss,
                        confidence=pick.confidence,
                        holding_period_days=pick.holding_period_days,
                    ))

        logger.info("Step 10 complete: %d engines reported", len(engine_results))

        # --- Step 10b: Cross-engine dataset health check ---
        from src.data.dataset_verification import verify_cross_engine
        cross_health = verify_cross_engine(
            engine_results=[
                {
                    "engine_name": er.engine_name,
                    "run_date": er.run_date,
                    "regime": er.regime,
                    "status": er.status,
                    "picks": [p.model_dump() for p in er.picks],
                    "candidates_screened": getattr(er, "candidates_screened", 0),
                }
                for er in engine_results
            ],
            regime_context=regime_context,
        )
        logger.info(
            "Cross-engine health: %s (%d/%d checks passed)",
            "PASS" if cross_health.passed else "WARN",
            cross_health.passed_count, cross_health.total_checks,
        )
        if not cross_health.passed:
            for w in cross_health.warnings:
                logger.warning("CrossEngine: %s", w)

        # --- Step 11: Resolve previous engine pick outcomes ---
        logger.info("Step 11: Resolving previous engine pick outcomes...")
        feedback = await resolve_engine_outcomes()
        for fb in feedback:
            logger.info(
                "Engine feedback — %s: %d resolved, %.0f%% hit rate, avg return %+.1f%%",
                fb["engine_name"], fb["resolved_count"],
                fb["hit_rate"] * 100, fb["avg_return_pct"],
            )

        # Compute credibility snapshot
        cred_snapshot = await compute_credibility_snapshot()

        # Build flat pick list for weighting (strict risk-schema gate)
        all_picks: list[dict] = []
        dropped_by_engine: dict[str, int] = defaultdict(int)
        drop_reasons: dict[str, list[str]] = defaultdict(list)
        for er in engine_results:
            for pick in er.picks:
                normalized_pick = {
                    "engine_name": er.engine_name,
                    "ticker": pick.ticker,
                    "strategy": pick.strategy,
                    "entry_price": pick.entry_price,
                    "stop_loss": pick.stop_loss,
                    "target_price": pick.target_price,
                    "confidence": pick.confidence,
                    "holding_period_days": pick.holding_period_days,
                    "thesis": pick.thesis,
                    "risk_factors": pick.risk_factors,
                }
                ok, reason = _validate_cross_engine_pick_risk(normalized_pick)
                if not ok:
                    dropped_by_engine[er.engine_name] += 1
                    if len(drop_reasons[er.engine_name]) < 5:
                        drop_reasons[er.engine_name].append(reason)
                    continue
                all_picks.append(normalized_pick)

        # Add screener's own picks with engine_name="multi_agentic_screener"
        for sp in screener_picks:
            normalized_pick = {
                "engine_name": "multi_agentic_screener",
                "ticker": sp.get("ticker", ""),
                "strategy": sp.get("signal_model", ""),
                "entry_price": sp.get("entry_price", 0),
                "stop_loss": sp.get("stop_loss", 0),
                "target_price": sp.get("target_1", 0),
                "confidence": sp.get("confidence", 0),
                "holding_period_days": sp.get("holding_period", 10),
                "thesis": sp.get("thesis", ""),
                "risk_factors": [],
            }
            ok, reason = _validate_cross_engine_pick_risk(normalized_pick)
            if not ok:
                dropped_by_engine["multi_agentic_screener"] += 1
                if len(drop_reasons["multi_agentic_screener"]) < 5:
                    drop_reasons["multi_agentic_screener"].append(reason)
                continue
            all_picks.append(normalized_pick)

        for engine_name, dropped in sorted(dropped_by_engine.items()):
            logger.warning(
                "Step 11 risk gate dropped %d picks from %s due to missing/invalid risk params: %s",
                dropped,
                engine_name,
                drop_reasons.get(engine_name, [])[:3],
            )

        logger.info("Step 11 complete: %d picks collected for weighting", len(all_picks))

        # --- Step 12: Cross-Engine Verifier ---
        verifier_output_dict = {}
        verifier_result = None
        if settings.cross_engine_verify_before_synthesize:
            logger.info("Step 12: Running cross-engine verifier...")
            verifier = CrossEngineVerifierAgent()

            engine_results_dicts = [_json_safe(er.model_dump()) for er in engine_results]
            cred_stats_dict = {
                name: {
                    "hit_rate": s.hit_rate,
                    "weight": s.weight,
                    "resolved_picks": s.resolved_picks,
                    "brier_score": s.brier_score,
                    "has_enough_data": s.has_enough_data,
                    "per_strategy": s.per_strategy,
                }
                for name, s in cred_snapshot.engine_stats.items()
            }

            verifier_result = await verifier.verify(
                engine_results=engine_results_dicts,
                credibility_stats=cred_stats_dict,
                regime_context=regime_context,
            )
            verifier_output_dict = _json_safe(verifier_result.model_dump())

            # Apply verifier weight adjustments to credibility stats before
            # computing weighted picks — this closes the gap where the verifier's
            # regime-aware adjustments were only seen as LLM context, never applied.
            if verifier_result.weight_adjustments:
                for engine_name, adj in verifier_result.weight_adjustments.items():
                    if engine_name in cred_snapshot.engine_stats:
                        old_w = cred_snapshot.engine_stats[engine_name].weight
                        new_w = old_w * adj.weight_multiplier
                        unclamped_w = new_w
                        new_w = max(0.1, min(3.0, new_w))  # clamp to [0.1, 3.0]
                        if unclamped_w != new_w:
                            logger.warning(
                                "Verifier weight for %s clamped: %.3f → %.3f "
                                "(requested ×%.2f would yield %.3f, outside [0.1, 3.0])",
                                engine_name, old_w, new_w,
                                adj.weight_multiplier, unclamped_w,
                            )
                        cred_snapshot.engine_stats[engine_name].weight = round(new_w, 3)
                        logger.info(
                            "Verifier adjusted %s weight: %.3f → %.3f (×%.2f, reason: %s)",
                            engine_name, old_w, new_w,
                            adj.weight_multiplier, adj.reason,
                        )

            logger.info(
                "Step 12 complete: %d verified picks, %d red flags, %d weight adjustments applied",
                len(verifier_result.verified_picks), len(verifier_result.red_flags),
                len(verifier_result.weight_adjustments),
            )
        else:
            logger.info("Step 12: Verifier disabled, skipping")

        # Compute weighted picks AFTER verifier adjustments are applied
        weighted_picks = compute_weighted_picks(all_picks, cred_snapshot.engine_stats)
        weighted_picks, gate_meta = apply_regime_strategy_gate(
            weighted_picks=weighted_picks,
            regime=regime_context.get("regime", "unknown"),
            settings=settings,
        )
        if gate_meta.get("applied"):
            logger.info(
                "Step 11.5 regime strategy gate applied: regime=%s, dropped=%d, penalized=%d",
                gate_meta.get("regime"),
                gate_meta.get("dropped", 0),
                gate_meta.get("penalized", 0),
            )
            if gate_meta.get("dropped_tickers"):
                logger.warning(
                    "Step 11.5 dropped bear-regime-incompatible tickers: %s",
                    gate_meta["dropped_tickers"][:5],
                )
        logger.info("Weighted picks computed: %d tickers ranked", len(weighted_picks))

        cred_weights_dict = {
            name: {"weight": s.weight, "hit_rate": s.hit_rate, "resolved_picks": s.resolved_picks}
            for name, s in cred_snapshot.engine_stats.items()
        }

        # --- Step 13: Cross-Engine Synthesizer ---
        if weighted_picks:
            logger.info("Step 13: Running cross-engine synthesizer...")
            synthesizer = CrossEngineSynthesizerAgent()
            synthesis = await synthesizer.synthesize(
                weighted_picks=weighted_picks,
                verifier_output=verifier_output_dict,
                credibility_weights=cred_weights_dict,
                regime_context=regime_context,
                screener_picks=screener_picks,
            )
        else:
            from src.agents.cross_engine_synthesizer import SynthesizerOutput

            logger.warning("Step 13 skipped: no risk-defined weighted picks available")
            synthesis = SynthesizerOutput(
                convergent_picks=[],
                unique_picks=[],
                portfolio=[],
                regime_consensus=(
                    "No risk-defined picks available after stop/target validation "
                    "across reporting engines."
                ),
                executive_summary=(
                    "NO TRADES THIS CYCLE — all candidate picks failed strict risk "
                    "parameter checks (missing/invalid stop-loss or target)."
                ),
            )

        # Final safety gate: drop any synthesized portfolio positions with
        # non-tradable/invalid ticker or invalid risk tuple.
        if synthesis.portfolio:
            valid_portfolio = []
            dropped_portfolio_reasons: list[str] = []
            for pos in synthesis.portfolio:
                ok, reason = _validate_cross_engine_pick_risk(pos.model_dump())
                if ok:
                    valid_portfolio.append(pos)
                else:
                    if len(dropped_portfolio_reasons) < 5:
                        dropped_portfolio_reasons.append(reason)
            if len(valid_portfolio) != len(synthesis.portfolio):
                logger.warning(
                    "Step 13 safety gate dropped %d synthesized positions due to "
                    "invalid ticker/risk schema: %s",
                    len(synthesis.portfolio) - len(valid_portfolio),
                    dropped_portfolio_reasons,
                )
                synthesis.portfolio = valid_portfolio

        # Low-overlap guardrail: when there are no convergent picks, force a
        # smaller, lighter portfolio rather than allowing broad unique-only risk.
        if not synthesis.convergent_picks and synthesis.portfolio:
            max_positions = max(1, int(settings.low_overlap_max_positions))
            max_total_weight = max(1.0, float(settings.low_overlap_max_total_weight_pct))

            original_count = len(synthesis.portfolio)
            original_total_weight = sum(float(p.weight_pct) for p in synthesis.portfolio)

            # Keep only top-N positions in declared order.
            if original_count > max_positions:
                synthesis.portfolio = synthesis.portfolio[:max_positions]

            # Cap total gross exposure in low-overlap sessions.
            new_total_weight = sum(float(p.weight_pct) for p in synthesis.portfolio)
            if new_total_weight > max_total_weight:
                scale = max_total_weight / new_total_weight
                for p in synthesis.portfolio:
                    p.weight_pct = round(float(p.weight_pct) * scale, 2)

            logger.info(
                "Low-overlap guardrail applied: convergent=0, positions %d→%d, "
                "weight %.1f%%→%.1f%%",
                original_count,
                len(synthesis.portfolio),
                original_total_weight,
                sum(float(p.weight_pct) for p in synthesis.portfolio),
            )

        logger.info(
            "Step 13 complete: %d convergent, %d portfolio positions",
            len(synthesis.convergent_picks), len(synthesis.portfolio),
        )

        # --- Step 13.5: Capital Guardian (portfolio-level risk defense) ---
        guardian_summary = ""
        if settings.guardian_enabled:
            from src.portfolio.capital_guardian import (
                compute_portfolio_risk_state,
                compute_guardian_verdict,
                apply_guardian_to_portfolio,
                format_guardian_summary,
            )

            logger.info("Step 13.5: Running Capital Guardian...")
            risk_state = await compute_portfolio_risk_state()
            guardian_verdict = compute_guardian_verdict(
                risk_state=risk_state,
                regime=regime_context.get("regime", "bull"),
            )

            # Apply sizing adjustments to the synthesis portfolio
            original_portfolio = [p.model_dump() for p in synthesis.portfolio]
            adjusted_portfolio = apply_guardian_to_portfolio(
                original_portfolio, guardian_verdict
            )

            # Update synthesis with guardian-adjusted portfolio
            from src.agents.cross_engine_synthesizer import PortfolioPosition
            synthesis.portfolio = [
                PortfolioPosition.model_validate(p) for p in adjusted_portfolio
            ]

            guardian_summary = format_guardian_summary(guardian_verdict)

            if guardian_verdict.halt:
                logger.warning("Capital Guardian HALTED trading: %s", guardian_verdict.halt_reason)
                # Keep persisted/alerted summary strictly consistent with guardian-enforced no-trade state.
                synthesis.executive_summary = (
                    f"HALT — NO TRADES THIS CYCLE. {guardian_verdict.halt_reason}. "
                    "All candidate positions are blocked until risk state improves."
                )
            else:
                # Reconcile narrative with executed portfolio after guardian sizing.
                original_total_weight = sum(float(p.get("weight_pct", 0.0)) for p in original_portfolio)
                adjusted_total_weight = sum(float(p.get("weight_pct", 0.0)) for p in adjusted_portfolio)
                changed = (
                    len(original_portfolio) != len(adjusted_portfolio)
                    or round(original_total_weight, 2) != round(adjusted_total_weight, 2)
                )
                if changed:
                    tickers = ", ".join(
                        f"{p['ticker']} ({p.get('weight_pct', 0):.2f}%)"
                        for p in adjusted_portfolio
                    ) if adjusted_portfolio else "none"
                    synthesis.executive_summary = (
                        f"Guardian-adjusted allocation: {len(adjusted_portfolio)} positions, "
                        f"{adjusted_total_weight:.1f}% gross exposure. Active names: {tickers}."
                    )
                logger.info(
                    "Step 13.5 complete: sizing=%.0f%%, %d→%d positions, warnings=%d",
                    guardian_verdict.sizing_multiplier * 100,
                    len(original_portfolio), len(adjusted_portfolio),
                    len(guardian_verdict.warnings),
                )
        else:
            logger.info("Step 13.5: Capital Guardian disabled, skipping")

        # --- Step 14: Save to DB + send Telegram ---
        logger.info("Step 14: Saving synthesis and sending alert...")
        synthesis_dict = _json_safe(synthesis.model_dump())
        should_send_alert = True
        async with get_session() as session:
            # Check for existing synthesis (re-run safety)
            existing_synth = await session.execute(
                select(CrossEngineSynthesis).where(CrossEngineSynthesis.run_date == today)
            )
            row = existing_synth.scalar_one_or_none()
            cross_health_dict = _json_safe(cross_health.to_dict())
            if row:
                existing_tickers = _portfolio_ticker_set(row.portfolio_recommendation)
                new_tickers = _portfolio_ticker_set(synthesis_dict.get("portfolio"))
                existing_halt = "HALT" in (row.executive_summary or "").upper()
                new_halt = "HALT" in (synthesis.executive_summary or "").upper()
                engines_increased = row.engines_reporting < len(engine_results)

                # Re-alert only when actionable state changes.
                should_send_alert = (
                    engines_increased
                    or existing_tickers != new_tickers
                    or existing_halt != new_halt
                )

                logger.info(
                    "Step 14 dedupe: engines %d→%d, tickers %s→%s, halt %s→%s, send=%s",
                    row.engines_reporting,
                    len(engine_results),
                    sorted(existing_tickers),
                    sorted(new_tickers),
                    existing_halt,
                    new_halt,
                    should_send_alert,
                )
                row.convergent_tickers = synthesis_dict.get("convergent_picks")
                row.portfolio_recommendation = synthesis_dict.get("portfolio")
                row.regime_consensus = synthesis.regime_consensus
                row.engines_reporting = len(engine_results)
                row.executive_summary = synthesis.executive_summary
                row.verifier_notes = verifier_output_dict
                row.credibility_weights = cred_weights_dict
                row.cross_engine_health = cross_health_dict
            else:
                session.add(CrossEngineSynthesis(
                    run_date=today,
                    convergent_tickers=synthesis_dict.get("convergent_picks"),
                    portfolio_recommendation=synthesis_dict.get("portfolio"),
                    regime_consensus=synthesis.regime_consensus,
                    engines_reporting=len(engine_results),
                    executive_summary=synthesis.executive_summary,
                    verifier_notes=verifier_output_dict,
                    credibility_weights=cred_weights_dict,
                    cross_engine_health=cross_health_dict,
                ))

        # Send cross-engine Telegram alert
        if should_send_alert:
            cross_alert = format_cross_engine_alert(
                synthesis={
                    **synthesis_dict,
                    "engines_reporting": len(engine_results),
                },
                credibility=cred_weights_dict,
            )
            if guardian_summary:
                cross_alert += guardian_summary
            await send_alert(cross_alert)
        else:
            logger.info(
                "Step 14: synthesis unchanged for %s; suppressing duplicate cross-engine alert",
                today,
            )

        logger.info(
            "Steps 10-14 complete: cross-engine synthesis saved; alert_sent=%s",
            should_send_alert,
        )

    except Exception as e:
        logger.error("Cross-engine steps failed (non-fatal): %s", e, exc_info=True)


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

    # Enrich with divergence stats (fail-closed)
    try:
        from src.output.performance import get_divergence_stats
        divergence_stats = await get_divergence_stats(days=30)
        if divergence_stats is not None:
            performance_data["divergence"] = divergence_stats
            logger.info(
                "Divergence stats added: %d events, %d resolved",
                divergence_stats.get("total_events", 0),
                divergence_stats.get("total_resolved", 0),
            )
    except Exception as e:
        logger.warning("Failed to fetch divergence stats (non-fatal): %s", e)

    # Enrich with near-miss stats (fail-closed)
    try:
        from src.output.performance import get_near_miss_stats
        near_miss_stats = await get_near_miss_stats(days=30)
        if near_miss_stats is not None:
            performance_data["near_misses"] = near_miss_stats
            logger.info(
                "Near-miss stats added: %d near-misses",
                near_miss_stats.get("total_near_misses", 0),
            )
    except Exception as e:
        logger.warning("Failed to fetch near-miss stats (non-fatal): %s", e)

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

        # Process threshold adjustments (dry-run by default)
        if result.threshold_adjustments:
            from src.governance.threshold_manager import process_adjustments
            adj_result = process_adjustments(
                adjustments=result.threshold_adjustments,
                run_date=str(today),
                dry_run=True,
            )
            logger.info(
                "Threshold proposals: %d applied (dry-run), %d rejected",
                len(adj_result.applied), len(adj_result.rejected),
            )
            async with get_session() as session:
                session.add(AgentLog(
                    run_date=today,
                    agent_name="threshold_manager",
                    model_used="deterministic",
                    ticker=None,
                    output_data=adj_result.snapshot.to_dict(),
                ))
    else:
        logger.warning("Meta-review returned no result")


async def run_afternoon_check() -> None:
    """Afternoon position check — runs at 4:30 PM ET."""
    logger.info("Running afternoon position check...")
    updates, health_cards, state_changes, resolved_near_misses = await check_open_positions()

    if updates:
        from src.output.telegram import format_outcome_alert, send_alert
        msg = format_outcome_alert(updates)
        if msg:
            await send_alert(msg)

    if state_changes:
        from src.output.telegram import format_health_alert, send_alert
        health_msg = format_health_alert(state_changes)
        if health_msg:
            await send_alert(health_msg)

    if resolved_near_misses:
        from src.output.telegram import format_near_miss_resolution_alert, send_alert
        nm_msg = format_near_miss_resolution_alert(resolved_near_misses)
        if nm_msg:
            await send_alert(nm_msg)

    logger.info(
        "Afternoon check complete: %d positions updated, %d health cards, "
        "%d state changes, %d near-misses resolved",
        len(updates), len(health_cards), len(state_changes),
        len(resolved_near_misses),
    )


async def run_evening_collection() -> None:
    """Evening cross-engine collection — runs at 9:30 PM ET.

    Standalone trigger for Steps 10-14 (cross-engine integration) that
    doesn't require the full morning pipeline to re-run.  Fetches today's
    screener picks from the DB so the synthesizer sees them alongside the
    three external engines.
    """
    import uuid

    requested_date = _trading_date_et()
    context_date = requested_date
    settings = get_settings()
    run_id = f"eve-{uuid.uuid4().hex[:8]}"

    logger.info("=" * 60)
    logger.info(
        "Starting evening cross-engine collection for %s (run_id=%s)",
        requested_date,
        run_id,
    )
    logger.info("=" * 60)

    if not settings.cross_engine_enabled:
        logger.info("Cross-engine integration disabled, nothing to do")
        return

    # Fetch screener context from DB. If requested date has no morning run yet
    # (for example manual trigger after midnight ET), fall back to latest prior
    # run_date so synthesis stays aligned with a real pipeline cycle.
    screener_picks: list[dict] = []
    regime_context: dict = {"regime": "unknown", "confidence": 0.0, "vix": None}
    try:
        async with get_session() as session:
            # Get the latest DailyRun for the requested date.
            daily_run_result = await session.execute(
                select(DailyRun).where(DailyRun.run_date == requested_date)
                .order_by(DailyRun.id.desc()).limit(1)
            )
            daily_run = daily_run_result.scalar_one_or_none()
            if not daily_run:
                fallback_daily_run_result = await session.execute(
                    select(DailyRun)
                    .where(DailyRun.run_date <= requested_date)
                    .order_by(DailyRun.run_date.desc(), DailyRun.id.desc())
                    .limit(1)
                )
                daily_run = fallback_daily_run_result.scalar_one_or_none()
                if daily_run:
                    context_date = daily_run.run_date
                    logger.warning(
                        "No morning run for requested date %s; using fallback context date %s",
                        requested_date,
                        context_date,
                    )
            if daily_run and daily_run.regime:
                regime_context["regime"] = daily_run.regime
                if daily_run.regime_details:
                    regime_context["confidence"] = daily_run.regime_details.get("confidence", 0.0)
                    regime_context["vix"] = daily_run.regime_details.get("vix")

            # Get approved picks from the aligned context date.
            artifact_result = await session.execute(
                select(PipelineArtifact).where(
                    PipelineArtifact.run_date == context_date,
                    PipelineArtifact.stage == StageName.FINAL_OUTPUT.value,
                ).order_by(PipelineArtifact.id.desc()).limit(1)
            )
            artifact = artifact_result.scalar_one_or_none()
            if artifact and artifact.payload:
                for pick in artifact.payload.get("picks", []):
                    targets = pick.get("targets") or []
                    screener_picks.append({
                        "ticker": pick.get("ticker", ""),
                        "direction": pick.get("direction", "LONG"),
                        "entry_price": pick.get("entry_price", pick.get("entry_zone", 0)),
                        "stop_loss": pick.get("stop_loss", 0),
                        "target_1": pick.get("target_1", targets[0] if targets else 0),
                        "confidence": pick.get("confidence", 0),
                        "signal_model": pick.get("signal_model", ""),
                        "thesis": pick.get("thesis", ""),
                        "holding_period": pick.get("holding_period", 10),
                    })
    except Exception as e:
        logger.warning("Failed to load today's screener context: %s", e)

    if regime_context["regime"] == "unknown" or not screener_picks:
        logger.warning(
            "Evening collection running with incomplete morning context: "
            "requested_date=%s, context_date=%s, regime=%s, screener_picks=%d. "
            "Morning pipeline may not have run or failed today. "
            "Cross-engine synthesis will proceed with external engines only.",
            requested_date, context_date, regime_context["regime"], len(screener_picks),
        )
    else:
        logger.info(
            "Evening collection context: requested_date=%s, context_date=%s, regime=%s, %d screener picks",
            requested_date, context_date, regime_context["regime"], len(screener_picks),
        )

    await _run_cross_engine_steps(
        today=context_date,
        run_id=run_id,
        regime_context=regime_context,
        screener_picks=screener_picks,
        settings=settings,
    )

    logger.info(
        "Evening cross-engine collection complete for requested_date=%s (context_date=%s)",
        requested_date,
        context_date,
    )


def start_scheduler() -> None:
    """Start the APScheduler with morning and afternoon jobs.

    Hardened with max_instances, misfire_grace_time, coalesce, and error listener.
    """
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.events import EVENT_JOB_ERROR

    settings = get_settings()
    scheduler = AsyncIOScheduler(timezone="US/Eastern")

    # Error listener — log and alert on job failures
    def _job_error_handler(event):
        job_id = event.job_id
        exc = event.exception
        logger.error("Scheduler job '%s' FAILED: %s", job_id, exc, exc_info=exc)
        # Fire-and-forget Telegram alert
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(send_alert(
                    f"SCHEDULER JOB FAILED\nJob: {job_id}\nError: {type(exc).__name__}: {exc}"
                ))
        except Exception as alert_exc:
            logger.error("Failed to send scheduler failure alert for job '%s': %s", job_id, alert_exc)

    scheduler.add_listener(_job_error_handler, EVENT_JOB_ERROR)

    # Morning pipeline
    scheduler.add_job(
        run_morning_pipeline,
        CronTrigger(
            hour=settings.morning_run_hour,
            minute=settings.morning_run_minute,
            day_of_week="mon-fri",
            timezone="US/Eastern",
        ),
        id="morning_pipeline",
        name="Daily Morning Pipeline",
        max_instances=1,
        misfire_grace_time=300,
        coalesce=True,
    )

    # Afternoon check
    scheduler.add_job(
        run_afternoon_check,
        CronTrigger(
            hour=settings.afternoon_check_hour,
            minute=settings.afternoon_check_minute,
            day_of_week="mon-fri",
            timezone="US/Eastern",
        ),
        id="afternoon_check",
        name="Afternoon Position Check",
        max_instances=1,
        misfire_grace_time=300,
        coalesce=True,
    )

    # Evening cross-engine collection (9:30 PM ET Mon-Fri)
    # Runs AFTER all 3 external engines finish (latest by ~8:15 PM ET)
    scheduler.add_job(
        run_evening_collection,
        CronTrigger(
            hour=21,
            minute=30,
            day_of_week="mon-fri",
            timezone="US/Eastern",
        ),
        id="evening_collection",
        name="Evening Cross-Engine Collection",
        max_instances=1,
        misfire_grace_time=300,
        coalesce=True,
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
        max_instances=1,
        misfire_grace_time=300,
        coalesce=True,
    )

    scheduler.start()
    logger.info(
        "Scheduler started: morning=%02d:%02d, afternoon=%02d:%02d, evening=21:30, weekly=Sun 19:00 (all ET, Mon-Fri)",
        settings.morning_run_hour, settings.morning_run_minute,
        settings.afternoon_check_hour, settings.afternoon_check_minute,
    )
    return scheduler


async def _debug_engines() -> None:
    """Debug mode: collect all engine results, run verifier + synthesizer, print comparison."""
    import json as _json
    from pathlib import Path

    from src.engines.collector import collect_engine_results
    from src.engines.credibility import compute_credibility_snapshot, compute_weighted_picks
    from src.agents.cross_engine_verifier import CrossEngineVerifierAgent
    from src.agents.cross_engine_synthesizer import CrossEngineSynthesizerAgent

    today = date.today()
    logger.info("=" * 60)
    logger.info("DEBUG ENGINES MODE — %s", today)
    logger.info("=" * 60)

    # Collect
    engine_results = await collect_engine_results()
    logger.info("Collected from %d engines", len(engine_results))

    # Credibility
    cred_snapshot = await compute_credibility_snapshot()

    # Build picks
    all_picks: list[dict] = []
    engine_summary: dict[str, dict] = {}
    for er in engine_results:
        tickers = [p.ticker for p in er.picks]
        engine_summary[er.engine_name] = {
            "picks": tickers,
            "regime": er.regime,
            "candidates_screened": er.candidates_screened,
            "status": er.status,
        }
        for pick in er.picks:
            all_picks.append({
                "engine_name": er.engine_name,
                "ticker": pick.ticker,
                "strategy": pick.strategy,
                "entry_price": pick.entry_price,
                "stop_loss": pick.stop_loss,
                "target_price": pick.target_price,
                "confidence": pick.confidence,
                "holding_period_days": pick.holding_period_days,
                "thesis": pick.thesis,
                "risk_factors": pick.risk_factors,
            })

    weighted = compute_weighted_picks(all_picks, cred_snapshot.engine_stats)

    # Verifier
    cred_stats_dict = {
        name: {
            "hit_rate": s.hit_rate, "weight": s.weight,
            "resolved_picks": s.resolved_picks, "brier_score": s.brier_score,
        }
        for name, s in cred_snapshot.engine_stats.items()
    }
    verifier = CrossEngineVerifierAgent()
    verifier_result = await verifier.verify(
        engine_results=[_json_safe(er.model_dump()) for er in engine_results],
        credibility_stats=cred_stats_dict,
        regime_context={"regime": "debug"},
    )

    # Synthesizer
    synthesizer = CrossEngineSynthesizerAgent()
    synthesis = await synthesizer.synthesize(
        weighted_picks=weighted,
        verifier_output=_json_safe(verifier_result.model_dump()),
        credibility_weights=cred_stats_dict,
        regime_context={"regime": "debug"},
    )

    # Build debug output
    debug_output = {
        "date": str(today),
        "engines": engine_summary,
        "credibility": cred_stats_dict,
        "weighted_picks": weighted,
        "verifier": _json_safe(verifier_result.model_dump()),
        "synthesis": _json_safe(synthesis.model_dump()),
    }

    # Print summary
    print("\n" + "=" * 60)
    print("CROSS-ENGINE DEBUG SUMMARY")
    print("=" * 60)
    print(f"\nEngines reporting: {len(engine_results)}")
    for name, info in engine_summary.items():
        print(f"\n  {name}:")
        print(f"    Regime: {info['regime']}")
        print(f"    Picks: {', '.join(info['picks']) or 'none'}")
    print("\nWeighted picks (top 10):")
    for wp in weighted[:10]:
        print(
            f"  {wp['ticker']}: score={wp['combined_score']:.0f}, "
            f"engines={wp['engine_count']}, convergence={wp['convergence_multiplier']:.1f}x"
        )
    print("\nSynthesis portfolio:")
    for pos in synthesis.portfolio:
        print(f"  {pos.ticker}: {pos.weight_pct:.0f}% [{pos.source}]")
    print(f"\nExecutive summary: {synthesis.executive_summary}")

    # Save debug JSON
    debug_dir = Path("outputs/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / f"cross_engine_{today}.json"
    debug_path.write_text(_json.dumps(debug_output, indent=2, default=str))
    logger.info("Debug output saved to %s", debug_path)


async def main():
    """Entry point — init DB, optionally run pipeline now, start scheduler."""
    import signal
    import sys
    import uvicorn

    await init_db()

    # Validate API keys for the configured execution mode
    settings = get_settings()
    try:
        settings.validate_keys_for_mode()
    except ValueError as e:
        logger.error("Startup validation failed:\n%s", e)
        sys.exit(1)

    # Parse --mode flag to override execution_mode at runtime
    for arg in sys.argv:
        if arg.startswith("--mode="):
            mode_value = arg.split("=", 1)[1]
            try:
                ExecutionMode(mode_value)  # validate
                settings = get_settings()
                settings.execution_mode = mode_value
                logger.info("Execution mode overridden to: %s", mode_value)
            except ValueError:
                logger.error("Invalid mode '%s'. Use: quant_only, hybrid, agentic_full", mode_value)
                return
        elif arg == "--mode" and sys.argv.index(arg) + 1 < len(sys.argv):
            mode_value = sys.argv[sys.argv.index(arg) + 1]
            try:
                ExecutionMode(mode_value)
                settings = get_settings()
                settings.execution_mode = mode_value
                logger.info("Execution mode overridden to: %s", mode_value)
            except ValueError:
                logger.error("Invalid mode '%s'. Use: quant_only, hybrid, agentic_full", mode_value)
                return

    if "--run-now" in sys.argv:
        await run_morning_pipeline()
        return

    if "--check-now" in sys.argv:
        await run_afternoon_check()
        return

    if "--meta-now" in sys.argv:
        await run_weekly_meta_review()
        return

    if "--debug-engines" in sys.argv:
        await _debug_engines()
        return

    # Start scheduler + API server
    scheduler = start_scheduler()

    # SIGTERM handler for graceful shutdown
    def _sigterm_handler(signum, frame):
        logger.info("Received SIGTERM — initiating graceful shutdown")
        scheduler.shutdown(wait=False)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

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
