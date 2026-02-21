"""Local Gemini STST pipeline runner — replaces HTTP fetch with in-process execution.

Isolates Gemini STST imports via temporary sys.path manipulation, runs the
momentum and mean-reversion screeners, queries results from Postgres, and maps
to MAS's ``EngineResultPayload`` contract with strategy tags.

Requires ``DATABASE_URL`` to be set (shared with MAS's own Postgres).

Fail-open: all exceptions are caught, logged, and ``None`` is returned.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
import time
from datetime import date, datetime

from src.config import get_settings, PROJECT_ROOT
from src.contracts import EnginePick, EngineResultPayload

logger = logging.getLogger(__name__)


def _is_transient(exc: Exception) -> bool:
    """Return True for DB/network errors worth retrying."""
    from sqlalchemy.exc import OperationalError, DisconnectionError
    transient_types = (OperationalError, DisconnectionError, ConnectionError, OSError, TimeoutError)
    return isinstance(exc, transient_types)


# Gemini STST root relative to MAS project root
_GEMINI_ROOT = PROJECT_ROOT / "Gemini STST"

_MOMENTUM_STOP_MULT = 3.5

# Track whether DB has been initialized this process
_db_initialized = False


# ---------------------------------------------------------------------------
# Risk parameter computation (reused from Gemini STST/app/engine_endpoint.py)
# ---------------------------------------------------------------------------

def _compute_momentum_risk_params(
    entry_price: float | None,
    atr_pct: float | None,
    min_rr: float = 1.8,
) -> tuple[float | None, float | None]:
    if not entry_price or entry_price <= 0:
        return None, None
    if atr_pct and atr_pct > 0:
        trail_frac = _MOMENTUM_STOP_MULT * atr_pct / (math.sqrt(5) * 100.0)
        trail_frac = max(0.04, min(0.20, trail_frac))
    else:
        trail_frac = 0.10
    target_frac = max(0.10, trail_frac * min_rr)
    target_frac = min(0.35, target_frac)
    stop = round(entry_price * (1 - trail_frac), 2)
    target = round(entry_price * (1 + target_frac), 2)
    return stop, target


def _build_scores_metadata(
    *,
    quality_score: float | None,
    confluence: bool | None,
    strategy: str,
) -> dict:
    scores: dict[str, float] = {}
    if quality_score is not None:
        scores["quality_score"] = float(quality_score)
    if confluence is not None:
        scores["confluence_bonus"] = 1.0 if bool(confluence) else 0.0
    if not scores:
        scores["quality_score"] = 50.0
    scores["strategy"] = 1.0 if strategy else 0.0
    return scores


# ---------------------------------------------------------------------------
# Strategy tag helpers
# ---------------------------------------------------------------------------

def _momentum_strategy_tags(signal) -> list[str]:
    """Build strategy tags for a momentum pick."""
    tags = ["gem_momentum_breakout"]
    opts = getattr(signal, "options_sentiment", None) or ""
    if opts.strip().lower() == "bullish":
        tags.append("gem_options_bullish")
    confluence = getattr(signal, "confluence", None)
    if confluence:
        tags.append("gem_confluence")
    return tags


def _reversion_strategy_tags(signal) -> list[str]:
    """Build strategy tags for a mean-reversion pick."""
    tags = ["gem_mean_reversion"]
    opts = getattr(signal, "options_sentiment", None) or ""
    if opts.strip().lower() == "bullish":
        tags.append("gem_options_bullish")
    confluence = getattr(signal, "confluence", None)
    if confluence:
        tags.append("gem_confluence")
    return tags


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def _ensure_gemini_db():
    """Initialize Gemini STST's DB tables on first call (with transient-error retry)."""
    global _db_initialized
    if _db_initialized:
        return

    gemini_src = str(_GEMINI_ROOT)
    if gemini_src not in sys.path:
        sys.path.insert(0, gemini_src)
    try:
        # Ensure Gemini STST sees MAS's DATABASE_URL
        settings = get_settings()
        if settings.database_url:
            db_url = settings.database_url
            # Heroku-style postgres:// → postgresql:// fix
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://", 1)
            os.environ["DATABASE_URL"] = db_url

        from app.database import init_db
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                init_db()
                _db_initialized = True
                logger.info("Gemini STST database initialized")
                return
            except Exception as exc:
                if attempt < max_attempts and _is_transient(exc):
                    delay = 2 ** attempt  # 2s, 4s
                    logger.warning(
                        "Gemini DB init failed (attempt %d/%d): %s — retrying in %ds",
                        attempt, max_attempts, exc, delay,
                    )
                    time.sleep(delay)
                else:
                    raise
    finally:
        if gemini_src in sys.path:
            sys.path.remove(gemini_src)


def _run_gemini_pipeline() -> EngineResultPayload | None:
    """Execute Gemini STST screeners synchronously and return mapped payload."""
    start = time.monotonic()
    gemini_src = str(_GEMINI_ROOT)

    # Initialize DB if needed
    _ensure_gemini_db()

    if gemini_src not in sys.path:
        sys.path.insert(0, gemini_src)
    try:
        from app.screener import run_screener
        from app.mean_reversion import run_reversion_screener
        from app.database import SessionLocal
        from app.models import Ticker, ScreenerSignal, ReversionSignal
        from sqlalchemy import func

        # Run both screeners
        logger.info("Running Gemini STST momentum screener locally")
        run_screener()

        logger.info("Running Gemini STST mean-reversion screener locally")
        run_reversion_screener()

        # Query results from DB (same logic as engine_endpoint.py)
        # Retry DB session on transient connection errors (screeners already ran).
        max_db_attempts = 3
        db = None
        for db_attempt in range(1, max_db_attempts + 1):
            try:
                db = SessionLocal()
                latest_momentum_date = db.query(func.max(ScreenerSignal.date)).scalar()
                latest_reversion_date = db.query(func.max(ReversionSignal.date)).scalar()
                latest_dates = [d for d in (latest_momentum_date, latest_reversion_date) if d is not None]
                asof_date = max(latest_dates) if latest_dates else date.today()

                picks: list[EnginePick] = []

                # Momentum signals
                momentum_query = (
                    db.query(ScreenerSignal, Ticker)
                    .join(Ticker, ScreenerSignal.ticker_id == Ticker.id)
                    .filter(ScreenerSignal.date == asof_date)
                    .order_by(ScreenerSignal.quality_score.desc().nullslast())
                    .all()
                )

                for signal, ticker in momentum_query:
                    confidence = signal.quality_score or 50.0
                    stop_loss, target_price = _compute_momentum_risk_params(
                        signal.trigger_price,
                        signal.atr_pct_at_trigger,
                    )
                    strat_tags = _momentum_strategy_tags(signal)

                    picks.append(EnginePick(
                        ticker=ticker.symbol,
                        strategy="momentum",
                        entry_price=signal.trigger_price or 0,
                        stop_loss=stop_loss,
                        target_price=target_price,
                        confidence=confidence,
                        holding_period_days=10,
                        thesis=(
                            f"RVOL={signal.rvol_at_trigger:.1f}x, ATR%={signal.atr_pct_at_trigger:.1f}%"
                            if signal.rvol_at_trigger and signal.atr_pct_at_trigger
                            else None
                        ),
                        risk_factors=[],
                        raw_score=signal.quality_score,
                        metadata={
                            "rvol": signal.rvol_at_trigger,
                            "atr_pct": signal.atr_pct_at_trigger,
                            "rsi_14": signal.rsi_14,
                            "options_sentiment": signal.options_sentiment,
                            "confluence": signal.confluence,
                            "scores": _build_scores_metadata(
                                quality_score=signal.quality_score,
                                confluence=signal.confluence,
                                strategy="momentum",
                            ),
                            "stop_method": "chandelier_proxy",
                            "strategies": strat_tags,
                        },
                    ))

                # Reversion signals
                reversion_query = (
                    db.query(ReversionSignal, Ticker)
                    .join(Ticker, ReversionSignal.ticker_id == Ticker.id)
                    .filter(ReversionSignal.date == asof_date)
                    .order_by(ReversionSignal.quality_score.desc().nullslast())
                    .all()
                )

                for signal, ticker in reversion_query:
                    confidence = signal.quality_score or 50.0
                    strat_tags = _reversion_strategy_tags(signal)

                    picks.append(EnginePick(
                        ticker=ticker.symbol,
                        strategy="mean_reversion",
                        entry_price=signal.trigger_price or 0,
                        stop_loss=round(signal.trigger_price * 0.95, 2) if signal.trigger_price else None,
                        target_price=round(signal.trigger_price * 1.10, 2) if signal.trigger_price else None,
                        confidence=confidence,
                        holding_period_days=3,
                        thesis=(
                            f"RSI2={signal.rsi2_at_trigger:.1f}, DD3d={signal.drawdown_3d_pct:.1f}%"
                            if signal.rsi2_at_trigger and signal.drawdown_3d_pct
                            else None
                        ),
                        risk_factors=[],
                        raw_score=signal.quality_score,
                        metadata={
                            "rsi2": signal.rsi2_at_trigger,
                            "drawdown_3d_pct": signal.drawdown_3d_pct,
                            "sma_distance_pct": signal.sma_distance_pct,
                            "options_sentiment": signal.options_sentiment,
                            "confluence": signal.confluence,
                            "scores": _build_scores_metadata(
                                quality_score=signal.quality_score,
                                confluence=signal.confluence,
                                strategy="mean_reversion",
                            ),
                            "strategies": strat_tags,
                        },
                    ))

                total_screened = len(momentum_query) + len(reversion_query)
                elapsed = time.monotonic() - start

                payload = EngineResultPayload(
                    engine_name="gemini_stst",
                    engine_version="7.0-local",
                    run_date=str(asof_date),
                    run_timestamp=datetime.utcnow().isoformat(),
                    regime=None,
                    picks=picks,
                    candidates_screened=total_screened,
                    pipeline_duration_s=elapsed,
                    status="success",
                )

                logger.info(
                    "Gemini STST local run complete: %d picks (%d momentum, %d reversion) in %.1fs",
                    len(picks), len(momentum_query), len(reversion_query), elapsed,
                )
                return payload

            except Exception as exc:
                if db:
                    db.close()
                    db = None
                if db_attempt < max_db_attempts and _is_transient(exc):
                    delay = 2 ** db_attempt
                    logger.warning(
                        "Gemini DB query failed (attempt %d/%d): %s — retrying in %ds",
                        db_attempt, max_db_attempts, exc, delay,
                    )
                    time.sleep(delay)
                    continue
                raise
            finally:
                if db:
                    db.close()

    finally:
        if gemini_src in sys.path:
            sys.path.remove(gemini_src)


async def run_gemini_locally() -> EngineResultPayload | None:
    """Run Gemini STST pipeline locally and return EngineResultPayload.

    Wraps the synchronous pipeline in asyncio.to_thread for non-blocking execution.
    Fail-open: catches all exceptions, logs, and returns None.
    """
    try:
        return await asyncio.to_thread(_run_gemini_pipeline)
    except Exception as e:
        logger.error("Gemini STST local runner failed: %s: %s", type(e).__name__, e, exc_info=True)
        return None
