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
from datetime import date, datetime, timezone

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
    opts = signal.get("options_sentiment", "") if isinstance(signal, dict) else (getattr(signal, "options_sentiment", None) or "")
    if opts.strip().lower() == "bullish":
        tags.append("gem_options_bullish")
    confluence = signal.get("confluence") if isinstance(signal, dict) else getattr(signal, "confluence", None)
    if confluence:
        tags.append("gem_confluence")
    return tags


def _reversion_strategy_tags(signal) -> list[str]:
    """Build strategy tags for a mean-reversion pick."""
    tags = ["gem_mean_reversion"]
    opts = signal.get("options_sentiment", "") if isinstance(signal, dict) else (getattr(signal, "options_sentiment", None) or "")
    if opts.strip().lower() == "bullish":
        tags.append("gem_options_bullish")
    confluence = signal.get("confluence") if isinstance(signal, dict) else getattr(signal, "confluence", None)
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


def _run_gemini_pipeline(target_date: date | None = None) -> EngineResultPayload | None:
    """Execute Gemini STST screeners synchronously and return mapped payload."""
    start = time.monotonic()
    gemini_src = str(_GEMINI_ROOT)
    screen_date = target_date or date.today()

    # Initialize DB if needed
    _ensure_gemini_db()

    if gemini_src not in sys.path:
        sys.path.insert(0, gemini_src)
    try:
        from app.screener import run_screener
        from app.mean_reversion import run_reversion_screener

        # Run both screeners
        logger.info("Running Gemini STST momentum screener locally")
        momentum_result = run_screener(screen_date=screen_date)

        logger.info("Running Gemini STST mean-reversion screener locally")
        reversion_result = run_reversion_screener(screen_date=screen_date)

        momentum_signals = momentum_result.get("signals", []) if isinstance(momentum_result, dict) else []
        reversion_signals = reversion_result.get("signals", []) if isinstance(reversion_result, dict) else []
        momentum_funnel = momentum_result.get("funnel", {}) if isinstance(momentum_result, dict) else {}
        reversion_funnel = reversion_result.get("funnel", {}) if isinstance(reversion_result, dict) else {}

        picks: list[EnginePick] = []

        for signal in momentum_signals:
            confidence = signal.get("quality_score") or 50.0
            trigger_price = signal.get("trigger_price")
            atr_pct = signal.get("atr_pct_at_trigger")
            stop_loss, target_price = _compute_momentum_risk_params(trigger_price, atr_pct)
            strat_tags = _momentum_strategy_tags(signal)
            thesis = None
            if signal.get("rvol_at_trigger") and atr_pct:
                thesis = f"RVOL={signal['rvol_at_trigger']:.1f}x, ATR%={atr_pct:.1f}%"

            picks.append(EnginePick(
                ticker=signal.get("symbol", ""),
                strategy="momentum",
                entry_price=trigger_price or 0,
                stop_loss=stop_loss,
                target_price=target_price,
                confidence=confidence,
                holding_period_days=10,
                thesis=thesis,
                risk_factors=[],
                raw_score=signal.get("quality_score"),
                metadata={
                    "rvol": signal.get("rvol_at_trigger"),
                    "atr_pct": atr_pct,
                    "rsi_14": signal.get("rsi_14"),
                    "options_sentiment": signal.get("options_sentiment"),
                    "confluence": signal.get("confluence"),
                    "scores": _build_scores_metadata(
                        quality_score=signal.get("quality_score"),
                        confluence=signal.get("confluence"),
                        strategy="momentum",
                    ),
                    "stop_method": "chandelier_proxy",
                    "strategies": strat_tags,
                },
            ))

        for signal in reversion_signals:
            confidence = signal.get("quality_score") or 50.0
            trigger_price = signal.get("trigger_price")
            strat_tags = _reversion_strategy_tags(signal)
            thesis = None
            if signal.get("rsi2") and signal.get("drawdown_3d_pct"):
                thesis = f"RSI2={signal['rsi2']:.1f}, DD3d={signal['drawdown_3d_pct']:.1f}%"

            picks.append(EnginePick(
                ticker=signal.get("symbol", ""),
                strategy="mean_reversion",
                entry_price=trigger_price or 0,
                stop_loss=round(trigger_price * 0.95, 2) if trigger_price else None,
                target_price=round(trigger_price * 1.10, 2) if trigger_price else None,
                confidence=confidence,
                holding_period_days=3,
                thesis=thesis,
                risk_factors=[],
                raw_score=signal.get("quality_score"),
                metadata={
                    "rsi2": signal.get("rsi2"),
                    "drawdown_3d_pct": signal.get("drawdown_3d_pct"),
                    "sma_distance_pct": signal.get("sma_distance_pct"),
                    "options_sentiment": signal.get("options_sentiment"),
                    "confluence": signal.get("confluence"),
                    "scores": _build_scores_metadata(
                        quality_score=signal.get("quality_score"),
                        confluence=signal.get("confluence"),
                        strategy="mean_reversion",
                    ),
                    "strategies": strat_tags,
                },
            ))

        total_screened = max(
            int(momentum_funnel.get("total_tickers", 0) or 0),
            int(reversion_funnel.get("total_tickers", 0) or 0),
        )
        elapsed = time.monotonic() - start

        payload = EngineResultPayload(
            engine_name="gemini_stst",
            engine_version="7.0-local",
            run_date=str(screen_date),
            run_timestamp=datetime.now(timezone.utc).isoformat(),
            regime=None,
            picks=picks,
            candidates_screened=total_screened,
            pipeline_duration_s=elapsed,
            status="success",
        )

        logger.info(
            "Gemini STST local run complete: %d picks (%d momentum, %d reversion) in %.1fs",
            len(picks), len(momentum_signals), len(reversion_signals), elapsed,
        )
        return payload

    finally:
        if gemini_src in sys.path:
            sys.path.remove(gemini_src)


async def run_gemini_locally(target_date: date | None = None) -> EngineResultPayload | None:
    """Run Gemini STST pipeline locally and return EngineResultPayload.

    Wraps the synchronous pipeline in asyncio.to_thread for non-blocking execution.
    Fail-open: catches all exceptions, logs, and returns None.
    """
    try:
        return await asyncio.to_thread(_run_gemini_pipeline, target_date)
    except Exception as e:
        logger.error("Gemini STST local runner failed: %s: %s", type(e).__name__, e, exc_info=True)
        return None
