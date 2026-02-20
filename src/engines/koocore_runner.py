"""Local KooCore-D pipeline runner — replaces HTTP fetch with in-process execution.

Runs KooCore-D's ``main.py all`` as a subprocess (avoids ``src`` namespace
collision with MAS), reads the ``hybrid_analysis_{date}.json`` output, and maps
it to MAS's ``EngineResultPayload`` contract with strategy tags.

Fail-open: all exceptions are caught, logged, and ``None`` is returned.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

from src.config import get_settings, PROJECT_ROOT
from src.contracts import EnginePick, EngineResultPayload

logger = logging.getLogger(__name__)

# KooCore-D root relative to MAS project root
_KOOCORE_ROOT = PROJECT_ROOT / "KooCore-D"

# ---------------------------------------------------------------------------
# Strategy tag mapping
# ---------------------------------------------------------------------------

_SOURCE_TO_TAG = {
    "weekly": "kc_weekly",
    "pro30": "kc_pro30",
    "swing": "kc_swing",
    "movers": "kc_movers",
}


def _normalize_source(source: str) -> str:
    """Normalize KooCore-D source names: 'Pro30(r1)' -> 'pro30'."""
    import re
    s = source.strip().lower()
    s = re.sub(r"\(.*\)$", "", s)  # strip rank suffix like (r1)
    return s


def _strategy_tags_from_sources(sources: list[str]) -> list[str]:
    """Map KooCore-D hybrid source names to strategy tags."""
    normalized = [_normalize_source(s) for s in sources]
    tags = [_SOURCE_TO_TAG.get(n, f"kc_{n}") for n in normalized]
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_tags: list[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            unique_tags.append(t)
    if len(set(normalized)) >= 2:
        unique_tags.append("kc_confluence")
    return unique_tags


# ---------------------------------------------------------------------------
# Mapping logic (reused from KooCore-D/src/api/engine_endpoint.py)
# ---------------------------------------------------------------------------

def _risk_profile_from_sources(sources: list[str]) -> tuple[float, float, str]:
    """Infer risk/target defaults from contributing source models."""
    src = {str(s).lower() for s in (sources or [])}
    if "pro30" in src:
        return 0.08, 0.18, "momentum"
    if "movers" in src:
        return 0.04, 0.08, "breakout"
    return 0.05, 0.10, "swing"


def _build_scores_metadata(item: dict, fallback_score: float) -> dict:
    raw_scores = item.get("scores")
    if isinstance(raw_scores, dict) and raw_scores:
        return raw_scores
    scores: dict[str, float] = {}
    for key in ("hybrid_score", "composite_score", "weekly_score", "pro30_score", "movers_score"):
        value = item.get(key)
        if value is None:
            continue
        try:
            scores[key] = float(value)
        except (TypeError, ValueError):
            continue
    if not scores:
        scores["hybrid_score"] = float(fallback_score)
    return scores


def _map_hybrid_to_payload(hybrid: dict, run_date: str, duration: float | None = None) -> EngineResultPayload:
    """Convert KooCore-D hybrid_analysis JSON to EngineResultPayload."""
    picks: list[EnginePick] = []

    hybrid_top3 = hybrid.get("hybrid_top3", [])
    weighted_picks = hybrid.get("weighted_picks", [])

    for item in hybrid_top3:
        ticker = item.get("ticker", "")
        composite_score = item.get("composite_score") or item.get("hybrid_score", 0)
        sources = item.get("sources", [])
        risk_pct, reward_pct, fallback_strategy = _risk_profile_from_sources(sources)
        strategy = "hybrid_" + "_".join(sources) if sources else fallback_strategy
        entry_price = float(item.get("current_price") or 0)
        if entry_price <= 0:
            continue

        confidence = min(max(composite_score * 10, 0), 100)
        explicit_target = item.get("target", {}).get("target_price_for_10pct")
        target_price = (
            float(explicit_target)
            if explicit_target is not None and float(explicit_target) > entry_price
            else round(entry_price * (1 + reward_pct), 2)
        )

        strat_tags = _strategy_tags_from_sources(sources)

        picks.append(EnginePick(
            ticker=ticker,
            strategy=strategy,
            entry_price=entry_price,
            stop_loss=round(entry_price * (1 - risk_pct), 2),
            target_price=target_price,
            confidence=round(confidence, 1),
            holding_period_days=14,
            thesis=item.get("verdict") or item.get("confidence"),
            risk_factors=[],
            raw_score=composite_score,
            metadata={
                "sources": sources,
                "rank": item.get("rank"),
                "scores": _build_scores_metadata(item, composite_score),
                "strategies": strat_tags,
            },
        ))

    # Include ALL weighted_picks not already in top3 (no score threshold or cap)
    seen_tickers = {p.ticker for p in picks}
    for item in weighted_picks:
        ticker = item.get("ticker", "")
        if not ticker or ticker in seen_tickers:
            continue
        composite_score = item.get("hybrid_score", 0)
        sources = item.get("sources", [])
        risk_pct, reward_pct, strategy = _risk_profile_from_sources(sources)
        entry_price = float(item.get("current_price") or 0)
        if entry_price <= 0:
            continue
        confidence = min(max(composite_score * 10, 0), 100)
        strat_tags = _strategy_tags_from_sources(sources)

        picks.append(EnginePick(
            ticker=ticker,
            strategy=strategy,
            entry_price=entry_price,
            stop_loss=round(entry_price * (1 - risk_pct), 2),
            target_price=round(entry_price * (1 + reward_pct), 2),
            confidence=round(confidence, 1),
            holding_period_days=14,
            thesis=None,
            risk_factors=[],
            raw_score=composite_score,
            metadata={
                "sources": sources,
                "scores": _build_scores_metadata(item, composite_score),
                "strategies": strat_tags,
            },
        ))
        seen_tickers.add(ticker)

    # Include primary_top5 (weekly/swing LLM-ranked picks) not already captured
    for item in hybrid.get("primary_top5", []):
        ticker = item.get("ticker", "")
        if not ticker or ticker in seen_tickers:
            continue
        composite_score = item.get("composite_score", 0)
        entry_price = float(item.get("current_price") or 0)
        if entry_price <= 0:
            continue
        confidence = min(max(composite_score * 10, 0), 100)
        picks.append(EnginePick(
            ticker=ticker,
            strategy="swing",
            entry_price=entry_price,
            stop_loss=round(entry_price * 0.94, 2),
            target_price=round(entry_price * 1.10, 2),
            confidence=round(confidence, 1),
            holding_period_days=10,
            thesis=item.get("confidence"),
            risk_factors=[],
            raw_score=composite_score,
            metadata={
                "sources": ["primary"],
                "rank": item.get("rank"),
                "scores": _build_scores_metadata(item, composite_score),
                "strategies": ["kc_weekly"],
            },
        ))
        seen_tickers.add(ticker)

    # NOTE: pro30_tickers are excluded from picks because they lack price data
    # (entry_price=0, no stop/target), which triggers the collector's quality
    # validation and rejects the entire engine payload.  They are counted in
    # candidates_screened for informational purposes only.

    summary = hybrid.get("summary", {})
    total_screened = (
        summary.get("weekly_top5_count", 0)
        + summary.get("pro30_candidates_count", 0)
        + summary.get("movers_count", 0)
    )

    return EngineResultPayload(
        engine_name="koocore_d",
        engine_version="2.0-local",
        run_date=run_date,
        run_timestamp=datetime.utcnow().isoformat(),
        regime=None,
        picks=picks,
        candidates_screened=total_screened,
        pipeline_duration_s=duration,
        status="success",
    )


# ---------------------------------------------------------------------------
# Fallback: read latest existing output without re-running the pipeline
# ---------------------------------------------------------------------------

def _find_latest_hybrid_output() -> tuple[Path, str] | None:
    """Find the most recent hybrid_analysis JSON in KooCore-D/outputs/."""
    outputs_dir = _KOOCORE_ROOT / "outputs"
    if not outputs_dir.exists():
        return None
    date_dirs = sorted(
        [d.name for d in outputs_dir.iterdir() if d.is_dir() and len(d.name) == 10],
        reverse=True,
    )
    for date_str in date_dirs:
        path = outputs_dir / date_str / f"hybrid_analysis_{date_str}.json"
        if path.exists():
            return path, date_str
    return None


# ---------------------------------------------------------------------------
# Pipeline execution via subprocess (avoids src namespace collision)
# ---------------------------------------------------------------------------

def _run_koocore_pipeline() -> EngineResultPayload | None:
    """Execute KooCore-D ``main.py all`` as a subprocess and return mapped payload."""
    settings = get_settings()
    config_path = settings.koocore_config_path
    start = time.monotonic()

    # Determine the output date (today's trading date)
    # Use a simple weekday check: if weekend, use Friday
    today = date.today()
    weekday = today.weekday()
    if weekday == 5:  # Saturday
        output_date = today.replace(day=today.day - 1)
    elif weekday == 6:  # Sunday
        output_date = today.replace(day=today.day - 2)
    else:
        output_date = today
    output_date_str = output_date.strftime("%Y-%m-%d")

    logger.info("Running KooCore-D pipeline via subprocess for %s", output_date_str)

    cmd = [
        sys.executable, "main.py", "all",
        "--config", config_path,
        "--date", output_date_str,
        "--no-movers",
    ]

    env = os.environ.copy()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(_KOOCORE_ROOT),
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute max
            env=env,
        )
        if result.returncode != 0:
            logger.warning(
                "KooCore-D subprocess exited with code %d\nstderr: %s",
                result.returncode,
                result.stderr[-500:] if result.stderr else "(empty)",
            )
            # Fall through to try reading existing output
    except subprocess.TimeoutExpired:
        logger.warning("KooCore-D subprocess timed out after 600s")
    except Exception as e:
        logger.warning("KooCore-D subprocess failed: %s: %s", type(e).__name__, e)

    # Read the hybrid_analysis output (from this run or a previous one)
    output_path = _KOOCORE_ROOT / "outputs" / output_date_str / f"hybrid_analysis_{output_date_str}.json"
    if not output_path.exists():
        # Try the latest available output as fallback
        latest = _find_latest_hybrid_output()
        if latest:
            output_path, output_date_str = latest
            logger.info("Using latest available KooCore-D output from %s", output_date_str)
        else:
            logger.warning("KooCore-D produced no hybrid_analysis output")
            return None

    with open(output_path, "r") as f:
        hybrid = json.load(f)

    elapsed = time.monotonic() - start
    payload = _map_hybrid_to_payload(hybrid, output_date_str, duration=elapsed)
    logger.info(
        "KooCore-D local run complete: %d picks in %.1fs",
        len(payload.picks), elapsed,
    )
    return payload


async def run_koocore_locally() -> EngineResultPayload | None:
    """Run KooCore-D pipeline locally and return EngineResultPayload.

    Wraps the synchronous subprocess call in asyncio.to_thread for non-blocking execution.
    Fail-open: catches all exceptions, logs, and returns None.
    """
    try:
        return await asyncio.to_thread(_run_koocore_pipeline)
    except Exception as e:
        logger.error("KooCore-D local runner failed: %s: %s", type(e).__name__, e, exc_info=True)
        return None
