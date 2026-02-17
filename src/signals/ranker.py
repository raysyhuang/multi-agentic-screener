"""Composite scoring, meta-model ranking, confluence detection, and signal cooldown.

Combines signals from breakout, mean_reversion, and catalyst models.
Ranks by composite score within the current regime context.
Includes confluence detection (multiple models flagging same ticker = high conviction)
and signal cooldown (suppress re-triggering for N days).

Confluence and cooldown patterns ported from KooCore-D and gemini_STST.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

from src.features.regime import Regime, get_regime_allowed_models
from src.signals.breakout import BreakoutSignal
from src.signals.mean_reversion import MeanReversionSignal
from src.signals.catalyst import CatalystSignal

logger = logging.getLogger(__name__)

# Type alias for any signal
AnySignal = BreakoutSignal | MeanReversionSignal | CatalystSignal

MODEL_MAP = {
    BreakoutSignal: "breakout",
    MeanReversionSignal: "mean_reversion",
    CatalystSignal: "catalyst",
}

# Regime multipliers: boost signals that work well in current regime
REGIME_MULTIPLIERS = {
    Regime.BULL: {"breakout": 1.2, "mean_reversion": 0.9, "catalyst": 1.0},
    Regime.BEAR: {"breakout": 0.5, "mean_reversion": 1.3, "catalyst": 0.7},
    Regime.CHOPPY: {"breakout": 0.6, "mean_reversion": 1.1, "catalyst": 1.1},
}

# Regime target multipliers: scale stop/target distances in adverse regimes
REGIME_TARGET_MULTIPLIERS = {
    Regime.BULL:   {"stop": 1.0, "target": 1.0},
    Regime.BEAR:   {"stop": 0.8, "target": 0.6},
    Regime.CHOPPY: {"stop": 0.9, "target": 0.75},
}


@dataclass
class RankedCandidate:
    ticker: str
    signal_model: str
    raw_score: float
    regime_adjusted_score: float
    direction: str
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float | None
    holding_period: int
    components: dict
    features: dict


def rank_candidates(
    signals: list[AnySignal],
    regime: Regime,
    features_by_ticker: dict[str, dict],
    top_n: int = 10,
) -> list[RankedCandidate]:
    """Rank all signal candidates, applying regime gate and adjustment.

    Steps:
      1. Filter out signals from models not allowed in current regime
      2. Apply regime multiplier to raw scores
      3. Sort by adjusted score descending
      4. Return top N
    """
    allowed_models = get_regime_allowed_models(regime)
    candidates = []

    for signal in signals:
        model_name = MODEL_MAP.get(type(signal), "unknown")

        # Regime gate: skip models that shouldn't fire in this regime
        if model_name not in allowed_models:
            logger.info(
                "Regime gate blocked %s signal for %s (regime=%s)",
                model_name, signal.ticker, regime.value,
            )
            continue

        # Apply regime multiplier
        multiplier = REGIME_MULTIPLIERS.get(regime, {}).get(model_name, 1.0)
        adjusted_score = signal.score * multiplier

        features = features_by_ticker.get(signal.ticker, {})

        # Apply regime-adaptive target scaling
        target_mults = REGIME_TARGET_MULTIPLIERS.get(regime, {"stop": 1.0, "target": 1.0})
        entry = signal.entry_price
        stop = signal.stop_loss
        t1 = signal.target_1
        t2 = getattr(signal, "target_2", None)

        adj_stop = round(entry - (entry - stop) * target_mults["stop"], 2)
        adj_t1 = round(entry + (t1 - entry) * target_mults["target"], 2)
        adj_t2 = round(entry + (t2 - entry) * target_mults["target"], 2) if t2 is not None else None

        candidates.append(RankedCandidate(
            ticker=signal.ticker,
            signal_model=model_name,
            raw_score=signal.score,
            regime_adjusted_score=round(adjusted_score, 2),
            direction=signal.direction,
            entry_price=entry,
            stop_loss=adj_stop,
            target_1=adj_t1,
            target_2=adj_t2,
            holding_period=signal.holding_period,
            components=signal.components,
            features=features,
        ))

    # Sort by regime-adjusted score
    candidates.sort(key=lambda c: c.regime_adjusted_score, reverse=True)

    logger.info(
        "Ranked %d candidates (regime=%s, allowed models=%s), returning top %d",
        len(candidates), regime.value, allowed_models, top_n,
    )
    return candidates[:top_n]


def filter_correlated_picks(
    candidates: list[RankedCandidate],
    price_data: dict[str, "pd.DataFrame"],
    max_correlation: float = 0.75,
) -> list[RankedCandidate]:
    """Remove highly-correlated candidates to prevent concentrated risk.

    Iterates top-down by score. If a candidate's 20-day return correlation
    with any already-accepted candidate exceeds max_correlation, it is dropped.
    """
    import math
    import pandas as pd
    import numpy as np

    if not candidates or not price_data:
        return candidates

    accepted: list[RankedCandidate] = []
    accepted_returns: dict[str, pd.Series] = {}

    for candidate in candidates:
        df = price_data.get(candidate.ticker)
        if df is None or df.empty or len(df) < 20:
            accepted.append(candidate)
            continue

        close = df["close"].astype(float)
        returns = close.pct_change().dropna().tail(20)

        if returns.empty or len(returns) < 10:
            accepted.append(candidate)
            continue

        # Check correlation with all accepted picks
        is_correlated = False
        for acc_ticker, acc_returns in accepted_returns.items():
            # Align by index length
            min_len = min(len(returns), len(acc_returns))
            if min_len < 10:
                continue
            corr = float(np.corrcoef(
                returns.values[-min_len:],
                acc_returns.values[-min_len:],
            )[0, 1])
            if math.isnan(corr):
                continue  # can't determine correlation, keep the pick
            if abs(corr) > max_correlation:
                logger.info(
                    "Correlation filter: dropping %s (corr=%.2f with %s)",
                    candidate.ticker, corr, acc_ticker,
                )
                is_correlated = True
                break

        if not is_correlated:
            accepted.append(candidate)
            accepted_returns[candidate.ticker] = returns

    if len(accepted) < len(candidates):
        logger.info(
            "Correlation filter: %d → %d candidates (max_corr=%.2f)",
            len(candidates), len(accepted), max_correlation,
        )

    return accepted


def deduplicate_signals(signals: list[AnySignal]) -> list[AnySignal]:
    """If multiple models flag the same ticker, keep the highest scoring one."""
    best_by_ticker: dict[str, AnySignal] = {}
    for signal in signals:
        ticker = signal.ticker
        if ticker not in best_by_ticker or signal.score > best_by_ticker[ticker].score:
            best_by_ticker[ticker] = signal
    return list(best_by_ticker.values())


# --- Confluence Detection ---


@dataclass
class ConfluenceResult:
    """Result of confluence analysis for a single ticker."""

    ticker: str
    signal_models: list[str]
    confluence_count: int  # How many models flag this ticker
    is_confluence: bool    # 2+ models = confluence
    confluence_bonus: float  # Score boost for multi-model agreement


def detect_confluence(signals: list[AnySignal]) -> dict[str, ConfluenceResult]:
    """Detect tickers flagged by multiple signal models (confluence).

    When 2+ models agree on the same ticker, hit rate improves significantly
    (KooCore-D data: 2 factors → 55-65%, 3+ factors → 65-75%).

    Returns a dict of ticker → ConfluenceResult.
    """
    by_ticker: dict[str, list[str]] = {}

    for signal in signals:
        model_name = MODEL_MAP.get(type(signal), "unknown")
        by_ticker.setdefault(signal.ticker, []).append(model_name)

    results: dict[str, ConfluenceResult] = {}
    for ticker, models in by_ticker.items():
        unique_models = list(set(models))
        count = len(unique_models)
        is_confluence = count >= 2

        # Bonus: 10% per additional model beyond 1
        bonus = (count - 1) * 0.10 if count > 1 else 0.0

        results[ticker] = ConfluenceResult(
            ticker=ticker,
            signal_models=unique_models,
            confluence_count=count,
            is_confluence=is_confluence,
            confluence_bonus=bonus,
        )

        if is_confluence:
            logger.info(
                "Confluence detected: %s flagged by %d models (%s) — +%.0f%% bonus",
                ticker, count, ", ".join(unique_models), bonus * 100,
            )

    return results


def apply_confluence_bonus(
    candidates: list[RankedCandidate],
    confluence: dict[str, ConfluenceResult],
) -> list[RankedCandidate]:
    """Apply confluence score bonus to candidates flagged by multiple models.

    Mutates regime_adjusted_score in place and re-sorts.
    """
    for c in candidates:
        cr = confluence.get(c.ticker)
        if cr and cr.is_confluence:
            old_score = c.regime_adjusted_score
            c.regime_adjusted_score = round(old_score * (1 + cr.confluence_bonus), 2)
            c.components["confluence_count"] = cr.confluence_count
            c.components["confluence_models"] = cr.signal_models

    # Re-sort after bonus application
    candidates.sort(key=lambda c: c.regime_adjusted_score, reverse=True)
    return candidates


# --- Signal Cooldown ---


COOLDOWN_DAYS = 5  # Suppress same ticker for N calendar days after signal


def apply_cooldown(
    signals: list[AnySignal],
    recent_signals: list[dict],
    cooldown_days: int = COOLDOWN_DAYS,
) -> list[AnySignal]:
    """Filter out signals for tickers that fired recently (cooldown window).

    Prevents re-triggering the same breakout on consecutive days.

    Args:
        signals: Current batch of raw signals.
        recent_signals: List of dicts with 'ticker' and 'run_date' keys
                       from recent pipeline runs (query from DB).
        cooldown_days: Number of calendar days to suppress.

    Returns:
        Filtered signals with recently-fired tickers removed.
    """
    if not recent_signals:
        return signals

    today = date.today()
    cooldown_cutoff = today - timedelta(days=cooldown_days)

    # Build set of tickers that are still in cooldown
    cooled_down: set[str] = set()
    for sig in recent_signals:
        sig_date = sig.get("run_date")
        if sig_date is None:
            continue
        if isinstance(sig_date, str):
            sig_date = date.fromisoformat(sig_date)
        if sig_date >= cooldown_cutoff:
            cooled_down.add(sig["ticker"])

    if not cooled_down:
        return signals

    filtered = []
    suppressed = 0
    for signal in signals:
        if signal.ticker in cooled_down:
            logger.info(
                "Cooldown: suppressing %s (%s) — fired within last %d days",
                signal.ticker, MODEL_MAP.get(type(signal), "?"), cooldown_days,
            )
            suppressed += 1
        else:
            filtered.append(signal)

    if suppressed > 0:
        logger.info("Cooldown filter: %d → %d signals (%d suppressed)", len(signals), len(filtered), suppressed)

    return filtered
