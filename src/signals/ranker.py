"""Composite scoring and meta-model ranking.

Combines signals from breakout, mean_reversion, and catalyst models.
Ranks by composite score within the current regime context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

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

        candidates.append(RankedCandidate(
            ticker=signal.ticker,
            signal_model=model_name,
            raw_score=signal.score,
            regime_adjusted_score=round(adjusted_score, 2),
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            target_1=signal.target_1,
            target_2=getattr(signal, "target_2", None),
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


def deduplicate_signals(signals: list[AnySignal]) -> list[AnySignal]:
    """If multiple models flag the same ticker, keep the highest scoring one."""
    best_by_ticker: dict[str, AnySignal] = {}
    for signal in signals:
        ticker = signal.ticker
        if ticker not in best_by_ticker or signal.score > best_by_ticker[ticker].score:
            best_by_ticker[ticker] = signal
    return list(best_by_ticker.values())
