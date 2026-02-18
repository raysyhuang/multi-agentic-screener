"""Cross-engine pick synthesizer — merges picks via credibility weighting.

Reuses the convergence-multiplier logic from
:func:`src.engines.credibility.compute_convergence_multiplier` and the
weighted-pick merging approach from
:func:`src.engines.credibility.compute_weighted_picks`, adapted for the
backtest context where engine weights are configurable (static or rolling).
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field

from src.backtest.multi_engine.adapters.base import NormalizedPick

logger = logging.getLogger(__name__)


@dataclass
class SynthesisConfig:
    """Configuration for cross-engine synthesis."""

    initial_weights: dict[str, float] = field(
        default_factory=lambda: {"mas": 1.0, "koocore_d": 1.0, "gemini_stst": 1.0}
    )
    convergence_multipliers: dict[int, float] = field(
        default_factory=lambda: {2: 1.3, 3: 1.0}
    )
    top_n_per_day: int = 5
    rolling_credibility: bool = False
    min_confidence: float = 35.0
    regime_convergence_overrides: dict[str, dict[int, float]] = field(
        default_factory=dict
    )


@dataclass
class SynthesisPick:
    """A synthesized pick from merging multiple engine outputs."""

    ticker: str
    combined_score: float
    avg_weighted_confidence: float
    convergence_multiplier: float
    engine_count: int
    engines: list[str]
    strategies: list[str]
    entry_price: float
    stop_loss: float | None
    target_price: float | None
    holding_period_days: int
    direction: str
    source_picks: list[NormalizedPick] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "combined_score": self.combined_score,
            "avg_weighted_confidence": self.avg_weighted_confidence,
            "convergence_multiplier": self.convergence_multiplier,
            "engine_count": self.engine_count,
            "engines": self.engines,
            "strategies": self.strategies,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target_price": self.target_price,
            "holding_period_days": self.holding_period_days,
            "direction": self.direction,
        }


# ── Rolling Credibility Tracker ──────────────────────────────────────────


@dataclass
class _EngineTradeRecord:
    """Single resolved trade outcome for credibility tracking."""

    pnl_pct: float
    is_win: bool


class RollingCredibilityTracker:
    """Tracks per-engine rolling performance to compute dynamic weights.

    Maintains a 20-day rolling window of trade outcomes per engine.
    Engines that outperform get higher weight (up to 2.5x), underperformers
    get lower weight (down to 0.3x).  Requires a minimum of 10 trades
    before overriding the default weights.
    """

    def __init__(
        self,
        window: int = 20,
        min_trades: int = 10,
        weight_floor: float = 0.3,
        weight_cap: float = 2.5,
    ):
        self.window = window
        self.min_trades = min_trades
        self.weight_floor = weight_floor
        self.weight_cap = weight_cap
        # engine_name -> deque of recent trade records
        self._history: dict[str, deque[_EngineTradeRecord]] = defaultdict(
            lambda: deque(maxlen=window)
        )

    def record_outcome(self, engine_name: str, pnl_pct: float) -> None:
        """Record a resolved trade outcome for *engine_name*."""
        self._history[engine_name].append(
            _EngineTradeRecord(pnl_pct=pnl_pct, is_win=pnl_pct > 0)
        )

    def get_rolling_weights(
        self, default_weights: dict[str, float]
    ) -> dict[str, float] | None:
        """Compute dynamic weights from rolling performance.

        Returns ``None`` if no engine has enough history yet (caller should
        fall back to *default_weights*).

        Algorithm:
          1. For each engine with >= min_trades history, compute:
             - rolling win rate (0-1)
             - rolling avg return
          2. Composite score = 0.6 * win_rate + 0.4 * sigmoid(avg_return)
          3. Normalize scores so they average to 1.0, then clamp to [floor, cap].
        """
        scores: dict[str, float] = {}

        for engine, records in self._history.items():
            if len(records) < self.min_trades:
                continue
            wins = sum(1 for r in records if r.is_win)
            win_rate = wins / len(records)
            avg_ret = sum(r.pnl_pct for r in records) / len(records)
            # Sigmoid-like scaling for avg return: maps (-inf,+inf) to (0,1)
            import math
            scaled_ret = 1.0 / (1.0 + math.exp(-avg_ret * 0.5))
            scores[engine] = 0.6 * win_rate + 0.4 * scaled_ret

        if not scores:
            return None

        # Normalize so average weight = 1.0
        avg_score = sum(scores.values()) / len(scores) if scores else 1.0
        weights: dict[str, float] = {}
        for engine in default_weights:
            if engine in scores and avg_score > 0:
                raw_w = scores[engine] / avg_score
                weights[engine] = max(
                    self.weight_floor, min(self.weight_cap, raw_w)
                )
            else:
                weights[engine] = default_weights[engine]

        return weights


# ── Convergence multiplier lookup ────────────────────────────────────────


def _get_convergence_multiplier(
    engine_count: int,
    multipliers: dict[int, float],
    regime: str | None = None,
    regime_overrides: dict[str, dict[int, float]] | None = None,
) -> float:
    """Look up convergence multiplier, with optional regime override."""
    # Use regime-specific multipliers if available
    if regime and regime_overrides and regime in regime_overrides:
        regime_mults = regime_overrides[regime]
        if engine_count in regime_mults:
            return regime_mults[engine_count]
        if engine_count >= max(regime_mults.keys(), default=3):
            return regime_mults.get(max(regime_mults.keys(), default=3), 1.0)

    # Fall back to default multipliers
    if engine_count >= max(multipliers.keys(), default=3):
        return multipliers.get(max(multipliers.keys(), default=3), 1.0)
    return multipliers.get(engine_count, 1.0)


# ── Main synthesis function ──────────────────────────────────────────────


def synthesize_picks(
    all_picks: list[NormalizedPick],
    config: SynthesisConfig,
    rolling_weights: dict[str, float] | None = None,
    regime: str | None = None,
) -> list[SynthesisPick]:
    """Merge picks across engines into a ranked synthesis list.

    Algorithm:
      1. Group all picks by ticker.
      2. For each ticker, compute weighted-average confidence using
         engine weights (initial or rolling).
      3. Apply convergence multiplier based on how many engines agree
         (regime-adaptive when *regime* is provided).
      4. ``combined_score = avg_weighted_confidence * convergence_mult``
      5. Filter out picks below *min_confidence* quality floor.
      6. Sort by combined_score and return top-N.
      7. Entry/stop/target come from the highest-weighted contributing pick.
    """
    if not all_picks:
        return []

    weights = rolling_weights if rolling_weights else config.initial_weights

    # Group by ticker
    by_ticker: dict[str, list[NormalizedPick]] = defaultdict(list)
    for pick in all_picks:
        by_ticker[pick.ticker].append(pick)

    synthesis_results: list[SynthesisPick] = []

    for ticker, ticker_picks in by_ticker.items():
        engine_names = list({p.engine_name for p in ticker_picks})
        engine_count = len(engine_names)
        convergence_mult = _get_convergence_multiplier(
            engine_count,
            config.convergence_multipliers,
            regime=regime,
            regime_overrides=config.regime_convergence_overrides,
        )

        # Weighted confidence
        total_weighted_conf = 0.0
        total_weight = 0.0
        strategies: list[str] = []

        for pick in ticker_picks:
            w = weights.get(pick.engine_name, 1.0)
            total_weighted_conf += pick.confidence * w
            total_weight += w
            strategies.append(pick.strategy)

        avg_weighted_conf = (
            total_weighted_conf / total_weight if total_weight > 0 else 0
        )

        # Quality floor: skip low-confidence picks
        if avg_weighted_conf < config.min_confidence:
            continue

        combined_score = avg_weighted_conf * convergence_mult

        # Best pick = highest weighted contribution
        best_pick = max(
            ticker_picks,
            key=lambda p: weights.get(p.engine_name, 1.0) * p.confidence,
        )

        synthesis_results.append(SynthesisPick(
            ticker=ticker,
            combined_score=round(combined_score, 2),
            avg_weighted_confidence=round(avg_weighted_conf, 2),
            convergence_multiplier=convergence_mult,
            engine_count=engine_count,
            engines=engine_names,
            strategies=strategies,
            entry_price=best_pick.entry_price,
            stop_loss=best_pick.stop_loss,
            target_price=best_pick.target_price,
            holding_period_days=best_pick.holding_period_days,
            direction=best_pick.direction,
            source_picks=ticker_picks,
        ))

    # Sort by combined score and take top-N
    synthesis_results.sort(key=lambda s: s.combined_score, reverse=True)
    top_n = synthesis_results[: config.top_n_per_day]

    if top_n:
        logger.info(
            "Synthesis: %d tickers → top %d (best: %s %.1f, %d engines)",
            len(synthesis_results),
            len(top_n),
            top_n[0].ticker,
            top_n[0].combined_score,
            top_n[0].engine_count,
        )

    return top_n
