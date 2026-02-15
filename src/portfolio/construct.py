"""Portfolio construction — builds trade plans from ranked candidates.

Implements three sizing strategies:
  - Kelly criterion (growth-optimal with drawdown control)
  - Volatility-scaled (equal risk per trade)
  - Equal weight (1/N)

Liquidity caps prevent oversized positions in illiquid names.
Regime multipliers adjust exposure based on market conditions.

Ported from KooCore-D portfolio/construct.py + gemini_STST position sizing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SizingMethod(str, Enum):
    KELLY = "kelly"
    VOLATILITY = "volatility"
    EQUAL = "equal"


@dataclass
class PortfolioConfig:
    """Configuration for portfolio construction."""

    portfolio_usd: float = 100_000.0
    max_positions: int = 5
    min_confidence: float = 0.0         # Minimum confidence (0-100) to include
    sizing_method: SizingMethod = SizingMethod.VOLATILITY
    target_risk_pct: float = 1.0        # Target risk per trade (% of portfolio)
    min_position_pct: float = 5.0       # Min position size (% of portfolio)
    max_position_pct: float = 20.0      # Max position size (% of portfolio)
    max_adv_fraction: float = 0.01      # Max position as fraction of ADV (1%)
    kelly_fraction: float = 0.25        # Quarter-Kelly for conservative sizing


# Regime adjusts gross exposure
REGIME_EXPOSURE = {
    "bull": 1.0,
    "bear": 0.5,
    "choppy": 0.75,
}

# Quality adjusts per-position weight
QUALITY_MULTIPLIERS = [
    (70, 1.25),  # confidence >= 70 → 125% weight
    (40, 1.0),   # confidence >= 40 → 100% weight
    (0, 0.75),   # confidence <  40 →  75% weight
]


@dataclass
class TradePlan:
    """A single position in the trade plan."""

    ticker: str
    direction: str
    entry_price: float
    stop_loss: float
    target_price: float
    confidence: float
    signal_model: str
    holding_period: int
    weight_pct: float           # Position size as % of portfolio
    notional_usd: float         # Dollar amount
    shares: int                 # Number of shares (rounded down)
    risk_per_share: float       # Entry - stop loss
    reward_risk_ratio: float    # (Target - entry) / (entry - stop)


def build_trade_plan(
    candidates: list[dict],
    config: PortfolioConfig | None = None,
    regime: str = "bull",
) -> list[TradePlan]:
    """Build a portfolio trade plan from ranked candidates.

    Steps:
      1. Filter by minimum confidence
      2. Sort by confidence descending
      3. Cap at max_positions
      4. Compute weights using the selected sizing method
      5. Apply liquidity caps
      6. Scale by regime exposure
      7. Clamp to min/max position size
    """
    if config is None:
        config = PortfolioConfig()

    # Step 1: Filter
    eligible = [
        c for c in candidates
        if (c.get("confidence", 0) or 0) >= config.min_confidence
    ]

    if not eligible:
        logger.info("No candidates above confidence threshold %.0f", config.min_confidence)
        return []

    # Step 2: Sort by confidence
    eligible.sort(key=lambda c: c.get("confidence", 0) or 0, reverse=True)

    # Step 3: Cap
    eligible = eligible[:config.max_positions]

    # Step 4: Compute raw weights
    if config.sizing_method == SizingMethod.KELLY:
        weights = _kelly_weights(eligible, config)
    elif config.sizing_method == SizingMethod.VOLATILITY:
        weights = _volatility_weights(eligible, config)
    else:
        weights = _equal_weights(eligible, config)

    # Step 5-7: Build plans
    regime_mult = REGIME_EXPOSURE.get(regime.lower(), 0.75)
    plans: list[TradePlan] = []

    for candidate, raw_weight in zip(eligible, weights):
        entry = candidate["entry_price"]
        stop = candidate["stop_loss"]
        target = candidate.get("target_1", entry * 1.10)
        confidence = candidate.get("confidence", 50)
        atr_pct = candidate.get("atr_pct", 0)
        adv = candidate.get("avg_daily_volume", 0) or candidate.get("vol_sma_20", 0)

        # Quality multiplier
        quality_mult = 0.75
        for threshold, mult in QUALITY_MULTIPLIERS:
            if confidence >= threshold:
                quality_mult = mult
                break

        # Adjusted weight
        weight_pct = raw_weight * regime_mult * quality_mult

        # Liquidity cap: position cannot exceed max_adv_fraction of ADV
        if adv and entry > 0:
            max_shares_by_liquidity = int(adv * config.max_adv_fraction)
            max_notional_by_liquidity = max_shares_by_liquidity * entry
            max_weight_by_liquidity = (max_notional_by_liquidity / config.portfolio_usd) * 100
            weight_pct = min(weight_pct, max_weight_by_liquidity)

        # Clamp to bounds
        weight_pct = max(config.min_position_pct, min(config.max_position_pct, weight_pct))

        notional = config.portfolio_usd * (weight_pct / 100)
        shares = int(notional / entry) if entry > 0 else 0
        risk_per_share = abs(entry - stop)
        rr = (target - entry) / risk_per_share if risk_per_share > 0 else 0

        plans.append(TradePlan(
            ticker=candidate["ticker"],
            direction=candidate.get("direction", "LONG"),
            entry_price=entry,
            stop_loss=stop,
            target_price=target,
            confidence=confidence,
            signal_model=candidate.get("signal_model", "unknown"),
            holding_period=candidate.get("holding_period", 10),
            weight_pct=round(weight_pct, 2),
            notional_usd=round(notional, 2),
            shares=shares,
            risk_per_share=round(risk_per_share, 2),
            reward_risk_ratio=round(rr, 2),
        ))

    total_weight = sum(p.weight_pct for p in plans)
    logger.info(
        "Trade plan: %d positions, total weight=%.1f%%, regime=%s (mult=%.2f)",
        len(plans), total_weight, regime, regime_mult,
    )
    return plans


def _kelly_weights(candidates: list[dict], config: PortfolioConfig) -> list[float]:
    """Kelly criterion: f* = (p*b - q) / b, then scale by kelly_fraction.

    p = estimated win probability (confidence / 100)
    b = payoff ratio (target gain / stop loss distance)
    q = 1 - p
    """
    weights = []
    for c in candidates:
        entry = c["entry_price"]
        stop = c["stop_loss"]
        target = c.get("target_1", entry * 1.10)
        prob = min(max((c.get("confidence", 50) or 50) / 100, 0.01), 0.99)

        gain_pct = abs(target - entry) / entry if entry > 0 else 0
        loss_pct = abs(entry - stop) / entry if entry > 0 else 0
        b = gain_pct / loss_pct if loss_pct > 0 else 1

        q = 1 - prob
        kelly_full = (prob * b - q) / b if b > 0 else 0
        kelly_full = max(kelly_full, 0)  # Never bet negative

        weight = kelly_full * config.kelly_fraction * 100  # Convert to %
        weights.append(weight)

    return weights


def _volatility_weights(candidates: list[dict], config: PortfolioConfig) -> list[float]:
    """Volatility-scaled: position_pct = target_risk / atr_pct.

    Keeps risk constant across trades regardless of volatility.
    """
    weights = []
    for c in candidates:
        atr_pct = c.get("atr_pct", 0) or 0
        if atr_pct <= 0:
            # Fallback: use stop distance as risk estimate
            entry = c["entry_price"]
            stop = c["stop_loss"]
            atr_pct = abs(entry - stop) / entry * 100 if entry > 0 else 5.0

        weight = (config.target_risk_pct / (atr_pct / 100)) if atr_pct > 0 else 10.0
        weights.append(weight)

    return weights


def _equal_weights(candidates: list[dict], config: PortfolioConfig) -> list[float]:
    """Equal weight: 1/N allocation."""
    n = len(candidates) or 1
    weight = 100.0 / n
    return [weight] * len(candidates)
