"""Performance monitoring and decay detection.

Maintains a sliding window over recent trade outcomes and detects model
deterioration: declining hit rate, expanding adverse excursion, or negative
expectancy. When triggered, the system should fall back to conservative
behavior (e.g., reduce position sizes, skip uncertain signals).

Ported from KooCore-D governance/performance_monitor.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DecayThresholds:
    """Configurable thresholds that define when a model is "decaying"."""

    hit_rate_ratio: float = 0.5      # Live hit rate < 50% of baseline → decay
    mae_multiplier: float = 1.5      # Max adverse excursion grows 50% over baseline → decay
    min_expectancy: float = 0.0      # Negative expectancy → decay
    min_trades_for_check: int = 10   # Need at least N trades before checking


@dataclass
class RollingMetrics:
    """Performance metrics over a sliding window."""

    hit_rate: float = 0.0
    avg_gain_pct: float = 0.0
    avg_loss_pct: float = 0.0
    avg_mae_pct: float = 0.0   # Average max adverse excursion
    avg_mfe_pct: float = 0.0   # Average max favorable excursion
    expectancy: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0


@dataclass
class DecayResult:
    """Result of a decay check."""

    is_decaying: bool = False
    triggers: list[str] = field(default_factory=list)
    metrics: RollingMetrics = field(default_factory=RollingMetrics)
    baseline: RollingMetrics | None = None


def compute_rolling_metrics(trades: list[dict], window: int | None = None) -> RollingMetrics:
    """Compute rolling performance metrics from trade outcome dicts.

    Each trade dict should have:
        - pnl_pct: float (realized P&L %)
        - max_adverse: float (max adverse excursion %, usually negative)
        - max_favorable: float (max favorable excursion %)
    """
    if not trades:
        return RollingMetrics()

    if window:
        trades = trades[-window:]

    pnls = [t.get("pnl_pct", 0) or 0 for t in trades]
    maes = [abs(t.get("max_adverse", 0) or 0) for t in trades]
    mfes = [t.get("max_favorable", 0) or 0 for t in trades]

    total = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    hit_rate = len(wins) / total if total > 0 else 0
    avg_gain = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0

    total_gains = sum(wins) if wins else 0
    total_losses = abs(sum(losses)) if losses else 0
    pf = total_gains / total_losses if total_losses > 0 else (float("inf") if total_gains > 0 else 0)

    expectancy = avg_gain * hit_rate + avg_loss * (1 - hit_rate)

    return RollingMetrics(
        hit_rate=round(hit_rate, 4),
        avg_gain_pct=round(avg_gain, 4),
        avg_loss_pct=round(avg_loss, 4),
        avg_mae_pct=round(sum(maes) / total, 4) if total > 0 else 0,
        avg_mfe_pct=round(sum(mfes) / total, 4) if total > 0 else 0,
        expectancy=round(expectancy, 4),
        profit_factor=round(pf, 4) if pf != float("inf") else 999.0,
        total_trades=total,
    )


def check_decay(
    live_metrics: RollingMetrics,
    baseline_metrics: RollingMetrics,
    thresholds: DecayThresholds | None = None,
) -> DecayResult:
    """Detect performance decay by comparing live metrics against baseline.

    Returns DecayResult with is_decaying=True if any threshold is breached.
    """
    if thresholds is None:
        thresholds = DecayThresholds()

    triggers: list[str] = []

    # Not enough data to check
    if live_metrics.total_trades < thresholds.min_trades_for_check:
        return DecayResult(
            is_decaying=False,
            metrics=live_metrics,
            baseline=baseline_metrics,
        )

    # Check 1: Hit rate decay
    if baseline_metrics.hit_rate > 0:
        ratio = live_metrics.hit_rate / baseline_metrics.hit_rate
        if ratio < thresholds.hit_rate_ratio:
            triggers.append(
                f"hit_rate_decay: live={live_metrics.hit_rate:.2%} vs "
                f"baseline={baseline_metrics.hit_rate:.2%} "
                f"(ratio={ratio:.2f} < {thresholds.hit_rate_ratio})"
            )

    # Check 2: MAE expansion
    if baseline_metrics.avg_mae_pct > 0:
        mae_ratio = live_metrics.avg_mae_pct / baseline_metrics.avg_mae_pct
        if mae_ratio > thresholds.mae_multiplier:
            triggers.append(
                f"mae_expansion: live={live_metrics.avg_mae_pct:.2f}% vs "
                f"baseline={baseline_metrics.avg_mae_pct:.2f}% "
                f"(ratio={mae_ratio:.2f} > {thresholds.mae_multiplier})"
            )

    # Check 3: Negative expectancy
    if live_metrics.expectancy < thresholds.min_expectancy:
        triggers.append(
            f"negative_expectancy: {live_metrics.expectancy:.4f} "
            f"< {thresholds.min_expectancy}"
        )

    is_decaying = len(triggers) > 0
    if is_decaying:
        logger.warning("DECAY DETECTED: %s", "; ".join(triggers))
    else:
        logger.info(
            "No decay: hit_rate=%.2f%%, expectancy=%.4f, MAE=%.2f%%",
            live_metrics.hit_rate * 100,
            live_metrics.expectancy,
            live_metrics.avg_mae_pct,
        )

    return DecayResult(
        is_decaying=is_decaying,
        triggers=triggers,
        metrics=live_metrics,
        baseline=baseline_metrics,
    )
