"""Capital Guardian — portfolio-level risk defense for compounding strategies.

Sits between synthesis and final output. Pure math, no LLM.
Computes portfolio risk state and outputs a sizing multiplier that
protects the compounding chain from drawdown-induced ruin.

Rules (applied in order, most restrictive wins):
  1. Drawdown circuit breaker: halt all trading if trailing drawdown > threshold
  2. Streak reduction: scale down after consecutive losses
  3. Regime scaling: reduce exposure in bear/choppy markets
  4. Sector concentration: cap combined weight in any one sector
  5. Per-trade risk cap: ensure no single trade risks > X% of capital
  6. Portfolio heat cap: total capital at risk across all positions < threshold
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlalchemy import select, and_

from src.config import get_settings
from src.db.models import Outcome, Signal
from src.db.session import get_session

logger = logging.getLogger(__name__)


@dataclass
class PortfolioRiskState:
    """Current portfolio risk state computed from recent trade history."""

    # Drawdown
    equity_curve_pct: list[float] = field(default_factory=list)
    peak_equity_pct: float = 0.0
    current_equity_pct: float = 0.0
    current_drawdown_pct: float = 0.0  # negative number

    # Streak
    recent_results: list[float] = field(default_factory=list)  # last N trade P&Ls
    consecutive_losses: int = 0
    consecutive_wins: int = 0

    # Portfolio heat
    open_position_count: int = 0
    total_open_risk_pct: float = 0.0  # sum of (entry - stop) / entry across open positions

    # Stats
    recent_win_rate: float = 0.0
    recent_avg_return: float = 0.0
    total_closed_trades: int = 0


@dataclass
class GuardianVerdict:
    """Output of the Capital Guardian — sizing directive for the pipeline."""

    sizing_multiplier: float = 1.0  # 0.0 (halt) to 1.0 (full size)
    halt: bool = False
    halt_reason: str = ""
    warnings: list[str] = field(default_factory=list)
    risk_state: PortfolioRiskState = field(default_factory=PortfolioRiskState)

    # Breakdown of what contributed to the multiplier
    drawdown_factor: float = 1.0
    streak_factor: float = 1.0
    regime_factor: float = 1.0
    heat_factor: float = 1.0


async def compute_portfolio_risk_state(lookback_days: int = 90) -> PortfolioRiskState:
    """Query recent outcomes and compute current portfolio risk state."""
    state = PortfolioRiskState()
    cutoff = date.today() - timedelta(days=lookback_days)

    async with get_session() as session:
        # Closed trades (ordered by exit date for equity curve)
        closed_result = await session.execute(
            select(Outcome).where(
                and_(
                    Outcome.still_open == False,  # noqa: E712
                    Outcome.entry_date >= cutoff,
                )
            ).order_by(Outcome.exit_date.asc())
        )
        closed = closed_result.scalars().all()

        # Open positions
        open_result = await session.execute(
            select(Outcome, Signal)
            .join(Signal, Outcome.signal_id == Signal.id)
            .where(Outcome.still_open == True)  # noqa: E712
        )
        open_positions = open_result.all()

    state.total_closed_trades = len(closed)

    if closed:
        # Build equity curve (cumulative P&L)
        cumulative = 0.0
        peak = 0.0
        for o in closed:
            pnl = o.pnl_pct or 0.0
            cumulative += pnl
            state.equity_curve_pct.append(cumulative)
            peak = max(peak, cumulative)

        state.peak_equity_pct = peak
        state.current_equity_pct = cumulative
        state.current_drawdown_pct = cumulative - peak  # negative if in drawdown

        # Recent results (last 20 trades for streak analysis)
        recent = closed[-20:]
        state.recent_results = [o.pnl_pct or 0.0 for o in recent]

        # Consecutive streak (from most recent backward)
        for pnl in reversed(state.recent_results):
            if pnl <= 0:
                if state.consecutive_wins == 0:
                    state.consecutive_losses += 1
                else:
                    break
            else:
                if state.consecutive_losses == 0:
                    state.consecutive_wins += 1
                else:
                    break

        # Recent stats
        wins = sum(1 for r in state.recent_results if r > 0)
        state.recent_win_rate = wins / len(state.recent_results) if state.recent_results else 0
        state.recent_avg_return = (
            sum(state.recent_results) / len(state.recent_results)
            if state.recent_results else 0
        )

    # Open position risk
    state.open_position_count = len(open_positions)
    for outcome, signal in open_positions:
        if signal.entry_price > 0 and signal.stop_loss > 0:
            risk_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price * 100
            state.total_open_risk_pct += risk_pct

    return state


def compute_guardian_verdict(
    risk_state: PortfolioRiskState,
    regime: str = "bull",
    portfolio_sectors: list[str] | None = None,
) -> GuardianVerdict:
    """Compute sizing directive from portfolio risk state.

    Pure deterministic math — no LLM calls.
    Returns a GuardianVerdict with sizing_multiplier in [0.0, 1.0].
    """
    settings = get_settings()
    verdict = GuardianVerdict(risk_state=risk_state)

    # --- 1. Drawdown circuit breaker ---
    max_dd = settings.guardian_max_drawdown_pct
    if risk_state.current_drawdown_pct <= -max_dd:
        verdict.halt = True
        verdict.halt_reason = (
            f"Drawdown circuit breaker: {risk_state.current_drawdown_pct:.1f}% "
            f"exceeds -{max_dd}% threshold"
        )
        verdict.sizing_multiplier = 0.0
        verdict.drawdown_factor = 0.0
        return verdict

    # Gradual drawdown reduction: linear scale from 1.0 at 0% DD to 0.25 at max_dd
    if risk_state.current_drawdown_pct < 0:
        dd_ratio = abs(risk_state.current_drawdown_pct) / max_dd  # 0.0 to 1.0
        verdict.drawdown_factor = max(0.25, 1.0 - (dd_ratio * 0.75))
        if dd_ratio > 0.5:
            verdict.warnings.append(
                f"Drawdown warning: {risk_state.current_drawdown_pct:.1f}% "
                f"(>{max_dd * 0.5:.0f}% threshold)"
            )

    # --- 2. Streak reduction ---
    streak_threshold = settings.guardian_streak_reduction_after
    halt_threshold = settings.guardian_halt_after_consecutive_losses

    if risk_state.consecutive_losses >= halt_threshold:
        verdict.halt = True
        verdict.halt_reason = (
            f"Loss streak circuit breaker: {risk_state.consecutive_losses} consecutive losses "
            f"(threshold: {halt_threshold})"
        )
        verdict.sizing_multiplier = 0.0
        verdict.streak_factor = 0.0
        return verdict

    if risk_state.consecutive_losses >= streak_threshold:
        # Reduce by 25% for each loss beyond the threshold
        excess = risk_state.consecutive_losses - streak_threshold
        verdict.streak_factor = max(0.25, 1.0 - (excess * 0.25))
        verdict.warnings.append(
            f"Loss streak: {risk_state.consecutive_losses} consecutive "
            f"(reducing size to {verdict.streak_factor:.0%})"
        )

    # Winning streak bonus (mild, capped at 1.25x — not Martingale)
    if risk_state.consecutive_wins >= 3:
        verdict.streak_factor = min(1.25, 1.0 + (risk_state.consecutive_wins - 2) * 0.05)

    # --- 3. Regime scaling ---
    regime_multipliers = {
        "bull": 1.0,
        "bear": settings.guardian_bear_sizing,
        "choppy": settings.guardian_choppy_sizing,
    }
    verdict.regime_factor = regime_multipliers.get(regime.lower(), 0.75)
    if regime.lower() in ("bear", "choppy"):
        verdict.warnings.append(
            f"Regime scaling: {regime} → {verdict.regime_factor:.0%} sizing"
        )

    # --- 4. Portfolio heat cap ---
    soft_heat = max(settings.guardian_max_portfolio_heat_pct, 0.1)
    hard_heat = max(settings.guardian_halt_portfolio_heat_pct, soft_heat + 0.1)
    overheat_floor = max(0.0, min(1.0, settings.guardian_overheat_sizing_floor))
    current_heat = risk_state.total_open_risk_pct

    # Hard heat breaker: portfolio already too loaded to add risk.
    if current_heat >= hard_heat:
        verdict.halt = True
        verdict.halt_reason = (
            f"Portfolio heat circuit breaker: {current_heat:.1f}% "
            f"exceeds {hard_heat:.1f}% threshold"
        )
        verdict.sizing_multiplier = 0.0
        verdict.heat_factor = 0.0
        return verdict

    # Soft heat zone: gradual de-risking, not full halt.
    if current_heat >= soft_heat:
        # 0.5x at soft cap, decays linearly to floor at hard cap.
        ratio = (current_heat - soft_heat) / (hard_heat - soft_heat)
        verdict.heat_factor = max(
            overheat_floor,
            0.5 - (0.5 - overheat_floor) * ratio,
        )
        verdict.warnings.append(
            f"Portfolio heat above soft cap: {current_heat:.1f}% "
            f"(soft {soft_heat:.1f}%, hard {hard_heat:.1f}%)"
        )
    elif current_heat > soft_heat * 0.7:
        # Pre-cap zone: taper from 1.0x at 70% cap utilization to 0.5x at soft cap.
        ratio = (current_heat - soft_heat * 0.7) / (soft_heat * 0.3)
        verdict.heat_factor = max(0.5, 1.0 - 0.5 * ratio)
        verdict.warnings.append(
            f"Approaching heat cap: {current_heat:.1f}% "
            f"of {soft_heat:.1f}% deployed"
        )

    # --- 5. Sector concentration (if sector data available) ---
    if portfolio_sectors:
        from collections import Counter
        sector_counts = Counter(portfolio_sectors)
        max_sector = settings.guardian_max_sector_concentration
        for sector, count in sector_counts.items():
            if count >= max_sector:
                verdict.warnings.append(
                    f"Sector concentration: {count} positions in {sector} "
                    f"(max {max_sector})"
                )

    # --- Combine factors (multiplicative — most restrictive wins) ---
    verdict.sizing_multiplier = (
        verdict.drawdown_factor
        * verdict.streak_factor
        * verdict.regime_factor
        * verdict.heat_factor
    )
    # Clamp to [0.0, 1.25] — mild winning-streak bonus is the only way above 1.0
    verdict.sizing_multiplier = max(0.0, min(1.25, verdict.sizing_multiplier))
    verdict.sizing_multiplier = round(verdict.sizing_multiplier, 3)

    if verdict.sizing_multiplier == 0.0 and not verdict.halt:
        verdict.halt = True
        verdict.halt_reason = "Combined risk factors reduce sizing to zero"

    return verdict


def apply_guardian_to_portfolio(
    portfolio: list[dict],
    verdict: GuardianVerdict,
    per_trade_risk_cap_pct: float | None = None,
) -> list[dict]:
    """Apply guardian sizing to a synthesis portfolio.

    Adjusts weight_pct on each position. If halt, returns empty portfolio.
    """
    settings = get_settings()
    risk_cap = per_trade_risk_cap_pct or settings.guardian_per_trade_risk_cap_pct

    if verdict.halt:
        logger.warning(
            "Capital Guardian HALT: %s — blocking all positions", verdict.halt_reason
        )
        return []

    adjusted = []
    for pos in portfolio:
        original_weight = pos.get("weight_pct", 10.0)

        # Apply guardian multiplier
        new_weight = original_weight * verdict.sizing_multiplier

        # Enforce per-trade risk cap
        entry = pos.get("entry_price", 0)
        stop = pos.get("stop_loss", 0)
        if entry > 0 and stop > 0:
            risk_per_unit = abs(entry - stop) / entry * 100  # risk as % of position
            if risk_per_unit > 0:
                # max_weight such that position_risk = risk_cap% of portfolio
                max_weight_by_risk = (risk_cap / risk_per_unit) * 100
                if new_weight > max_weight_by_risk:
                    new_weight = max_weight_by_risk

        new_weight = round(max(0.0, new_weight), 2)

        adjusted_pos = {**pos, "weight_pct": new_weight}
        if new_weight != original_weight:
            adjusted_pos["guardian_adjusted"] = True
            adjusted_pos["original_weight_pct"] = original_weight

        adjusted.append(adjusted_pos)

    # Filter out positions reduced to near-zero
    adjusted = [p for p in adjusted if p["weight_pct"] >= 1.0]

    total_original = sum(p.get("original_weight_pct", p["weight_pct"]) for p in adjusted)
    total_adjusted = sum(p["weight_pct"] for p in adjusted)
    logger.info(
        "Capital Guardian: multiplier=%.3f, portfolio %d→%d positions, "
        "weight %.1f%%→%.1f%%, factors=[dd=%.2f, streak=%.2f, regime=%.2f, heat=%.2f]",
        verdict.sizing_multiplier, len(portfolio), len(adjusted),
        total_original, total_adjusted,
        verdict.drawdown_factor, verdict.streak_factor,
        verdict.regime_factor, verdict.heat_factor,
    )

    return adjusted


def format_guardian_summary(verdict: GuardianVerdict) -> str:
    """Format guardian verdict for Telegram alert."""
    rs = verdict.risk_state
    lines = ["\n--- Capital Guardian ---"]

    if verdict.halt:
        lines.append(f"HALT: {verdict.halt_reason}")
        return "\n".join(lines)

    lines.append(
        f"Sizing: {verdict.sizing_multiplier:.0%} "
        f"[DD={verdict.drawdown_factor:.2f} "
        f"Streak={verdict.streak_factor:.2f} "
        f"Regime={verdict.regime_factor:.2f} "
        f"Heat={verdict.heat_factor:.2f}]"
    )

    if rs.current_drawdown_pct < 0:
        lines.append(f"Drawdown: {rs.current_drawdown_pct:.1f}% from peak")
    if rs.consecutive_losses > 0:
        lines.append(f"Streak: {rs.consecutive_losses} consecutive losses")
    if rs.consecutive_wins >= 3:
        lines.append(f"Streak: {rs.consecutive_wins} consecutive wins")
    if rs.open_position_count > 0:
        lines.append(
            f"Open: {rs.open_position_count} positions, "
            f"{rs.total_open_risk_pct:.1f}% heat"
        )
    if rs.total_closed_trades > 0:
        lines.append(
            f"Recent: {rs.recent_win_rate:.0%} win rate, "
            f"{rs.recent_avg_return:+.1f}% avg ({rs.total_closed_trades} trades)"
        )

    for w in verdict.warnings:
        lines.append(f"  {w}")

    return "\n".join(lines)
