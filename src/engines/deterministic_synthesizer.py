"""Deterministic cross-engine synthesizer — replaces LLM-based synthesis.

Takes weighted picks (from credibility.compute_weighted_picks) and produces
a SynthesizerOutput using pure deterministic logic. No LLM calls, zero cost.

Replaces CrossEngineSynthesizerAgent + CrossEngineVerifierAgent.
"""

from __future__ import annotations

import logging

from src.agents.cross_engine_synthesizer import (
    ConvergentPick,
    PortfolioPosition,
    SynthesizerOutput,
    UniquePick,
)

logger = logging.getLogger(__name__)

# Max portfolio positions before low-overlap guardrail
_MAX_PORTFOLIO = 5
# Equal weight per position (guardian adjusts later)
_BASE_WEIGHT_PCT = 10.0


def deterministic_regime_weight_adjust(
    engine_stats: dict,
    regime: str,
) -> dict:
    """Apply regime-based weight adjustments deterministically.

    In bear regime: boost mean-reversion engines, penalize momentum.
    In bull regime: boost momentum engines, penalize mean-reversion.

    Returns dict of {engine_name: adjusted_weight} with adjustment reasons.
    """
    adjustments: dict[str, dict] = {}
    for name, stats in engine_stats.items():
        per_strat = getattr(stats, "per_strategy", None) or {}
        has_mean_rev = any("reversion" in s.lower() for s in per_strat)
        has_momentum = any("momentum" in s.lower() or "breakout" in s.lower() for s in per_strat)

        multiplier = 1.0
        reason = "no adjustment"

        if regime == "bear":
            if has_mean_rev:
                multiplier = 1.1
                reason = "bear regime: mean-reversion boost"
            if has_momentum:
                multiplier *= 0.8
                reason = "bear regime: momentum penalty"
        elif regime == "bull":
            if has_momentum:
                multiplier = 1.1
                reason = "bull regime: momentum boost"
            if has_mean_rev:
                multiplier *= 0.9
                reason = "bull regime: mean-reversion slight penalty"

        # Clamp
        multiplier = max(0.5, min(1.5, multiplier))
        old_weight = stats.weight
        new_weight = round(old_weight * multiplier, 3)
        new_weight = max(0.1, min(3.0, new_weight))

        if abs(multiplier - 1.0) > 0.01:
            stats.weight = new_weight
            adjustments[name] = {
                "old_weight": old_weight,
                "new_weight": new_weight,
                "multiplier": multiplier,
                "reason": reason,
            }
            logger.info(
                "Deterministic weight adjust %s: %.3f → %.3f (×%.2f, %s)",
                name, old_weight, new_weight, multiplier, reason,
            )

    return adjustments


def _apply_veto_board(
    weighted_picks: list[dict],
    engine_results: list | None = None,
) -> list[dict]:
    """Apply veto-first architecture: penalize picks that conflict with other engines.

    If an engine screened a ticker but rejected it (especially for distribution/trend
    reasons), that's a veto signal. Penalize confidence for conflicting picks.

    Also applies idiosyncratic bonus: single-engine picks where the stock is oversold
    but the sector is not (stock_rsi2 < 10 AND sector_rsi2 > 30) get a small boost.
    """
    from src.config import get_settings
    settings = get_settings()

    if not settings.veto_board_enabled:
        return weighted_picks

    # Build a set of tickers that were screened-but-rejected by any engine
    rejected_tickers: dict[str, list[str]] = {}  # ticker -> [rejecting_engine_names]
    if engine_results:
        for er in engine_results:
            # Support EngineResultPayload objects, raw dicts, or dicts with nested payload
            if isinstance(er, dict):
                payload = er.get("payload", er)
            else:
                payload = getattr(er, "payload", None) or {}
            if isinstance(payload, dict):
                screened_rejected = payload.get("screened_but_rejected", [])
                for item in screened_rejected:
                    if isinstance(item, dict):
                        t = item.get("ticker", "")
                        if t:
                            rejected_tickers.setdefault(t, []).append(
                                payload.get("engine_name", "unknown")
                            )

    for wp in weighted_picks:
        ticker = wp["ticker"]
        engine_count = wp.get("engine_count", 1)
        current_conf = wp.get("avg_weighted_confidence", wp.get("combined_score", 50))

        # Veto: penalize picks that other engines explicitly rejected
        if ticker in rejected_tickers and len(rejected_tickers[ticker]) >= 1:
            wp["avg_weighted_confidence"] = current_conf * settings.veto_penalty
            wp["veto_applied"] = True
            wp["veto_engines"] = rejected_tickers[ticker]
            logger.info(
                "Veto board: %s penalized (%.1f → %.1f), rejected by %s",
                ticker, current_conf, wp["avg_weighted_confidence"],
                rejected_tickers[ticker],
            )

        # Idiosyncratic bonus: single-engine pick with stock oversold but sector not
        if settings.idiosyncratic_bonus_enabled and engine_count == 1:
            metadata = wp.get("metadata", {}) or {}
            stock_rsi2 = metadata.get("rsi_2") or metadata.get("stock_rsi2")
            sector_rsi2 = metadata.get("sector_rsi2")
            if (stock_rsi2 is not None and float(stock_rsi2) < 10
                    and sector_rsi2 is not None and float(sector_rsi2) > 30):
                old_conf = wp.get("avg_weighted_confidence", wp.get("combined_score", 50))
                wp["avg_weighted_confidence"] = old_conf * settings.idiosyncratic_bonus_multiplier
                wp["idiosyncratic_bonus"] = True
                logger.info(
                    "Idiosyncratic bonus: %s boosted (%.1f → %.1f), "
                    "stock_rsi2=%.1f sector_rsi2=%.1f",
                    ticker, old_conf, wp["avg_weighted_confidence"],
                    float(stock_rsi2), float(sector_rsi2),
                )

    return weighted_picks


def synthesize_deterministic(
    weighted_picks: list[dict],
    regime: str,
    engines_reporting: int,
    engine_results: list | None = None,
) -> SynthesizerOutput:
    """Build portfolio from weighted picks without LLM.

    Logic:
    0. Apply veto board (penalize conflicting picks, bonus idiosyncratic)
    1. Separate convergent (2+ engines) from unique (1 engine) picks
    2. Sort each group by combined_score descending
    3. Fill portfolio: convergent first, then unique, up to _MAX_PORTFOLIO
    4. Assign equal weights (Capital Guardian adjusts later)
    """
    # Veto board step
    weighted_picks = _apply_veto_board(weighted_picks, engine_results)

    convergent = []
    unique = []

    for wp in weighted_picks:
        engine_count = wp.get("engine_count", 1)
        if engine_count >= 2:
            convergent.append(wp)
        else:
            unique.append(wp)

    # Sort by score
    convergent.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
    unique.sort(key=lambda x: x.get("combined_score", 0), reverse=True)

    # Build convergent pick objects
    convergent_picks = [
        ConvergentPick(
            ticker=wp["ticker"],
            engines=wp.get("engines", []),
            combined_score=wp.get("combined_score", 0),
            strategy_consensus=", ".join(wp.get("strategies", [])),
            entry_price=float(wp.get("entry_price") or 0),
            stop_loss=float(wp.get("stop_loss") or 0),
            target_price=float(wp.get("target_price") or 0),
            holding_period_days=int(wp.get("holding_period_days") or 7),
            thesis=wp.get("thesis") or "",
        )
        for wp in convergent
    ]

    # Build unique pick objects (top 10)
    unique_picks = [
        UniquePick(
            ticker=wp["ticker"],
            engine=wp.get("engines", ["unknown"])[0],
            confidence=wp.get("avg_weighted_confidence", 0),
            strategy=", ".join(wp.get("strategies", [])),
            justification=f"Score {wp.get('combined_score', 0):.1f}, {wp.get('convergence_type', 'unique')}",
        )
        for wp in unique[:10]
    ]

    # Build portfolio: convergent first, then unique
    portfolio = []
    seen_tickers: set[str] = set()

    for wp in convergent + unique:
        if len(portfolio) >= _MAX_PORTFOLIO:
            break
        ticker = wp["ticker"]
        if ticker in seen_tickers:
            continue
        seen_tickers.add(ticker)

        source = "convergent" if wp.get("engine_count", 1) >= 2 else "unique"
        portfolio.append(
            PortfolioPosition(
                ticker=ticker,
                weight_pct=_BASE_WEIGHT_PCT,
                source=source,
                entry_price=float(wp.get("entry_price") or 0),
                stop_loss=float(wp.get("stop_loss") or 0),
                target_price=float(wp.get("target_price") or 0),
                holding_period_days=int(wp.get("holding_period_days") or 7),
            )
        )

    # Executive summary
    n_conv = len(convergent_picks)
    n_uniq = len(unique_picks)
    n_portfolio = len(portfolio)
    tickers_str = ", ".join(p.ticker for p in portfolio) if portfolio else "none"

    if n_portfolio == 0:
        summary = "No tradeable picks this cycle."
    else:
        summary = (
            f"Deterministic synthesis: {n_conv} convergent, {n_uniq} unique, "
            f"{n_portfolio} portfolio positions ({tickers_str}). "
            f"Regime: {regime}, {engines_reporting} engines reporting."
        )

    return SynthesizerOutput(
        convergent_picks=convergent_picks,
        unique_picks=unique_picks,
        portfolio=portfolio,
        regime_consensus=regime,
        executive_summary=summary,
    )
