"""Deterministic regime-aware strategy gate for weighted cross-engine picks."""

from __future__ import annotations

from typing import Any

from src.config import get_settings

_PROTECTIVE_STRATEGIES = {"mean_reversion", "defensive", "value"}


def _norm_strategy(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _as_strategy_set(raw: Any) -> set[str]:
    if isinstance(raw, str):
        return {_norm_strategy(raw)} if raw.strip() else set()
    if isinstance(raw, list):
        out = set()
        for item in raw:
            if isinstance(item, str) and item.strip():
                out.add(_norm_strategy(item))
        return out
    return set()


def _csv_to_set(csv_value: str, fallback: set[str]) -> set[str]:
    parsed = {_norm_strategy(s) for s in csv_value.split(",") if s.strip()}
    return parsed or set(fallback)


def apply_regime_strategy_gate(
    weighted_picks: list[dict],
    regime: str,
    settings: Any | None = None,
) -> tuple[list[dict], dict]:
    """Apply deterministic strategy gating before LLM synthesis.

    In bear regime:
      - Drop pure blocked-strategy picks (default: momentum-only).
      - Penalize risky strategies (default: breakout/swing) unless they
        include a protective strategy (e.g. mean_reversion).
    """
    cfg = settings or get_settings()
    regime_label = (regime or "").strip().lower()
    gate_meta = {
        "applied": False,
        "regime": regime_label,
        "dropped": 0,
        "penalized": 0,
        "dropped_tickers": [],
    }

    if not weighted_picks:
        return [], gate_meta
    if not getattr(cfg, "regime_strategy_gate_enabled", True):
        return [dict(p) for p in weighted_picks], gate_meta
    if regime_label != "bear":
        return [dict(p) for p in weighted_picks], gate_meta

    blocked = _csv_to_set(
        getattr(cfg, "regime_gate_bear_blocked_strategies", "momentum"),
        {"momentum"},
    )
    penalized = _csv_to_set(
        getattr(cfg, "regime_gate_bear_penalized_strategies", "breakout,swing"),
        {"breakout", "swing"},
    )
    penalty_mult = float(getattr(cfg, "regime_gate_bear_penalty_multiplier", 0.65))
    penalty_mult = max(0.1, min(1.0, penalty_mult))

    filtered: list[dict] = []
    for pick in weighted_picks:
        strategies = _as_strategy_set(pick.get("strategies", []))
        if not strategies:
            filtered.append(dict(pick))
            continue

        # Hard block: all contributing strategies are blocked in bear regime.
        if strategies.issubset(blocked):
            gate_meta["applied"] = True
            gate_meta["dropped"] += 1
            ticker = str(pick.get("ticker", ""))
            if ticker:
                gate_meta["dropped_tickers"].append(ticker)
            continue

        needs_penalty = bool(strategies & penalized) and not bool(
            strategies & _PROTECTIVE_STRATEGIES
        )
        if needs_penalty:
            adjusted = dict(pick)
            adjusted["combined_score"] = round(
                float(adjusted.get("combined_score", 0.0)) * penalty_mult, 2
            )
            adjusted["avg_weighted_confidence"] = round(
                float(adjusted.get("avg_weighted_confidence", 0.0)) * penalty_mult, 2
            )
            adjusted["regime_gate"] = f"bear_penalty_x{penalty_mult:.2f}"
            filtered.append(adjusted)
            gate_meta["applied"] = True
            gate_meta["penalized"] += 1
            continue

        filtered.append(dict(pick))

    filtered.sort(key=lambda p: float(p.get("combined_score", 0.0)), reverse=True)
    return filtered, gate_meta
