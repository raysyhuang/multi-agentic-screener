"""Deterministic regime-aware strategy gate for weighted cross-engine picks.

Two layers:
1. Bear-regime blocking/penalty (existing) — hard-blocks pure momentum, penalizes
   breakout/swing unless paired with a protective strategy.
2. Proactive regime-strategy weighting (new) — multiplies combined_score by a
   regime-dependent weight per strategy category, applied in all regimes.
"""

from __future__ import annotations

from typing import Any

from src.config import get_settings

_PROTECTIVE_STRATEGIES = {"mean_reversion", "defensive", "value"}

# ---------------------------------------------------------------------------
# Proactive regime-strategy weight tables
# ---------------------------------------------------------------------------
# Keys are normalized strategy keywords matched against pick strategies.
# Weights > 1.0 boost, < 1.0 penalize.  Default weight = 1.0 for unlisted.

_REGIME_STRATEGY_WEIGHTS: dict[str, dict[str, float]] = {
    "bull": {
        "breakout": 1.20,
        "momentum": 1.15,
        "swing": 1.10,
        "mean_reversion": 0.85,
    },
    "bear": {
        "mean_reversion": 1.25,
        "defensive": 1.20,
        "breakout": 0.65,
        "momentum": 0.60,
    },
    "choppy": {
        "mean_reversion": 1.10,
        "swing": 0.95,
        "momentum": 0.90,
        "breakout": 0.85,
    },
}


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


def _override_to_set(value: Any, fallback: set[str]) -> set[str]:
    if value is None:
        return set(fallback)
    if isinstance(value, list):
        parsed = {_norm_strategy(str(v)) for v in value if str(v).strip()}
        return parsed or set(fallback)
    if isinstance(value, str):
        return _csv_to_set(value, fallback)
    return set(fallback)


def apply_regime_strategy_gate(
    weighted_picks: list[dict],
    regime: str,
    settings: Any | None = None,
    overrides: dict[str, Any] | None = None,
) -> tuple[list[dict], dict]:
    """Apply deterministic strategy gating before LLM synthesis.

    In bear regime:
      - Drop pure blocked-strategy picks (default: momentum-only).
      - Penalize risky strategies (default: breakout/swing) unless they
        include a protective strategy (e.g. mean_reversion).
    """
    cfg = settings or get_settings()
    overrides = overrides or {}
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

    # Non-bear regimes skip blocking/penalty but still get proactive weighting
    if regime_label != "bear":
        filtered = [dict(p) for p in weighted_picks]
        weight_table = _get_regime_weight_table(regime_label, overrides)
        if weight_table:
            filtered = _apply_regime_weights(
                filtered, weight_table,
                mode=str(overrides.get("strategy_weight_selection_mode", "best")),
                uplift_cap=float(overrides.get("strategy_weight_uplift_cap", 1.10)),
            )
            gate_meta["regime_weights_applied"] = True
        else:
            gate_meta["regime_weights_applied"] = False
        return filtered, gate_meta

    blocked = _override_to_set(
        overrides.get("bear_blocked_strategies", getattr(cfg, "regime_gate_bear_blocked_strategies", "momentum")),
        {"momentum"},
    )
    penalized = _override_to_set(
        overrides.get("bear_penalized_strategies", getattr(cfg, "regime_gate_bear_penalized_strategies", "breakout,swing")),
        {"breakout", "swing"},
    )
    penalty_mult = float(
        overrides.get(
            "bear_penalty_multiplier",
            getattr(cfg, "regime_gate_bear_penalty_multiplier", 0.65),
        )
    )
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

    # --- Proactive regime-strategy weighting (all regimes) ---
    weight_table = _get_regime_weight_table(regime_label, overrides)
    if weight_table:
        filtered = _apply_regime_weights(
            filtered, weight_table,
            mode=str(overrides.get("strategy_weight_selection_mode", "best")),
            uplift_cap=float(overrides.get("strategy_weight_uplift_cap", 1.10)),
        )
        gate_meta["regime_weights_applied"] = True
    else:
        gate_meta["regime_weights_applied"] = False

    return filtered, gate_meta


def _get_regime_weight_table(regime_label: str, overrides: dict[str, Any]) -> dict[str, float]:
    raw_table = overrides.get("weights")
    if isinstance(raw_table, dict):
        regime_table = raw_table.get(regime_label)
        if isinstance(regime_table, dict):
            return {_norm_strategy(str(k)): float(v) for k, v in regime_table.items()}
    return _REGIME_STRATEGY_WEIGHTS.get(regime_label, {})


def _best_regime_weight(strategies: set[str], weight_table: dict[str, float]) -> float:
    """Pick the most favorable regime weight from a pick's strategies.

    If a pick has multiple strategies, we use the highest weight so that
    protective strategies (e.g. mean_reversion paired with swing) benefit
    from the protective uplift rather than being dragged down.
    """
    weights = [weight_table.get(s, 1.0) for s in strategies]
    return max(weights) if weights else 1.0


def _regime_weight_for_pick(
    strategies: set[str],
    weight_table: dict[str, float],
    *,
    mode: str = "best",
    uplift_cap: float = 1.10,
) -> float:
    """Compute regime multiplier for a pick from its strategy set.

    Modes:
      - ``best``: highest weight among strategies (existing behavior)
      - ``average``: arithmetic mean of strategy weights
      - ``capped_best``: ``best`` with upside capped to avoid over-boosting mixed-tag picks
    """
    weights = [weight_table.get(s, 1.0) for s in strategies]
    if not weights:
        return 1.0
    if mode == "average":
        return sum(weights) / len(weights)
    if mode == "capped_best":
        best = max(weights)
        return min(best, max(1.0, uplift_cap))
    return max(weights)


def _apply_regime_weights(
    picks: list[dict],
    weight_table: dict[str, float],
    *,
    mode: str = "best",
    uplift_cap: float = 1.10,
) -> list[dict]:
    """Multiply combined_score by regime-strategy weight and re-sort."""
    weighted: list[dict] = []
    for pick in picks:
        adjusted = dict(pick)
        strategies = _as_strategy_set(pick.get("strategies", []))
        mult = _regime_weight_for_pick(
            strategies, weight_table, mode=mode, uplift_cap=uplift_cap
        )
        if mult != 1.0:
            adjusted["combined_score"] = round(
                float(adjusted.get("combined_score", 0.0)) * mult, 2
            )
            adjusted["regime_weight"] = round(mult, 2)
        weighted.append(adjusted)
    weighted.sort(key=lambda p: float(p.get("combined_score", 0.0)), reverse=True)
    return weighted
