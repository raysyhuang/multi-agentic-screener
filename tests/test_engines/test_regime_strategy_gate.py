"""Tests for deterministic bear-regime strategy gating."""

from __future__ import annotations

from src.engines.regime_gate import apply_regime_strategy_gate


class _Cfg:
    regime_strategy_gate_enabled = True
    regime_gate_bear_blocked_strategies = "momentum"
    regime_gate_bear_penalized_strategies = "breakout,swing"
    regime_gate_bear_penalty_multiplier = 0.65


def _pick(ticker: str, score: float, strategies: list[str]) -> dict:
    return {
        "ticker": ticker,
        "combined_score": score,
        "avg_weighted_confidence": score,
        "strategies": strategies,
    }


def test_non_bear_regime_applies_proactive_weights():
    picks = [_pick("AAA", 90, ["momentum"]), _pick("BBB", 80, ["breakout"])]
    gated, meta = apply_regime_strategy_gate(picks, regime="bull", settings=_Cfg())
    # Bull proactive weights: momentum=1.15 (90*1.15=103.5), breakout=1.20 (80*1.20=96.0)
    assert [p["ticker"] for p in gated] == ["AAA", "BBB"]
    assert not meta["applied"]  # No bear blocking/penalty
    assert meta.get("regime_weights_applied") is True
    assert gated[0]["combined_score"] == 103.5
    assert gated[1]["combined_score"] == 96.0


def test_bear_drops_pure_momentum():
    picks = [
        _pick("MOMO", 88, ["momentum"]),
        _pick("MRVR", 70, ["mean_reversion"]),
    ]
    gated, meta = apply_regime_strategy_gate(picks, regime="bear", settings=_Cfg())
    assert [p["ticker"] for p in gated] == ["MRVR"]
    assert meta["applied"]
    assert meta["dropped"] == 1
    assert "MOMO" in meta["dropped_tickers"]


def test_bear_penalizes_breakout_without_protective_strategy():
    picks = [
        _pick("BRKO", 100, ["breakout"]),
        _pick("SAFE", 90, ["mean_reversion"]),
    ]
    gated, meta = apply_regime_strategy_gate(picks, regime="bear", settings=_Cfg())
    by_ticker = {p["ticker"]: p for p in gated}
    # Bear penalty (0.65) then proactive regime weight for breakout (0.65): 100 * 0.65 * 0.65 = 42.25
    assert by_ticker["BRKO"]["combined_score"] == 42.25
    assert by_ticker["BRKO"]["regime_gate"] == "bear_penalty_x0.65"
    assert meta["penalized"] == 1


def test_bear_keeps_breakout_if_mean_reversion_context_present():
    picks = [_pick("MIXD", 85, ["breakout", "mean_reversion"])]
    gated, meta = apply_regime_strategy_gate(picks, regime="bear", settings=_Cfg())
    assert gated[0]["ticker"] == "MIXD"
    # No penalty; proactive weight uses best strategy: mean_reversion=1.25, so 85 * 1.25 = 106.25
    assert gated[0]["combined_score"] == 106.25
    assert not meta["applied"]


def test_gate_reorders_after_penalty():
    picks = [
        _pick("BRKO", 100, ["breakout"]),
        _pick("SAFE", 80, ["mean_reversion"]),
    ]
    gated, _ = apply_regime_strategy_gate(picks, regime="bear", settings=_Cfg())
    # BRKO drops to 65 after penalty and should rank below SAFE (80).
    assert [p["ticker"] for p in gated] == ["SAFE", "BRKO"]
