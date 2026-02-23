from __future__ import annotations

from datetime import date

from src.backtest.multi_engine.adapters.base import NormalizedPick
from src.backtest.multi_engine.report_generator import generate_report
from src.backtest.multi_engine.portfolio_sim import DailyPickRecord, TrackResult
from src.backtest.multi_engine.synthesizer import (
    RollingCredibilityTracker,
    SynthesisConfig,
    SynthesisPick,
    synthesize_picks,
)
from src.engines.regime_gate import apply_regime_strategy_gate


def _pick(
    ticker: str,
    engine: str,
    strategy: str,
    confidence: float,
) -> NormalizedPick:
    return NormalizedPick(
        ticker=ticker,
        engine_name=engine,
        strategy=strategy,
        entry_price=100.0,
        stop_loss=95.0,
        target_price=110.0,
        confidence=confidence,
        holding_period_days=5,
        direction="LONG",
    )


def _synth_pick(ticker: str, score: float, strategies: list[str], engines: list[str]) -> SynthesisPick:
    return SynthesisPick(
        ticker=ticker,
        combined_score=score,
        avg_weighted_confidence=score,
        convergence_multiplier=1.0,
        diversity_multiplier=1.0,
        engine_count=len(engines),
        engines=engines,
        strategies=strategies,
        entry_price=100.0,
        stop_loss=95.0,
        target_price=110.0,
        holding_period_days=5,
        direction="LONG",
    )


def test_synthesize_picks_respects_diversity_config_knobs():
    picks = [
        _pick("AAA", "mas_quant_screener", "breakout", 80),
        _pick("AAA", "koocore_d", "mean_reversion", 80),
        _pick("BBB", "mas_quant_screener", "breakout", 80),
        _pick("BBB", "koocore_d", "momentum", 80),
        _pick("BBB", "gemini_stst", "breakout", 80),
    ]
    cfg = SynthesisConfig(
        min_confidence=0,
        diversity_enabled=True,
        diversity_boost_multi_category=1.05,
        diversity_penalty_homogeneous_3plus=0.85,
        convergence_multipliers={2: 1.0, 3: 1.0},
    )

    out = synthesize_picks(picks, cfg)
    by_ticker = {p.ticker: p for p in out}
    assert by_ticker["AAA"].diversity_multiplier == 1.05
    assert by_ticker["BBB"].diversity_multiplier == 0.85

    cfg_off = SynthesisConfig(
        min_confidence=0,
        diversity_enabled=False,
        convergence_multipliers={2: 1.0, 3: 1.0},
    )
    out_off = synthesize_picks(picks, cfg_off)
    by_ticker_off = {p.ticker: p for p in out_off}
    assert by_ticker_off["AAA"].diversity_multiplier == 1.0
    assert by_ticker_off["BBB"].diversity_multiplier == 1.0


def test_rolling_credibility_tracker_custom_limits_apply():
    tracker = RollingCredibilityTracker(window=5, min_trades=2, weight_floor=0.5, weight_cap=1.2)
    for pnl in [5.0, 6.0]:
        tracker.record_outcome("mas_quant_screener", pnl)
    for pnl in [-4.0, -3.0]:
        tracker.record_outcome("gemini_stst", pnl)
    weights = tracker.get_rolling_weights({"mas_quant_screener": 1.0, "gemini_stst": 1.0})
    assert weights is not None
    assert 0.5 <= weights["mas_quant_screener"] <= 1.2
    assert 0.5 <= weights["gemini_stst"] <= 1.2


class _Cfg:
    regime_strategy_gate_enabled = True
    regime_gate_bear_blocked_strategies = "momentum"
    regime_gate_bear_penalized_strategies = "breakout,swing"
    regime_gate_bear_penalty_multiplier = 0.65


def test_regime_gate_override_average_mode_changes_weighting():
    picks = [{
        "ticker": "MIXD",
        "combined_score": 100.0,
        "avg_weighted_confidence": 100.0,
        "strategies": ["breakout", "mean_reversion"],
    }]
    best, _ = apply_regime_strategy_gate(
        picks, regime="bear", settings=_Cfg(),
        overrides={"strategy_weight_selection_mode": "best"},
    )
    avg, _ = apply_regime_strategy_gate(
        picks, regime="bear", settings=_Cfg(),
        overrides={"strategy_weight_selection_mode": "average"},
    )
    capped, _ = apply_regime_strategy_gate(
        picks, regime="bear", settings=_Cfg(),
        overrides={
            "strategy_weight_selection_mode": "capped_best",
            "strategy_weight_uplift_cap": 1.10,
        },
    )
    assert best[0]["combined_score"] == 125.0  # best uses mean_reversion=1.25
    assert avg[0]["combined_score"] == 95.0    # avg of 1.25 and 0.65
    assert capped[0]["combined_score"] == 110.0


def test_report_generator_emits_tuning_meta_and_synthesis_diagnostics():
    rec = DailyPickRecord(
        screen_date=date(2025, 1, 2),
        regime="bull",
        engine_picks={},
        synthesis_eq=[_synth_pick("AAA", 100, ["breakout"], ["mas_quant_screener", "koocore_d"])],
        synthesis_cred=[_synth_pick("BBB", 90, ["mean_reversion"], ["gemini_stst"])],
        synthesis_regime_gated=[],
    )
    report = generate_report(
        track_results={
            "eq_synth": TrackResult("eq_synth", [], [], []),
            "cred_synth": TrackResult("cred_synth", [], [], []),
            "regime_gated": TrackResult("regime_gated", [], [], []),
            "spy_benchmark": TrackResult("spy_benchmark", [], [], []),
        },
        daily_records=[rec],
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        engine_names=[],
        config={"tuning_meta": {"config_label": "candidate_a"}},
        elapsed_s=1.0,
    )
    assert report["tuning_meta"]["config_label"] == "candidate_a"
    diag = report["synthesis_diagnostics"]
    assert diag["equal_weight"]["total_picks"] == 1
    assert diag["equal_weight"]["avg_engine_count_per_pick"] == 2.0
    assert diag["equal_weight"]["source_engine_counts"]["mas_quant_screener"] == 1
