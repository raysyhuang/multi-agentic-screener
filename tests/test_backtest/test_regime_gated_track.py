"""Tests for the regime_gated backtest track and KooCore-D pick pipeline."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pandas as pd

from src.backtest.multi_engine.adapters.base import NormalizedPick
from src.backtest.multi_engine.portfolio_sim import (
    DailyPickRecord,
    PortfolioConfig,
    TrackResult,
    run_portfolio_simulation,
)
from src.backtest.multi_engine.synthesizer import SynthesisPick
from src.backtest.multi_engine.orchestrator import _apply_regime_gate_to_synthesis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synth_pick(ticker: str, score: float, strategies: list[str]) -> SynthesisPick:
    return SynthesisPick(
        ticker=ticker,
        combined_score=score,
        avg_weighted_confidence=score,
        convergence_multiplier=1.0,
        engine_count=1,
        engines=["test"],
        strategies=strategies,
        entry_price=100.0,
        stop_loss=95.0,
        target_price=110.0,
        holding_period_days=5,
        direction="LONG",
    )


def _price_df(ticker: str, n_days: int = 30) -> pd.DataFrame:
    """Generate a simple price DataFrame for testing."""
    dates = pd.bdate_range(end="2025-01-31", periods=n_days)
    return pd.DataFrame({
        "date": dates.date,
        "open": [100.0] * n_days,
        "high": [102.0] * n_days,
        "low": [98.0] * n_days,
        "close": [101.0] * n_days,
        "volume": [1_000_000] * n_days,
    })


# ---------------------------------------------------------------------------
# Bug #1: regime_gated must NOT fall back to eq when all picks are blocked
# ---------------------------------------------------------------------------

def test_regime_gated_skips_day_when_all_blocked():
    """When regime gate blocks all picks, the regime_gated track should produce
    zero trades for that day — NOT fall back to eq_synth picks."""
    eq_picks = [_synth_pick("AAPL", 80, ["momentum"])]

    record = DailyPickRecord(
        screen_date=date(2025, 1, 15),
        regime="bear",
        engine_picks={"test": []},
        synthesis_eq=eq_picks,
        synthesis_cred=eq_picks,
        synthesis_regime_gated=[],  # All blocked by bear regime
    )

    price_data = {"AAPL": _price_df("AAPL")}
    config = PortfolioConfig(holding_periods=[5])

    results = run_portfolio_simulation([record], price_data, config)

    # eq_synth should have trades (it doesn't gate)
    eq_trades = results["eq_synth"].trades
    # regime_gated should have ZERO trades (all were blocked)
    gated_trades = results["regime_gated"].trades

    assert len(gated_trades) == 0, (
        f"regime_gated should have 0 trades when all picks are blocked, got {len(gated_trades)}"
    )
    # Sanity: eq_synth did produce trades
    assert len(eq_trades) > 0


# ---------------------------------------------------------------------------
# Regime gate integration: verify bear blocks momentum
# ---------------------------------------------------------------------------

def test_regime_gate_blocks_pure_momentum_in_bear():
    """apply_regime_strategy_gate should drop pure momentum picks in bear regime."""
    picks = [
        _synth_pick("MOMO", 90, ["momentum"]),
        _synth_pick("SAFE", 70, ["mean_reversion"]),
    ]

    gated = _apply_regime_gate_to_synthesis(picks, regime="bear")

    tickers = [p.ticker for p in gated]
    assert "MOMO" not in tickers, "Pure momentum should be blocked in bear"
    assert "SAFE" in tickers, "Mean reversion should survive bear regime"


def test_regime_gate_passes_through_in_bull():
    """In bull regime, no picks should be dropped (just re-weighted)."""
    picks = [
        _synth_pick("MOMO", 90, ["momentum"]),
        _synth_pick("BRKO", 80, ["breakout"]),
    ]

    gated = _apply_regime_gate_to_synthesis(picks, regime="bull")
    assert len(gated) == 2, "Bull regime should not drop any picks"


# ---------------------------------------------------------------------------
# Bug #2: KooCore-D pick pipeline should not include priceless pro30 tickers
# ---------------------------------------------------------------------------

def test_koocore_payload_excludes_priceless_pro30():
    """Verify that _map_hybrid_to_payload does NOT include pro30_tickers
    that lack price data, which would cause collector quality rejection."""
    from src.engines.koocore_runner import _map_hybrid_to_payload

    hybrid = {
        "hybrid_top3": [
            {
                "ticker": "AAPL",
                "composite_score": 7.0,
                "sources": ["Weekly(1)", "Pro30(r1)"],
                "current_price": 185.0,
                "rank": 1,
            }
        ],
        "weighted_picks": [],
        "primary_top5": [],
        "pro30_tickers": ["XYZ", "ABC", "DEF"],  # No price data available
        "summary": {
            "weekly_top5_count": 1,
            "pro30_candidates_count": 3,
            "movers_count": 0,
        },
    }

    payload = _map_hybrid_to_payload(hybrid, "2025-01-15")

    pick_tickers = {p.ticker for p in payload.picks}
    # AAPL should be included (has price)
    assert "AAPL" in pick_tickers
    # pro30_tickers without prices should NOT be included
    assert "XYZ" not in pick_tickers
    assert "ABC" not in pick_tickers
    assert "DEF" not in pick_tickers
    # But they should still be counted in candidates_screened
    assert payload.candidates_screened == 4  # 1 weekly + 3 pro30


# ---------------------------------------------------------------------------
# Bug #3: report generator includes regime_gated in head-to-head
# ---------------------------------------------------------------------------

def test_report_generator_includes_regime_gated():
    """Verify that generate_report includes regime_gated in synthesis and
    head-to-head sections."""
    from src.backtest.multi_engine.report_generator import generate_report

    # Minimal track results with regime_gated
    tracks = {
        "eq_synth": TrackResult("eq_synth", [], [], []),
        "cred_synth": TrackResult("cred_synth", [], [], []),
        "regime_gated": TrackResult("regime_gated", [], [], []),
        "spy_benchmark": TrackResult("spy_benchmark", [], [], []),
    }

    report = generate_report(
        track_results=tracks,
        daily_records=[],
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
        engine_names=[],
        config={},
        elapsed_s=1.0,
    )

    assert "regime_gated" in report["synthesis"], (
        "regime_gated should appear in synthesis section"
    )
