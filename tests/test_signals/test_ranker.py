"""Tests for signal ranking, confluence detection, and cooldown."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.features.regime import Regime
from src.signals.breakout import BreakoutSignal
from src.signals.mean_reversion import MeanReversionSignal
from src.signals.sniper import SniperSignal
from src.signals.ranker import (
    rank_candidates,
    deduplicate_signals,
    filter_correlated_picks,
    detect_confluence,
    apply_confluence_bonus,
    apply_cooldown,
    RankedCandidate,
)


def _make_breakout(ticker: str, score: float) -> BreakoutSignal:
    return BreakoutSignal(
        ticker=ticker, score=score, direction="LONG",
        entry_price=100, stop_loss=95, target_1=110, target_2=115,
        holding_period=10, components={},
    )


def _make_mean_rev(ticker: str, score: float) -> MeanReversionSignal:
    return MeanReversionSignal(
        ticker=ticker, score=score, direction="LONG",
        entry_price=50, stop_loss=48, target_1=53, target_2=55,
        holding_period=5, components={},
    )


def test_rank_respects_regime_gate():
    signals = [
        _make_breakout("AAPL", 80),
        _make_mean_rev("MSFT", 75),
    ]
    # In BEAR regime, breakout should be filtered out
    result = rank_candidates(signals, Regime.BEAR, {}, top_n=10)
    tickers = [r.ticker for r in result]
    assert "AAPL" not in tickers  # breakout blocked in bear
    assert "MSFT" in tickers


def test_rank_sorts_by_adjusted_score():
    signals = [
        _make_breakout("LOW", 55),
        _make_breakout("HIGH", 90),
        _make_breakout("MID", 70),
    ]
    result = rank_candidates(signals, Regime.BULL, {}, top_n=10)
    assert result[0].ticker == "HIGH"
    assert result[-1].ticker == "LOW"


def test_rank_respects_top_n():
    signals = [_make_breakout(f"T{i}", 60 + i) for i in range(20)]
    result = rank_candidates(signals, Regime.BULL, {}, top_n=5)
    assert len(result) == 5


def test_deduplicate_keeps_best():
    signals = [
        _make_breakout("AAPL", 80),
        _make_mean_rev("AAPL", 90),  # same ticker, higher score
    ]
    result = deduplicate_signals(signals)
    assert len(result) == 1
    assert result[0].score == 90


def _make_ranked(ticker: str, score: float) -> RankedCandidate:
    return RankedCandidate(
        ticker=ticker, signal_model="breakout", raw_score=score,
        regime_adjusted_score=score, direction="LONG", entry_price=100,
        stop_loss=95, target_1=110, target_2=None, holding_period=10,
        components={}, features={},
    )


def test_correlation_filter_drops_identical():
    """Two tickers with identical returns should be filtered."""
    np.random.seed(42)
    returns = np.random.randn(30).cumsum() + 100
    price_data = {
        "A": pd.DataFrame({"close": returns}),
        "B": pd.DataFrame({"close": returns}),  # identical
    }
    candidates = [_make_ranked("A", 90), _make_ranked("B", 80)]
    result = filter_correlated_picks(candidates, price_data, max_correlation=0.75)
    assert len(result) == 1
    assert result[0].ticker == "A"


def test_correlation_filter_keeps_uncorrelated():
    """Two uncorrelated tickers should both survive."""
    np.random.seed(42)
    price_data = {
        "A": pd.DataFrame({"close": np.random.randn(30).cumsum() + 100}),
        "B": pd.DataFrame({"close": np.random.randn(30).cumsum() + 200}),
    }
    candidates = [_make_ranked("A", 90), _make_ranked("B", 80)]
    result = filter_correlated_picks(candidates, price_data, max_correlation=0.75)
    assert len(result) == 2


# --- Confluence Detection ---


def test_confluence_single_model():
    """Single model per ticker = no confluence."""
    signals = [_make_breakout("AAPL", 80)]
    result = detect_confluence(signals)
    assert result["AAPL"].confluence_count == 1
    assert result["AAPL"].is_confluence is False
    assert result["AAPL"].confluence_bonus == 0.0


def test_confluence_two_models():
    """Two models flagging same ticker = confluence."""
    signals = [
        _make_breakout("AAPL", 80),
        _make_mean_rev("AAPL", 75),
    ]
    result = detect_confluence(signals)
    assert result["AAPL"].confluence_count == 2
    assert result["AAPL"].is_confluence is True
    assert result["AAPL"].confluence_bonus == 0.10  # 10% per additional model


def test_confluence_bonus_applied():
    """Confluence bonus should increase adjusted scores."""
    signals = [
        _make_breakout("AAPL", 80),
        _make_mean_rev("AAPL", 75),
        _make_breakout("MSFT", 70),
    ]
    confluence = detect_confluence(signals)
    candidates = [
        _make_ranked("AAPL", 80),
        _make_ranked("MSFT", 85),  # Higher base score
    ]
    result = apply_confluence_bonus(candidates, confluence)
    # AAPL gets 10% bonus: 80 * 1.10 = 88 → now beats MSFT's 85
    assert result[0].ticker == "AAPL"
    assert result[0].regime_adjusted_score == 88.0


# --- Signal Cooldown ---


def test_cooldown_no_recent_signals():
    """No recent signals → nothing suppressed."""
    signals = [_make_breakout("AAPL", 80), _make_breakout("MSFT", 70)]
    result = apply_cooldown(signals, recent_signals=[])
    assert len(result) == 2


def test_cooldown_suppresses_recent():
    """Tickers fired within cooldown window should be suppressed."""
    signals = [_make_breakout("AAPL", 80), _make_breakout("MSFT", 70)]
    recent = [
        {"ticker": "AAPL", "run_date": date.today() - timedelta(days=2)},
    ]
    result = apply_cooldown(signals, recent_signals=recent, cooldown_days=5)
    assert len(result) == 1
    assert result[0].ticker == "MSFT"


def test_cooldown_allows_expired():
    """Tickers that fired beyond the cooldown window should pass."""
    signals = [_make_breakout("AAPL", 80)]
    recent = [
        {"ticker": "AAPL", "run_date": date.today() - timedelta(days=10)},
    ]
    result = apply_cooldown(signals, recent_signals=recent, cooldown_days=5)
    assert len(result) == 1
    assert result[0].ticker == "AAPL"


def test_cooldown_string_dates():
    """Cooldown should handle string dates from DB."""
    signals = [_make_breakout("AAPL", 80)]
    recent = [
        {"ticker": "AAPL", "run_date": str(date.today() - timedelta(days=2))},
    ]
    result = apply_cooldown(signals, recent_signals=recent, cooldown_days=5)
    assert len(result) == 0


def test_rank_candidates_snapshots_signal_source():
    """RankedCandidate must capture the signal_source set on the signal at
    rank time. Regression test for the MR Manual Sleeve bug where the same
    MR signal object lived in both the MAS list and the sleeve list, and
    later setattr calls on the shared reference clobbered the source for
    one of the streams.
    """
    mr = _make_mean_rev("XOM", 80)
    # Tag as sleeve, rank — RankedCandidate should snapshot 'mr_manual_sleeve'.
    mr.signal_source = "mr_manual_sleeve"
    sleeve_ranked = rank_candidates([mr], Regime.BULL, {}, top_n=10)
    assert len(sleeve_ranked) == 1
    assert sleeve_ranked[0].signal_source == "mr_manual_sleeve"

    # Now overwrite the source on the underlying object (simulating MAS
    # annotation clobbering the sleeve source on a shared reference).
    mr.signal_source = "mas_official"
    # The previously-built RankedCandidate must NOT change — it has its own copy.
    assert sleeve_ranked[0].signal_source == "mr_manual_sleeve"
    # And re-ranking now snapshots the new value.
    mas_ranked = rank_candidates([mr], Regime.BULL, {}, top_n=10)
    assert mas_ranked[0].signal_source == "mas_official"


# ---------------------------------------------------------------------------
# RankedCandidate.persisted_features() — sniper composite score persistence
# ---------------------------------------------------------------------------

def _make_sniper_candidate(
    *,
    ticker: str = "MU",
    raw_score: float = 78.5,
    regime_adjusted_score: float = 65.42,
    components: dict | None = None,
    features: dict | None = None,
) -> RankedCandidate:
    return RankedCandidate(
        ticker=ticker,
        signal_model="sniper",
        raw_score=raw_score,
        regime_adjusted_score=regime_adjusted_score,
        direction="LONG",
        entry_price=100.0,
        stop_loss=92.0,
        target_1=110.0,
        target_2=115.0,
        holding_period=7,
        components=components if components is not None else {
            "bb_squeeze": 85.0,
            "vol_compression": 70.0,
            "relative_strength": 60.0,
            "trend_alignment": 75.0,
            "momentum_base": 50.0,
        },
        features=features if features is not None else {"close": 100.0, "atr_14": 3.5},
    )


def test_persisted_features_includes_score_metadata():
    c = _make_sniper_candidate()
    pf = c.persisted_features()
    assert pf["model_raw_score"] == 78.5
    assert pf["model_adjusted_score"] == 65.42
    assert pf["score_source"] == "score_sniper"
    assert pf["model_components"]["bb_squeeze"] == 85.0
    assert pf["model_components"]["vol_compression"] == 70.0


def test_persisted_features_preserves_original_feature_keys():
    c = _make_sniper_candidate(features={"close": 100.0, "rsi_14": 55.0, "atr_14": 3.5})
    pf = c.persisted_features()
    assert pf["close"] == 100.0
    assert pf["rsi_14"] == 55.0
    assert pf["atr_14"] == 3.5


def test_persisted_features_does_not_mutate_source():
    """Caller must be able to call persisted_features() repeatedly without
    accumulating side effects on the underlying features dict."""
    original_features = {"close": 100.0, "rsi_14": 55.0}
    c = _make_sniper_candidate(features=original_features)
    pf1 = c.persisted_features()
    pf2 = c.persisted_features()
    assert "model_raw_score" not in original_features
    assert "model_components" not in original_features
    assert pf1 == pf2


def test_persisted_features_components_isolated_from_source():
    """Mutating the returned components must not leak back into the candidate."""
    c = _make_sniper_candidate()
    pf = c.persisted_features()
    pf["model_components"]["bb_squeeze"] = -999
    assert c.components["bb_squeeze"] == 85.0


def test_persisted_features_handles_empty_inputs():
    c = RankedCandidate(
        ticker="X", signal_model="sniper", raw_score=70.0,
        regime_adjusted_score=70.0, direction="LONG",
        entry_price=10.0, stop_loss=9.0, target_1=12.0, target_2=None,
        holding_period=7, components={}, features={},
    )
    pf = c.persisted_features()
    assert pf["model_raw_score"] == 70.0
    assert pf["model_components"] == {}
    assert pf["score_source"] == "score_sniper"


def test_persisted_features_score_source_per_model():
    c_mr = _make_sniper_candidate()
    c_mr.signal_model = "mean_reversion"
    assert c_mr.persisted_features()["score_source"] == "score_mean_reversion"

    c_bo = _make_sniper_candidate()
    c_bo.signal_model = "breakout"
    assert c_bo.persisted_features()["score_source"] == "score_breakout"

    # Unknown model falls back to the model name itself
    c_x = _make_sniper_candidate()
    c_x.signal_model = "experimental_v2"
    assert c_x.persisted_features()["score_source"] == "experimental_v2"


def test_rank_candidates_round_trip_to_persisted_features():
    """End-to-end: a SniperSignal goes through rank_candidates and the result's
    persisted_features() carries the score components from the original signal."""
    sniper = SniperSignal(
        ticker="MU",
        score=78.5,
        direction="LONG",
        entry_price=100.0,
        stop_loss=92.0,
        target_1=110.0,
        target_2=115.0,
        holding_period=7,
        components={"bb_squeeze": 85, "vol_compression": 70, "relative_strength": 60,
                    "trend_alignment": 75, "momentum_base": 50},
    )
    ranked = rank_candidates([sniper], Regime.BULL, {"MU": {"close": 100.0}}, top_n=10)
    assert len(ranked) == 1
    pf = ranked[0].persisted_features()
    assert pf["model_raw_score"] == 78.5
    # bull regime sniper multiplier is 1.3 → adjusted ≈ 102.05
    assert pf["model_adjusted_score"] == ranked[0].regime_adjusted_score
    assert pf["score_source"] == "score_sniper"
    assert set(pf["model_components"].keys()) >= {
        "bb_squeeze", "vol_compression", "relative_strength",
        "trend_alignment", "momentum_base",
    }
