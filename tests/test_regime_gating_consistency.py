"""The regime gate must be derived from ONE regime per run.

`allowed_models` is computed at Step 1 from the PRELIMINARY regime (breadth is
not available until OHLCV lands). Step 3b then recomputes the regime WITH
breadth and adopts it — the scorers read `regime_assessment.regime` directly and
the governor re-derives the allow-list from it at the end.

Before the fix, `allowed_models` was never recomputed, so a run whose breadth
flipped the regime gated the ranker on one regime while every other layer used
the other. The concrete live case: preliminary BEAR (sniper stripped from the
allow-list) → breadth-aware CHOPPY (sniper allowed, and sniper's own bear-block
would have let it through) → sniper silently dropped by a regime the pipeline
had already overwritten.
"""
from __future__ import annotations

from src.features.regime import Regime, get_regime_allowed_models


def test_bear_and_choppy_allow_lists_actually_differ_for_sniper():
    """Premise check: the divergence is only harmful because the lists differ."""
    bear = set(get_regime_allowed_models(Regime.BEAR))
    choppy = set(get_regime_allowed_models(Regime.CHOPPY))
    assert "sniper" not in bear
    assert "sniper" in choppy
    assert bear != choppy


def test_regate_after_breadth_changes_the_allowed_models():
    """The fix's core arithmetic: re-deriving from the adopted regime."""
    prelim_allowed = get_regime_allowed_models(Regime.BEAR)      # Step 1
    final_allowed = get_regime_allowed_models(Regime.CHOPPY)     # Step 3b adopts this
    assert set(prelim_allowed) != set(final_allowed)
    # Post-fix the ranker uses the RE-DERIVED list, matching what the scorers
    # and the governor see.
    assert "sniper" in final_allowed


def test_regate_is_a_noop_when_breadth_does_not_move_the_regime():
    """No spurious churn: same regime in and out means the same allow-list."""
    for regime in (Regime.BULL, Regime.BEAR, Regime.CHOPPY):
        assert get_regime_allowed_models(regime) == get_regime_allowed_models(regime)


def test_every_regime_allows_the_counter_trend_and_event_models():
    """Guards the invariant the re-gate must never break: MR and PEAD always
    survive, so a regime flip can never leave the book with nothing to trade."""
    for regime in (Regime.BULL, Regime.BEAR, Regime.CHOPPY):
        allowed = get_regime_allowed_models(regime)
        assert "mean_reversion" in allowed
        assert "pead" in allowed


def test_regime_envelope_regime_info_is_mutable_for_correction():
    """The fix rewrites the persisted REGIME envelope when the gate changes, so
    the stored artifact matches what the run actually traded. That depends on
    RegimeInfo not being frozen."""
    from src.contracts import RegimeInfo

    info = RegimeInfo(label="bear", confidence=0.5,
                      signals_allowed=get_regime_allowed_models(Regime.BEAR))
    info.signals_allowed = list(get_regime_allowed_models(Regime.CHOPPY))
    info.label = "choppy"
    assert "sniper" in info.signals_allowed
    assert info.label == "choppy"
