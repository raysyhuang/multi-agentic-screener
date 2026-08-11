"""Why a candidate was not picked must be recorded, not inferred.

The sniper concurrency cap was a bare `continue` in `_build_quant_only_result`,
so a candidate dropped because the book was full was indistinguishable in stored
data from one ranked below the daily quota. The question the next few weeks of
observation exist to answer —

    Of candidates that passed eligibility and ranking, how many were denied
    solely because every sniper slot was occupied?

— could not be answered at all. These tests pin the distinctions the ledger has
to keep: stage separate from reason, and slot state read at the moment of the
decision rather than reconstructed afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.main import _build_quant_only_result


@dataclass
class _Candidate:
    """Minimal stand-in for a ranked candidate."""
    ticker: str
    signal_model: str
    raw_score: float = 90.0
    regime_adjusted_score: float = 90.0
    entry_price: float = 100.0
    stop_loss: float = 95.0
    target_1: float = 110.0
    target_2: float | None = None
    holding_period: int = 7
    suppressed_by_cross_model_ranking: bool = False
    components: dict = field(default_factory=lambda: {"score": 1.0})
    features: dict = field(default_factory=lambda: {"vol_sma_20": 1_000_000})
    direction: str = "LONG"

    def persisted_features(self) -> dict:
        """A method on the real candidate, not an attribute."""
        return dict(self.features)


def _regime() -> dict:
    return {"regime": "bull", "confidence": 0.8}


def test_capacity_censoring_is_distinguishable_from_the_quota() -> None:
    """The distinction the whole ledger exists for.

    Three sniper candidates, no free slots. All three are denied by capacity —
    none of them because the daily quota was exhausted, since nothing was
    picked at all.
    """
    ranked = [_Candidate(t, "sniper") for t in ("AAA", "BBB", "CCC")]
    ledger: dict = {}

    result = _build_quant_only_result(
        ranked, _regime(), max_picks=2, max_sniper=0, ledger=ledger
    )

    assert result.approved == []
    for ticker in ("AAA", "BBB", "CCC"):
        assert ledger[ticker]["rejection_stage"] == "capacity"
        assert ledger[ticker]["rejection_reason"] == "capacity_censored"
        assert ledger[ticker]["selection_stage_reached"] is True
        assert ledger[ticker]["selected"] is False


def test_below_quota_is_not_labelled_capacity() -> None:
    """A full quota is not a full book — mislabelling would inflate the answer."""
    ranked = [_Candidate(t, "mean_reversion") for t in ("AAA", "BBB", "CCC")]
    ledger: dict = {}

    result = _build_quant_only_result(
        ranked, _regime(), max_picks=2, max_sniper=None, ledger=ledger
    )

    assert len(result.approved) == 2
    assert ledger["AAA"]["selected"] is True
    assert ledger["BBB"]["selected"] is True
    assert ledger["CCC"]["rejection_stage"] == "quota"
    assert ledger["CCC"]["rejection_reason"] == "below_quota"


def test_both_causes_are_separable_in_one_run() -> None:
    """The realistic shape: capacity bites first, then the quota fills."""
    ranked = [
        _Candidate("SNIP1", "sniper"),
        _Candidate("MR1", "mean_reversion"),
        _Candidate("MR2", "mean_reversion"),
        _Candidate("MR3", "mean_reversion"),
    ]
    ledger: dict = {}

    _build_quant_only_result(
        ranked, _regime(), max_picks=2, max_sniper=0, ledger=ledger
    )

    assert ledger["SNIP1"]["rejection_reason"] == "capacity_censored"
    assert ledger["MR1"]["selected"] is True
    assert ledger["MR2"]["selected"] is True
    assert ledger["MR3"]["rejection_reason"] == "below_quota"


def test_rank_is_recorded_so_marginality_is_visible() -> None:
    """A censored top-ranked candidate is a different fact from a censored 9th."""
    ranked = [_Candidate(f"T{i}", "sniper") for i in range(1, 6)]
    ledger: dict = {}

    _build_quant_only_result(
        ranked, _regime(), max_picks=2, max_sniper=0, ledger=ledger
    )

    assert ledger["T1"]["strategy_rank"] == 1
    assert ledger["T5"]["strategy_rank"] == 5


def test_the_ledger_is_optional() -> None:
    """Existing callers pass no ledger and must be unaffected."""
    ranked = [_Candidate("AAA", "sniper")]

    result = _build_quant_only_result(ranked, _regime(), max_picks=2, max_sniper=1)

    assert len(result.approved) == 1


def test_selection_behaviour_is_unchanged_by_recording() -> None:
    """Instrumentation must not alter which candidates are picked.

    This PR is explicitly not allowed to change strategy thresholds, ranking,
    position limits or execution — so the picks must be identical with and
    without a ledger attached.
    """
    def _ranked():
        return [
            _Candidate("SNIP1", "sniper"),
            _Candidate("SNIP2", "sniper"),
            _Candidate("MR1", "mean_reversion"),
        ]

    without = _build_quant_only_result(_ranked(), _regime(), max_picks=2, max_sniper=1)
    with_ledger = _build_quant_only_result(
        _ranked(), _regime(), max_picks=2, max_sniper=1, ledger={}
    )

    assert [p.ticker for p in without.approved] == [p.ticker for p in with_ledger.approved]


def test_correlation_drops_record_rank_and_counterpart() -> None:
    """A correlation row with NULL rank and NULL counterpart answers nothing.

    Both facts exist inside the filter at decision time and were previously
    only logged. A drop at rank 2 is a materially different fact from one at
    rank 9, and "correlated with what?" is the whole content of the decision.
    """
    import numpy as np
    import pandas as pd

    from src.signals.ranker import filter_correlated_picks

    # Two tickers that move together, one that does not.
    rng = np.random.default_rng(7)
    base = rng.normal(0, 0.02, 60)
    price_data = {
        "LEAD": pd.DataFrame({"close": 100 * np.cumprod(1 + base)}),
        "TWIN": pd.DataFrame({"close": 100 * np.cumprod(1 + base)}),  # identical
        "INDY": pd.DataFrame({"close": 100 * np.cumprod(1 + rng.normal(0, 0.02, 60))}),
    }
    ranked = [
        _Candidate("LEAD", "sniper"),
        _Candidate("TWIN", "sniper"),
        _Candidate("INDY", "sniper"),
    ]

    rejections: dict = {}
    kept = filter_correlated_picks(ranked, price_data, rejections=rejections)

    assert [c.ticker for c in kept] == ["LEAD", "INDY"], "TWIN should be dropped"
    assert "TWIN" in rejections, "the drop must be reported, not only logged"
    assert rejections["TWIN"]["correlated_with"] == "LEAD"
    assert rejections["TWIN"]["pre_filter_rank"] == 2
    assert rejections["TWIN"]["correlation"] > 0.9


def test_the_rejections_out_parameter_is_optional() -> None:
    """Existing callers pass none and must behave identically."""
    import numpy as np
    import pandas as pd

    from src.signals.ranker import filter_correlated_picks

    rng = np.random.default_rng(3)
    price_data = {
        t: pd.DataFrame({"close": 100 * np.cumprod(1 + rng.normal(0, 0.02, 60))})
        for t in ("AAA", "BBB")
    }
    ranked = [_Candidate("AAA", "sniper"), _Candidate("BBB", "sniper")]

    assert filter_correlated_picks(ranked, price_data) is not None
