from src.research.variant_counts import VARIANTS_TESTED_FLOOR, variants_tested_for


def test_live_model_families_never_disable_multiple_testing_penalty():
    for model in ("mean_reversion", "sniper", "pead"):
        assert variants_tested_for(model) > 1


def test_counts_match_checked_in_search_records():
    assert VARIANTS_TESTED_FLOOR == {
        "mean_reversion": 162,
        "sniper": 19,
        "pead": 12,
    }


def test_unknown_model_gets_conservative_nontrivial_floor():
    assert variants_tested_for("future_model") == 2
