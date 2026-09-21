"""Auditable lower bounds for model-family search multiplicity.

These are deliberately conservative floors, not claims that every historical
trial has been recovered. A floor still produces a stricter (and more honest)
deflated Sharpe than the former production constant of one.
"""

from __future__ import annotations


# Sources:
# - mean_reversion: the checked-in 3*3*3*3*2 grid in signal_backtest.py.
# - sniper: the July trail/stop sweep reports 19 variants in the registry.
# - pead: the author's registered E1/quality search count used by the validation
#   card in scripts/pead_neglected_beat_valcard.py.
VARIANTS_TESTED_FLOOR: dict[str, int] = {
    "mean_reversion": 162,
    "sniper": 19,
    "pead": 12,
}


def variants_tested_for(signal_model: str) -> int:
    """Return the registered family floor; unknown models fail conservatively."""
    return VARIANTS_TESTED_FLOOR.get(signal_model, 2)
