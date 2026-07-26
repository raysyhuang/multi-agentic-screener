"""PEAD paper variant routing: decelerating-growth beats get their own
signal_source ("pead_neglected") so the validated stronger cohort accrues a
separate forward track record; the rest stay "pead_paper". Both are quarantined."""
from __future__ import annotations

from types import SimpleNamespace

from src.main import _pead_variant_source


def _sig(neglected):
    return SimpleNamespace(components={"neglected_beat": neglected, "eps_surprise_pct": 12.0})


def test_neglected_beat_routes_to_variant_source():
    assert _pead_variant_source(_sig(True)) == "pead_neglected"


def test_non_neglected_stays_base_paper():
    assert _pead_variant_source(_sig(False)) == "pead_paper"


def test_missing_tag_defaults_to_base_paper():
    # No components / no tag → base paper (fail-safe; never routes to the variant).
    assert _pead_variant_source(SimpleNamespace(components={})) == "pead_paper"
    assert _pead_variant_source(SimpleNamespace()) == "pead_paper"
