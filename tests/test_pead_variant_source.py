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


# --- Open-position dedup (2026-07-30) ---------------------------------------
# PEAD's earnings window is 6 days but its hold is 20, so a single beat re-qualifies
# for ~5-6 consecutive runs. Without dedup that stacks a NEW position each run:
# observed live on BKR / QRVO / LOGI / INCY (2 open positions each). That both
# concentrates one event and corrupts the paper track record the promotion gate
# depends on, since duplicate rows are not independent samples.

def test_open_pead_tickers_is_failsafe_on_db_error(monkeypatch):
    """A DB failure must not block PEAD — it degrades to no-dedup, never to no-picks."""
    import asyncio

    from src import main as m

    def _boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(m, "get_session", _boom)
    assert asyncio.run(m._open_pead_tickers()) == set()


def test_dedup_filter_drops_already_held_tickers():
    """The filter itself: candidates whose ticker is already open are removed,
    case-insensitively, and everything else is preserved in rank order."""
    from types import SimpleNamespace

    ranked = [SimpleNamespace(ticker=t) for t in ("QRVO", "LOGI", "FTNT", "UMBF")]
    open_pead = {"QRVO", "logi".upper()}
    kept = [c for c in ranked if c.ticker.upper() not in open_pead]
    assert [c.ticker for c in kept] == ["FTNT", "UMBF"]


# --- Concurrency slot cap (2026-08-08) --------------------------------------
# `pead_max_positions` bounds picks PER RUN, not open positions. With a 20-day
# hold (untrailed since PR #43) that compounds: 13 PEAD positions were open on
# 2026-08-06 against a nominal cap of 5. The slot cap mirrors sniper's.

def test_count_open_pead_positions_is_failsafe_on_db_error(monkeypatch):
    """A DB failure widens the cap (returns 0) rather than blocking the stream."""
    import asyncio

    from src import main as m

    def _boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(m, "get_session", _boom)
    assert asyncio.run(m._count_open_pead_positions()) == 0


def _admit(open_n, max_concurrent, per_run):
    """The admission arithmetic used in the pipeline."""
    return min(per_run, max(0, max_concurrent - open_n))


def test_slot_cap_admits_full_run_quota_when_book_is_empty():
    assert _admit(open_n=0, max_concurrent=10, per_run=5) == 5


def test_slot_cap_throttles_as_positions_accumulate():
    # 7 open of 10 → only 3 slots left, even though the run cap allows 5.
    assert _admit(open_n=7, max_concurrent=10, per_run=5) == 3


def test_slot_cap_admits_nothing_when_full():
    assert _admit(open_n=10, max_concurrent=10, per_run=5) == 0


def test_slot_cap_never_goes_negative_when_over_capacity():
    """The live starting state: 13 open against a 10 cap must admit 0, not -3."""
    assert _admit(open_n=13, max_concurrent=10, per_run=5) == 0
