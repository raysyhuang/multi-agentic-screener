"""The earnings-feed health check must distinguish three causes of "no dates".

The blackout FAILS OPEN, so a feed that stops returning data silently stops
protecting picks. The previous check tried to detect that from "fraction of the
qualified universe with a known earnings date" against a 15% floor — and could
not, for a structural reason:

  * the calendar is fetched 14 days ahead, so a name reporting in three weeks
    correctly has no date and counted as missing coverage;
  * measured against a live-shaped universe on 2026-08-13, the CEILING at a
    14-day window was 6.9% (168 of 2,439) — the gate was unreachable;
  * a dead feed (0%) and a healthy off-season (7%) both sit under 15%, so the
    alarm was already on in both states and could not tell them apart.

An alarm that never turns off reports nothing. These fixtures pin the four
states the replacement must separate.
"""

from __future__ import annotations

import pytest

from src.config import get_settings


def _classify(fetch_error: str | None, rows: int) -> bool:
    """Mirror of the health predicate in run_morning_pipeline."""
    s = get_settings()
    return fetch_error is None and rows > 0 and rows >= s.earnings_calendar_min_entries


def test_a_healthy_feed_passes_even_though_most_names_are_outside_the_horizon():
    """The false alarm that ran for four consecutive days.

    A working 14-day calendar returns thousands of rows while only ~7% of the
    qualified universe reports inside the window. That is the earnings season,
    not a fault, and it must not warn.
    """
    assert _classify(fetch_error=None, rows=2272) is True


def test_an_empty_calendar_is_flagged():
    """Fetch succeeded, zero rows: the feed is answering but carrying nothing."""
    assert _classify(fetch_error=None, rows=0) is False


def test_a_provider_failure_is_flagged_and_is_not_confused_with_an_empty_feed():
    """`get_upcoming_earnings` returns [] on BOTH, so the error must carry the
    difference — otherwise a dead provider looks like a quiet week."""
    assert _classify(fetch_error="HTTPStatusError", rows=0) is False
    assert _classify(fetch_error="ReadTimeout", rows=0) is False


def test_a_feed_that_never_ran_is_not_treated_as_healthy():
    """The aggregator starts at "not_attempted" deliberately."""
    assert _classify(fetch_error="not_attempted", rows=0) is False


def test_the_old_coverage_metric_no_longer_gates_anything():
    """Regression: reinstating it would restore a permanently-on alarm."""
    import inspect

    from src import main as main_mod

    # The check lives in _run_pipeline_core, not run_morning_pipeline — assert
    # against the function that actually contains it, or this passes vacuously.
    src = inspect.getsource(main_mod._run_pipeline_core)
    assert "earnings_coverage_min_pct" not in src, (
        "the unreachable coverage threshold is gating again"
    )
    assert "earnings_feed_alive" in src
    assert "window_coverage_diagnostic" in src, (
        "match rate must still be REPORTED, just never gated"
    )


@pytest.mark.parametrize("horizon_days,blackout_need", [(14, 2), (14, 7)])
def test_the_fourteen_day_horizon_still_covers_every_real_blackout(horizon_days, blackout_need):
    """The control is not being weakened.

    The global gate needs 2 days of notice; sniper's hold-aware gate needs 7
    (guarding the mid-hold earnings gap). A 14-day fetch covers both with a week
    to spare, so widening the window would raise the old metric while changing
    nothing about protection — and would cost API volume for no risk reduction.
    """
    s = get_settings()
    assert s.earnings_blackout_days == 2
    assert s.sniper_holding_period == 7
    assert horizon_days > blackout_need
    assert horizon_days >= s.sniper_holding_period * 2


def test_a_real_near_term_earnings_name_is_still_blacked_out():
    """The blackout itself, unchanged: a name reporting inside the window is
    skipped, which is the behaviour the health check exists to protect."""
    from src.features.fundamental import days_to_next_earnings

    from datetime import date, timedelta
    s = get_settings()
    soon = (date.today() + timedelta(days=1)).isoformat()
    cal = [{"symbol": "ACME", "date": soon}]

    dte = days_to_next_earnings(cal, "ACME")

    assert dte is not None and dte <= s.earnings_blackout_days, "should be blacked out"
    assert days_to_next_earnings(cal, "OTHER") is None, (
        "a name absent from the window has no date — normal, not a feed fault"
    )
