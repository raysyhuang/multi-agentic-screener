"""Tests for the portfolio equity engine (src/backtest/portfolio.py)."""

from __future__ import annotations

from datetime import date

import pytest

from src.backtest.portfolio import BookTrade, _sharpe, exit_day_overlap, simulate_book


def _t(entry, exit_, pnl):
    return BookTrade(entry=date(2026, 1, entry), exit=date(2026, 1, exit_), pnl_pct=pnl)


def test_empty_and_single_trade():
    empty = simulate_book([])
    assert empty["taken"] == 0 and empty["total_return_pct"] == 0.0 and empty["equity_curve"] == []

    # One +10% trade at equal-slots/10 deploys 1/10 of capital → +1.0% on the account.
    r = simulate_book([_t(1, 5, 10.0)], max_concurrent=10, start_capital=100_000.0)
    assert r["taken"] == 1 and r["skipped"] == 0
    assert r["total_return_pct"] == round((1.01 - 1.0) * 100, 10) or abs(r["total_return_pct"] - 1.0) < 1e-9
    assert r["max_drawdown_pct"] == 0.0  # monotonic up


def test_concurrency_cap_skips_when_slots_full():
    # 3 trades all open on day 1, cap of 2 → the 3rd is skipped (no free slot).
    trades = [_t(1, 9, 5.0), _t(1, 9, 5.0), _t(1, 9, 5.0)]
    r = simulate_book(trades, max_concurrent=2, start_capital=100_000.0)
    assert r["taken"] == 2 and r["skipped"] == 1
    assert r["peak_concurrent"] == 2


def test_exits_free_capital_before_same_day_entries():
    # Trade A exits on day 5; trade B enters on day 5. With cap=1, B must still
    # be taken because A's exit is processed before B's entry same day.
    trades = [_t(1, 5, 4.0), _t(5, 9, 4.0)]
    r = simulate_book(trades, max_concurrent=1, start_capital=100_000.0)
    assert r["taken"] == 2 and r["skipped"] == 0


def test_same_day_round_trip_enters_before_its_own_exit():
    # A trade that enters and stops on day 5 must contribute its loss. The
    # generic "exits before entries" ordering used to ignore the exit, leave
    # the position at cost, and silently report zero return.
    r = simulate_book([_t(5, 5, -10.0)], max_concurrent=10, start_capital=100_000.0)

    assert r["taken"] == 1 and r["skipped"] == 0
    assert abs(r["total_return_pct"] - (-1.0)) < 1e-9
    assert len(r["equity_curve"]) == 2  # seed + exactly one accounted exit


def test_sharpe_uses_the_last_equity_point_as_the_same_day_close():
    # Sorting full (date, equity) tuples reorders same-day points by value and
    # can replace the actual close with that day's high-water mark.
    curve = [
        (date(2026, 1, 1), 100_000.0),
        (date(2026, 1, 2), 102_000.0),
        (date(2026, 1, 2), 99_000.0),
        (date(2026, 1, 3), 100_000.0),
    ]
    daily_closes = [curve[0], curve[2], curve[3]]

    assert _sharpe(curve) == _sharpe(daily_closes)


def test_binding_cap_preserves_input_order_for_same_day_entries():
    # Exit date is future information and must not decide which same-day signal
    # gets the only slot. The caller's frozen order owns that decision.
    trades = [_t(1, 9, 10.0), _t(1, 1, -10.0)]

    r = simulate_book(trades, max_concurrent=1, start_capital=100_000.0)

    assert r["taken"] == 1 and r["skipped"] == 1
    assert abs(r["total_return_pct"] - 10.0) < 1e-9


def test_exit_before_entry_is_rejected_before_replay():
    with pytest.raises(ValueError, match="exit before entry"):
        simulate_book([_t(9, 1, -10.0)])


def test_repeated_trade_object_is_accounted_as_two_input_rows():
    trade = _t(1, 5, 10.0)

    r = simulate_book([trade, trade], max_concurrent=2, start_capital=100_000.0)

    assert r["taken"] == 2 and r["skipped"] == 0
    assert abs(r["total_return_pct"] - 10.0) < 1e-9
    assert len(r["equity_curve"]) == 3


def test_prior_exit_frees_slot_before_same_day_round_trip_and_new_entry():
    trades = [_t(1, 5, 10.0), _t(5, 5, -10.0), _t(5, 9, 5.0)]

    r = simulate_book(trades, max_concurrent=1, start_capital=100_000.0)

    assert r["taken"] == 2 and r["skipped"] == 1
    assert abs(r["total_return_pct"] - (-1.0)) < 1e-9
    assert len(r["equity_curve"]) == 3


def test_drawdown_and_sharpe_present_on_a_losing_then_winning_path():
    # A loss then a recovery → a real drawdown and a computable Sharpe.
    trades = [_t(1, 5, -10.0), _t(6, 10, -10.0), _t(11, 20, 40.0)]
    r = simulate_book(trades, max_concurrent=1, start_capital=100_000.0)
    assert r["max_drawdown_pct"] > 0.0
    assert r["sharpe"] is not None  # enough spread + span to annualise
    assert len(r["equity_curve"]) == 4  # seed + 3 exits


def test_exit_day_overlap():
    streams = {
        "sniper": [{"exit_date": "2026-01-05"}, {"exit_date": "2026-01-06"}, {"exit_date": None}],
        "mr": [{"exit_date": "2026-01-06"}, {"exit_date": "2026-01-09"}],
    }
    ov = exit_day_overlap(streams)
    assert ov["sniper_exit_days"] == 2  # None dropped
    assert ov["mr_exit_days"] == 2
    assert ov["shared_days"] == 1  # only 2026-01-06
