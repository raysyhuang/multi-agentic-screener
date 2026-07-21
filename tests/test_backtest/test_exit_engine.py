"""Characterization tests for the shared pure exit engine (walk_exit).

Hand-constructed ExitBar sequences pin down exact exit_index / exit_reason /
exit_price for every load-bearing branch. These lock the canonical LIVE
semantics (check_entry_bar=True, gap_through=True) plus the legacy-backtest
toggles (check_entry_bar=False, gap_through=False).
"""

from __future__ import annotations

from datetime import date, timedelta

from src.backtest.exit_engine import ExitBar, ExitParams, walk_exit

D0 = date(2026, 3, 10)


def _bars(rows: list[tuple[float, float, float, float]]) -> list[ExitBar]:
    """Build ExitBars from (open, high, low, close) tuples on consecutive days."""
    return [
        ExitBar(date=D0 + timedelta(days=i), open=o, high=h, low=lo, close=c)
        for i, (o, h, lo, c) in enumerate(rows)
    ]


def _params(**kw) -> ExitParams:
    defaults = dict(stop=95.0, target=110.0, max_hold=10, slippage=0.0)
    defaults.update(kw)
    return ExitParams(**defaults)


# ---------------------------------------------------------------------------
# Gap-through fills (open-based)
# ---------------------------------------------------------------------------

def test_gap_down_through_stop_exits_at_open():
    bars = _bars([
        (100.0, 101.0, 99.5, 100.5),   # entry bar
        (96.0, 97.0, 95.5, 96.5),      # gap down: open 96 < stop 98
    ])
    out = walk_exit(bars, 100.0, _params(stop=98.0))
    assert out.exited
    assert out.exit_index == 1
    assert out.exit_reason == "stop"
    assert out.exit_price == 96.0  # open, not stop 98


def test_gap_up_through_target_exits_at_open():
    bars = _bars([
        (100.0, 101.0, 99.5, 100.5),
        (105.0, 106.0, 104.5, 105.5),  # gap up: open 105 > target 103
    ])
    out = walk_exit(bars, 100.0, _params(target=103.0))
    assert out.exited
    assert out.exit_index == 1
    assert out.exit_reason == "target"
    assert out.exit_price == 105.0  # open, not target 103


# ---------------------------------------------------------------------------
# Intraday fills (high/low)
# ---------------------------------------------------------------------------

def test_intraday_stop_low_touches_close_recovers():
    bars = _bars([
        (100.0, 101.0, 99.5, 100.5),
        (99.5, 100.0, 97.5, 99.0),     # low 97.5 <= stop 98, close recovers to 99
    ])
    out = walk_exit(bars, 100.0, _params(stop=98.0))
    assert out.exit_index == 1
    assert out.exit_reason == "stop"
    assert out.exit_price == 98.0  # exact stop


def test_intraday_target_high_touches_close_fades():
    bars = _bars([
        (100.0, 101.0, 99.5, 100.5),
        (101.0, 103.5, 100.5, 101.0),  # high 103.5 >= target 103
    ])
    out = walk_exit(bars, 100.0, _params(target=103.0))
    assert out.exit_index == 1
    assert out.exit_reason == "target"
    assert out.exit_price == 103.0


def test_same_bar_stop_and_target_resolves_to_stop():
    bars = _bars([
        (100.0, 101.0, 99.5, 100.5),
        (99.0, 104.0, 97.0, 100.0),    # both stop (97<=98) and target (104>=103) reachable
    ])
    out = walk_exit(bars, 100.0, _params(stop=98.0, target=103.0))
    assert out.exit_reason == "stop"  # stop checked first


def test_same_bar_resolver_can_pick_target():
    """When minute data says the target came first, the resolver overrides the
    conservative stop-first assumption on the ambiguous bar."""
    from src.backtest.exit_engine import resolve_first_touch

    bars = _bars([
        (100.0, 101.0, 99.5, 100.5),
        (99.0, 104.0, 97.0, 100.0),    # ambiguous: low 97<=98 and high 104>=103
    ])
    # Minute path: rises to target BEFORE dipping to stop.
    minute = [(99.0, 100.0), (103.5, 104.0), (97.0, 98.0)]  # (low, high) per minute

    def resolver(d, stop, tgt):
        return resolve_first_touch(minute, stop, tgt)

    out = walk_exit(bars, 100.0, _params(stop=98.0, target=103.0, same_bar_resolver=resolver))
    assert out.exit_reason == "target"
    assert out.exit_price == 103.0


def test_same_bar_resolver_none_keeps_conservative_stop():
    """An unresolved verdict (None) falls back to the conservative stop."""
    bars = _bars([
        (100.0, 101.0, 99.5, 100.5),
        (99.0, 104.0, 97.0, 100.0),
    ])
    def resolver(d, stop, tgt):
        return None

    out = walk_exit(bars, 100.0, _params(stop=98.0, target=103.0, same_bar_resolver=resolver))
    assert out.exit_reason == "stop"


def test_resolve_first_touch_orders():
    from src.backtest.exit_engine import resolve_first_touch

    # stop touched first
    assert resolve_first_touch([(97.0, 99.0), (100.0, 104.0)], 98.0, 103.0) == "stop"
    # target touched first
    assert resolve_first_touch([(99.0, 100.0), (103.5, 104.0)], 98.0, 103.0) == "target"
    # neither touched
    assert resolve_first_touch([(99.0, 100.0), (99.5, 100.5)], 98.0, 103.0) is None
    # single minute straddles both → conservative stop
    assert resolve_first_touch([(97.0, 104.0)], 98.0, 103.0) == "stop"


# ---------------------------------------------------------------------------
# Trailing stop guards
# ---------------------------------------------------------------------------

def test_newly_armed_trail_does_not_exit_same_bar():
    # Entry bar rallies (arms trail) then dips below the freshly-computed trail
    # level — must NOT exit on the arming bar via open or low.
    bars = _bars([
        (100.0, 102.0, 100.3, 101.5),  # arms trail (2% gain), low 100.3 < trail 101.694
        (101.55, 101.80, 101.50, 101.60),  # ratchet path — trail fires here
    ])
    out = walk_exit(
        bars, 100.0,
        _params(stop=90.0, trail_activate_pct=0.5, trail_distance_pct=0.3),
    )
    assert out.exit_index == 1  # day 2, not day 1
    assert out.exit_reason == "trail_stop"
    assert out.exit_price > 101.0  # near yesterday's ratcheted trail ~101.694


def test_already_active_trail_ratchets_and_exits_same_bar():
    bars = _bars([
        (100.0, 102.0, 100.0, 101.5),  # day1: arms trail (newly armed, no enforce)
        (105.7, 106.0, 105.0, 105.5),  # day2: new high 106 ratchets trail to 105.682; low 105 exits
    ])
    out = walk_exit(
        bars, 100.0,
        _params(stop=90.0, target=130.0, trail_activate_pct=0.5, trail_distance_pct=0.3),
    )
    assert out.exit_index == 1
    assert out.exit_reason == "trail_stop"
    assert 105.4 < out.exit_price < 106.0  # ratcheted level ~105.68, not stale day-1 trail


# ---------------------------------------------------------------------------
# Two-leg partial + breakeven defer
# ---------------------------------------------------------------------------

def test_leg1_fill_defers_breakeven_same_bar():
    # Leg1 fills on day 1 (high >= 106); low 99.8 dips below entry but above base
    # stop. Same-bar breakeven must NOT fire. Position survives to expiry.
    bars = _bars([
        (100.0, 107.0, 99.8, 100.5),   # fills leg1, low below entry
        (100.5, 100.7, 100.3, 100.5),
    ])
    out = walk_exit(
        bars, 100.0,
        _params(stop=95.0, target=115.0, max_hold=2, partial_tp_target=106.0),
    )
    assert out.leg1_filled
    assert out.leg1_index == 0
    # No same-bar exit on the fill bar.
    assert out.exit_index == 1
    assert out.exit_reason == "expiry"


def test_leg1_prior_bar_enforces_breakeven_next_bar():
    bars = _bars([
        (100.0, 107.0, 100.5, 106.5),  # day1: fills leg1 cleanly (low above entry)
        (101.0, 101.5, 99.5, 100.0),   # day2: low 99.5 < breakeven 100 → stop
    ])
    out = walk_exit(
        bars, 100.0,
        _params(stop=95.0, target=115.0, max_hold=5, partial_tp_target=106.0),
    )
    assert out.leg1_filled
    assert out.exit_index == 1
    assert out.exit_reason == "stop"  # breakeven floor, trail inactive
    assert out.exit_price == 100.0  # entry-level breakeven


def test_leg1_prefilled_enforces_breakeven_from_first_bar():
    bars = _bars([
        (100.0, 101.0, 99.5, 100.5),   # day1: open 100 <= breakeven 100 -> immediate stop
    ])
    out = walk_exit(
        bars, 100.0,
        _params(stop=95.0, max_hold=5, partial_tp_target=106.0, leg1_prefilled=True),
    )
    assert out.exit_reason == "stop"
    assert out.leg1_index is None  # filled before this walk


# ---------------------------------------------------------------------------
# Sniper time stop
# ---------------------------------------------------------------------------

def test_time_stop_does_not_fire_on_entry_bar():
    # Entry bar closes red — must NOT time_stop (i==0). Next bars rally.
    bars = _bars([
        (100.0, 100.4, 99.5, 99.6),    # red entry bar
        (99.8, 100.5, 99.7, 100.4),    # green — above entry, no time_stop
        (100.5, 101.0, 100.3, 100.8),
    ])
    out = walk_exit(
        bars, 100.0,
        _params(stop=90.0, target=115.0, max_hold=7,
                time_stop_days=1, time_stop_eligible=True),
    )
    # Must not exit via time_stop on bar 0.
    assert not (out.exit_reason == "time_stop" and out.exit_index == 0)


def test_time_stop_fires_on_next_red_bar():
    bars = _bars([
        (100.0, 100.4, 99.5, 99.6),    # red entry bar (no fire, i==0)
        (99.6, 100.0, 99.0, 99.2),     # red, close <= entry → time_stop
    ])
    out = walk_exit(
        bars, 100.0,
        _params(stop=90.0, target=115.0, max_hold=7,
                time_stop_days=1, time_stop_eligible=True),
    )
    assert out.exit_index == 1
    assert out.exit_reason == "time_stop"
    assert out.exit_price == 99.2  # bar close


def test_time_stop_green_bar_does_not_fire():
    bars = _bars([
        (100.0, 100.4, 99.5, 99.6),    # red entry bar
        (99.6, 101.0, 99.4, 100.5),    # green: close 100.5 > entry → no time_stop
        (100.5, 101.0, 100.3, 100.8),
    ])
    out = walk_exit(
        bars, 100.0,
        _params(stop=90.0, target=115.0, max_hold=3,
                time_stop_days=1, time_stop_eligible=True),
    )
    # No time_stop anywhere; resolves via expiry at max_hold.
    assert out.exit_reason == "expiry"


# ---------------------------------------------------------------------------
# Expiry
# ---------------------------------------------------------------------------

def test_expiry_at_close():
    bars = _bars([
        (100.5, 101.0, 100.0, 100.5),
        (100.5, 101.0, 100.0, 100.5),
        (100.5, 101.0, 100.0, 100.7),  # day3 close = expiry
    ])
    out = walk_exit(bars, 100.0, _params(stop=90.0, target=115.0, max_hold=3))
    assert out.exit_index == 2
    assert out.exit_reason == "expiry"
    assert out.exit_price == 100.7


# ---------------------------------------------------------------------------
# Legacy toggles
# ---------------------------------------------------------------------------

def test_check_entry_bar_false_skips_entry_bar_exits():
    # Entry bar would stop out under live semantics (low 90 <= stop 95), but
    # legacy mode (check_entry_bar=False) never checks bar 0.
    bars = _bars([
        (100.0, 101.0, 90.0, 96.0),    # entry bar: low 90 < stop 95
        (96.0, 97.0, 96.0, 96.5),      # calm follow bar
    ])
    live = walk_exit(bars, 100.0, _params(stop=95.0, max_hold=2, check_entry_bar=True))
    legacy = walk_exit(bars, 100.0, _params(stop=95.0, max_hold=2, check_entry_bar=False))
    assert live.exit_index == 0 and live.exit_reason == "stop"
    assert legacy.exit_index == 1 and legacy.exit_reason == "expiry"


def test_gap_through_false_ignores_open_gap_fill():
    # Open gaps below stop. With gap_through the fill is at the (worse) open;
    # without it, only the intraday low <= stop triggers, filling at the stop.
    bars = _bars([
        (100.0, 101.0, 99.5, 100.5),
        (96.0, 97.0, 95.0, 96.5),      # open 96 < stop 98; low 95 <= stop too
    ])
    gap = walk_exit(bars, 100.0, _params(stop=98.0, max_hold=2, gap_through=True))
    nogap = walk_exit(bars, 100.0, _params(stop=98.0, max_hold=2, gap_through=False))
    assert gap.exit_price == 96.0   # fills at open
    assert nogap.exit_price == 98.0  # fills at stop, not the gapped-through open
    assert gap.exit_reason == nogap.exit_reason == "stop"
