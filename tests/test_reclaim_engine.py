"""RECLAIM engine — the pure rules shared by the replay backtest and any collector.

Covers RECLAIM_MAS_SPEC_v1.1 §8.9 tests 1-19 and 26-27 that do not need a
database: state machine, actual-open decision, earnings gate, exits, controls,
censoring, flags, plus the QNT parity and four-name golden fixtures. No network.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.signals import reclaim as rc
from src.utils.trading_calendar import is_trading_day, trading_sessions

FIXTURES = Path(__file__).parent / "fixtures"


# ── helpers ──────────────────────────────────────────────────────────────────

def _sessions(n: int, start: date = date(2026, 1, 2)) -> list[date]:
    out = trading_sessions(start, start + timedelta(days=n * 2 + 10))
    return out[:n]


def _bars(close, high=None, low=None, open_=None) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    high = close + 0.5 if high is None else np.asarray(high, dtype=float)
    low = close - 0.5 if low is None else np.asarray(low, dtype=float)
    open_ = close if open_ is None else np.asarray(open_, dtype=float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": np.full(len(close), 1e6), "vwap": np.full(len(close), np.nan)})


def _episode(start: int, floor: float, sessions, lane=rc.DRIFT, sym="TEST") -> rc.Episode:
    return rc.Episode(lane=lane, symbol=sym, start=start, start_date=sessions[start],
                      floor=floor, floor_type=rc.FLOOR_TYPE[lane],
                      episode_id=rc.episode_id(lane, sym, sessions[start]))


def _run(close, start, floor, *, high=None, low=None, inclusions=None, last=None):
    sessions = _sessions(len(close))
    bars = _bars(close, high, low)
    ep = _episode(start, floor, sessions)
    sma50 = rc.sma(bars["close"].to_numpy(dtype=float), 50)
    return rc.run_episode(ep, bars, sma50, inclusions or {start}, last if last is not None else len(close) - 1,
                          sessions), sma50


def _events(res):
    return [(i, e) for i, e, _ in res.events]


def _base(n=120, level=100.0):
    """Flat at `level` so SMA50 == level once defined."""
    return np.full(n, level)


# ── 1. SMA50 window and missing-bar reset ────────────────────────────────────

def test_sma50_needs_fifty_consecutive_closes_and_a_gap_resets_it():
    close = np.arange(1.0, 131.0)
    s = rc.sma(close, 50)
    assert np.isnan(s[48]) and s[49] == pytest.approx(np.mean(close[:50]))
    close[70] = np.nan
    s = rc.sma(close, 50)
    assert np.isnan(s[70]) and np.isnan(s[119])       # every window that holds the gap
    assert s[120] == pytest.approx(np.mean(close[71:121]))


# ── 2. Floors ────────────────────────────────────────────────────────────────

def test_drift_floor_is_trailing_20_low_including_start():
    low = np.arange(100.0, 160.0)
    low[40] = 50.0          # the start bar itself is the minimum
    assert rc.drift_floor(low, 40) == 50.0
    assert rc.drift_floor(low, 59) == 50.0      # 40 is the first bar of the 20-session window
    assert rc.drift_floor(low[:60], 39) == 120.0  # window 20..39 excludes the dip at 40


def test_drift_floor_unavailable_with_19_bars():
    low = np.arange(100.0, 160.0)
    assert rc.drift_floor(low, 18) is None          # fewer than 20 sessions exist
    low[30] = np.nan
    assert rc.drift_floor(low, 40) is None          # 19 non-null bars in the window


def test_range_box_low_excludes_t():
    sessions = _sessions(80)
    close = np.full(80, 20.0)
    low = close - 1.0
    low[79] = 5.0                                    # t's own low must not enter box_low
    s = rc.range_tech_native_series(_bars(close, close + 1.0, low), 5e9)
    assert s["box_low"].iloc[79] == 19.0
    assert len(sessions) == 80


def test_floor_unavailable_episode_is_ledgered_not_evaluated():
    res, _ = _run(_base(), 60, float("nan"))
    assert res.status == "FLOOR_UNAVAILABLE" and res.trigger is None


# ── 3. Break / recover / SUPPORT_FAILED ──────────────────────────────────────

def test_break_recovers_at_t_plus_1_and_t_plus_2_and_blocked_sessions_recognise_nothing():
    close = _base()
    close[62] = 95.0   # break vs floor 98
    close[63] = 96.0   # still below (t+1)
    close[64] = 101.0  # recover at t+2, and also a would-be reclaim — must be ignored
    res, _ = _run(close, 60, 98.0, last=70)
    ev = _events(res)
    assert (62, "BREAK") in ev and (64, "RECOVER") in ev
    assert not any(e == "RECLAIM" and i in (62, 63, 64) for i, e in ev)
    assert 62 not in res.risk_states and 64 not in res.risk_states


def test_support_failed_when_no_recovery_within_two_sessions():
    close = _base()
    close[62] = 95.0
    close[63] = 96.0
    close[64] = 97.0
    res, _ = _run(close, 60, 98.0)
    assert res.status == "SUPPORT_FAILED" and res.end == 64


def test_a_later_break_opens_a_new_window_against_the_same_floor():
    close = _base()
    close[62], close[63] = 95.0, 99.0      # break, recover t+1
    close[70], close[71], close[72] = 97.0, 97.5, 99.0   # break again, recover at t+2
    res, _ = _run(close, 60, 98.0, last=80)
    ev = _events(res)
    assert (70, "BREAK") in ev and (72, "RECOVER") in ev and res.status == "CENSORED_ALIVE"


# ── 4. Reclaim ───────────────────────────────────────────────────────────────

def _reclaim_series():
    """Below SMA50 then a cross up at 70."""
    close = _base(140)
    close[55:70] = 97.0
    close[70] = 103.0
    return close


def test_reclaim_requires_prior_close_at_or_below_sma50():
    close = _reclaim_series()
    res, sma50 = _run(close, 56, 90.0, last=72)
    assert (70, "RECLAIM") in _events(res)
    assert close[69] <= sma50[69] and close[70] > sma50[70]


def test_earliest_armed_reclaim_wins():
    close = _reclaim_series()
    close[71] = 96.0       # dips below
    close[72] = 104.0      # second cross while armed — ignored
    res, _ = _run(close, 56, 90.0, last=73)
    reclaims = [i for i, e in _events(res) if e == "RECLAIM"]
    assert reclaims == [70]


# ── 5. Retest ────────────────────────────────────────────────────────────────

def _with_retest(at: int):
    close = _reclaim_series()
    close[71:90] = 106.0
    sessions = len(close)
    high = close + 0.5
    low = close - 0.5
    sma50 = rc.sma(close, 50)
    low[at] = sma50[at] * 1.005          # within 1.01 band
    close[at] = sma50[at] + 0.2          # close >= SMA50
    high[at] = close[at] + 0.3
    low[at] = min(low[at], close[at])
    return close, high, low, sessions


@pytest.mark.parametrize("offset", [1, 10])
def test_retest_at_r_plus_1_and_r_plus_10(offset):
    close, high, low, _ = _with_retest(70 + offset)
    res, _ = _run(close, 56, 90.0, high=high, low=low, last=70 + offset)
    assert (70 + offset, "RETEST") in _events(res)


def test_retest_timeout_at_r_plus_10():
    close = _reclaim_series()
    close[71:90] = 110.0    # far above SMA50: no retest band touch
    res, _ = _run(close, 56, 90.0, last=85)
    assert (80, "RETEST_TIMEOUT") in _events(res)


# ── 6. Confirm ───────────────────────────────────────────────────────────────

def test_confirm_q_plus_1_to_q_plus_5_and_never_on_the_retest_bar():
    close, high, low, _ = _with_retest(72)
    # retest bar closes ABOVE its own high? impossible; ensure no same-bar trigger
    close[73:78] = close[72] - 0.1         # below high_q for 5 sessions
    high[73:78] = close[73:78] + 0.05
    res, _ = _run(close, 56, 90.0, high=high, low=low, last=80)
    ev = _events(res)
    assert (72, "RETEST") in ev and (77, "CONFIRM_TIMEOUT") in ev and res.trigger is None


def test_confirm_triggers_when_close_exceeds_retest_high():
    close, high, low, _ = _with_retest(72)
    close[73] = close[72] - 0.1
    high[73] = close[73] + 0.05
    close[74] = high[72] + 0.5
    high[74] = close[74] + 0.1
    res, _ = _run(close, 56, 90.0, high=high, low=low, last=80)
    assert res.status == "TRIGGERED" and res.trigger.k == 74 and res.trigger.q == 72


# ── 7. setup_low ─────────────────────────────────────────────────────────────

def test_setup_low_is_min_low_from_reclaim_through_trigger():
    close, high, low, _ = _with_retest(72)
    # 73 closes at 106 > high_q, so it is the trigger bar; its own low counts.
    low[73] = 95.5
    res, _ = _run(close, 56, 90.0, high=high, low=low, last=80)
    assert res.trigger.k == 73
    assert res.trigger.setup_low == pytest.approx(min(low[70:74]))
    assert res.trigger.setup_low == 95.5


# ── 8. Expiry vs cap; one trigger per episode ────────────────────────────────

def test_expiry_30_after_last_inclusion():
    res, _ = _run(_base(200), 60, 90.0, inclusions={60}, last=199)
    assert res.status == "EXPIRED_30" and res.end == 90


def test_expiry_cap_at_start_plus_60_despite_later_inclusions():
    res, _ = _run(_base(200), 60, 90.0, inclusions={60, 85, 110}, last=199)
    assert res.status == "EXPIRED_CAP" and res.end == 120


def test_one_trigger_consumes_the_episode_and_a_new_one_needs_a_20_session_gap():
    close, high, low, _ = _with_retest(72)
    sessions = _sessions(len(close))
    bars = _bars(close, high, low)
    eps = rc.build_episodes(rc.DRIFT, "TEST", [56, 60, 80, 95, 120], bars, sessions, len(close) - 1)
    assert eps[0].status == "TRIGGERED" and eps[0].end == 73
    # 80 is after the end but only 19 sessions from 60; 95 is 14 from 80; 120 is >= 20 from 95
    assert [e.episode.start for e in eps] == [56, 120]


# ── 9. Missing XNYS bar ──────────────────────────────────────────────────────

def test_missing_bar_advances_timers_and_recognises_nothing():
    close = _reclaim_series()
    close[71:90] = 110.0
    close[75] = np.nan
    res, _ = _run(close, 56, 90.0, last=85)
    ev = _events(res)
    assert (75, "MISSING_BAR") in ev
    assert 75 not in res.risk_states
    assert (80, "RETEST_TIMEOUT") in ev        # the timer kept counting through 75


def test_missing_bar_on_the_last_recovery_session_fails_support():
    close = _base()
    close[62] = 95.0
    close[63] = 96.0
    close[64] = np.nan
    res, _ = _run(close, 60, 98.0)
    assert res.status == "SUPPORT_FAILED" and res.end == 64


# ── 10. Per-bloodline episodes and overlap_desc ──────────────────────────────

def test_episode_ids_are_per_bloodline_and_stable():
    d = date(2026, 9, 15)
    assert rc.episode_id(rc.RANGE, "DUOL", d) != rc.episode_id(rc.DRIFT, "DUOL", d)
    assert rc.episode_id(rc.DRIFT, "LFUS", d) == "LFUS#DRIFT@2026-09-15"


def test_overlap_desc_flags_an_alive_episode_in_the_other_parent():
    from src.research import reclaim_replay as rr
    close, high, low, _ = _with_retest(72)
    close[74] = high[72] + 0.5
    high[74] = close[74] + 0.1
    sessions = _sessions(len(close))
    bars = _bars(close, high, low)
    drift = rc.build_episodes(rc.DRIFT, "X", [56], bars, sessions, 80)
    rng = rc.build_episodes(rc.RANGE, "X", [60], bars, sessions, 80, floor_at={60: 90.0})
    other = rr.LaneEpisodes(rc.RANGE, rng, {})
    assert rr.overlap_desc(drift[0].trigger, other, 80) is True
    assert rr.overlap_desc(drift[0].trigger, rr.LaneEpisodes(rc.RANGE, [], {}), 80) is False


# ── 11. Open decision precedence ─────────────────────────────────────────────

def test_open_equal_to_stop_is_unfillable():
    d = rc.open_decision(100.0, 100.0, 100.0)
    assert d.mech_status == rc.UNFILLABLE_GAP_THROUGH


def test_risk_exactly_8_percent_passes_and_8_0068_skips():
    assert rc.open_decision(100.0, 92.0, 100.0).mech_status == rc.PASS
    duol = rc.open_decision(146.00, 134.31, 141.9878)
    assert duol.risk_pct == pytest.approx(0.080068, abs=1e-6)
    assert duol.mech_status == rc.SKIP_RISK


def test_extension_exactly_1_10_passes():
    assert rc.open_decision(110.0, 105.0, 100.0).mech_status == rc.PASS
    assert rc.open_decision(110.01, 105.0, 100.0).mech_status == rc.SKIP_EXTENDED


def test_acmr_lists_both_extended_and_risk():
    d = rc.open_decision(86.09, 72.40, 76.6455)
    assert d.reasons == [rc.SKIP_EXTENDED, rc.SKIP_RISK] and d.mech_status == rc.SKIP_EXTENDED


def test_no_entry_bar_is_censored():
    assert rc.open_decision(None, 10.0, 11.0).mech_status == rc.CENSORED_NO_ENTRY_BAR


# ── 12. Earnings gate fails closed ───────────────────────────────────────────

E = date(2026, 10, 2)
BEFORE_OPEN = datetime(2026, 10, 2, 10, 17, tzinfo=timezone.utc)   # 06:17 ET
DATES_CLEAR = [date(2026, 7, 30), date(2026, 11, 4)]


def test_timely_bracketed_snapshot_without_a_window_hit_is_clear():
    assert rc.earnings_status(True, BEFORE_OPEN, DATES_CLEAR, E, is_trading_day) == rc.CLEAR


def test_date_inside_window_blocks():
    assert rc.earnings_status(True, BEFORE_OPEN, DATES_CLEAR + [date(2026, 10, 6)], E,
                              is_trading_day) == rc.BLOCK_EARNINGS


def test_late_failed_and_unbracketed_are_never_clear():
    at_open = datetime(2026, 10, 2, 13, 30, tzinfo=timezone.utc)    # 09:30 ET exactly
    assert rc.earnings_status(True, at_open, DATES_CLEAR, E, is_trading_day) == rc.INACTIVE_LATE
    assert rc.earnings_status(False, BEFORE_OPEN, DATES_CLEAR, E, is_trading_day) == rc.INACTIVE_FAILED
    assert rc.earnings_status(None, None, [], E, is_trading_day) == rc.INACTIVE_FAILED
    assert rc.earnings_status(True, BEFORE_OPEN, [date(2026, 7, 30)], E,
                              is_trading_day) == rc.BLOCK_EARNINGS_UNKNOWN


# ── 13-15. Execution ─────────────────────────────────────────────────────────

def _exit_bars(rows):
    sessions = _sessions(len(rows))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["volume"], df["vwap"] = 1e6, np.nan
    return sessions, df


def test_gap_below_stop_fills_at_the_open_not_the_stop():
    sessions, df = _exit_bars([(100, 101, 99, 100), (90, 91, 88, 89), (89, 90, 88, 89)])
    r = rc.clone_exit(df, sessions, 0, 100.0, 95.0, 5, 2)
    assert r.exited and r.exit_reason == "stop" and r.exit_price == pytest.approx(90 * (1 - rc.COST))
    assert r.gap_through


def test_same_bar_touch_exits_at_the_stop_and_entry_bar_checks_low_only():
    sessions, df = _exit_bars([(100, 104, 94, 103), (103, 104, 102, 103)])
    r = rc.clone_exit(df, sessions, 0, 100.0, 95.0, 5, 1)
    assert r.exit_reason == "stop" and r.exit_price == pytest.approx(95 * (1 - rc.COST))
    assert r.exit_date == sessions[0]


def test_three_clones_share_open_and_stop_and_differ_only_in_horizon():
    rows = [(100, 101, 99.5, 100.5)] + [(100 + i, 101 + i, 99.5 + i, 100.5 + i) for i in range(1, 25)]
    sessions, df = _exit_bars(rows)
    res = {h: rc.clone_exit(df, sessions, 0, 100.0, 95.0, h, len(rows) - 1) for h in rc.HORIZONS}
    for h, r in res.items():
        assert r.exit_reason == "expiry" and r.exit_date == sessions[h - 1]
    assert res[5].net_return < res[10].net_return < res[20].net_return


def test_slippage_is_applied_once_on_entry_and_once_on_exit():
    rows = [(100, 101, 99.5, 100)] * 5
    sessions, df = _exit_bars(rows)
    r = rc.clone_exit(df, sessions, 0, 100.0, 90.0, 5, 4)
    assert r.net_return == pytest.approx(100 * (1 - rc.COST) / (100 * (1 + rc.COST)) - 1)


def test_lfus_h1_matches_the_worked_example():
    sessions, df = _exit_bars([(448.31, 457.0, 447.0, 456.53)])
    r = rc.clone_exit(df, sessions, 0, 448.31, 420.86, 1, 0)
    assert r.net_return == pytest.approx(0.01630, abs=5e-5)


# ── 14. No other exit levers ─────────────────────────────────────────────────

def test_settings_disable_tiers_and_trail_for_reclaim():
    from src.config import Settings
    s = Settings()
    assert s.uses_score_tiered_stops("reclaim") is False
    assert s.trail_for_model("reclaim") == (0.0, 0.0)


def test_big_run_up_never_exits_early_without_a_target_or_trail():
    rows = [(100, 100.5, 99.8, 100)] + [(100 + 5 * i, 101 + 5 * i, 99.9 + 5 * i, 100 + 5 * i) for i in range(1, 6)]
    sessions, df = _exit_bars(rows)
    r = rc.clone_exit(df, sessions, 0, 100.0, 95.0, 5, len(rows) - 1)
    assert r.exit_reason == "expiry" and r.exit_date == sessions[4]


# ── 16. Sentinel ─────────────────────────────────────────────────────────────

def test_persisted_features_are_json_safe_and_carry_the_no_target_marker():
    import json
    trig = rc.Trigger("E#DRIFT@2026-09-15", rc.DRIFT, "E", 1, 2, 3, date(2026, 9, 25), date(2026, 9, 28),
                      date(2026, 10, 1), date(2026, 10, 2), 420.86, 432.14, 435.55, 424.9641, float("nan"))
    ep = rc.Episode(rc.DRIFT, "E", 0, date(2026, 9, 15), 393.0, rc.FLOOR_TYPE[rc.DRIFT], trig.episode_id)
    sig = rc.ReclaimSignal.from_trigger(trig, ep, 10, overlap_desc=False, score_pct_value=None)
    feats = sig.persisted_features()
    text = json.dumps(feats, allow_nan=False)
    assert feats["no_target"] is True and feats["exit_policy"] == rc.EXIT_POLICY
    assert sig.target_1 == sig.target_2 == rc.NO_TARGET_SENTINEL and "Infinity" not in text
    assert sig.max_entry_price == pytest.approx(min(1.10 * 424.9641, 420.86 / 0.92))


# ── 17. Uniform censoring ────────────────────────────────────────────────────

def test_an_early_stop_stays_incomplete_until_the_terminal_bar_exists():
    sessions, df = _exit_bars([(100, 101, 94, 95), (95, 96, 94, 95), (95, 96, 94, 95)])
    r = rc.clone_exit(df, sessions, 0, 100.0, 95.0, 10, 2)
    assert r.exited and r.stopped and not r.complete


def test_reporter_drops_incomplete_horizons_from_statistics():
    from src.research import reclaim_report as rp
    row = rp.HorizonRow(rc.DRIFT, "a", "A", date(2026, 1, 2), date(2026, 1, 5), 10, True,
                        complete=False, exited=True, net=None, stopped=True, gap_through=False,
                        bench_spy=None, bench_sector=None, nn_net=None)
    s = rp.summarize([row], rc.DRIFT, 10, clean_only=True)
    assert s["complete"] == 0 and s["stopped_before_end"] == 1 and s["censored"] == 1


# ── 18. Controls ─────────────────────────────────────────────────────────────

def _cand(eid, sym, sector="Tech", mcap=1e10, pct=0.5, dist=0.02, age=10, pre=rc.VALID, bar=True):
    return rc.ControlCandidate(eid, sym, sector, mcap, pct, dist, age, pre, bar)


def test_controls_prefer_same_sector_nearest_and_break_ties_by_episode_id():
    trig = _cand("T", "T", dist=0.02)
    pool = [_cand("b", "B", dist=0.02), _cand("a", "A", dist=0.02), _cand("c", "C", dist=0.5),
            _cand("z", "Z", sector="Energy", dist=0.02), _cand("d", "D", dist=0.03)]
    picked = rc.nn_controls(trig, pool, rc.RANGE)
    assert [c.episode_id for c in picked] == ["a", "b", "d"]


def test_drift_lane_drops_the_score_dimension():
    trig = _cand("T", "T", pct=0.9)
    near_score = _cand("s", "S", pct=0.9, dist=0.10)
    near_dist = _cand("d", "D", pct=0.1, dist=0.02)
    assert rc.nn_controls(trig, [near_score, near_dist], rc.DRIFT, k=1)[0].episode_id == "d"


def test_missing_candidate_feature_adds_one_and_controls_need_an_entry_bar():
    trig = _cand("T", "T", dist=0.0)
    pool = [_cand("m", "M", dist=None), _cand("n", "N", dist=0.5), _cand("o", "O", dist=-0.5),
            _cand("x", "X", bar=False, dist=0.0)]
    # SD over the full risk set (ddof=0) is ~0.41, so n and o sit ~1.22 SD away
    # (squared 1.5); m's missing dimension costs exactly 1.0 -> m is nearest.
    # x has no entry bar and can never be chosen, despite matching exactly.
    assert [c.episode_id for c in rc.nn_controls(trig, pool, rc.DRIFT)] == ["m", "n", "o"]


def test_valid_trigger_uses_valid_controls_only_when_three_exist():
    trig = _cand("T", "T")
    pool = [_cand("i1", "I1", pre=rc.INSUFFICIENT_PREHISTORY, dist=0.02),
            _cand("v1", "V1", dist=0.4), _cand("v2", "V2", dist=0.5), _cand("v3", "V3", dist=0.6)]
    assert {c.episode_id for c in rc.nn_controls(trig, pool, rc.DRIFT)} == {"v1", "v2", "v3"}
    pool2 = pool[:3]
    assert "i1" in {c.episode_id for c in rc.nn_controls(trig, pool2, rc.DRIFT)}


def test_control_stop_mirrors_the_trigger_risk_fraction():
    assert rc.control_stop(50.0, 100.0, 94.0) == pytest.approx(47.0)


# ── 19. KILL clock ───────────────────────────────────────────────────────────

def test_kill_clock_fires_at_26_weeks_or_30_clean_triggers():
    start = date(2026, 10, 5)
    assert not rc.kill_clock_due(5, start, start + timedelta(weeks=26) - timedelta(days=1))
    assert rc.kill_clock_due(5, start, start + timedelta(weeks=26))
    assert rc.kill_clock_due(30, start, start)


def test_kill_flag_needs_negative_excess_at_both_horizons_and_high_stop_touch():
    rows = [rc.GateInput(date(2026, 1, 2) + timedelta(days=i), "UP_LOVOL", -0.01, -0.01, i % 2 == 0, True, True)
            for i in range(30)]
    assert rc.kill_flag(rows, date(2026, 1, 1), date(2026, 3, 1)) is True
    rows_low_stop = [rc.GateInput(r.entry_date, r.regime, r.excess_h5, r.excess_h10, False, True, True)
                     for r in rows]
    assert rc.kill_flag(rows_low_stop, date(2026, 1, 1), date(2026, 3, 1)) is False


def test_promotion_needs_dates_clusters_and_positive_excess_at_h5_and_h10():
    rows = [rc.GateInput(date(2026, 1, 2) + timedelta(days=i), "UP_LOVOL" if i % 2 else "DOWN_HIVOL",
                         0.01, 0.01, False, True, True) for i in range(30)]
    assert rc.promotion_flag(rows) is True
    assert rc.promotion_flag(rows[:29]) is False
    neg5 = [rc.GateInput(r.entry_date, r.regime, -0.01, 0.01, False, True, True) for r in rows]
    assert rc.promotion_flag(neg5) is False


def test_date_clustered_bootstrap_is_seeded_and_resamples_dates():
    vals = [0.01, 0.02, -0.01, 0.03, 0.00, 0.015]
    dates = [date(2026, 1, 2)] * 3 + [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
    a = rc.date_clustered_bootstrap_ci(vals, dates)
    assert a == rc.date_clustered_bootstrap_ci(vals, dates)
    assert rc.date_clustered_mean(vals, dates) == pytest.approx(np.mean([0.02 / 3 * 1, 0.03, 0.0, 0.015]))


# ── 26. QNT ATR/ADX parity with Range gate_inputs ────────────────────────────

def test_qnt_atr14_and_adx14_match_range_on_2026_10_01():
    df = pd.read_csv(FIXTURES / "reclaim_qnt_2026-10-01.csv")
    sessions = trading_sessions(date(2026, 5, 1), date(2026, 10, 1))
    s = rc.range_tech_native_series(rc.align_to_sessions(df, sessions), 5e9).iloc[-1]
    assert round(s["atr14"], 5) == 3.31369
    assert round(s["adx14"], 4) == 12.5032
    assert bool(s["member"]) is True        # QNT was in Range's 10-01 pass list (Top5 #1)


# ── 27. Four-name golden test (canonical V0.2 forward, trigger 10-01) ────────

GOLDEN = {
    # symbol: (lane, start, floor, status, reasons, stop, sma50_k, slope5_pct, open, ext, risk_pct)
    "LFUS": (rc.DRIFT, "2026-09-15", 393.00, rc.PASS, [], 420.86, 424.9641, 0.542, 448.31, 1.0549, 6.123),
    "ACMR": (rc.DRIFT, "2026-09-04", 66.4201, rc.SKIP_EXTENDED, [rc.SKIP_EXTENDED, rc.SKIP_RISK],
             72.40, 76.6455, -1.083, 86.09, 1.1232, 15.902),
    "DUOL": (rc.DRIFT, "2026-09-28", 133.68, rc.SKIP_RISK, [rc.SKIP_RISK], 134.31, 141.9878, 0.974,
             146.00, 1.0283, 8.0068),
    "KSS": (rc.RANGE, "2026-09-16", 16.02, rc.SKIP_RISK, [rc.SKIP_RISK], 16.89, 18.2950, 0.242,
            19.16, 1.0473, 11.848),
}


@pytest.mark.parametrize("symbol", sorted(GOLDEN))
def test_four_name_golden(symbol):
    lane, start, floor, status, reasons, stop, sma_k, slope_pct, open_e, ext, risk = GOLDEN[symbol]
    allbars = pd.read_csv(FIXTURES / "reclaim_golden_2026-10-02.csv")
    sessions = trading_sessions(date(2026, 5, 1), date(2026, 10, 2))
    bars = rc.align_to_sessions(allbars[allbars.symbol == symbol].drop(columns="symbol"), sessions)
    s_idx = sessions.index(date.fromisoformat(start))
    c_idx = sessions.index(date(2026, 10, 1))
    if lane == rc.DRIFT:
        assert rc.drift_floor(bars["low"].to_numpy(dtype=float), s_idx) == pytest.approx(floor)
    ep = rc.Episode(lane, symbol, s_idx, sessions[s_idx], floor, rc.FLOOR_TYPE[lane],
                    rc.episode_id(lane, symbol, sessions[s_idx]))
    res = rc.run_episode(ep, bars, rc.sma(bars["close"].to_numpy(dtype=float), 50), {s_idx}, c_idx, sessions)
    trig = res.trigger
    assert res.status == "TRIGGERED" and trig.trigger_date == date(2026, 10, 1)
    assert trig.setup_low == pytest.approx(stop)
    assert trig.sma50_k == pytest.approx(sma_k, abs=5e-5)
    assert trig.slope5 * 100 == pytest.approx(slope_pct, abs=5e-4)
    e = sessions.index(date(2026, 10, 2))
    dec = rc.open_decision(float(bars["open"].iloc[e]), trig.setup_low, trig.sma50_k)
    assert dec.open_e == pytest.approx(open_e)
    assert dec.ext_ratio == pytest.approx(ext, abs=5e-5)
    assert dec.risk_pct * 100 == pytest.approx(risk, abs=5e-4)
    assert dec.mech_status == status and dec.reasons == reasons
    if symbol == "ACMR":
        assert ("BREAK" in [e for _, e, _ in res.events]) and ("RECOVER" in [e for _, e, _ in res.events])


# ── Calendar (§5: the state machine iterates XNYS sessions) ──────────────────

@pytest.mark.parametrize("year, sessions", [(2020, 253), (2021, 252), (2022, 251), (2023, 250),
                                            (2024, 252), (2025, 250)])
def test_xnys_session_counts_match_the_published_calendar(year, sessions):
    assert len(trading_sessions(date(year, 1, 1), date(year, 12, 31))) == sessions


def test_unscheduled_2025_01_09_closure_is_not_a_session():
    assert not is_trading_day(date(2025, 1, 9))
