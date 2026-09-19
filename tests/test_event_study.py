"""Event-study primitives — synthetic data, no network.

What these pin: the liquidity screen cannot see the entry bar; an event is
compared with same-day, same-bucket stocks (so market/size beta cancels); a
missing bar yields no return rather than a flat one; and the cluster bootstrap
collapses when every observation shares one date.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.research import event_study as es


def _bars(closes, *, volume=1_000_000, start=date(2026, 1, 5), opens=None):
    days, d = [], start
    while len(days) < len(closes):
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    opens = opens if opens is not None else closes
    return pd.DataFrame({"date": days, "open": opens, "high": closes, "low": closes,
                         "close": closes, "volume": volume})


def test_liquidity_screen_uses_only_bars_before_the_entry_date():
    """A volume spike ON day D must not make D eligible; it can only affect D+1."""
    n = 30
    vol = [10_000] * n                      # $100 * 10k = $1M/day < $2M floor
    vol[25] = 50_000_000                    # spike on row 25
    panel = es.build_panel({"AAA": _bars([100.0] * n, volume=vol)})
    d = panel.dates
    assert not panel.eligible.at[d[25], "AAA"]      # spike day itself: still ineligible
    assert panel.eligible.at[d[26], "AAA"]          # next day: the spike is now history


def test_price_floor_reads_the_prior_close_not_the_entry_bar():
    closes = [4.0] * 25 + [10.0] * 5
    panel = es.build_panel({"AAA": _bars(closes, volume=5_000_000)})
    d = panel.dates
    assert not panel.eligible.at[d[25], "AAA"]      # prior close 4.0 < $5
    assert panel.eligible.at[d[26], "AAA"]


def test_excess_cancels_a_market_wide_move():
    """Every name rises 1%/day; the event name rises the same. Excess must be ~0,
    which is the whole reason the base rate is same-day and same-bucket."""
    n = 40
    path = [100.0 * 1.01 ** i for i in range(n)]
    prices = {t: _bars(path, volume=1_000_000) for t in ("AAA", "BBB", "CCC", "DDD")}
    panel = es.build_panel(prices, n_buckets=1)
    ev = [{"ticker": "AAA", "entry_date": panel.dates[25]}]
    out = es.event_excess(panel, ev, [5])
    assert out.loc[0, "eligible"]
    assert out.loc[0, "fwd_5"] == pytest.approx((1.01 ** 5 - 1) * 100, rel=1e-6)
    assert out.loc[0, "excess_5"] == pytest.approx(0.0, abs=1e-9)


def test_event_outperformance_shows_up_as_excess():
    n = 40
    flat = [100.0] * n
    jump = [100.0] * 26 + [110.0] * (n - 26)        # +10% the day after entry
    prices = {"EVT": _bars(jump, volume=1_000_000),
              **{t: _bars(flat, volume=1_000_000) for t in ("B1", "B2", "B3")}}
    panel = es.build_panel(prices, n_buckets=1)
    out = es.event_excess(panel, [{"ticker": "EVT", "entry_date": panel.dates[25]}], [5])
    # base rate includes the event name itself: (10 + 0 + 0 + 0) / 4 = 2.5
    assert out.loc[0, "fwd_5"] == pytest.approx(10.0)
    assert out.loc[0, "base_5"] == pytest.approx(2.5)
    assert out.loc[0, "excess_5"] == pytest.approx(7.5)


def test_missing_exit_bar_gives_no_return_not_a_flat_one():
    full = _bars([100.0] * 40, volume=1_000_000)
    short = full.iloc[:28].copy()                   # stops trading after row 27
    panel = es.build_panel({"DEAD": short, "LIVE": full}, n_buckets=1)
    out = es.event_excess(panel, [{"ticker": "DEAD", "entry_date": panel.dates[25]}], [5])
    assert out.loc[0, "eligible"]
    assert np.isnan(out.loc[0, "fwd_5"]) and np.isnan(out.loc[0, "excess_5"])


def test_ineligible_event_is_reported_not_dropped():
    thin = _bars([100.0] * 40, volume=100)          # far below the dollar-volume floor
    liquid = _bars([100.0] * 40, volume=1_000_000)
    panel = es.build_panel({"THIN": thin, "OK": liquid}, n_buckets=1)
    out = es.event_excess(panel, [{"ticker": "THIN", "entry_date": panel.dates[25]}], [5])
    assert len(out) == 1 and not out.loc[0, "eligible"] and np.isnan(out.loc[0, "excess_5"])


def test_cluster_ci_collapses_when_all_observations_share_one_date():
    x = [1.0, -2.0, 3.5, 0.5]
    lo, hi = es.cluster_boot_ci(x, ["2026-01-05"] * len(x))
    assert lo == pytest.approx(np.mean(x)) and hi == pytest.approx(np.mean(x))
    ilo, ihi = es.boot_ci(x)
    assert ilo < np.mean(x) < ihi                   # the iid interval does not collapse


def test_market_regime_lag_shifts_labels_by_one_completed_session():
    closes = [100.0] * 60 + [130.0] * 25            # regime changes late in the series
    spy = _bars(closes)
    same, lagged = es.market_regime(spy), es.market_regime(spy, lag_sessions=1)
    keys = list(same)
    assert lagged[keys[0]] == "unknown"
    assert all(lagged[keys[i]] == same[keys[i - 1]] for i in range(1, len(keys)))


def test_block_ci_moves_overlapping_dates_together():
    """Adjacent dates with the same sign move as one unit in a block resample, so
    a run of positive days followed by a run of negative days yields a WIDER
    interval than treating each date as independent."""
    cal = pd.to_datetime(pd.bdate_range("2026-01-05", periods=200))
    x = [1.0] * 100 + [-1.0] * 100
    dates = list(cal)
    blo, bhi = es.block_boot_ci(x, dates, cal, block=20, n_boot=4000)
    clo, chi = es.cluster_boot_ci(x, [str(d)[:10] for d in dates], n_boot=4000)
    assert blo < clo and bhi > chi
    assert blo < 0 < bhi


def test_block_ci_collapses_for_a_constant_series():
    cal = pd.to_datetime(pd.bdate_range("2026-01-05", periods=60))
    lo, hi = es.block_boot_ci([0.7] * 60, list(cal), cal, block=10, n_boot=500)
    assert lo == pytest.approx(0.7) and hi == pytest.approx(0.7)


def test_block_ci_handles_a_calendar_shorter_than_the_block():
    """block > n used to raise IndexError; the wrap is now modular."""
    cal = pd.to_datetime(pd.bdate_range("2026-01-05", periods=7))
    lo, hi = es.block_boot_ci([1.0, 2.0, 3.0], list(cal[:3]), cal, block=20, n_boot=200)
    assert 1.0 <= lo <= hi <= 3.0


def test_block_ci_wraparound_and_weighting_match_an_explicit_computation():
    """Prefix-sum block sums must equal an explicit circular walk, with unequal
    event counts per date (observation weighting)."""
    import numpy as _np
    cal = pd.to_datetime(pd.bdate_range("2026-01-05", periods=9))
    x = [1.0, 1.0, 5.0, -2.0, 4.0, 0.5]
    d = [cal[0], cal[0], cal[3], cal[7], cal[8], cal[8]]
    lo, hi = es.block_boot_ci(x, d, cal, block=4, n_boot=3000, seed=7)
    sums = _np.zeros(9)
    cnts = _np.zeros(9)
    for v, dd in zip(x, d):
        i = list(cal).index(dd)
        sums[i] += v
        cnts[i] += 1
    rng = _np.random.default_rng(7)
    starts = rng.integers(0, 9, size=(3000, 3))
    means = []
    for row in starts:
        tot = cnt = 0.0
        for st in row:
            for k in range(4):
                j = (st + k) % 9
                tot += sums[j]
                cnt += cnts[j]
        if cnt:
            means.append(tot / cnt)
    means.sort()
    assert lo == pytest.approx(means[int(0.025 * len(means))])
    assert hi == pytest.approx(means[int(0.975 * len(means))])


def test_block_diff_ci_collapses_for_constant_cohorts_and_detects_a_gap():
    cal = pd.to_datetime(pd.bdate_range("2026-01-05", periods=80))
    d = list(cal)
    point, lo, hi = es.block_diff_ci([2.0] * 80, d, [0.5] * 80, d, cal, block=10, n_boot=500)
    assert point == pytest.approx(1.5) and lo == pytest.approx(1.5) and hi == pytest.approx(1.5)
    rng = np.random.default_rng(1)
    a = list(1.0 + rng.normal(0, 1, 80))
    b = list(rng.normal(0, 1, 80))
    point, lo, hi = es.block_diff_ci(a, d, b, d, cal, block=10, n_boot=2000)
    assert lo > 0 and hi > lo
