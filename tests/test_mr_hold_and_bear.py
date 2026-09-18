"""scripts/mr_hold_and_bear.py — the regime join and the hold override.

The join must use the SPY MARKET regime on the trade's entry date, never the
per-ticker `regime` the backtest stamps; and gen_mr_trades' --holding-period
must be the only key that changes relative to LIVE_MR.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from scripts.choppy_sniper_regime_test import spy_market_regime
from scripts.gen_mr_trades import LIVE_MR
from scripts.mr_hold_and_bear import (
    bear_rule, boot_ci, cluster_boot_ci, equity, stamp_market_regime, summarize,
)


def _spy(closes: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=len(closes))
    return pd.DataFrame({"date": dates, "close": closes})


def test_spy_market_regime_labels_by_date_from_a_frame():
    # 60 bars: first 49 are "unknown" (SMA50 not ready); a monotone rise then
    # puts close > SMA50 and SMA20 > SMA50 -> bull; a collapse -> bear.
    closes = [100 + i for i in range(60)] + [40.0] * 30
    reg = spy_market_regime(spy=_spy(closes))
    keys = sorted(reg)
    assert reg[keys[0]] == "unknown"
    assert reg[keys[59]] == "bull"
    assert reg[keys[-1]] == "bear"
    assert all(k == k[:10] and len(k) == 10 for k in keys)  # YYYY-MM-DD keys


def test_lagged_regime_labels_an_entry_with_the_previous_sessions_regime():
    """An entry at D's open cannot know D's close. With the regime flipping ON
    day D, lag_sessions=1 must label D with D-1's regime (the unlagged series
    labels D with D's own close-derived regime — the leak Codex caught)."""
    closes = [100 + i for i in range(60)] + [40.0] * 30   # bull ... then collapse
    unlagged = spy_market_regime(spy=_spy(closes))
    lagged = spy_market_regime(spy=_spy(closes), lag_sessions=1)
    keys = sorted(unlagged)
    # Find the first day the unlagged label leaves "bull" (the flip day D).
    flip = next(i for i in range(1, len(keys))
                if unlagged[keys[i]] != unlagged[keys[i - 1]] and unlagged[keys[i - 1]] == "bull")
    d_flip, d_prev = keys[flip], keys[flip - 1]
    assert unlagged[d_flip] != "bull" and unlagged[d_prev] == "bull"
    assert lagged[d_flip] == "bull"                      # D carries D-1's label
    assert lagged[keys[flip + 1]] == unlagged[d_flip]    # D+1 carries D's
    assert lagged[keys[0]] == "unknown"                  # nothing completed before day 1
    assert all(lagged[keys[i]] == unlagged[keys[i - 1]] for i in range(1, len(keys)))
    assert set(lagged) == set(unlagged)                  # same date keys, shifted labels


def test_cluster_bootstrap_collapses_when_every_trade_shares_one_date():
    """All trades on one entry date = one cluster: every resample is the whole
    sample, so the cluster CI is the point estimate, while the iid CI is not."""
    x = [-2.0, -1.0, 0.0, 1.0, 2.0, 5.0, -3.0, 0.5]
    lo, hi = cluster_boot_ci(x, ["2025-03-03"] * len(x))
    mean = sum(x) / len(x)
    assert lo == hi == mean
    ilo, ihi = boot_ci(x)
    assert ilo < mean < ihi and (ihi - ilo) > 0


def test_cluster_bootstrap_reduces_to_iid_with_one_trade_per_date():
    x = [-2.0, -1.0, 0.0, 1.0, 2.0, 5.0, -3.0, 0.5]
    dates = [f"2025-03-{d:02d}" for d in range(3, 3 + len(x))]
    # Same seed, same draw sequence over the same number of units -> identical.
    assert cluster_boot_ci(x, dates) == boot_ci(x)


def test_cluster_bootstrap_is_wider_than_iid_when_dates_cluster():
    """Ten identical-sign trades per date across two opposite dates: the iid
    interval sees 20 draws, the cluster interval sees 2 — it must be wider."""
    x = [1.0] * 10 + [-1.0] * 10
    dates = ["2025-03-03"] * 10 + ["2025-03-04"] * 10
    ilo, ihi = boot_ci(x)
    clo, chi = cluster_boot_ci(x, dates)
    assert (chi - clo) > (ihi - ilo)


def test_stamp_uses_entry_date_not_ticker_regime():
    reg = {"2025-03-03": "bear", "2025-03-04": "bull"}
    trades = [
        {"ticker": "AAA", "entry_date": date(2025, 3, 3), "regime": "bull", "pnl_pct": 1.0,
         "exit_date": date(2025, 3, 4), "exit_reason": "target", "holding_days": 1},
        {"ticker": "BBB", "entry_date": "2025-03-04", "regime": "bear", "pnl_pct": -1.0,
         "exit_date": date(2025, 3, 5), "exit_reason": "stop", "holding_days": 1},
        {"ticker": "CCC", "entry_date": "2025-03-05", "regime": "bull", "pnl_pct": 0.5,
         "exit_date": date(2025, 3, 6), "exit_reason": "expiry", "holding_days": 1},
    ]
    out = stamp_market_regime(trades, reg)
    assert [t["mkt"] for t in out] == ["bear", "bull", "unknown"]
    # The per-ticker label is carried but never consulted.
    assert [t["regime"] for t in out] == ["bull", "bear", "bull"]


def test_summarize_splits_bear_from_the_rest():
    reg = {"2025-03-03": "bear", "2025-03-04": "bull"}
    rows = []
    for i in range(12):
        rows.append({"ticker": f"T{i}", "entry_date": "2025-03-03", "exit_date": date(2025, 3, 4),
                     "pnl_pct": -1.0, "exit_reason": "stop", "holding_days": 1})
        rows.append({"ticker": f"U{i}", "entry_date": "2025-03-04", "exit_date": date(2025, 3, 5),
                     "pnl_pct": 2.0, "exit_reason": "target", "holding_days": 1})
    s = summarize(stamp_market_regime(rows, reg))
    assert s["bear"]["n"] == 12 and s["bear"]["avg"] == -1.0 and s["bear"]["wr"] == 0.0
    assert s["ex_bear"]["n"] == 12 and s["ex_bear"]["avg"] == 2.0
    assert s["all"]["n"] == 24 and s["all"]["avg"] == 0.5
    assert s["bear"]["ci_lo"] <= s["bear"]["avg"] <= s["bear"]["ci_hi"]
    # One entry date per cell -> the cluster interval collapses to the mean.
    assert s["bear"]["entry_dates"] == 1
    assert s["bear"]["cluster_ci_lo"] == s["bear"]["cluster_ci_hi"] == s["bear"]["avg"]


def test_bear_rule_is_the_pre_registered_one():
    assert bear_rule({"n": 99, "avg": -3.0}).startswith("NOT EVALUABLE")
    assert bear_rule({"n": 100, "avg": 0.5}).startswith("NO GATE")
    assert bear_rule({"n": 100, "avg": 0.0}).startswith("PROPOSE BEAR-BLOCK")
    assert bear_rule({"n": 100, "avg": 0.25}).startswith("NO ACTION")


def test_equity_accepts_iso_and_date_rows():
    rows = [
        {"ticker": "A", "entry_date": "2025-03-03", "exit_date": "2025-03-04", "pnl_pct": 1.0},
        {"ticker": "B", "entry_date": date(2025, 3, 4), "exit_date": date(2025, 3, 6), "pnl_pct": -0.5},
    ]
    eq = equity(rows)
    assert eq["taken"] == 2 and eq["skipped"] == 0
    assert set(eq) == {"taken", "skipped", "peak_concurrent", "total_return_pct",
                       "max_drawdown_pct", "sharpe"}


def test_gen_mr_trades_hold_override_is_the_only_change(monkeypatch, tmp_path):
    import scripts.gen_mr_trades as gm

    seen: dict = {}

    class _Res:
        class metrics:
            total_trades = 0
            win_rate = 0.0
            avg_return_pct = 0.0
            expectancy = 0.0
        trades: list = []

    def _fake_backtest(model, price, params):
        seen["model"], seen["params"] = model, dict(params)
        return _Res()

    monkeypatch.setattr(gm, "run_model_backtest", _fake_backtest)
    pq = tmp_path / "c.parquet"
    pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=3), "open": 1.0, "high": 1.0,
                  "low": 1.0, "close": 1.0, "volume": 1, "_ticker": "AAA"}).to_parquet(pq)
    monkeypatch.setattr("sys.argv", ["gen_mr_trades", "--cache-file", str(pq),
                                     "--out", str(tmp_path / "o.csv"), "--holding-period", "7"])
    gm.main()
    assert seen["model"] == "mean_reversion"
    assert seen["params"] == {**LIVE_MR, "holding_period": 7}
    assert LIVE_MR["holding_period"] == 3  # module constant untouched
