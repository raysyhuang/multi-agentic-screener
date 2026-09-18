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
from scripts.mr_hold_and_bear import bear_rule, equity, stamp_market_regime, summarize


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
