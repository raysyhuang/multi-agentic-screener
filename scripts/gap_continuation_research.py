"""Gap-continuation alpha research — the first Phase-4 candidate.

Thesis: a stock that gaps up strongly on heavy volume and CLOSES near its high
(the gap held; buyers in control) tends to continue higher over the next several
days. We enter T+1 open (never chase the gap day) with ATR stops/targets/trail.

Discipline (the whole point of the truth-matrix work): every run goes through
the UNIFIED exit engine with realistic gap-through fills (gap_through=True) and a
concurrency-capped equity curve. No optimistic-simulator self-deception, and a
deliberately SMALL parameter sweep so we don't overfit a bull-heavy window.

Efficiency: features are computed ONCE per ticker (all indicators are causal),
not per expanding window — so this runs in seconds, not the truth matrix's
minutes.

Usage:
  python scripts/gap_continuation_research.py --cache-file <parquet>
  python scripts/gap_continuation_research.py --cache-file <parquet> --sweep
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd

from src.backtest.metrics import compute_metrics
from src.features.technical import compute_all_technical_features
from src.research.signal_backtest import simulate_trade
import scripts.sniper_equity_curve as eq

MIN_HISTORY = 60


@dataclass
class GapParams:
    gap_min: float = 3.0        # min gap-up % vs prior close
    gap_max: float = 15.0       # reject unstable mega-gaps (news/earnings blowups)
    vol_min: float = 1.5        # min relative volume (rvol) on the gap day
    close_pos_min: float = 0.5  # close must be in upper half of the day's range
    rsi_max: float = 82.0       # not a blow-off top
    require_uptrend: bool = True  # close > sma_50
    stop_atr: float = 1.5
    target_atr: float = 3.0
    hold: int = 7
    trail_activate: float = 1.0
    trail_distance: float = 0.5
    entry_gap_atr: float = 1.0  # reject if T+1 open gaps > entry + this*ATR (chase guard)


def scan_gap_continuation(e: pd.DataFrame, p: GapParams) -> list[dict]:
    """Return signal dicts from a PRE-ENRICHED frame (features already computed).

    All indicators are causal, so features are computed once per ticker upstream
    and reused across every sweep config — equivalent to per-bar recompute but
    far faster.
    """
    if len(e) < MIN_HISTORY:
        return []
    out: list[dict] = []
    for i in range(MIN_HISTORY, len(e)):
        row = e.iloc[i]
        gap = row.get("gap_pct")
        rvol = row.get("rvol")
        atr = row.get("atr_14")
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        sma50 = row.get("sma_50")
        rsi = row.get("rsi_14")

        if gap is None or pd.isna(gap) or not (p.gap_min <= float(gap) <= p.gap_max):
            continue
        if rvol is None or pd.isna(rvol) or float(rvol) < p.vol_min:
            continue
        rng = high - low
        close_pos = (close - low) / rng if rng > 0 else 0.0
        if close_pos < p.close_pos_min:
            continue
        if p.require_uptrend and (sma50 is None or pd.isna(sma50) or close <= float(sma50)):
            continue
        if rsi is not None and not pd.isna(rsi) and float(rsi) > p.rsi_max:
            continue
        if atr is None or pd.isna(atr) or float(atr) <= 0:
            continue
        atr = max(float(atr), close * 0.005)

        out.append({
            "signal_date": row["date"],
            "entry_ref": close,
            "stop": round(close - p.stop_atr * atr, 2),
            "target": round(close + p.target_atr * atr, 2),
            "atr": atr,
            "max_entry_price": round(close + p.entry_gap_atr * atr, 2),
        })
    return out


def build_enriched(price_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Compute technical features once per ticker (reused across all sweep configs)."""
    out: dict[str, pd.DataFrame] = {}
    for ticker, df in price_data.items():
        if ticker == "SPY" or len(df) < MIN_HISTORY:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        out[ticker] = compute_all_technical_features(df).reset_index(drop=True)
    return out


def backtest(enriched: dict[str, pd.DataFrame], p: GapParams) -> tuple[list, list[eq.Trade], dict]:
    """Run the gap-continuation scan + unified-engine simulation across tickers."""
    returns: list[float] = []
    eq_trades: list[eq.Trade] = []
    exit_reasons: dict[str, int] = {}
    for ticker, df in enriched.items():
        for sig in scan_gap_continuation(df, p):
            result = simulate_trade(
                df, sig["signal_date"],
                stop_loss=sig["stop"], target=sig["target"], max_hold=p.hold,
                trail_activate_pct=p.trail_activate, trail_distance_pct=p.trail_distance,
                atr_value=sig["atr"], max_entry_price=sig["max_entry_price"],
                gap_through=True,  # realistic fills — the whole point
            )
            if result is None:
                continue
            returns.append(result["pnl_pct"])
            exit_reasons[result["exit_reason"]] = exit_reasons.get(result["exit_reason"], 0) + 1
            eq_trades.append(eq.Trade(
                ticker=ticker, entry=result["entry_date"], exit=result["exit_date"],
                pnl_pct=result["pnl_pct"], regime="", score=0.0,
            ))
    return returns, eq_trades, exit_reasons


def _row(label: str, p: GapParams, enriched: dict) -> str:
    returns, eq_trades, exits = backtest(enriched, p)
    if not returns:
        return f"{label:<22}{'0 trades':>10}"
    m = compute_metrics(returns)
    eq_res = eq.simulate(eq_trades, mode="fixed_fraction", fraction=0.20,
                         max_concurrent=5, start_capital=100_000.0)
    return (f"{label:<22}{m.total_trades:>6}{m.win_rate:>8.1%}{m.avg_return_pct:>8.2f}"
            f"{m.expectancy:>9.3f}{m.profit_factor:>7.2f}{m.sharpe_ratio:>7.2f}"
            f"{eq_res.multiple:>8.2f}{eq_res.cagr_pct:>8.1f}{eq_res.max_drawdown_pct:>7.1f}"
            f"{eq_res.skipped_saturation:>6}")


def forward_drift_diagnostic(enriched: dict, p: GapParams) -> None:
    """Decisive test: does the ENTRY signal have raw forward drift (no stops)?

    Measures forward 1/3/5/10-day returns from the T+1 open for gap signals, and
    compares to the universe base rate (every bar's forward return). If the signal
    drift is not above base rate, the entry has no edge and no exit tuning can
    save it — the family is dead on daily bars. If it IS above base rate, the edge
    is real but daily stops destroy it → worth the minute-bar (intraday) test.
    """
    import numpy as np
    horizons = [1, 3, 5, 10]
    sig_fwd = {h: [] for h in horizons}
    base_fwd = {h: [] for h in horizons}
    for df in enriched.values():
        closes = df["close"].to_numpy(dtype=float)
        opens = df["open"].to_numpy(dtype=float)
        n = len(df)
        # base rate: forward close-to-close return from every bar
        for h in horizons:
            if n > h:
                base_fwd[h].extend(((closes[h:] - closes[:-h]) / closes[:-h] * 100).tolist())
        # signal drift: from T+1 open to T+1+h close
        for sig in scan_gap_continuation(df, p):
            idx = df.index[df["date"] == sig["signal_date"]]
            if len(idx) == 0:
                continue
            i = int(idx[0])
            if i + 1 >= n:
                continue
            entry = opens[i + 1]
            for h in horizons:
                if i + 1 + h < n:
                    sig_fwd[h].append((closes[i + 1 + h] - entry) / entry * 100)
    print("\nForward-drift diagnostic (raw returns, NO stops) — is the entry edge real?")
    print(f"{'horizon':<10}{'signal avg%':>13}{'base avg%':>12}{'edge(bps)':>11}{'sig N':>8}")
    print("-" * 54)
    for h in horizons:
        s = float(np.mean(sig_fwd[h])) if sig_fwd[h] else 0.0
        b = float(np.mean(base_fwd[h])) if base_fwd[h] else 0.0
        print(f"{h:<10}{s:>13.3f}{b:>12.3f}{(s - b) * 100:>11.1f}{len(sig_fwd[h]):>8}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-file", required=True)
    ap.add_argument("--sweep", action="store_true", help="run a small principled parameter sweep")
    ap.add_argument("--diagnostic", action="store_true", help="raw forward-drift edge test")
    ap.add_argument("--drift", action="store_true", help="time-based drift-capture exit test")
    args = ap.parse_args()

    combined = pd.read_parquet(args.cache_file)
    price_data = {t: g.drop(columns=["_ticker"]).reset_index(drop=True)
                  for t, g in combined.groupby("_ticker")}
    print(f"Loaded {len(price_data)} tickers; computing features once...")
    enriched = build_enriched(price_data)

    header = (f"{'config':<22}{'N':>6}{'WR':>8}{'avg%':>8}{'expect%':>9}{'PF':>7}"
              f"{'Sharpe':>7}{'equity×':>8}{'CAGR%':>8}{'eqDD%':>7}{'skip':>6}")
    print("\n" + header)
    print("-" * len(header))
    print(_row("baseline", GapParams(), enriched))

    if args.sweep:
        # Small, principled sweep — one dimension at a time off the baseline.
        for g in (2.0, 4.0, 5.0):
            print(_row(f"gap_min={g}", GapParams(gap_min=g), enriched))
        for t in (2.0, 4.0):
            print(_row(f"target_atr={t}", GapParams(target_atr=t), enriched))
        for h in (3, 5, 10):
            print(_row(f"hold={h}", GapParams(hold=h), enriched))
        print(_row("no-uptrend-filter", GapParams(require_uptrend=False), enriched))
        print(_row("tight close>=0.7", GapParams(close_pos_min=0.7), enriched))

    if args.diagnostic:
        forward_drift_diagnostic(enriched, GapParams())

    if args.drift:
        # Drift-capture: the edge is a small 1-3 day drift, so exit at close on a
        # time stop with only a catastrophe stop (5xATR) and no target/trail —
        # don't let ATR stops eat a 13-bps edge. This is how you monetize drift.
        print("\nDrift-capture (wide catastrophe stop, exit at close on hold):")
        print(header)
        print("-" * len(header))
        for h in (1, 2, 3, 5):
            p = GapParams(stop_atr=5.0, target_atr=100.0, hold=h,
                          trail_activate=0.0, trail_distance=0.0)
            print(_row(f"drift hold={h}", p, enriched))
        for h in (2, 3):
            p = GapParams(stop_atr=5.0, target_atr=100.0, hold=h, gap_min=5.0,
                          trail_activate=0.0, trail_distance=0.0)
            print(_row(f"drift hold={h} gap>=5", p, enriched))


if __name__ == "__main__":
    main()
