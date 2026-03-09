#!/usr/bin/env python
"""Sniper sleeve simulator — capital-aware portfolio-level backtest.

Takes the raw trade log from the sniper backtester and simulates a real
portfolio with constraints:
  - Max N concurrent positions (default 3)
  - Equal-weight sizing per position (e.g. 33% each)
  - Sleeve allocation as % of total portfolio (default 50%)
  - FIFO signal selection: highest score wins when capacity is full
  - Capital Guardian halt on consecutive losses

Produces:
  - Sleeve equity curve with real DD
  - 10-day campaign window analysis
  - Guardian trigger frequency
  - Promotion gate evaluation

Usage:
    python scripts/simulate_sniper_sleeve.py                          # default: 1Y trades
    python scripts/simulate_sniper_sleeve.py --trades outputs/research/backtest_sniper_v3_1Y_trades.csv
    python scripts/simulate_sniper_sleeve.py --trades outputs/research/backtest_sniper_3Y_trades.csv --window 10
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class SleeveConfig:
    """Portfolio constraints for the sniper sleeve."""
    max_positions: int = 3
    sleeve_allocation_pct: float = 50.0  # % of total portfolio allocated to sleeve
    guardian_halt_after_losses: int = 3   # halt sleeve after N consecutive losses
    guardian_cooldown_trades: int = 2     # resume after N wins post-halt


@dataclass
class SleeveResult:
    """Results from the sleeve simulation."""
    total_trades_taken: int
    total_trades_skipped: int
    win_rate: float
    avg_return_pct: float
    sleeve_total_return_pct: float
    sleeve_max_drawdown_pct: float
    sharpe: float
    profit_factor: float
    max_consecutive_losses: int
    guardian_halts: int
    time_stop_rate: float
    equity_curve: list[dict]
    campaign_windows: list[dict]


def load_trades(path: str | Path) -> pd.DataFrame:
    """Load trade log CSV from backtester."""
    df = pd.read_csv(path)
    # Ensure date columns are parsed
    for col in ["signal_date", "entry_date", "exit_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.date
    return df.sort_values("entry_date").reset_index(drop=True)


def simulate_sleeve(
    trades: pd.DataFrame,
    config: SleeveConfig | None = None,
) -> SleeveResult:
    """Simulate the sniper sleeve with position limits and guardian rules.

    Processes trades chronologically. On each entry date, if capacity is
    available, takes the highest-scoring signal. Tracks equity curve with
    proper position sizing.
    """
    config = config or SleeveConfig()

    # Position weight = sleeve allocation / max positions
    per_position_pct = config.sleeve_allocation_pct / config.max_positions

    # Group trades by entry_date (multiple signals can fire same day)
    trades_by_entry = {}
    for _, row in trades.iterrows():
        entry = row["entry_date"]
        trades_by_entry.setdefault(entry, []).append(row)

    # Sort each day's trades by score descending (best first)
    for d in trades_by_entry:
        trades_by_entry[d] = sorted(trades_by_entry[d], key=lambda r: r.get("score", 0), reverse=True)

    # State
    open_positions: list[dict] = []  # {entry_date, exit_date, pnl_pct, ticker}
    completed_trades: list[dict] = []
    equity = 100.0  # start at 100
    peak_equity = 100.0
    max_dd = 0.0
    equity_curve = [{"date": str(min(trades_by_entry.keys())), "equity": 100.0}]

    consecutive_losses = 0
    halted = False
    guardian_halts = 0
    halt_wins_since = 0
    skipped = 0

    # Process day by day
    all_dates = sorted(trades_by_entry.keys())
    if not all_dates:
        return _empty_result()

    # Build a complete date range for equity tracking
    first_date = all_dates[0]
    last_date = max(max(t["exit_date"] for t in day_trades) for day_trades in trades_by_entry.values())
    current = first_date

    while current <= last_date:
        # Close positions that exit today
        still_open = []
        for pos in open_positions:
            if pos["exit_date"] <= current:
                # Position closed
                trade_return = pos["pnl_pct"] * per_position_pct / 100.0
                equity += equity * trade_return / 100.0
                completed_trades.append(pos)

                # Update consecutive losses
                if pos["pnl_pct"] <= 0:
                    consecutive_losses += 1
                    halt_wins_since = 0
                else:
                    halt_wins_since += 1
                    consecutive_losses = 0

                # Guardian halt check
                if consecutive_losses >= config.guardian_halt_after_losses and not halted:
                    halted = True
                    guardian_halts += 1

                # Guardian resume check
                if halted and halt_wins_since >= config.guardian_cooldown_trades:
                    halted = False
                    consecutive_losses = 0
            else:
                still_open.append(pos)
        open_positions = still_open

        # Open new positions if we have signals today and capacity
        if current in trades_by_entry and not halted:
            day_signals = trades_by_entry[current]
            available_slots = config.max_positions - len(open_positions)

            for sig in day_signals[:available_slots]:
                open_positions.append({
                    "ticker": sig.get("ticker", "?"),
                    "entry_date": sig["entry_date"],
                    "exit_date": sig["exit_date"],
                    "pnl_pct": sig["pnl_pct"],
                    "score": sig.get("score", 0),
                    "exit_reason": sig.get("exit_reason", ""),
                })

            skipped += max(0, len(day_signals) - available_slots)
        elif current in trades_by_entry and halted:
            skipped += len(trades_by_entry[current])

        # Track equity
        peak_equity = max(peak_equity, equity)
        dd = (equity - peak_equity) / peak_equity * 100
        max_dd = min(max_dd, dd)

        equity_curve.append({"date": str(current), "equity": round(equity, 4)})

        # Next trading day (skip weekends)
        current += timedelta(days=1)
        while current.weekday() >= 5 and current <= last_date:
            current += timedelta(days=1)

    # Compute metrics
    taken = len(completed_trades)
    if taken == 0:
        return _empty_result()

    returns = [t["pnl_pct"] * per_position_pct / 100.0 for t in completed_trades]
    wins = sum(1 for r in returns if r > 0)
    wr = wins / taken
    avg_ret = float(np.mean(returns))

    gross_profit = sum(r for r in returns if r > 0)
    gross_loss = abs(sum(r for r in returns if r < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0.0

    time_stops = sum(1 for t in completed_trades if t.get("exit_reason") == "time_stop")
    ts_rate = time_stops / taken if taken > 0 else 0

    max_consec_l = 0
    curr_l = 0
    for r in returns:
        if r <= 0:
            curr_l += 1
            max_consec_l = max(max_consec_l, curr_l)
        else:
            curr_l = 0

    # Campaign window analysis
    campaign_windows = _compute_campaign_windows(equity_curve, window_days=10)

    sleeve_return = (equity - 100.0)

    return SleeveResult(
        total_trades_taken=taken,
        total_trades_skipped=skipped,
        win_rate=round(wr, 4),
        avg_return_pct=round(avg_ret, 4),
        sleeve_total_return_pct=round(sleeve_return, 2),
        sleeve_max_drawdown_pct=round(abs(max_dd), 2),
        sharpe=round(sharpe, 2),
        profit_factor=round(pf, 2),
        max_consecutive_losses=max_consec_l,
        guardian_halts=guardian_halts,
        time_stop_rate=round(ts_rate, 4),
        equity_curve=equity_curve,
        campaign_windows=campaign_windows,
    )


def _compute_campaign_windows(
    equity_curve: list[dict],
    window_days: int = 10,
) -> list[dict]:
    """Slice equity curve into rolling N-day campaign windows."""
    if len(equity_curve) < 2:
        return []

    df = pd.DataFrame(equity_curve)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    windows = []
    dates = df.index.tolist()

    i = 0
    while i < len(dates) - 1:
        start = dates[i]
        end = start + timedelta(days=window_days)

        # Find closest date to window end
        window_data = df.loc[start:end]
        if len(window_data) < 2:
            i += 1
            continue

        start_eq = float(window_data["equity"].iloc[0])
        end_eq = float(window_data["equity"].iloc[-1])
        peak = float(window_data["equity"].max())
        trough = float(window_data["equity"].min())

        window_return = (end_eq - start_eq) / start_eq * 100 if start_eq > 0 else 0
        window_dd = (trough - peak) / peak * 100 if peak > 0 else 0

        windows.append({
            "start": str(start.date()),
            "end": str(window_data.index[-1].date()),
            "return_pct": round(window_return, 2),
            "max_dd_pct": round(abs(window_dd), 2),
            "hit_7pct": window_return >= 7.0,
            "hit_5pct": window_return >= 5.0,
            "positive": window_return > 0,
        })

        # Step forward by window_days (non-overlapping)
        i += max(1, len(window_data) - 1)

    return windows


def _empty_result() -> SleeveResult:
    return SleeveResult(
        total_trades_taken=0, total_trades_skipped=0,
        win_rate=0, avg_return_pct=0, sleeve_total_return_pct=0,
        sleeve_max_drawdown_pct=0, sharpe=0, profit_factor=0,
        max_consecutive_losses=0, guardian_halts=0, time_stop_rate=0,
        equity_curve=[], campaign_windows=[],
    )


def print_report(
    result: SleeveResult,
    config: SleeveConfig,
    raw_metrics: dict | None = None,
) -> None:
    """Print formatted sleeve simulation report with hybrid promotion gates."""
    print(f"\n{'='*60}")
    print(f"  SNIPER SLEEVE SIMULATION")
    print(f"{'='*60}")
    print(f"  Sleeve allocation:   {config.sleeve_allocation_pct}% of portfolio")
    print(f"  Max positions:       {config.max_positions}")
    print(f"  Per-position weight: {config.sleeve_allocation_pct / config.max_positions:.1f}%")
    print(f"  Guardian halt after: {config.guardian_halt_after_losses} consecutive losses")
    print()
    print(f"  Trades taken:        {result.total_trades_taken}")
    print(f"  Trades skipped:      {result.total_trades_skipped} (capacity full or halted)")
    print(f"  Win rate:            {result.win_rate:.1%}")
    print(f"  Avg return (sized):  {result.avg_return_pct:+.4f}%")
    print(f"  Sleeve total return: {result.sleeve_total_return_pct:+.2f}%")
    print(f"  Sleeve max DD:       {result.sleeve_max_drawdown_pct:.2f}%")
    print(f"  Sharpe:              {result.sharpe:.2f}")
    print(f"  Profit factor:       {result.profit_factor:.2f}")
    print(f"  Max consec losses:   {result.max_consecutive_losses}")
    print(f"  Guardian halts:      {result.guardian_halts}")
    print(f"  Time-stop rate:      {result.time_stop_rate:.1%}")

    # Campaign windows
    if result.campaign_windows:
        positive = sum(1 for w in result.campaign_windows if w["positive"])
        hit_5 = sum(1 for w in result.campaign_windows if w["hit_5pct"])
        hit_7 = sum(1 for w in result.campaign_windows if w["hit_7pct"])
        total_w = len(result.campaign_windows)
        avg_w_ret = float(np.mean([w["return_pct"] for w in result.campaign_windows]))
        avg_w_dd = float(np.mean([w["max_dd_pct"] for w in result.campaign_windows]))

        # Consecutive positive windows
        max_consec_pos = 0
        curr_pos = 0
        for w in result.campaign_windows:
            if w["positive"]:
                curr_pos += 1
                max_consec_pos = max(max_consec_pos, curr_pos)
            else:
                curr_pos = 0

        print(f"\n  Campaign Windows (10-day):")
        print(f"    Total windows:     {total_w}")
        print(f"    Positive:          {positive}/{total_w} ({positive/total_w:.0%})")
        print(f"    Hit 5%+:           {hit_5}/{total_w} ({hit_5/total_w:.0%})")
        print(f"    Hit 7%+:           {hit_7}/{total_w} ({hit_7/total_w:.0%})")
        print(f"    Avg window return: {avg_w_ret:+.2f}%")
        print(f"    Avg window DD:     {avg_w_dd:.2f}%")
        print(f"    Max consec +ve:    {max_consec_pos}")

    # Hybrid Promotion Gates (raw signal quality + sleeve portfolio reality)
    print(f"\n  Promotion Gates:")

    # Layer 1: Raw signal quality gates (from backtest metrics, not sized)
    print(f"    --- Layer 1: Signal Quality (raw trades) ---")
    raw_gates = {}
    if raw_metrics:
        raw_gates = {
            "raw WR >= 50%": (raw_metrics.get("win_rate", 0) >= 0.50, raw_metrics.get("win_rate", 0)),
            "raw avg_return >= 2.0%": (raw_metrics.get("avg_return_pct", 0) >= 2.0, raw_metrics.get("avg_return_pct", 0)),
            "raw PF >= 1.5": (raw_metrics.get("profit_factor", 0) >= 1.5, raw_metrics.get("profit_factor", 0)),
            "raw Sharpe >= 0.5": (raw_metrics.get("sharpe_ratio", 0) >= 0.5, raw_metrics.get("sharpe_ratio", 0)),
        }
    else:
        print(f"    (no raw metrics provided — skipping signal gates)")

    # Layer 2: Sleeve portfolio gates (capital-aware)
    sleeve_gates = {
        "sleeve WR >= 50%": (result.win_rate >= 0.50, result.win_rate),
        "sleeve PF >= 1.5": (result.profit_factor >= 1.5, result.profit_factor),
        "sleeve DD <= 15%": (result.sleeve_max_drawdown_pct <= 15.0, result.sleeve_max_drawdown_pct),
        "sleeve Sharpe >= 1.0": (result.sharpe >= 1.0, result.sharpe),
        "time_stop_rate < 40%": (result.time_stop_rate < 0.40, result.time_stop_rate),
    }

    if result.campaign_windows:
        max_cp = 0
        cp = 0
        for w in result.campaign_windows:
            if w["positive"]:
                cp += 1
                max_cp = max(max_cp, cp)
            else:
                cp = 0
        sleeve_gates["2+ consec positive windows"] = (max_cp >= 2, max_cp)

    all_pass = True
    for name, (passed, value) in raw_gates.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"    {name:35s}: {status}  ({value:.4f})")

    if raw_gates:
        print(f"    --- Layer 2: Portfolio Reality (sized sleeve) ---")

    for name, (passed, value) in sleeve_gates.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"    {name:35s}: {status}  ({value:.4f})")

    if all_pass:
        print(f"\n  PROMOTION ELIGIBLE")
    else:
        print(f"\n  NOT READY FOR PROMOTION")


def _load_raw_metrics(trades_path: str) -> dict | None:
    """Try to load raw backtest metrics from companion JSON file."""
    # Convention: trades file is backtest_sniper_*_trades.csv
    # Metrics file is sniper_v3_*.json or sniper_*.json
    trades_p = Path(trades_path)
    parent = trades_p.parent

    # Try common naming patterns
    for pattern in ["sniper_v3_*.json", "sniper_*.json"]:
        candidates = sorted(parent.glob(pattern))
        for c in candidates:
            if "sleeve" in c.name or "baseline" in c.name:
                continue
            try:
                data = json.loads(c.read_text())
                if "metrics" in data:
                    print(f"  Loaded raw metrics from {c.name}")
                    return data["metrics"]
            except (json.JSONDecodeError, KeyError):
                continue

    return None


def main():
    parser = argparse.ArgumentParser(description="Sniper sleeve simulator")
    parser.add_argument("--trades", type=str,
                        default="outputs/research/backtest_sniper_v3_1Y_trades.csv",
                        help="Path to trade log CSV")
    parser.add_argument("--raw-metrics", type=str, default=None,
                        help="Path to raw backtest JSON (auto-detected if omitted)")
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--sleeve-pct", type=float, default=50.0,
                        help="Sleeve allocation as %% of total portfolio")
    parser.add_argument("--halt-after", type=int, default=3,
                        help="Guardian halt after N consecutive losses")
    parser.add_argument("--window", type=int, default=10,
                        help="Campaign window size in days")
    args = parser.parse_args()

    config = SleeveConfig(
        max_positions=args.max_positions,
        sleeve_allocation_pct=args.sleeve_pct,
        guardian_halt_after_losses=args.halt_after,
    )

    # Load raw metrics for hybrid gates
    raw_metrics = None
    if args.raw_metrics:
        data = json.loads(Path(args.raw_metrics).read_text())
        raw_metrics = data.get("metrics", data)
    else:
        raw_metrics = _load_raw_metrics(args.trades)

    trades = load_trades(args.trades)
    print(f"Loaded {len(trades)} trades from {args.trades}")

    result = simulate_sleeve(trades, config)
    print_report(result, config, raw_metrics=raw_metrics)

    # Save results
    out_path = Path(args.trades).stem + "_sleeve.json"
    out_file = Path("outputs/research") / out_path
    out_data = {
        "config": {
            "max_positions": config.max_positions,
            "sleeve_allocation_pct": config.sleeve_allocation_pct,
            "guardian_halt_after_losses": config.guardian_halt_after_losses,
        },
        "total_trades_taken": result.total_trades_taken,
        "total_trades_skipped": result.total_trades_skipped,
        "win_rate": result.win_rate,
        "avg_return_pct": result.avg_return_pct,
        "sleeve_total_return_pct": result.sleeve_total_return_pct,
        "sleeve_max_drawdown_pct": result.sleeve_max_drawdown_pct,
        "sharpe": result.sharpe,
        "profit_factor": result.profit_factor,
        "max_consecutive_losses": result.max_consecutive_losses,
        "guardian_halts": result.guardian_halts,
        "time_stop_rate": result.time_stop_rate,
        "campaign_windows": result.campaign_windows,
        "raw_metrics": raw_metrics,
    }
    out_file.write_text(json.dumps(out_data, indent=2))
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
