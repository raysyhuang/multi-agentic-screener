"""Convert a sniper trade list into a real account equity curve.

The frozen sniper baselines report `total_return_pct` as a SUM of per-trade
returns across non-concurrency-adjusted trades — that is not an account
multiple and cannot be compared to an equity-curve number (e.g. Dragon Pulse
"+26% 3Y"). This script replays the trade list as an event-driven portfolio
with a real position-sizing rule and a max-concurrent-positions cap, so we get
an actual compounded account multiple, CAGR, and equity-based max drawdown.

Sizing modes:
  fixed_fraction  each new position = min(fraction * equity, available cash)
  equal_slots     each new position = equity / max_concurrent (capped by cash)

Capital model: cash earns 0%. Open positions are held at cost (we only know
entry/exit, not daily marks), so the equity curve is sampled at exit events —
standard for trade-list backtests. Concurrency saturation (signals dropped
because all slots were full) is reported explicitly: a real account can't take
every signal a per-trade sum implicitly assumes.

Usage:
  python scripts/sniper_equity_curve.py <trades.csv> \
      [--mode fixed_fraction|equal_slots] [--fraction 0.2] \
      [--max-concurrent 10] [--start-capital 100000] \
      [--wr-haircut 0.69]  # optional: rescale to a target live win rate
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class Trade:
    ticker: str
    entry: date
    exit: date
    pnl_pct: float
    regime: str
    score: float = 0.0


@dataclass
class Result:
    start_capital: float
    final_equity: float
    multiple: float
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    taken: int
    skipped_saturation: int
    years: float
    peak_concurrent: int
    equity_curve: list[tuple[date, float]] = field(default_factory=list)


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def load_trades(path: str) -> list[Trade]:
    trades: list[Trade] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if not row.get("entry_date") or not row.get("exit_date"):
                continue
            trades.append(
                Trade(
                    ticker=row["ticker"],
                    entry=_parse_date(row["entry_date"]),
                    exit=_parse_date(row["exit_date"]),
                    pnl_pct=float(row["pnl_pct"]),
                    regime=row.get("regime", ""),
                    score=float(row.get("score") or 0.0),
                )
            )
    return trades


def throttle_per_day(trades: list[Trade], max_per_day: int) -> list[Trade]:
    """Keep at most `max_per_day` entries per entry_date, ranked by score desc
    (Neo's 'max N new Sniper/day' rule). Mirrors a manual operator who can only
    action a handful of fresh signals a day."""
    by_day: dict[date, list[Trade]] = {}
    for t in trades:
        by_day.setdefault(t.entry, []).append(t)
    kept: list[Trade] = []
    for day, day_trades in by_day.items():
        day_trades.sort(key=lambda t: t.score, reverse=True)
        kept.extend(day_trades[:max_per_day])
    return kept


def simulate(
    trades: list[Trade],
    *,
    mode: str = "fixed_fraction",
    fraction: float = 0.20,
    max_concurrent: int = 10,
    start_capital: float = 100_000.0,
    wr_haircut: float | None = None,
    max_per_day: int | None = None,
) -> Result:
    """Event-driven replay. Exits processed before entries on the same day so
    freed capital can be redeployed immediately."""
    if max_per_day is not None:
        trades = throttle_per_day(trades, max_per_day)

    trades = sorted(trades, key=lambda t: (t.entry, t.exit))

    if wr_haircut is not None:
        trades = _apply_wr_haircut(trades, wr_haircut)

    # Event list: (date, kind, trade). kind 0 = exit, 1 = entry → exits first.
    events: list[tuple[date, int, Trade]] = []
    for t in trades:
        events.append((t.exit, 0, t))
        events.append((t.entry, 1, t))
    events.sort(key=lambda e: (e[0], e[1]))

    cash = start_capital
    # open positions: trade -> notional deployed at entry
    open_positions: dict[int, float] = {}
    taken = 0
    skipped = 0
    peak_concurrent = 0
    equity_curve: list[tuple[date, float]] = [(events[0][0], start_capital)]

    def equity() -> float:
        return cash + sum(open_positions.values())

    for ev_date, kind, t in events:
        tid = id(t)
        if kind == 0:  # exit
            if tid in open_positions:
                notional = open_positions.pop(tid)
                cash += notional * (1.0 + t.pnl_pct / 100.0)
                equity_curve.append((ev_date, equity()))
        else:  # entry
            if len(open_positions) >= max_concurrent:
                skipped += 1
                continue
            eq = equity()
            if mode == "equal_slots":
                target = eq / max_concurrent
            else:  # fixed_fraction
                target = eq * fraction
            notional = min(target, cash)
            if notional <= 0:
                skipped += 1
                continue
            cash -= notional
            open_positions[tid] = notional
            taken += 1
            peak_concurrent = max(peak_concurrent, len(open_positions))

    final_eq = equity()
    # Max drawdown on the realized (exit-sampled) equity curve.
    peak = -float("inf")
    max_dd = 0.0
    for _, eq in equity_curve:
        peak = max(peak, eq)
        if peak > 0:
            max_dd = max(max_dd, (peak - eq) / peak)

    first_day = min(t.entry for t in trades)
    last_day = max(t.exit for t in trades)
    years = (last_day - first_day).days / 365.25
    multiple = final_eq / start_capital
    cagr = (multiple ** (1.0 / years) - 1.0) * 100.0 if years > 0 else 0.0

    return Result(
        start_capital=start_capital,
        final_equity=final_eq,
        multiple=multiple,
        total_return_pct=(multiple - 1.0) * 100.0,
        cagr_pct=cagr,
        max_drawdown_pct=max_dd * 100.0,
        taken=taken,
        skipped_saturation=skipped,
        years=years,
        peak_concurrent=peak_concurrent,
        equity_curve=equity_curve,
    )


def _apply_wr_haircut(trades: list[Trade], target_wr: float) -> list[Trade]:
    """Rescale outcomes toward a target win rate by flipping the worst-MFE
    wins into losses is overkill; simpler honest proxy: deterministically
    convert the smallest-margin wins into break-even/small losses until the
    realized win rate matches target. Returns a new list (order preserved)."""
    wins = [t for t in trades if t.pnl_pct > 0]
    cur_wr = len(wins) / len(trades) if trades else 0.0
    if target_wr >= cur_wr:
        return trades
    # number of wins to demote
    n_target_wins = int(round(target_wr * len(trades)))
    n_demote = len(wins) - n_target_wins
    if n_demote <= 0:
        return trades
    # demote the smallest wins (closest to break-even) — most marginal edge
    win_sorted = sorted(wins, key=lambda t: t.pnl_pct)
    demote = set(id(t) for t in win_sorted[:n_demote])
    out: list[Trade] = []
    for t in trades:
        if id(t) in demote:
            # marginal win becomes a typical loss (use median loss size)
            out.append(Trade(t.ticker, t.entry, t.exit, -abs(t.pnl_pct), t.regime))
        else:
            out.append(t)
    return out


def _fmt(r: Result, label: str) -> str:
    return (
        f"{label}\n"
        f"  signals taken / dropped(saturation) : {r.taken} / {r.skipped_saturation}\n"
        f"  peak concurrent positions           : {r.peak_concurrent}\n"
        f"  window (years)                      : {r.years:.2f}\n"
        f"  start capital                       : ${r.start_capital:,.0f}\n"
        f"  final equity                        : ${r.final_equity:,.0f}\n"
        f"  account multiple                    : {r.multiple:.2f}x\n"
        f"  TOTAL RETURN (equity)               : {r.total_return_pct:+.1f}%\n"
        f"  CAGR                                : {r.cagr_pct:+.1f}%\n"
        f"  max drawdown (equity)               : {r.max_drawdown_pct:.1f}%\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="trade list CSV")
    ap.add_argument("--mode", default="fixed_fraction",
                    choices=["fixed_fraction", "equal_slots"])
    ap.add_argument("--fraction", type=float, default=0.20)
    ap.add_argument("--max-concurrent", type=int, default=10)
    ap.add_argument("--start-capital", type=float, default=100_000.0)
    ap.add_argument("--wr-haircut", type=float, default=None,
                    help="target live win rate, e.g. 0.69; demotes marginal wins")
    ap.add_argument("--max-per-day", type=int, default=None,
                    help="max new entries/day, ranked by score (Neo's 1/day rule)")
    ap.add_argument("--grid", action="store_true",
                    help="run a scenario grid instead of a single config")
    args = ap.parse_args()

    trades = load_trades(args.csv)
    print(f"loaded {len(trades)} trades from {args.csv}\n")

    if args.grid:
        scenarios = [
            ("equal_slots  N=20 (5%/pos)",   dict(mode="equal_slots", max_concurrent=20)),
            ("equal_slots  N=10 (10%/pos)",  dict(mode="equal_slots", max_concurrent=10)),
            ("equal_slots  N=5  (20%/pos)",  dict(mode="equal_slots", max_concurrent=5)),
            ("fixed 20%/pos, cap 8",         dict(mode="fixed_fraction", fraction=0.20, max_concurrent=8)),
            ("fixed 35%/pos, cap 5 (live-ish)", dict(mode="fixed_fraction", fraction=0.35, max_concurrent=5)),
            ("fixed 50%/pos, cap 4 (aggressive)", dict(mode="fixed_fraction", fraction=0.50, max_concurrent=4)),
        ]
        for label, kw in scenarios:
            r = simulate(trades, start_capital=args.start_capital, **kw)
            print(_fmt(r, label))
        if args.wr_haircut:
            print(f"=== same grid, win rate haircut to {args.wr_haircut:.0%} (live proxy) ===\n")
            for label, kw in scenarios:
                r = simulate(trades, start_capital=args.start_capital,
                             wr_haircut=args.wr_haircut, **kw)
                print(_fmt(r, label + f" [WR→{args.wr_haircut:.0%}]"))
    else:
        r = simulate(
            trades,
            mode=args.mode,
            fraction=args.fraction,
            max_concurrent=args.max_concurrent,
            start_capital=args.start_capital,
            wr_haircut=args.wr_haircut,
            max_per_day=args.max_per_day,
        )
        cfg = (f"mode={args.mode} fraction={args.fraction} "
               f"max_concurrent={args.max_concurrent}"
               + (f" max_per_day={args.max_per_day}" if args.max_per_day else "")
               + (f" wr_haircut={args.wr_haircut}" if args.wr_haircut else ""))
        print(_fmt(r, cfg))


if __name__ == "__main__":
    main()
