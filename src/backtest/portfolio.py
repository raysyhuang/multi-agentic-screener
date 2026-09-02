"""Portfolio-level equity replay — the book vs each stream alone.

Turns a list of closed trades into a real, concurrency-capped account equity
curve so we can talk in the terms a desk actually uses — compounded return,
max drawdown, Sharpe — instead of a per-trade sum. The point of combining
weakly-correlated streams (sniper + MR) is that the *book's* risk-adjusted
return exceeds either stream alone; this module is what measures that.

Event-driven, equal-weight slots. Previously open positions exit before new
entries so freed capital redeploys immediately; a new position's own same-day
exit is processed after its entry. Cash earns 0%; open positions are held at
cost (we only know entry/exit, not daily marks), so equity is sampled at exit
events — standard for a trade-list backtest. This is the Dashboard's production
accounting engine. The research CLI ``scripts/sniper_equity_curve.py`` retains a
separate event loop and must not be treated as evidence-equivalent unless its
ordering is reconciled independently.

Sharpe is computed on a calendar-day forward-filled equity curve and annualised
at sqrt(252). At small trade counts it is DIRECTIONAL only: a stream that rarely
deploys leaves capital idle, flattening vol and inflating Sharpe — trust the
ranking across configs more than the absolute number, and never annualise CAGR
off a sub-year window.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from statistics import mean, pstdev

TRADING_DAYS = 252


@dataclass
class BookTrade:
    entry: date
    exit: date
    pnl_pct: float


def _sharpe(equity_curve: list[tuple[date, float]]) -> float | None:
    """Annualised Sharpe from a calendar-day forward-filled equity curve.
    None if the curve is too short or has zero variance."""
    if len(equity_curve) < 3:
        return None
    # Sort by date only. Python's sort is stable, so same-day points retain
    # execution order and the last one remains that day's close.
    pts = sorted(equity_curve, key=lambda point: point[0])
    start, end = pts[0][0], pts[-1][0]
    span = (end - start).days
    if span < 2:
        return None
    daily: list[float] = []
    i, cur = 0, pts[0][1]
    for k in range(span + 1):
        day = date.fromordinal(start.toordinal() + k)
        while i < len(pts) and pts[i][0] <= day:
            cur = pts[i][1]
            i += 1
        daily.append(cur)
    rets = [daily[j] / daily[j - 1] - 1.0 for j in range(1, len(daily)) if daily[j - 1]]
    if len(rets) < 2 or pstdev(rets) == 0:
        return None
    return (mean(rets) / pstdev(rets)) * (TRADING_DAYS ** 0.5)


def simulate_book(
    trades: list[BookTrade],
    *,
    max_concurrent: int = 10,
    start_capital: float = 100_000.0,
) -> dict:
    """Replay ``trades`` as one equal-slots account with a concurrency cap.

    Returns compounded metrics plus the exit-sampled equity curve (a list of
    ``(date, equity)`` points, seeded at ``start_capital``). ``skipped`` counts
    signals dropped because every slot was full — a real account cannot take
    every trade a per-trade sum implicitly assumes.
    """
    trades = [t for t in trades if t.entry and t.exit and t.pnl_pct is not None]
    invalid = next((t for t in trades if t.exit < t.entry), None)
    if invalid is not None:
        raise ValueError(
            "portfolio trade has exit before entry "
            f"({invalid.entry.isoformat()} > {invalid.exit.isoformat()})"
        )
    if not trades:
        return {
            "taken": 0, "skipped": 0, "peak_concurrent": 0, "multiple": 1.0,
            "total_return_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe": None,
            "equity_curve": [],
        }
    # Preserve the caller's frozen order among same-day entries. Sorting by exit
    # would use future holding duration to decide which signal gets a scarce slot.
    trades = sorted(trades, key=lambda t: t.entry)

    # Event order within a day:
    #   0. exits for positions opened on an earlier day (free capital),
    #   1. new entries,
    #   2. exits for positions opened that same day.
    # A blanket exits-before-entries rule silently drops same-day round trips:
    # their exit is visited before the position exists and is never retried.
    events: list[tuple[date, int, int, BookTrade]] = []
    for trade_id, t in enumerate(trades):
        exit_kind = 2 if t.entry == t.exit else 0
        events.append((t.exit, exit_kind, trade_id, t))
        events.append((t.entry, 1, trade_id, t))
    events.sort(key=lambda e: (e[0], e[1], e[2]))

    cash = start_capital
    open_positions: dict[int, float] = {}
    taken = skipped = peak_concurrent = 0
    equity_curve: list[tuple[date, float]] = [(events[0][0], start_capital)]

    def equity() -> float:
        return cash + sum(open_positions.values())

    for ev_date, kind, trade_id, t in events:
        if kind in (0, 2):  # prior-day exit or same-day round-trip exit
            if trade_id in open_positions:
                notional = open_positions.pop(trade_id)
                cash += notional * (1.0 + t.pnl_pct / 100.0)
                equity_curve.append((ev_date, equity()))
        else:  # entry
            if len(open_positions) >= max_concurrent:
                skipped += 1
                continue
            notional = min(equity() / max_concurrent, cash)
            if notional <= 0:
                skipped += 1
                continue
            cash -= notional
            open_positions[trade_id] = notional
            taken += 1
            peak_concurrent = max(peak_concurrent, len(open_positions))

    # Every accepted input is a closed trade. Fail loudly rather than publish a
    # portfolio card if an event-ordering regression strands one at cost again.
    if open_positions or len(equity_curve) != taken + 1:
        raise RuntimeError(
            "portfolio accounting did not close every taken trade "
            f"(taken={taken}, exits={len(equity_curve) - 1}, open={len(open_positions)})"
        )

    final_eq = equity()
    peak = -float("inf")
    max_dd = 0.0
    for _, eq in equity_curve:
        peak = max(peak, eq)
        if peak > 0:
            max_dd = max(max_dd, (peak - eq) / peak)

    multiple = final_eq / start_capital
    return {
        "taken": taken,
        "skipped": skipped,
        "peak_concurrent": peak_concurrent,
        "multiple": multiple,
        "total_return_pct": (multiple - 1.0) * 100.0,
        "max_drawdown_pct": max_dd * 100.0,
        "sharpe": _sharpe(equity_curve),
        "equity_curve": equity_curve,
    }


def exit_day_overlap(streams: dict[str, list]) -> dict:
    """Count distinct exit-days per stream and the pairwise shared-day count.

    Low overlap between two streams is consistent with diversification (they
    seldom fire together), which is the mechanism behind a book Sharpe that
    exceeds either stream alone. ``streams`` maps a label to a list of trade
    dicts carrying ``exit_date`` (ISO string or None)."""
    days = {k: {r["exit_date"] for r in v if r.get("exit_date")} for k, v in streams.items()}
    out = {f"{k}_exit_days": len(v) for k, v in days.items()}
    labels = list(days)
    if len(labels) == 2:
        out["shared_days"] = len(days[labels[0]] & days[labels[1]])
    return out
