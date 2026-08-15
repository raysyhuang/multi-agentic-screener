"""PEAD concurrency + entry-limit sweep — what caps keep the book size sane?

The PEAD paper stream has 20-day untrailed holds (since PR #43). With only
same-ticker dedup for concurrency control, earnings-season clustering stacked
13 open positions on 2026-08-06 against a "cap" of 5 (which is actually a
per-run pick quota). Now that `pead_max_concurrent` landed, this answers what
the concurrency cap should be AND whether an entries-per-week pre-filter is
necessary to keep PEAD from becoming the book's largest exposure.

Questions:
1. Peak concurrent open PEAD positions when uncapped (20-day holds)?
2. At open-slot caps {3, 5, 8, 10, uncapped}, what are taken/skipped/peak/return/maxDD?
   (10 is the live pead_max_concurrent on main; include for ranking vs shipping default.)
3. Crossed grid (§2.6): max-entries-per-week {2, 3, 5} × sector cap {None, 2} ×
   slot caps {3, 5, 8, 10, uncapped}. Weekly/sector are PRE-FILTERS on the trade
   list, then replayed through capped-equity sim at each slot cap.
4. Pre-registered kill note: if a cap would exclude >30% of the backtest cohort,
   write that the 30-trade promotion clock restarts under that cap (§0.2).

Relative arms only (same lesson as sniper_pick_count): the comparison is
capital-aware compounded return, not summed per-trade P&L. Absolute returns
depend on the backtest universe, cost assumption, and time window. The RANKING
between cap configs on a fixed population is what matters.

Usage:
    python scripts/pead_slot_sweep.py [--cohort PATH]
    # Exits 0 with a message if cohort is missing; tests lock mechanism only.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.portfolio import BookTrade, simulate_book


def load_cohort(path: Path) -> list[dict]:
    """Load PEAD trade cohort. Expected columns: ticker, entry_date, exit_date,
    pnl_pct, and optionally sector, signal_date."""
    import pandas as pd
    df = pd.read_csv(path)
    required = {"ticker", "entry_date", "exit_date", "pnl_pct"}
    if not required.issubset(df.columns):
        raise ValueError(f"cohort missing required columns: {required - set(df.columns)}")
    df = df.sort_values("entry_date").reset_index(drop=True)
    return df.to_dict("records")


def to_book_trades(records: list[dict]) -> list[BookTrade]:
    """Convert cohort records to BookTrade list."""
    out = []
    for r in records:
        try:
            out.append(BookTrade(
                entry=date.fromisoformat(str(r["entry_date"])[:10]),
                exit=date.fromisoformat(str(r["exit_date"])[:10]),
                pnl_pct=float(r["pnl_pct"]),
            ))
        except (ValueError, TypeError, KeyError):
            continue
    return out


def filter_weekly_entry_limit(records: list[dict], max_per_week: int) -> list[dict]:
    """Pre-filter: admit at most `max_per_week` entries per 7-day rolling window.
    A trade list pre-filter, not a portfolio sim change. Earliest entries win."""
    if max_per_week is None or max_per_week <= 0:
        return records
    out = []
    window: list[date] = []
    for r in records:
        try:
            entry = date.fromisoformat(str(r["entry_date"])[:10])
        except (ValueError, TypeError):
            continue
        # Slide the 7-day window: drop entries older than 7 days before `entry`.
        cutoff = entry - timedelta(days=7)
        window = [d for d in window if d > cutoff]
        if len(window) < max_per_week:
            window.append(entry)
            out.append(r)
    return out


def filter_sector_cap(records: list[dict], max_per_sector: int) -> tuple[list[dict], bool]:
    """Pre-filter: at most `max_per_sector` open positions per sector concurrently.
    Returns (filtered_records, sector_available). If sector column is missing or
    empty for >50% of records, returns (records, False) — do not fabricate sectors."""
    if max_per_sector is None or max_per_sector <= 0:
        return records, False
    if "sector" not in records[0] if records else True:
        return records, False
    # Check availability: >50% must have a non-null sector.
    valid = sum(1 for r in records if r.get("sector") and str(r["sector"]).strip())
    if valid < len(records) * 0.5:
        return records, False

    # Event-driven sim: track open positions by sector.
    # Build event list: (date, kind, record). kind 0=exit, 1=entry.
    events = []
    for r in records:
        try:
            entry = date.fromisoformat(str(r["entry_date"])[:10])
            exit_d = date.fromisoformat(str(r["exit_date"])[:10])
        except (ValueError, TypeError):
            continue
        events.append((exit_d, 0, r))
        events.append((entry, 1, r))
    events.sort(key=lambda e: (e[0], e[1]))

    open_by_sector: dict[str, list[int]] = {}
    out = []
    for _, kind, r in events:
        tid = id(r)
        sector = str(r.get("sector", "")).strip() or "UNKNOWN"
        if kind == 0:  # exit
            if sector in open_by_sector and tid in open_by_sector[sector]:
                open_by_sector[sector].remove(tid)
        else:  # entry
            if sector not in open_by_sector:
                open_by_sector[sector] = []
            if len(open_by_sector[sector]) < max_per_sector:
                open_by_sector[sector].append(tid)
                out.append(r)
    return out, True


def compute_metrics(trades: list[BookTrade], max_concurrent: int | None) -> dict:
    """Run simulate_book and return taken/skipped/peak/return/maxDD/sharpe."""
    if max_concurrent is None:
        max_concurrent = 999
    res = simulate_book(trades, max_concurrent=max_concurrent)
    return {
        "taken": res["taken"],
        "skipped": res["skipped"],
        "peak_concurrent": res["peak_concurrent"],
        "return_pct": res["total_return_pct"],
        "max_dd_pct": res["max_drawdown_pct"],
        "sharpe": res["sharpe"],
    }


def print_row(label: str, total: int, m: dict) -> None:
    """Format a single result row."""
    sharpe = "–" if m["sharpe"] is None else f"{m['sharpe']:.2f}"
    print(f"{label:<25}{total:>7}{m['taken']:>7}{m['skipped']:>9}{m['peak_concurrent']:>6}"
          f"{m['return_pct']:>+10.1f}%{m['max_dd_pct']:>9.1f}%{sharpe:>8}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="outputs/research/pead_trades_cohort.csv",
                    help="PEAD trade list (ticker, entry_date, exit_date, pnl_pct, sector)")
    args = ap.parse_args()

    cohort_path = Path(args.cohort)
    if not cohort_path.exists():
        print(f"cohort file not found: {cohort_path}")
        print("Synthetic unit tests still lock the mechanism (see tests/test_pead_slot_sweep.py).")
        print("Full cohort sweep requires a PEAD trade cache in this checkout.")
        sys.exit(0)

    records = load_cohort(cohort_path)
    print(f"cohort: {len(records)} trades\n")

    hdr = (f"{'config':<25}{'total':>7}{'taken':>7}{'skipped':>9}{'peak':>6}"
           f"{'return':>10}{'maxDD':>9}{'Sharpe':>8}")

    # 1. Baseline: uncapped (measure peak concurrent).
    print("=" * 95)
    print("BASELINE — uncapped (measures peak concurrent open positions)")
    print("=" * 95)
    print(hdr)
    print("-" * 95)
    m_uncap = compute_metrics(to_book_trades(records), max_concurrent=None)
    print_row("uncapped", len(records), m_uncap)
    print()

    # 2. Open-slot cap sweep (baseline: no pre-filters).
    print("=" * 95)
    print("OPEN-SLOT CAP SWEEP — {3, 5, 8, 10, uncapped} (no pre-filters)")
    print("=" * 95)
    print(hdr)
    print("-" * 95)
    caps = [3, 5, 8, 10, None]
    results_cap = {}
    for cap in caps:
        label = "uncapped" if cap is None else f"cap={cap}"
        m = compute_metrics(to_book_trades(records), max_concurrent=cap)
        print_row(label, len(records), m)
        results_cap[cap] = m
    print()

    # Check 30% exclusion threshold (pre-registered kill note from §0.2).
    for cap in [3, 5, 8, 10]:
        m = results_cap[cap]
        exclusion_pct = (m["skipped"] / len(records)) * 100.0 if len(records) else 0
        if exclusion_pct > 30.0:
            print(f"⚠  cap={cap} excluded {exclusion_pct:.1f}% of cohort (>30% threshold)")
            print("   → would restart PEAD promotion clock; do NOT ship this cap")
    print()

    # 3. CROSSED GRID: weekly × sector × slot caps (§2.6 request).
    # Check if sector data is available once.
    _, sector_ok = filter_sector_cap(records, max_per_sector=2)
    
    print("=" * 95)
    print("CROSSED GRID — weekly {2,3,5} × sector × slot caps {3,5,8,10,uncapped} (§2.6)")
    print("=" * 95)
    
    # Grid: weekly limits (None = no limit), sector (if available), slot caps.
    weekly_limits = [2, 3, 5]
    sector_caps = [2] if sector_ok else []
    slot_caps = [3, 5, 8, 10, None]
    
    # For each weekly limit (with and without sector cap if available):
    for weekly in weekly_limits:
        filtered_weekly = filter_weekly_entry_limit(records, weekly)
        weekly_exclusion = ((len(records) - len(filtered_weekly)) / len(records)) * 100.0 if len(records) else 0
        
        # Weekly only (no sector cap).
        print(f"\nWeekly limit = {weekly} (no sector cap)")
        print(f"  pre-filter: {len(filtered_weekly)}/{len(records)} trades "
              f"({weekly_exclusion:.1f}% excluded)")
        print(hdr)
        print("-" * 95)
        for cap in slot_caps:
            label = f"  slot={'∞' if cap is None else cap}"
            m = compute_metrics(to_book_trades(filtered_weekly), max_concurrent=cap)
            print_row(label, len(filtered_weekly), m)
        
        # Weekly + sector cap (if sector available).
        if sector_ok:
            filtered_weekly_sector, _ = filter_sector_cap(filtered_weekly, max_per_sector=2)
            combined_exclusion = ((len(records) - len(filtered_weekly_sector)) / len(records)) * 100.0 if len(records) else 0
            print(f"\nWeekly limit = {weekly} + sector cap = 2")
            print(f"  pre-filter: {len(filtered_weekly_sector)}/{len(records)} trades "
                  f"({combined_exclusion:.1f}% excluded)")
            print(hdr)
            print("-" * 95)
            for cap in slot_caps:
                label = f"  slot={'∞' if cap is None else cap}"
                m = compute_metrics(to_book_trades(filtered_weekly_sector), max_concurrent=cap)
                print_row(label, len(filtered_weekly_sector), m)
    
    if not sector_ok:
        print("\n(sector cap grid skipped: sector column not available or >50% null)")
    print()

    # 5. Summary verdict.
    print("=" * 95)
    print("VERDICT INPUT (compounded return vs summed per-trade P&L)")
    print("=" * 95)
    print("Report taken/skipped, peak concurrent, compounded return, maxDD.")
    print("Win rate is NOT the result (same lesson as sniper_pick_count).")
    print("Summed per-trade P&L is not a portfolio result.")


if __name__ == "__main__":
    main()
