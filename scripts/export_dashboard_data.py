"""Export the MAS dashboard data snapshot from the DB to JSON.

Runs as the final step of the GitHub Actions pipeline jobs (where DATABASE_URL
points at the production Postgres) and writes a single self-contained
`data.json` consumed by the static dashboard (dashboard/). No secrets reach the
page — this bakes a read-only snapshot.

Streams are ALWAYS kept separate (official vs manual sleeve — never blended;
see CLAUDE.md). Baseline expectation bands are the honest, truth-matrix /
reconciliation numbers, not the retired optimistic labels.

Usage:
  python scripts/export_dashboard_data.py [--out dashboard/data.json] [--days 90]
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date, datetime, timedelta, timezone

import bisect
from datetime import date as _date

from sqlalchemy import select

from src.backtest.portfolio import BookTrade, exit_day_overlap, simulate_book
from src.db.models import Candidate, DailyRun, Outcome, Signal
from src.db.session import get_session

# The "book" = the two systematic official streams run together. The manual
# sleeve is deliberately excluded: it reproduces the official MR picks verbatim
# and adds only negative-alpha breadth (see the 2026-07 manual-sleeve forensic),
# so it dilutes the book rather than diversifying it.
BOOK_STREAMS = ["sniper|mas_official", "mean_reversion|mas_official"]
PORTFOLIO_MAX_CONCURRENT = 10
PORTFOLIO_START_CAPITAL = 100_000.0

# Benchmarks: SPY = S&P 500 proxy, QQQ = Nasdaq-100 proxy. Per-trade alpha is
# the pick's return MINUS the benchmark's return over the SAME entry->exit dates
# — the honest "did I beat just holding the index for those days?" (alpha vs beta),
# not a buy-and-hold overlay (the strategy sits in cash between short-hold trades).
BENCHMARKS = {"spy": "SPY", "qqq": "QQQ"}


async def _benchmark_closes(days: int) -> dict[str, dict]:
    """Return {key: {date: close}} for each benchmark over the window, via Polygon.
    Empty on any failure (no key locally / offline) — alpha fields degrade to None."""
    out: dict[str, dict] = {}
    try:
        from src.data.polygon_client import PolygonClient
        client = PolygonClient()
    except Exception:
        return {k: {} for k in BENCHMARKS}
    frm = date.today() - timedelta(days=days + 20)
    to = date.today()
    for key, ticker in BENCHMARKS.items():
        try:
            df = await client.get_ohlcv(ticker, frm, to)
            out[key] = {r["date"]: float(r["close"]) for _, r in df.iterrows()} if not df.empty else {}
        except Exception:
            out[key] = {}
    return out


def _alpha_summary(alphas: list[float]) -> dict | None:
    """Per-stream alpha stats with a seeded bootstrap 95% CI of the mean.

    The CI is the anti-over-excitement guard: a positive mean whose CI still
    crosses zero is a lean, not an established edge (a 64%-beat on n=25 is a
    coin-flip run away from chance). Seeded so the dashboard number is stable
    across runs given the same trades.
    """
    import random
    a = [x for x in alphas if x is not None]
    if len(a) < 3:
        return None
    mean = sum(a) / len(a)
    beat = sum(1 for x in a if x > 0) / len(a)
    rng = random.Random(20260723)
    n = len(a)
    means = sorted(sum(rng.choices(a, k=n)) / n for _ in range(10_000))
    lo, hi = means[249], means[9749]
    return {
        "n": n,
        "mean": round(mean, 4),
        "ci_lo": round(lo, 4),
        "ci_hi": round(hi, 4),
        "beat_pct": round(beat, 4),
        "significant": bool(lo > 0 or hi < 0),  # CI excludes zero
    }


def _bench_return(closes: dict, entry: _date | None, exit_: _date | None) -> float | None:
    """Benchmark % return over [entry, exit] using close-to-close, nearest trading
    day on-or-before each date. None if unavailable."""
    if not closes or entry is None or exit_ is None:
        return None
    days_sorted = sorted(closes)
    def on_or_before(d):
        i = bisect.bisect_right(days_sorted, d) - 1
        return days_sorted[i] if i >= 0 else None
    de, dx = on_or_before(entry), on_or_before(exit_)
    if de is None or dx is None:
        return None
    c0 = closes[de]
    return (closes[dx] - c0) / c0 * 100 if c0 else None

# Honest expectation bands (per-trade), from the 2026-07 truth work:
#  - sniper: truth-matrix Run E (live-faithful fills): ~54.3% WR / +0.54%/trade
#  - MR official: reconciled 90d live +0.46%/trade (n=23, provisional)
#  - MR sleeve: reconciled ~breakeven (-0.01%)
BASELINES = {
    "sniper|mas_official": {"label": "Sniper (official)", "wr": 0.543, "avg": 0.54,
                            "source": "truth-matrix Run E (2026-07-19)"},
    "mean_reversion|mas_official": {"label": "MR (official)", "wr": 0.522, "avg": 0.46,
                                    "source": "90d reconciliation (provisional, n=23)"},
    "mean_reversion|mr_manual_sleeve": {"label": "MR (manual sleeve)", "wr": 0.493, "avg": -0.01,
                                        "source": "90d reconciliation"},
    # PEAD paper trial — the band is the BACKTEST target we're forward-testing
    # against (pead_FINDINGS.md), NOT a validated live number. Paper until it
    # clears ~30 trades / 4-6 weeks live.
    "pead|pead_paper": {"label": "PEAD (paper)", "wr": 0.57, "avg": 1.80,
                        "source": "backtest target — paper, unproven live"},
    # Neglected-beat variant (>10% beat + DECELERATING YoY revenue growth): the
    # stronger PEAD cohort that cleared the validation card (N=669, deflated
    # Sharpe 1.0). Band is the backtest target — paper, forward-testing now.
    "pead|pead_neglected": {"label": "PEAD (neglected-beat)", "wr": 0.58, "avg": 2.42,
                            "source": "validation card — paper, unproven live"},
}


def _iso(d) -> str | None:
    return d.isoformat() if d is not None else None


def _position_days_held(entry: date | None, observed_sessions: set[date]) -> int | None:
    """Count completed market sessions from observed benchmark bars only."""
    if entry is None or not observed_sessions:
        return None
    held = sum(session >= entry for session in observed_sessions)
    return held or None


def _stream_key(model: str | None, source: str | None) -> str:
    return f"{model or 'unknown'}|{source or 'unknown'}"


def _portfolio(trades: dict[str, list]) -> dict | None:
    """The book (sniper + MR official) vs each stream alone, as a real
    concurrency-capped account. Front-end assigns colors; we ship data only."""
    sniper_key, mr_key = "sniper|mas_official", "mean_reversion|mas_official"
    if not (trades.get(sniper_key) or trades.get(mr_key)):
        return None

    def book_trades(rows: list) -> list[BookTrade]:
        out = []
        for r in rows:
            e, x = r.get("entry_date"), r.get("exit_date")
            if e and x and r.get("pnl_pct") is not None:
                out.append(BookTrade(entry=date.fromisoformat(e),
                                     exit=date.fromisoformat(x), pnl_pct=r["pnl_pct"]))
        return out

    specs = [
        ("sniper", "Sniper only", [sniper_key]),
        ("mr", "MR official only", [mr_key]),
        ("book", "Book (sniper + MR)", BOOK_STREAMS),
    ]
    configs, equity = [], {}
    for ckey, label, streams in specs:
        rows = [r for k in streams for r in trades.get(k, [])]
        if not rows:
            continue
        res = simulate_book(book_trades(rows), max_concurrent=PORTFOLIO_MAX_CONCURRENT,
                            start_capital=PORTFOLIO_START_CAPITAL)
        configs.append({
            "key": ckey, "label": label, "streams": streams,
            "trades": res["taken"], "skipped": res["skipped"],
            "return_pct": round(res["total_return_pct"], 3),
            "max_dd_pct": round(res["max_drawdown_pct"], 3),
            "sharpe": round(res["sharpe"], 2) if res["sharpe"] is not None else None,
        })
        equity[ckey] = [{"date": d.isoformat(),
                         "ret": round((eq / PORTFOLIO_START_CAPITAL - 1.0) * 100.0, 4)}
                        for d, eq in res["equity_curve"]]
    if not configs:
        return None
    return {
        "book_streams": BOOK_STREAMS,
        "start_capital": PORTFOLIO_START_CAPITAL,
        "max_concurrent": PORTFOLIO_MAX_CONCURRENT,
        "configs": configs,
        "equity": equity,
        "overlap": exit_day_overlap({"sniper": trades.get(sniper_key, []),
                                     "mr": trades.get(mr_key, [])}),
    }


def _candidates_payload(candidates, signals, outcome_by_sig) -> list[dict]:
    """Ranked candidates per run, tagged with whether each became an official pick.

    Rank is assigned by composite_score within a run_date, which is the order the
    pipeline itself used. `picked` lets the audit compare the chosen top-2 against
    the ranks that were passed over.

    That intent was not met until now. This function re-sorted by score and
    renumbered from 1, exporting a DERIVED ordering while `Candidate.strategy_rank`
    — the order selection actually used — was never emitted at all, and the
    ledger's rejection columns were dropped. An audit built on the exported rank
    was auditing an ordering the pipeline never used, and its first conclusions
    ("rank-1 candidates skipped", "unpicked beats picked") were artifacts of
    exactly that. The two orderings are now separate, explicitly named fields.

    Identity must be (run_date, ticker, MODEL) and must count only OFFICIAL,
    non-shadow signals. Keying on (run_date, ticker) alone silently corrupted the
    audit three ways:
      * a shadow-booked, validation-BLOCKED signal marked its candidate "picked"
        — inverting the exact quantity the audit measures;
      * a same-ticker manual-sleeve or PEAD paper signal marked an official
        candidate picked;
      * two models proposing one ticker collapsed into a single identity.
    `Candidate` rows come from the OFFICIAL ranked list, so the pick side has to
    be restricted the same way.
    """
    picked = set()
    for s in signals:
        if s.signal_source != "mas_official":
            continue                      # manual sleeve + PEAD paper are separate books
        o = outcome_by_sig.get(s.id)
        if o is not None and o.skip_reason:
            continue                      # shadow-booked / gap-rejected: never taken
        picked.add((s.run_date, s.ticker.upper(), s.signal_model))
    by_run: dict = {}
    for c in candidates:
        by_run.setdefault(c.run_date, []).append(c)
    out = []
    for run_date, rows in sorted(by_run.items()):
        # Ordered by strategy_rank where recorded, score otherwise, so the export
        # is stable. The ordering is presentational; the RANK FIELDS below carry
        # the meaning, and they are deliberately two separate fields.
        # score_rank must be derived from SCORE, independently of the emission
        # order. Computing it from an enumeration over a strategy_rank-sorted
        # list silently makes the two fields the same number under a different
        # name — which is the defect this change exists to remove.
        by_score = sorted(rows, key=lambda r: -(r.composite_score or 0))
        score_rank = {id(r): i for i, r in enumerate(by_score, start=1)}
        rows.sort(key=lambda r: (r.strategy_rank if r.strategy_rank is not None else 10**6,
                                 -(r.composite_score or 0)))
        for c in rows:
            out.append({
                "run_date": run_date.isoformat(),
                # The order selection ACTUALLY used. None for rows written before
                # the selection ledger (PR #76) began recording it: null means
                # "not recorded", never "rank unknown but presumed 1".
                "strategy_rank": c.strategy_rank,
                # Ordering by score alone. NOT the selection order — a
                # correlation-dropped name can hold score_rank 1 with
                # picked=false, which is precisely what made an audit built on
                # this field conclude that rank-1 candidates were being skipped.
                "score_rank": score_rank[id(c)],
                "ticker": c.ticker,
                "model": c.signal_model,
                "score": round(float(c.composite_score), 4) if c.composite_score is not None else None,
                "picked": (run_date, c.ticker.upper(), c.signal_model) in picked,
                # Why a candidate was not taken. Without these, picked=false
                # conflates below-quota, capacity-censored, correlation-dropped
                # and fragility-blocked into one undifferentiated flag, and no
                # selection-quality question can be answered from the export.
                "selection_stage_reached": c.selection_stage_reached,
                "rejection_stage": c.rejection_stage,
                "rejection_reason": c.rejection_reason,
                "slots_total": c.slots_total,
                "slots_occupied": c.slots_occupied,
                "slots_available": c.slots_available,
                "correlated_with": c.correlated_with,
                "correlation": (round(float(c.correlation), 4)
                                if c.correlation is not None else None),
            })
    return out


async def build_snapshot(days: int = 90, bench_closes: dict | None = None) -> dict:
    cutoff = date.today() - timedelta(days=days)
    if bench_closes is None:
        bench_closes = await _benchmark_closes(days)

    async with get_session() as session:
        runs = (await session.execute(
            select(DailyRun).where(DailyRun.run_date >= cutoff).order_by(DailyRun.run_date)
        )).scalars().all()

        signals = (await session.execute(
            select(Signal).where(Signal.run_date >= cutoff)
        )).scalars().all()

        sig_ids = [s.id for s in signals]
        outcomes = (await session.execute(
            select(Outcome).where(Outcome.signal_id.in_(sig_ids))
        )).scalars().all() if sig_ids else []

        # Ranked candidates per run — the input to the top-2-of-N choice. The
        # pipeline has persisted these all along (main.py: "Save candidates"),
        # but nothing ever read them back, so the question "does our rank
        # ordering predict outcomes, or are we picking noise?" was unanswerable
        # on live data. Exporting them makes the selection-quality audit
        # runnable (see scripts/rank_quality_audit.py).
        candidates = (await session.execute(
            select(Candidate).where(Candidate.run_date >= cutoff)
            .order_by(Candidate.run_date, Candidate.composite_score.desc())
        )).scalars().all()

    sig_by_id = {s.id: s for s in signals}
    # One outcome per signal — used to tell a taken pick from a shadow-booked or
    # gap-rejected one when tagging candidates.
    outcome_by_sig = {o.signal_id: o for o in outcomes}
    latest_run = runs[-1] if runs else None

    # --- Today: picks of the latest run date ---
    today_picks = []
    if latest_run:
        for s in signals:
            if s.run_date != latest_run.run_date:
                continue
            feats = s.features or {}
            today_picks.append({
                "ticker": s.ticker,
                "model": s.signal_model,
                "source": s.signal_source,
                "confidence": s.confidence,
                "raw_score": feats.get("model_raw_score"),
                "components": feats.get("model_components") or {},
                "entry": s.entry_price,
                "stop": s.stop_loss,
                "target1": s.target_1,
                "hold_days": s.holding_period_days,
                "regime": s.regime,
            })

    # --- Trades: closed, skipped, open — per stream, never blended ---
    trades: dict[str, list[dict]] = {}
    open_positions = []
    skip_counts: dict[str, int] = {}
    for o in outcomes:
        s = sig_by_id.get(o.signal_id)
        if s is None:
            continue
        key = _stream_key(s.signal_model, s.signal_source)
        if o.skip_reason:
            skip_counts[key] = skip_counts.get(key, 0) + 1
            continue
        if o.still_open:
            # Carry the levels and the planned horizon so the dashboard can show
            # how far a live position is from its stop, its target and its expiry
            # — the three things that decide whether it needs attention today.
            entry_px = o.entry_price or s.entry_price
            days_held = _position_days_held(o.entry_date, set(bench_closes.get("spy", {})))
            open_positions.append({
                "ticker": o.ticker, "stream": key,
                "entry_date": _iso(o.entry_date),
                "days_held": days_held,
                "days_held_basis": "observed_spy_sessions" if days_held is not None else "unavailable",
                "unrealized_pnl_pct": o.pnl_pct,
                "entry_price": entry_px,
                "stop_loss": s.stop_loss,
                "target_1": s.target_1,
                "hold_days": s.holding_period_days,
            })
            continue
        if o.pnl_pct is None:
            continue
        row = {
            "ticker": o.ticker,
            "signal_date": _iso(s.run_date),
            "entry_date": _iso(o.entry_date),
            "exit_date": _iso(o.exit_date),
            "exit_reason": o.exit_reason,
            "pnl_pct": round(o.pnl_pct, 4),
            "mfe": o.max_favorable,
            "mae": o.max_adverse,
            "hold_days": (o.exit_date - o.entry_date).days
                         if (o.exit_date and o.entry_date) else None,
        }
        # Per-trade alpha vs each benchmark over the SAME holding window.
        for bk, closes in bench_closes.items():
            br = _bench_return(closes, o.entry_date, o.exit_date)
            row[f"bench_{bk}"] = round(br, 4) if br is not None else None
            row[f"alpha_{bk}"] = round(o.pnl_pct - br, 4) if br is not None else None
        trades.setdefault(key, []).append(row)
    for key in trades:
        trades[key].sort(key=lambda t: (t["exit_date"] or "", t["ticker"]))

    # Per-stream alpha summary with bootstrap CI, per benchmark.
    alpha_summary: dict[str, dict] = {}
    for key, rows in trades.items():
        per_bench = {}
        for bk in BENCHMARKS:
            s = _alpha_summary([r.get(f"alpha_{bk}") for r in rows])
            if s:
                per_bench[bk] = s
        if per_bench:
            alpha_summary[key] = per_bench

    # --- System: run history + health ---
    run_history = [{
        "date": _iso(r.run_date),
        "regime": r.regime,
        "universe": r.universe_size,
        "candidates": r.candidates_scored,
        "duration_s": r.pipeline_duration_s,
        "mode": r.execution_mode,
        "health": (r.pipeline_health or {}).get("status")
                  if isinstance(r.pipeline_health, dict) else None,
        "warnings": (r.pipeline_health or {}).get("warnings")
                    if isinstance(r.pipeline_health, dict) else None,
    } for r in runs]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": days,
        "latest_run": {
            "date": _iso(latest_run.run_date) if latest_run else None,
            "regime": latest_run.regime if latest_run else None,
            "universe": latest_run.universe_size if latest_run else None,
            "candidates": latest_run.candidates_scored if latest_run else None,
        },
        "today_picks": today_picks,
        "trades": trades,
        "open_positions": open_positions,
        "skip_counts": skip_counts,
        "run_history": run_history,
        "baselines": BASELINES,
        "benchmarks": {"spy": "S&P 500 (SPY)", "qqq": "Nasdaq-100 (QQQ)"},
        "benchmark_available": any(bench_closes.get(k) for k in BENCHMARKS),
        "alpha_summary": alpha_summary,
        "portfolio": _portfolio(trades),
        "candidates": _candidates_payload(candidates, signals, outcome_by_sig),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dashboard/data.json")
    ap.add_argument("--days", type=int, default=90)
    args = ap.parse_args()

    snap = await build_snapshot(days=args.days)
    with open(args.out, "w") as f:
        json.dump(snap, f, default=str)
    n_trades = sum(len(v) for v in snap["trades"].values())
    print(f"Wrote {args.out}: {len(snap['today_picks'])} picks today, "
          f"{n_trades} closed trades ({args.days}d), {len(snap['open_positions'])} open, "
          f"{len(snap['run_history'])} runs")


if __name__ == "__main__":
    asyncio.run(main())
