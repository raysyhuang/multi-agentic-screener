"""RECLAIM evaluation (RECLAIM_MAS_SPEC_v1.1 §6, §7) — pure, shared by both reporters.

`scripts/reclaim_shadow_report.py` feeds it persisted forward rows;
`scripts/reclaim_backtest.py` feeds it an in-memory replay. Same arithmetic, so
the two can only differ by input. The reporter — not the generic Outcome table —
is the authority on completeness: a horizon is complete only when bar e+h-1
exists, even if the stop fired earlier (uniform censoring, R22).

Flags never alter the book, routines or other lanes.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from src.signals import reclaim as rc


@dataclass
class TriggerObs:
    lane: str
    symbol: str
    episode_id: str
    trigger_date: date
    entry_session: date
    start_date: date
    sector: str | None
    mcap: float | None
    score_pct: float | None
    sma_dist: float | None
    prehistory: str
    mech_status: str
    earnings_status: str
    forward_recorded: bool
    open_e: float | None
    stop: float
    regime: str | None

    @property
    def primary_eligible(self) -> bool:
        return self.prehistory == rc.VALID and self.mech_status == rc.PASS and self.lane in rc.LANES

    @property
    def clean(self) -> bool:
        return self.primary_eligible and self.earnings_status == rc.CLEAR and self.forward_recorded

    @property
    def key(self) -> tuple[str, date]:
        return (self.episode_id, self.trigger_date)


@dataclass
class HorizonRow:
    lane: str
    episode_id: str
    symbol: str
    trigger_date: date
    entry_date: date
    horizon: int
    clean: bool
    complete: bool
    exited: bool
    net: float | None
    stopped: bool
    gap_through: bool
    bench_spy: float | None
    bench_sector: float | None
    nn_net: float | None
    controls: list[str] = field(default_factory=list)
    regime: str | None = None

    @property
    def excess_spy(self) -> float | None:
        return None if self.net is None or self.bench_spy is None else self.net - self.bench_spy

    @property
    def excess_sector(self) -> float | None:
        return None if self.net is None or self.bench_sector is None else self.net - self.bench_sector

    @property
    def excess_nn(self) -> float | None:
        return None if self.net is None or self.nn_net is None else self.net - self.nn_net


def _age(sessions_idx: dict[date, int], start: date, k: date) -> int | None:
    a, b = sessions_idx.get(start), sessions_idx.get(k)
    return None if a is None or b is None else b - a


def evaluate(
    obs: list[TriggerObs],
    risk_sets: dict[tuple[str, date], list[rc.ControlCandidate]],
    bars: dict[str, pd.DataFrame],
    benchmarks: dict[str, pd.DataFrame],
    sessions: list[date],
) -> list[HorizonRow]:
    """Every PASS trigger x horizon, with SPY, sector and NN-control comparisons."""
    idx = {d: i for i, d in enumerate(sessions)}
    last_idx = len(sessions) - 1
    rows: list[HorizonRow] = []
    for o in obs:
        if o.mech_status != rc.PASS or o.open_e is None:
            continue
        e = idx.get(o.entry_session)
        b = bars.get(o.symbol)
        if e is None or b is None:
            continue
        trig_cand = rc.ControlCandidate(
            episode_id=o.episode_id, symbol=o.symbol, sector=o.sector, mcap=o.mcap,
            score_pct=o.score_pct, sma_dist=o.sma_dist,
            age=_age(idx, o.start_date, o.trigger_date), prehistory=o.prehistory,
        )
        pool = []
        for c in risk_sets.get(o.key, []):
            cb = bars.get(c.symbol)
            has_bar = cb is not None and np.isfinite(cb["open"].iloc[e])
            pool.append(rc.ControlCandidate(**{**asdict(c), "has_entry_bar": bool(has_bar and c.has_entry_bar)}))
        controls = rc.nn_controls(trig_cand, pool, o.lane)
        sector_etf = rc.SECTOR_ETF.get(o.sector or "")
        for h in rc.HORIZONS:
            res = rc.clone_exit(b, sessions, e, o.open_e, o.stop, h, last_idx)
            nn_vals = []
            for c in controls:
                cb = bars[c.symbol]
                c_open = float(cb["open"].iloc[e])
                c_stop = rc.control_stop(c_open, o.open_e, o.stop)
                cr = rc.clone_exit(cb, sessions, e, c_open, c_stop, h, last_idx)
                if cr.complete and cr.net_return is not None:
                    nn_vals.append(cr.net_return)
            rows.append(HorizonRow(
                lane=o.lane, episode_id=o.episode_id, symbol=o.symbol,
                trigger_date=o.trigger_date, entry_date=o.entry_session, horizon=h,
                clean=o.clean, complete=res.complete, exited=res.exited,
                net=res.net_return if res.complete else None,
                stopped=res.stopped, gap_through=res.gap_through,
                bench_spy=rc.benchmark_return(benchmarks["SPY"], e, h, last_idx) if "SPY" in benchmarks else None,
                bench_sector=(rc.benchmark_return(benchmarks[sector_etf], e, h, last_idx)
                              if sector_etf in benchmarks else None),
                nn_net=float(np.mean(nn_vals)) if nn_vals and res.complete else None,
                controls=[c.episode_id for c in controls], regime=o.regime,
            ))
    return rows


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    arr = np.array(values)
    return {"n": len(values), "mean": float(arr.mean()), "median": float(np.median(arr)),
            "pct_pos": float((arr > 0).mean())}


def summarize(rows: list[HorizonRow], lane: str, horizon: int, *, clean_only: bool) -> dict:
    """The canonical V0.1r §5 table for one bloodline x horizon."""
    cell = [r for r in rows if r.lane == lane and r.horizon == horizon and (r.clean or not clean_only)]
    complete = [r for r in cell if r.complete and r.net is not None]
    nets = [r.net for r in complete]
    dates = [r.entry_date for r in complete]
    out = {
        "lane": lane, "horizon": horizon, "cohort": "clean" if clean_only else "descriptive_all_pass",
        "entered": len(cell),
        "complete": len(complete),
        "censored": len(cell) - len(complete),
        "stopped_before_end": sum(1 for r in cell if not r.complete and r.stopped),
        "distinct_entry_dates": len(set(dates)),
        "net": _stats(nets),
        "net_date_clustered_mean": rc.date_clustered_mean(nets, dates),
        "net_bootstrap_ci95": rc.date_clustered_bootstrap_ci(nets, dates),
        "excess_spy": _stats([r.excess_spy for r in complete if r.excess_spy is not None]),
        "excess_sector": _stats([r.excess_sector for r in complete if r.excess_sector is not None]),
        "excess_nn": _stats([r.excess_nn for r in complete if r.excess_nn is not None]),
        "stop_touch_rate": (sum(1 for r in complete if r.stopped) / len(complete)) if complete else None,
        "gap_through_count": sum(1 for r in complete if r.gap_through),
    }
    used = [c for r in complete for c in r.controls]
    out["control_reuse"] = {
        "slots": len(used), "distinct": len(set(used)),
        "max_uses": max((used.count(c) for c in set(used)), default=0),
    }
    out["descriptive_only"] = len(complete) < rc.GATE_MIN_CLEAN
    return out


def gate_inputs(rows: list[HorizonRow], lane: str) -> list[rc.GateInput]:
    by_trigger: dict[tuple[str, date], dict[int, HorizonRow]] = {}
    for r in rows:
        if r.lane == lane and r.clean:
            by_trigger.setdefault((r.episode_id, r.trigger_date), {})[r.horizon] = r
    out = []
    for hs in by_trigger.values():
        r5, r10 = hs.get(5), hs.get(10)
        any_r = r10 or r5
        if any_r is None:
            continue
        out.append(rc.GateInput(
            entry_date=any_r.entry_date, regime=any_r.regime,
            excess_h5=r5.excess_nn if r5 else None,
            excess_h10=r10.excess_nn if r10 else None,
            stopped_h10=bool(r10 and r10.stopped),
            complete_h5=bool(r5 and r5.complete), complete_h10=bool(r10 and r10.complete),
        ))
    return out


def lane_report(rows: list[HorizonRow], lane: str, forward_start: date | None, as_of: date) -> dict:
    gi = gate_inputs(rows, lane)
    return {
        "lane": lane,
        "forward_start": forward_start.isoformat() if forward_start else None,
        "clean": [summarize(rows, lane, h, clean_only=True) for h in rc.HORIZONS],
        "descriptive": [summarize(rows, lane, h, clean_only=False) for h in rc.HORIZONS],
        "clean_complete_h10": sum(1 for g in gi if g.complete_h10),
        "kill_clock_due": rc.kill_clock_due(sum(1 for g in gi if g.complete_h10), forward_start, as_of),
        "kill_checkpoint_entry_date": rc.kill_checkpoint(gi, forward_start, as_of),
        "flags": {
            "PROMOTION": rc.promotion_flag(gi),
            "KILL": rc.kill_flag(gi, forward_start, as_of),
            "note": "Flags only. They never alter the book; promotion needs Ray's explicit decision.",
        },
    }


def pooled_descriptive(rows: list[HorizonRow], horizon: int) -> dict:
    """One line across lanes, each (symbol, entry_date) counted once. Never a gate."""
    seen: dict[tuple[str, date], HorizonRow] = {}
    for r in rows:
        if r.horizon == horizon and r.complete and r.net is not None:
            seen.setdefault((r.symbol, r.entry_date), r)
    vals = [r.net for r in seen.values()]
    return {"horizon": horizon, "unique_symbol_entries": len(vals), "net": _stats(vals)}


def json_default(o):
    if isinstance(o, date):
        return o.isoformat()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return str(o)
