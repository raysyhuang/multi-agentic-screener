"""RECLAIM native-lane historical replay — RECONSTRUCTED_DESCRIPTIVE, never evidence.

Runs both native lanes (RANGE_TECH_NATIVE, DRIFT_G3_NATIVE) over years of
Polygon daily bars through exactly the code the live collector would use: the
pools, the XNYS state machine, the actual-open decision, the structural-stop
fixed-horizon exits through `walk_exit`, and the frozen NN-control matching.
Only the inputs differ from a forward run:

- the universe is TODAY's FMP universe and TODAY's market cap, replayed
  backwards — survivor-biased, so names that died or shrank are missing;
- every inclusion is reconstructed, so no episode can be VALID under §4A. A
  replay-equivalent label is used instead: the 20 prior sessions lie inside the
  replay window and the symbol is absent from all of them;
- earnings use FMP's historical report dates (not point-in-time vintages), so
  the gate here is a sensitivity cut, not the live fail-closed gate.

    python scripts/reclaim_backtest.py --start 2023-01-03 --end 2026-10-02 \
        --out outputs/research/reclaim

Spec: RECLAIM_MAS_SPEC_v1.1. Output is descriptive; the §7 flags printed here
are what the rule WOULD say on reconstructed data, nothing more.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from src.research import reclaim_replay as rr
from src.research import reclaim_report as rp
from src.signals import reclaim as rc
from src.utils.trading_calendar import is_trading_day, trading_sessions

logger = logging.getLogger("reclaim_backtest")


async def load_universe(cap: int) -> tuple[list[str], dict[str, dict]]:
    from src.data.aggregator import DataAggregator
    from src.research.reclaim_replay import select_reclaim_universe
    from src.signals.filter import filter_universe

    agg = DataAggregator()
    try:
        rows = filter_universe(await agg.get_universe())
    finally:
        agg.close()
    # FMP's screener still returns ~20% delisted names (TWTR, ABMD, SGEN, ...)
    # flagged isActivelyTrading=False; they have no bars in the window and would
    # only burn cap slots. Dropping them keeps the universe a survivor set, which
    # is the stated bias of this replay anyway.
    dead = sum(1 for r in rows if r.get("isActivelyTrading") is False)
    rows = [r for r in rows if r.get("isActivelyTrading") is not False]
    logger.info("universe: dropped %d rows flagged isActivelyTrading=False", dead)
    symbols, meta, cap_hits = select_reclaim_universe(rows, cap)
    logger.info("universe: %d symbols (cap %d, %d cut)", len(symbols), cap, cap_hits)
    return symbols, meta


async def load_bars(symbols: list[str], sessions: list[date]) -> dict:
    from src.data.aggregator import DataAggregator

    agg = DataAggregator()
    agg.reset_data_provenance()
    try:
        raw = await agg.get_bulk_ohlcv(symbols, sessions[0], sessions[-1])
        prov = agg.get_data_provenance()
    finally:
        agg.close()
    return {"raw": raw, "provenance": prov}


async def load_earnings(symbols: list[str]) -> dict[str, list[date]]:
    from src.data.earnings_cache import get_earnings

    out: dict[str, list[date]] = {}
    for sym in symbols:
        try:
            out[sym] = rc.parse_earnings_dates(await get_earnings(sym))
        except Exception as exc:  # noqa: BLE001 — recorded as unknown, never as clear
            logger.warning("earnings fetch failed for %s: %s", sym, exc)
    return out


def replay_equivalent_prehistory(start: int, lane: str, inclusions: set[int]) -> tuple[str, str | None]:
    """Replay stand-in for VALID: 20 prior sessions in-window and symbol absent.

    Never written as VALID anywhere persistent; it only selects the cohort that
    WOULD be primary-eligible if the reconstructed lists had been production.
    """
    lo = start - rc.PREHIST
    if lo < rc.LANE_WARMUP[lane]:
        return rc.INSUFFICIENT_PREHISTORY, rc.LEFT_CENSORED_FIRST_OBS
    if any(i in inclusions for i in range(lo, start)):
        return rc.INSUFFICIENT_PREHISTORY, "NOT_ABSENT"
    return rc.VALID, "REPLAY_EQUIVALENT"


def earnings_label(dates: list[date] | None, entry: date) -> str:
    """Same window rule as the live gate, on non-PIT historical dates."""
    if dates is None:
        return rc.INACTIVE_FAILED
    lo = rc.session_offset(entry, -rc.EARN_WIN, is_trading_day)
    hi = rc.session_offset(entry, rc.EARN_WIN, is_trading_day)
    if not (any(d < lo for d in dates) and any(d > hi for d in dates)):
        return rc.BLOCK_EARNINGS_UNKNOWN
    if any(lo <= d <= hi for d in dates):
        return rc.BLOCK_EARNINGS
    return rc.CLEAR


def run_replay(bars_raw: dict, symbols: list[str], meta: dict, sessions: list[date]) -> dict:
    """Pools -> episodes -> triggers -> open decisions -> risk sets. Earnings are
    labelled afterwards (``apply_earnings``) so only triggered names are fetched."""
    bars = {s: rc.align_to_sessions(bars_raw.get(s), sessions) for s in symbols}
    benches = {b: rc.align_to_sessions(bars_raw.get(b), sessions) for b in rc.BENCHMARK_TICKERS}
    last_idx = len(sessions) - 1
    spy = benches["SPY"]

    pools, lane_eps, incl_by_lane = {}, {}, {}
    for lane in rc.LANES:
        pool = rr.compute_lane_pool(lane, bars, meta)
        incl = rr.reconstructed_inclusions(pool)
        floor_at = {s: {i: float(pool.box_low[s].iloc[i]) for i in incl[s]} for s in incl}
        pools[lane], incl_by_lane[lane] = pool, incl

        def pre_fn(start, _lane=lane):
            return (rc.INSUFFICIENT_PREHISTORY, rc.RECONSTRUCTED_HISTORY)

        lane_eps[lane] = rr.replay_lane(lane, bars, sessions, last_idx, incl, floor_at, pre_fn)
        logger.info("%s: %d member-days, %d episodes", lane,
                    int(pool.member.to_numpy().sum()), len(lane_eps[lane].results))

    obs: list[rp.TriggerObs] = []
    risk_sets: dict = {}
    ledger: dict = defaultdict(lambda: defaultdict(int))
    for lane in rc.LANES:
        for res in lane_eps[lane].results:
            trig = res.trigger
            if trig is None or trig.k + 1 > last_idx:
                continue
            pre, _ = replay_equivalent_prehistory(res.episode.start, lane, incl_by_lane[lane][trig.symbol])
            res.episode.prehistory = pre
            e = trig.k + 1
            o = bars[trig.symbol]["open"].iloc[e]
            dec = rc.open_decision(float(o) if np.isfinite(o) else None, trig.setup_low, trig.sma50_k)
            ledger[lane][dec.mech_status] += 1
            entry = sessions[e]
            m = meta.get(trig.symbol) or {}
            obs.append(rp.TriggerObs(
                lane=lane, symbol=trig.symbol, episode_id=trig.episode_id,
                trigger_date=trig.trigger_date, entry_session=entry,
                start_date=res.episode.start_date, sector=m.get("sector"), mcap=m.get("mcap"),
                score_pct=rr.latest_score_pct(pools[lane], trig.symbol, trig.k),
                sma_dist=trig.close_k / trig.sma50_k - 1.0, prehistory=pre,
                mech_status=dec.mech_status, earnings_status=rc.INACTIVE_FAILED, forward_recorded=True,
                open_e=dec.open_e, stop=trig.setup_low, regime=rc.spy_regime_label(spy, e),
            ))
            risk_sets[(trig.episode_id, trig.trigger_date)] = [
                rc.ControlCandidate(
                    episode_id=r.control_episode_id, symbol=r.control_symbol, sector=r.sector,
                    mcap=r.mcap, score_pct=r.score_pct, sma_dist=r.sma_dist, age=r.age,
                    prehistory=replay_equivalent_prehistory(
                        _start_idx(lane_eps[lane], r.control_episode_id), lane,
                        incl_by_lane[lane][r.control_symbol])[0],
                )
                for r in rr.risk_rows(trig, lane_eps[lane], pools[lane], meta)
                if not r.triggered_on_k
            ]
    return {"obs": obs, "risk_sets": risk_sets, "bars": bars, "benches": benches,
            "ledger": {k: dict(v) for k, v in ledger.items()},
            "member_days": {lane: int(pools[lane].member.to_numpy().sum()) for lane in rc.LANES},
            "episodes": {lane: len(lane_eps[lane].results) for lane in rc.LANES}}


def apply_earnings(obs: list[rp.TriggerObs], earnings: dict[str, list[date]] | None) -> None:
    """Label each trigger; with --no-earnings every trigger reads CLEAR (and says so)."""
    for o in obs:
        if earnings is None:
            o.earnings_status = rc.CLEAR
        else:
            o.earnings_status = earnings_label(earnings.get(o.symbol), o.entry_session)


_START_CACHE: dict[int, dict[str, int]] = {}


def _start_idx(lane_eps: rr.LaneEpisodes, episode_id: str) -> int:
    key = id(lane_eps)
    if key not in _START_CACHE:
        _START_CACHE[key] = {r.episode.episode_id: r.episode.start for r in lane_eps.results}
    return _START_CACHE[key][episode_id]


def split_table(rows: list[rp.HorizonRow], lane: str, horizon: int, key) -> dict:
    groups: dict = defaultdict(list)
    for r in rows:
        if r.lane == lane and r.horizon == horizon and r.complete and r.net is not None:
            groups[key(r)].append(r)
    out = {}
    for g, rs in sorted(groups.items(), key=lambda x: str(x[0])):
        ex = [r.excess_nn for r in rs if r.excess_nn is not None]
        out[str(g)] = {
            "n": len(rs), "dates": len({r.entry_date for r in rs}),
            "net_mean": float(np.mean([r.net for r in rs])),
            "excess_spy_mean": float(np.mean([r.excess_spy for r in rs if r.excess_spy is not None]))
            if any(r.excess_spy is not None for r in rs) else None,
            "excess_nn_mean": float(np.mean(ex)) if ex else None,
            "excess_nn_date_clustered": rc.date_clustered_mean(ex, [r.entry_date for r in rs if r.excess_nn is not None]),
            "stop_touch": sum(r.stopped for r in rs) / len(rs),
        }
    return out


def build_report(result: dict, sessions: list[date], provenance: dict, universe_n: int) -> dict:
    rows, obs = result["rows"], result["obs"]
    as_of = sessions[-1]
    lanes = []
    for lane in rc.LANES:
        lr = rp.lane_report(rows, lane, None, as_of)
        lr["flags"]["note"] = ("What the §7 rule WOULD say on reconstructed, survivor-biased "
                               "data. Not a gate result; nothing here is evidence.")
        # Primary excess for the cohorts, split by year and by regime.
        for cohort, flt in (("clean_equivalent", lambda r: r.clean), ("all_pass", lambda r: True)):
            sub = [r for r in rows if flt(r)]
            lr[f"by_year_{cohort}"] = {h: split_table(sub, lane, h, lambda r: r.entry_date.year)
                                      for h in rc.HORIZONS}
            lr[f"by_regime_{cohort}"] = {h: split_table(sub, lane, h, lambda r: r.regime)
                                        for h in rc.HORIZONS}
        lanes.append(lr)
    return {
        "label": "RECONSTRUCTED_DESCRIPTIVE — survivor-biased universe, non-PIT earnings, not evidence",
        "spec": "RECLAIM_MAS_SPEC_v1.1",
        "window": {"from": sessions[0].isoformat(), "to": as_of.isoformat(), "sessions": len(sessions)},
        "universe_symbols": universe_n,
        "provenance": provenance,
        "member_days": result["member_days"],
        "episodes": result["episodes"],
        "trigger_ledger": result["ledger"],
        "triggers": len(obs),
        "clean_equivalent_triggers": sum(1 for o in obs if o.clean),
        "lanes": lanes,
        "pooled_descriptive": [rp.pooled_descriptive(rows, h) for h in rc.HORIZONS],
    }


async def main_async(args) -> None:
    sessions = trading_sessions(date.fromisoformat(args.start), date.fromisoformat(args.end))
    symbols, meta = await load_universe(args.cap)
    data = await load_bars(symbols + list(rc.BENCHMARK_TICKERS), sessions)
    result = run_replay(data["raw"], symbols, meta, sessions)
    obs = result["obs"]
    earnings = None if args.no_earnings else await load_earnings(
        sorted({o.symbol for o in obs if o.mech_status == rc.PASS}))
    apply_earnings(obs, earnings)
    result["rows"] = rp.evaluate(obs, result["risk_sets"], result["bars"], result["benches"], sessions)
    report = build_report(result, sessions, data["provenance"], len(symbols))
    report["earnings_basis"] = ("none (--no-earnings: every trigger labelled CLEAR)" if earnings is None
                                else "FMP historical report dates, non-PIT")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"reclaim_backtest_{sessions[0]}_{sessions[-1]}.json"
    path.write_text(json.dumps(report, indent=2, default=rp.json_default))
    trig_path = out / f"reclaim_backtest_rows_{sessions[0]}_{sessions[-1]}.csv"
    import pandas as pd
    pd.DataFrame([{
        "lane": r.lane, "symbol": r.symbol, "trigger_date": r.trigger_date, "entry_date": r.entry_date,
        "horizon": r.horizon, "clean_equivalent": r.clean, "complete": r.complete, "net": r.net,
        "bench_spy": r.bench_spy, "bench_sector": r.bench_sector, "nn_net": r.nn_net,
        "stopped": r.stopped, "gap_through": r.gap_through, "regime": r.regime,
        "controls": ";".join(r.controls),
    } for r in result["rows"]]).to_csv(trig_path, index=False)
    print(f"Wrote {path}\nWrote {trig_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-03")
    ap.add_argument("--end", default="2026-10-02")
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--no-earnings", action="store_true")
    ap.add_argument("--out", default="outputs/research/reclaim")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
