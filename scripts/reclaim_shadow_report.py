"""Offline, read-only RECLAIM shadow report (RECLAIM_MAS_SPEC_v1.1 §7, §8.8).

Reads the persisted reclaim tables, refetches bars for triggers, controls and
benchmarks, and writes the per-bloodline x horizon tables plus PROMOTION/KILL
flags. It never writes to the database and never feeds `run_validation_checks`
or any book gate. Controls come only from `reclaim_risk_snapshots` (R20).

    python scripts/reclaim_shadow_report.py --out outputs/research/reclaim
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select

from src.db.models import (
    ReclaimLaneRegistry,
    ReclaimOpenDecision,
    ReclaimRiskSnapshot,
    ReclaimTrigger,
)
from src.db.session import get_session
from src.output.reclaim_tracker import _default_bars, last_complete_session
from src.research import reclaim_report as rp
from src.signals import reclaim as rc
from src.utils.trading_calendar import trading_sessions


async def build_report(now_utc: datetime | None = None) -> dict:
    now_utc = now_utc or datetime.now(timezone.utc)
    as_of = last_complete_session(now_utc)
    async with get_session() as session:
        registry = {r.lane: r.forward_start_date for r in
                    (await session.execute(select(ReclaimLaneRegistry))).scalars().all()}
        triggers = (await session.execute(select(ReclaimTrigger))).scalars().all()
        decisions = {(d.episode_id, d.trigger_date): d for d in
                     (await session.execute(select(ReclaimOpenDecision))).scalars().all()}
        snaps = (await session.execute(select(ReclaimRiskSnapshot))).scalars().all()
    if not triggers:
        return {"as_of": as_of.isoformat(), "forward_start": {k: v.isoformat() for k, v in registry.items()},
                "lanes": [], "note": "no forward triggers yet"}

    risk_sets: dict = {}
    for s in snaps:
        if s.triggered_on_k:
            continue
        risk_sets.setdefault((s.trigger_episode_id, s.trigger_date), []).append(rc.ControlCandidate(
            episode_id=s.control_episode_id, symbol=s.control_symbol, sector=s.sector, mcap=s.mcap,
            score_pct=s.score_pct, sma_dist=s.sma_dist, age=s.age, prehistory=s.prehistory,
            has_entry_bar=s.has_entry_bar is not False,
        ))
    symbols = sorted({t.symbol for t in triggers} | {s.control_symbol for s in snaps})
    start = min(t.trigger_date for t in triggers) - timedelta(days=10)
    sessions = trading_sessions(start, as_of)
    raw = await _default_bars(symbols + list(rc.BENCHMARK_TICKERS), sessions[0], as_of)
    bars = {s: rc.align_to_sessions(raw.get(s), sessions) for s in symbols}
    benches = {b: rc.align_to_sessions(raw.get(b), sessions) for b in rc.BENCHMARK_TICKERS}

    obs = []
    for t in triggers:
        d = decisions.get((t.episode_id, t.trigger_date))
        obs.append(rp.TriggerObs(
            lane=t.lane, symbol=t.symbol, episode_id=t.episode_id, trigger_date=t.trigger_date,
            entry_session=t.entry_session, start_date=_episode_start(t.episode_id),
            sector=t.sector, mcap=t.mcap, score_pct=t.score_pct,
            sma_dist=(t.close_k / t.sma50_k - 1.0) if t.sma50_k else None,
            prehistory=t.prehistory,
            mech_status=d.mech_status if d else rc.CENSORED_NO_ENTRY_BAR,
            earnings_status=d.earnings_status if d else rc.INACTIVE_FAILED,
            forward_recorded=t.forward_recorded, open_e=d.open_e if d else None,
            stop=t.setup_low, regime=t.spy_regime,
        ))
    rows = rp.evaluate(obs, risk_sets, bars, benches, sessions)
    ledger = {}
    for o in obs:
        ledger.setdefault(o.lane, {}).setdefault(o.mech_status, 0)
        ledger[o.lane][o.mech_status] += 1
    return {
        "as_of": as_of.isoformat(),
        "spec": "RECLAIM_MAS_SPEC_v1.1",
        "status": "SHADOW — not evidence until the §7 gate passes on clean forward triggers",
        "forward_start": {k: v.isoformat() for k, v in registry.items()},
        "trigger_ledger": ledger,
        "clean_triggers": sum(1 for o in obs if o.clean),
        "lanes": [rp.lane_report(rows, lane, registry.get(lane), as_of) for lane in rc.LANES],
        "pooled_descriptive": [rp.pooled_descriptive(rows, h) for h in rc.HORIZONS],
    }


def _episode_start(episode_id: str):
    from datetime import date
    return date.fromisoformat(episode_id.rsplit("@", 1)[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/research/reclaim")
    args = ap.parse_args()
    report = asyncio.run(build_report())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"reclaim_shadow_report_{report['as_of']}.json"
    path.write_text(json.dumps(report, indent=2, default=rp.json_default))
    print(f"Wrote {path}")
    for lane in report.get("lanes", []):
        c10 = next(c for c in lane["clean"] if c["horizon"] == rc.ANCHOR_HORIZON)
        print(f"{lane['lane']}: clean complete h10 = {lane['clean_complete_h10']}, "
              f"excess vs NN = {c10['excess_nn']}, flags = {lane['flags']}")


if __name__ == "__main__":
    main()
