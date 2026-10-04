"""Pool computation and episode replay shared by the live runner and the backtest.

Pure: bars and metadata in, episodes / triggers / risk sets out. The live runner
(`src/reclaim_shadow.py`) calls this with production inclusions overriding the
reconstructed ones from the lane's forward start on; the replay backtest
(`scripts/reclaim_backtest.py`) calls it with reconstructed inclusions only. One
implementation of the rules, two callers, so a backtest number and a live number
can only differ by their inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from src.signals import reclaim as rc


@dataclass
class LanePool:
    """One lane's membership over the session grid."""
    lane: str
    member: pd.DataFrame          # sessions x symbols, bool
    score: pd.DataFrame           # sessions x symbols, float (NaN for DRIFT)
    score_pct: pd.DataFrame       # sessions x symbols, percentile among that day's members
    box_low: pd.DataFrame         # sessions x symbols (RANGE floor source)
    series: dict[str, pd.DataFrame] = field(default_factory=dict)


def compute_lane_pool(
    lane: str,
    bars_by_symbol: dict[str, pd.DataFrame],
    meta: dict[str, dict],
) -> LanePool:
    """Membership for every symbol and session, plus the per-day score percentile."""
    member: dict[str, np.ndarray] = {}
    score: dict[str, np.ndarray] = {}
    box_low: dict[str, np.ndarray] = {}
    series: dict[str, pd.DataFrame] = {}
    n = None
    for sym, bars in bars_by_symbol.items():
        s = rc.lane_series(lane, bars, (meta.get(sym) or {}).get("mcap"))
        series[sym] = s
        member[sym] = s["member"].to_numpy(dtype=bool)
        score[sym] = s["score"].to_numpy(dtype=float)
        box_low[sym] = s["box_low"].to_numpy(dtype=float) if "box_low" in s else np.full(len(s), np.nan)
        n = len(s)
    if n is None:
        empty = pd.DataFrame()
        return LanePool(lane, empty, empty, empty, empty, series)
    member_df = pd.DataFrame(member)
    score_df = pd.DataFrame(score)
    box_df = pd.DataFrame(box_low)
    member_scores = score_df.where(member_df)
    # #{v <= own} / n among that day's members == rank(method="max") / count.
    ranks = member_scores.rank(axis=1, method="max")
    counts = member_scores.notna().sum(axis=1)
    pct = ranks.div(counts.replace(0, np.nan), axis=0)
    return LanePool(lane, member_df, score_df, pct, box_df, series)


@dataclass
class LaneEpisodes:
    lane: str
    results: list[rc.EpisodeResult]
    # session idx -> [(EpisodeResult, RiskState)] for every at-risk episode
    at_risk: dict[int, list[tuple[rc.EpisodeResult, rc.RiskState]]]


def replay_lane(
    lane: str,
    bars_by_symbol: dict[str, pd.DataFrame],
    sessions: list[date],
    last_idx: int,
    inclusions: dict[str, set[int]],
    floor_at: dict[str, dict[int, float]],
    prehistory_fn,
) -> LaneEpisodes:
    """Run every (symbol) bloodline of one lane through the state machine.

    ``prehistory_fn(start_idx) -> (status, reason)`` labels each episode.
    """
    results: list[rc.EpisodeResult] = []
    at_risk: dict[int, list[tuple[rc.EpisodeResult, rc.RiskState]]] = {}
    for sym in sorted(inclusions):
        incs = inclusions[sym]
        if not incs:
            continue
        bars = bars_by_symbol.get(sym)
        if bars is None:
            continue
        for res in rc.build_episodes(lane, sym, sorted(incs), bars, sessions, last_idx,
                                     floor_at=floor_at.get(sym)):
            res.episode.prehistory, res.episode.prehistory_reason = prehistory_fn(res.episode.start)
            results.append(res)
            for t, state in res.risk_states.items():
                at_risk.setdefault(t, []).append((res, state))
    return LaneEpisodes(lane, results, at_risk)


def reconstructed_inclusions(pool: LanePool, before_idx: int | None = None) -> dict[str, set[int]]:
    """Session indices where each symbol is a member (optionally only before an index)."""
    out: dict[str, set[int]] = {}
    for sym in pool.member.columns:
        idx = np.flatnonzero(pool.member[sym].to_numpy())
        if before_idx is not None:
            idx = idx[idx < before_idx]
        out[sym] = set(int(i) for i in idx)
    return out


def alive_at(res: rc.EpisodeResult, k: int, last_idx: int) -> bool:
    end = res.end if res.end is not None else last_idx
    return res.episode.start <= k <= end and res.status != "FLOOR_UNAVAILABLE"


def overlap_desc(trigger: rc.Trigger, other_lane: LaneEpisodes | None, last_idx: int) -> bool:
    """True when the same symbol has an alive episode in the other parent on k."""
    if other_lane is None:
        return False
    return any(
        r.episode.symbol == trigger.symbol and alive_at(r, trigger.k, last_idx)
        for r in other_lane.results
    )


def latest_score_pct(pool: LanePool, symbol: str, k: int) -> float | None:
    """score_pct from the latest inclusion <= k in that parent (RANGE only)."""
    if pool.lane != rc.RANGE or symbol not in pool.member.columns:
        return None
    member = pool.member[symbol].to_numpy()[: k + 1]
    hits = np.flatnonzero(member)
    if len(hits) == 0:
        return None
    v = pool.score_pct[symbol].iloc[int(hits[-1])]
    return float(v) if np.isfinite(v) else None


@dataclass
class RiskRow:
    """One at-risk control candidate on a trigger date (persisted shape)."""
    control_episode_id: str
    control_symbol: str
    sector: str | None
    mcap: float | None
    score_pct: float | None
    sma_dist: float | None
    age: int
    prehistory: str
    retest_pending: bool
    confirm_pending: bool
    triggered_on_k: bool


def risk_rows(
    trigger: rc.Trigger,
    lane_eps: LaneEpisodes,
    pool: LanePool,
    meta: dict[str, dict],
) -> list[RiskRow]:
    """Same-bloodline at-risk episodes on k, excluding the trigger symbol (§7)."""
    rows: list[RiskRow] = []
    for res, state in lane_eps.at_risk.get(trigger.k, []):
        ep = res.episode
        if ep.episode_id == trigger.episode_id or ep.symbol == trigger.symbol:
            continue
        m = meta.get(ep.symbol) or {}
        rows.append(RiskRow(
            control_episode_id=ep.episode_id,
            control_symbol=ep.symbol,
            sector=m.get("sector"),
            mcap=m.get("mcap"),
            score_pct=latest_score_pct(pool, ep.symbol, trigger.k),
            sma_dist=state.sma_dist,
            age=state.age,
            prehistory=ep.prehistory,
            retest_pending=state.retest_pending,
            confirm_pending=state.confirm_pending,
            triggered_on_k=res.trigger is not None and res.trigger.k == trigger.k,
        ))
    return rows


def select_reclaim_universe(universe_rows: list[dict], cap: int) -> tuple[list[str], dict[str, dict], int]:
    """Native-lane candidates from the run's filtered universe.

    Both lanes need mcap >= $1B (Range; Drift needs $3B), so that is the only
    pre-filter; everything else is decided on bars. Ranked by snapshot dollar
    volume and cut at the OHLCV cap — the cut is recorded, not hidden.
    """
    rows = []
    for r in universe_rows:
        sym = r.get("symbol") or r.get("ticker")
        mcap = r.get("marketCap")
        if not sym or mcap is None:
            continue
        try:
            mcap = float(mcap)
        except (TypeError, ValueError):
            continue
        if mcap < rc.RANGE_MIN_MCAP:
            continue
        price = float(r.get("price") or r.get("lastSale") or 0.0)
        volume = float(r.get("volume") or 0.0)
        rows.append((sym, mcap, r.get("sector"), price * volume))
    rows.sort(key=lambda x: (-x[3], x[0]))
    cap_hits = max(len(rows) - cap, 0)
    kept = rows[:cap]
    meta = {sym: {"mcap": mcap, "sector": sector} for sym, mcap, sector, _ in kept}
    return [r[0] for r in kept], meta, cap_hits
