"""RECLAIM research watch — today's triggers for the Telegram alert, nothing more.

Ray chose a watch list over the full collector (2026-10-05): each morning the
verified engine (`src/signals/reclaim.py`) scans both native lanes and lists the
triggers that confirmed on the signal date C, with the buy-zone and structural
stop that would apply at today's open. It writes nothing, tracks no outcome and
never touches the official pipeline; it only feeds one labelled section of the
alert.

The label matters: the 2022-2026 replay found no edge in either lane
(`outputs/research/reclaim_native_replay_FINDINGS.md`): DRIFT_G3_NATIVE is
wrong-signed against its controls and RANGE_TECH_NATIVE meets the spec's KILL
rule. Every pool here is reconstructed from today's universe, so no trigger is
ever "clean" in the spec's sense.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from src.research import reclaim_replay as rr
from src.signals import reclaim as rc
from src.utils.trading_calendar import previous_trading_day, trading_sessions

logger = logging.getLogger(__name__)

LANE_LABEL = {rc.RANGE: "Range-tech", rc.DRIFT: "Drift-G3"}


@dataclass
class WatchItem:
    lane: str
    symbol: str
    trigger_date: date
    close_k: float
    sma50_k: float
    stop: float
    max_entry: float          # PASS band at today's open: stop < open <= max_entry
    risk_at_close: float      # (close_k - stop) / close_k, a preview of SKIP_RISK
    earnings_soon: bool
    overlap: bool

    def to_dict(self) -> dict:
        return {
            "lane": LANE_LABEL[self.lane], "ticker": self.symbol,
            "trigger_date": self.trigger_date.isoformat(), "close": self.close_k,
            "sma50": self.sma50_k, "stop": self.stop, "max_entry": self.max_entry,
            "risk_at_close": self.risk_at_close, "earnings_soon": self.earnings_soon,
            "overlap": self.overlap,
        }


@dataclass
class WatchResult:
    signal_date: date
    items: list[WatchItem] = field(default_factory=list)
    alive: dict[str, int] = field(default_factory=dict)   # lane -> episodes alive at C
    symbols: int = 0            # universe scanned
    current: int = 0            # of which have a bar on C (the rest cannot trigger)

    def to_alert(self) -> dict:
        return {
            "signal_date": self.signal_date.isoformat(),
            "items": [i.to_dict() for i in self.items],
            "alive": {LANE_LABEL[k]: v for k, v in self.alive.items()},
            "symbols": self.symbols,
            "current": self.current,
        }


def _earnings_soon(symbol: str, entry: date, calendar: list[dict] | None) -> bool:
    """An upcoming report within EARN_WIN sessions of the entry open."""
    if not calendar:
        return False
    hi = rc.session_offset(entry, rc.EARN_WIN, _is_session)
    for row in calendar:
        if str(row.get("symbol", "")).upper() != symbol:
            continue
        try:
            d = date.fromisoformat(str(row.get("date", ""))[:10])
        except ValueError:
            continue
        if entry <= d <= hi:
            return True
    return False


def _is_session(d: date) -> bool:
    from src.utils.trading_calendar import is_trading_day
    return is_trading_day(d)


def scan(
    bars_raw: dict[str, pd.DataFrame],
    meta: dict[str, dict],
    sessions: list[date],
    earnings_calendar: list[dict] | None = None,
) -> WatchResult:
    """Triggers that confirm on the last session of ``sessions`` (pure)."""
    c_idx = len(sessions) - 1
    C = sessions[-1]
    entry = sessions[-1] + timedelta(days=1)
    while not _is_session(entry):
        entry += timedelta(days=1)
    bars = {s: rc.align_to_sessions(bars_raw.get(s), sessions) for s in meta}
    result = WatchResult(signal_date=C, symbols=len(bars), current=sum(
        1 for b in bars.values() if pd.notna(b["close"].iloc[c_idx])))
    lanes: dict[str, rr.LaneEpisodes] = {}
    for lane in rc.LANES:
        pool = rr.compute_lane_pool(lane, bars, meta)
        incl = rr.reconstructed_inclusions(pool)
        floor_at = {s: {i: float(pool.box_low[s].iloc[i]) for i in incl[s]} for s in incl}
        lanes[lane] = rr.replay_lane(
            lane, bars, sessions, c_idx, incl, floor_at,
            lambda _start: (rc.INSUFFICIENT_PREHISTORY, rc.RECONSTRUCTED_HISTORY),
        )
        result.alive[lane] = sum(1 for r in lanes[lane].results if r.end is None)
    for lane in rc.LANES:
        other = lanes[rc.DRIFT if lane == rc.RANGE else rc.RANGE]
        for res in lanes[lane].results:
            trig = res.trigger
            if trig is None or trig.k != c_idx:
                continue
            result.items.append(WatchItem(
                lane=lane, symbol=trig.symbol, trigger_date=trig.trigger_date,
                close_k=trig.close_k, sma50_k=trig.sma50_k, stop=trig.setup_low,
                max_entry=trig.max_entry,
                risk_at_close=(trig.close_k - trig.setup_low) / trig.close_k,
                earnings_soon=_earnings_soon(trig.symbol, entry, earnings_calendar),
                overlap=rr.overlap_desc(trig, other, c_idx),
            ))
    result.items.sort(key=lambda i: (i.lane, i.risk_at_close, i.symbol))
    return result


async def run_reclaim_watch(
    today: date,
    settings,
    universe_rows: list[dict],
    earnings_calendar: list[dict] | None = None,
    fetch=None,
) -> WatchResult:
    """Fetch history for the native universe and scan.

    ``fetch(tickers, start, end)`` is injected by the pipeline so the watch uses
    the same provider class the run (and its smoke test) uses; the default
    builds its own aggregator for offline callers.
    """
    C = previous_trading_day(today)
    sessions = trading_sessions(C - timedelta(days=settings.reclaim_history_days), C)
    symbols, meta, _ = rr.select_reclaim_universe(universe_rows, settings.max_ohlcv_tickers)
    if not symbols:
        # Both lanes need market cap. The Polygon fallback universe has none, so
        # on those days the honest answer is "could not scan", not "no triggers".
        raise RuntimeError(
            f"reclaim watch: none of {len(universe_rows)} universe rows carry a market cap >= $1B")
    if fetch is None:
        from src.data.aggregator import DataAggregator

        async def fetch(tickers, start, end):
            agg = DataAggregator()
            try:
                return await agg.get_bulk_ohlcv(tickers, start, end)
            finally:
                agg.close()
    raw = await fetch(symbols, sessions[0], C)
    # Coverage is measured on the session-aligned frames, so a date column and a
    # date index are read the same way scan() reads them.
    result = scan(raw, meta, sessions, earnings_calendar)
    if result.current / result.symbols < 0.9:
        raise RuntimeError(
            f"reclaim watch: only {result.current}/{result.symbols} symbols have a bar on {C}")
    logger.info("Reclaim watch %s: %d trigger(s), alive=%s", C, len(result.items),
                {LANE_LABEL[k]: v for k, v in result.alive.items()})
    return result

