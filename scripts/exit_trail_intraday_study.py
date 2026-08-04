"""Does the same-bar trail guard cost us money? (Answer: no — it earns it.)

`exit_engine.walk_exit` forbids a trailing stop from enforcing on the bar it
arms, because daily OHLC cannot prove the high preceded the low. With a median
hold of 1 day the arming bar IS usually the entry bar, so single-bar round trips
are forced to the hard stop no matter how high the bar peaked.

This script answers two separate questions from Polygon 1-minute data:

  1. ORDERING — on those bars, did the high actually come first? (measurement,
     not assumption). Result 2026-08-04: 25 high-first vs 13 low-first, so the
     engine's forced assumption is wrong roughly 2:1.

  2. P&L — what happens if the trail is enforced intraday instead? This must be
     run over WINNERS AND LOSERS. Replaying only the losers (the obvious thing to
     do, since they look like the victims) shows a huge fake gain. The full
     replay shows the opposite: WR jumps to ~90% while avg pnl/trade collapses to
     ~0, because a 0.3% trail that truly enforces caps every winner at ~+0.3%.

     >>> Win rate is purchasable and nearly worthless. Optimize avg P&L/trade. <<<

Conclusion: do NOT "fix" the guard. It is conservative in the right direction,
which also means recorded live P&L is the pessimistic branch.

Usage:
    python scripts/exit_trail_intraday_study.py            # both phases
    python scripts/exit_trail_intraday_study.py --phase ordering
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from src.config import get_settings

DASHBOARD = "https://raysyhuang.github.io/multi-agentic-screener/data.json"
CACHE = Path("data/cache/intraday_study")
STREAMS = ("mean_reversion|mas_official", "sniper|mas_official",
           "mean_reversion|mr_manual_sleeve")


def _settings():
    s = get_settings()
    return s.polygon_api_key, s.slippage_pct, s.trail_activate_pct, s.trail_distance_pct


def _agg(key: str, sym: str, span: str, d0: str, d1: str, tag: str) -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"{tag}.json"
    if cf.exists():
        try:
            return json.loads(cf.read_text())
        except Exception:
            pass
    url = (f"https://api.polygon.io/v2/aggs/ticker/{sym.replace('-', '.')}"
           f"/range/1/{span}/{d0}/{d1}")
    params = {"apiKey": key, "adjusted": "true", "sort": "asc", "limit": 50000}
    # Hundreds of calls per run — a single transient SSL/connect blip must not
    # abort the study half-way through.
    for attempt in range(4):
        try:
            with httpx.Client(timeout=60) as c:
                r = c.get(url, params=params)
            if r.status_code != 200:
                return []
            res = r.json().get("results", []) or []
            cf.write_text(json.dumps(res))
            return res
        except (httpx.TransportError, httpx.HTTPError):
            if attempt == 3:
                return []
            time.sleep(1.5 * (attempt + 1))
    return []


def _et(ms: int) -> str:
    return (datetime.fromtimestamp(ms / 1000, timezone.utc)
            - timedelta(hours=4)).strftime("%H:%M")


def _rth(bars: list[dict]) -> list[dict]:
    """Regular hours only, so the minute path matches the daily OHLC envelope."""
    return [b for b in bars if "09:30" <= _et(b["t"]) <= "16:00"]


def load_trades(src: str) -> dict[str, list[dict]]:
    if src.startswith("http"):
        data = httpx.get(src, timeout=60).json()
    else:
        data = json.loads(Path(src).read_text())
    return {s: data["trades"][s] for s in STREAMS if s in data["trades"]}


def replay(tr: dict, key: str, slip: float, act: float, dist: float) -> tuple[float, float | None, str]:
    """(recorded_pnl, intraday_pnl, tag) for one trade under an enforced trail."""
    rec = tr["pnl_pct"]
    if (tr.get("mfe") or 0) < act:
        return rec, rec, "never-armed"          # trail never arms => identical

    tkr, ed, xd = tr["ticker"], tr["entry_date"], tr["exit_date"]
    d1 = (date.fromisoformat(xd) + timedelta(days=2)).isoformat()
    daily = _agg(key, tkr, "day", ed, d1, f"d_{tkr}_{ed}_{d1}")
    if not daily:
        return rec, None, "no-daily"
    entry = round(daily[0]["o"] * (1 + slip), 4)
    walk = daily[: tr["hold_days"] + 1]

    # Hard stop: exact where the recorded exit WAS the stop; otherwise the
    # recorded outcome proves it sat below the observed low, so place it there.
    if tr["exit_reason"] == "stop":
        stop = entry * (1 + rec / 100) / (1 - slip)
    else:
        stop = min(b["l"] for b in walk) * 0.9999

    hwm, active = entry, False
    for b in walk:
        day = datetime.fromtimestamp(b["t"] / 1000, timezone.utc).date().isoformat()
        mb = _rth(_agg(key, tkr, "minute", day, day, f"m_{tkr}_{day}"))
        if not mb:
            return rec, None, "no-minute"
        for m in mb:
            if not active and m["l"] <= stop:
                return rec, (stop * (1 - slip) - entry) / entry * 100, "stop"
            hwm = max(hwm, m["h"])
            if not active and (hwm - entry) / entry * 100 >= act:
                active = True
                continue            # same guard spirit, at minute resolution
            if active:
                trail = max(stop, hwm * (1 - dist / 100))
                if m["l"] <= trail:
                    return rec, (trail * (1 - slip) - entry) / entry * 100, "trail"
    return rec, (walk[-1]["c"] * (1 - slip) - entry) / entry * 100, "expiry"


def phase_ordering(trades: dict[str, list[dict]], key: str, slip: float, act: float) -> None:
    print("=" * 78)
    print("PHASE 1 — did the arming high actually precede the stop-breaching low?")
    print("=" * 78)
    seen: set[tuple] = set()
    hi = lo = other = 0
    for stream, rows in trades.items():
        for t in rows:
            k = (t["ticker"], t["entry_date"], t["exit_date"])
            if k in seen or t["pnl_pct"] > 0 or (t.get("mfe") or 0) < act:
                continue
            seen.add(k)
            ed, xd = t["entry_date"], t["exit_date"]
            d1 = (date.fromisoformat(xd) + timedelta(days=4)).isoformat()
            daily = _agg(key, t["ticker"], "day", ed, d1, f"d_{t['ticker']}_{ed}_{d1}")
            if not daily:
                continue
            entry = round(daily[0]["o"] * (1 + slip), 4)
            walk = daily[: t["hold_days"] + 1]
            arm = next((b for b in walk if (b["h"] - entry) / entry * 100 >= act), None)
            if arm is None:
                continue
            day = datetime.fromtimestamp(arm["t"] / 1000, timezone.utc).date().isoformat()
            mb = _rth(_agg(key, t["ticker"], "minute", day, day, f"m_{t['ticker']}_{day}"))
            if not mb:
                continue
            stop = entry * (1 + t["pnl_pct"] / 100) / (1 - slip)
            t_arm = next((_et(b["t"]) for b in mb if b["h"] >= entry * (1 + act / 100)), None)
            t_stop = next((_et(b["t"]) for b in mb if b["l"] <= stop), None)
            if t_arm and t_stop:
                if t_arm < t_stop:
                    hi += 1
                else:
                    lo += 1
            else:
                other += 1
    print(f"  high first (trail armed before the stop was touched): {hi}")
    print(f"  low  first (stop genuinely hit first)               : {lo}")
    print(f"  unresolved                                           : {other}\n")


def phase_pnl(trades: dict[str, list[dict]], key: str, slip: float,
              act: float, dist: float) -> None:
    print("=" * 78)
    print("PHASE 2 — enforce the trail intraday, over WINNERS AND LOSERS")
    print("=" * 78)
    for stream, rows in trades.items():
        with ThreadPoolExecutor(max_workers=8) as ex:
            res = list(ex.map(lambda t: replay(t, key, slip, act, dist), rows))
        pairs = [(r, i) for r, i, _ in res if i is not None]
        if not pairs:
            continue
        rec = [r for r, _ in pairs]
        intr = [i for _, i in pairs]
        print(f"{stream}  n={len(pairs)} (dropped {len(rows) - len(pairs)})")
        for tag, v in (("RECORDED (deferred trail)", rec), ("INTRADAY (enforced)", intr)):
            print(f"  {tag:26s} WR={100 * sum(1 for x in v if x > 0) / len(v):5.1f}%  "
                  f"avg={st.mean(v):+.3f}%  sum={sum(v):+7.1f}pp")
        w = [(r, i) for r, i in pairs if r > 0]
        losers = [(r, i) for r, i in pairs if r <= 0]
        if w:
            print(f"    winners n={len(w):3d}: {sum(r for r, _ in w):+7.1f}pp -> "
                  f"{sum(i for _, i in w):+7.1f}pp")
        if losers:
            print(f"    losers  n={len(losers):3d}: {sum(r for r, _ in losers):+7.1f}pp -> "
                  f"{sum(i for _, i in losers):+7.1f}pp")
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DASHBOARD, help="dashboard data.json URL or path")
    ap.add_argument("--phase", choices=("ordering", "pnl", "both"), default="both")
    args = ap.parse_args()

    key, slip, act, dist = _settings()
    if not key:
        raise SystemExit("polygon_api_key required (intraday plan)")
    trades = load_trades(args.source)

    if args.phase in ("ordering", "both"):
        phase_ordering(trades, key, slip, act)
    if args.phase in ("pnl", "both"):
        phase_pnl(trades, key, slip, act, dist)


if __name__ == "__main__":
    main()
