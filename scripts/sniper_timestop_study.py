"""E1 — sniper time_stop: parity forensic, then variant sweep.

PART 1 (forensic): replay the 10 LIVE time_stop trades through walk_exit under
three stop conventions (flat 1.5xATR designed; tier 1.7x; tier 2.5x). Whichever
reproduces the recorded exits is live truth. Clears the 0/67-hard-stops anomaly
before any time_stop conclusion is trusted.

PART 2 (sweep): re-walk the truth-matrix cohort (1335 trades) at C-config and
verify it reproduces the C CSV (control). Then variants:
  ts_days in {1,2,3,5,0} x trigger in {plain close<=entry, cond close<=entry-1xATR}
Variant outcomes derived from the ts=0 walk + first-trigger-day logic, which is
exactly engine bar-order semantics (stop/trail beats time_stop within a bar).
"""
from __future__ import annotations

import json
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import pandas as pd

from src.backtest.exit_engine import ExitBar, ExitParams, walk_exit
from src.config import get_settings

S = get_settings()
SLIP = S.slippage_pct
HERE = Path(__file__).parent
CACHE = Path("data/cache/timestop_study")
CACHE.mkdir(exist_ok=True)

STOP_M, TGT_M, HOLD = 1.5, 3.0, 7  # sniper design


def _get(url: str, tag: str) -> list[dict]:
    cf = CACHE / f"{tag}.json"
    if cf.exists():
        try:
            return json.loads(cf.read_text())
        except Exception:
            pass
    import time
    for attempt in range(4):
        try:
            with httpx.Client(timeout=60) as c:
                r = c.get(url, params={"apiKey": S.polygon_api_key, "adjusted": "true",
                                       "sort": "asc", "limit": 50000})
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


def daily(t: str, d0: str, d1: str) -> list[dict]:
    return _get(f"https://api.polygon.io/v2/aggs/ticker/{t.replace('-', '.')}"
                f"/range/1/day/{d0}/{d1}", f"d_{t}_{d0}_{d1}")


def atr14(bars: list[dict]) -> float:
    if len(bars) < 15:
        return 0.0
    trs = [max(bars[i]["h"] - bars[i]["l"], abs(bars[i]["h"] - bars[i - 1]["c"]),
               abs(bars[i]["l"] - bars[i - 1]["c"])) for i in range(1, len(bars))]
    a = sum(trs[:14]) / 14
    for tr in trs[14:]:
        a = (a * 13 + tr) / 14
    return a


def prep(ticker: str, entry_date: str) -> dict | None:
    """bars from entry day forward + ATR at signal (bar before entry)."""
    d0 = (date.fromisoformat(entry_date) - timedelta(days=110)).isoformat()
    d1 = (date.fromisoformat(entry_date) + timedelta(days=21)).isoformat()
    bars = daily(ticker, d0, d1)
    if not bars:
        return None
    ei = next((i for i, b in enumerate(bars)
               if date.fromtimestamp(b["t"] / 1000).isoformat() >= entry_date), None)
    if ei is None or ei < 16:
        return None
    a = atr14(bars[:ei])              # ATR through the signal bar (pre-entry)
    if a <= 0:
        return None
    entry = round(bars[ei]["o"] * (1 + SLIP), 4)
    fwd = [ExitBar(date=date.fromtimestamp(b["t"] / 1000), open=b["o"], high=b["h"],
                   low=b["l"], close=b["c"]) for b in bars[ei: ei + HOLD + 2]]
    return {"entry": entry, "atr": a, "bars": fwd}


def walk(rec: dict, stop_mult: float, ts_days: int) -> dict:
    p = ExitParams(
        stop=rec["entry"] - stop_mult * rec["atr"],
        target=rec["entry"] + TGT_M * rec["atr"],
        max_hold=HOLD, slippage=SLIP,
        trail_activate_pct=S.trail_activate_pct, trail_distance_pct=S.trail_distance_pct,
        time_stop_days=ts_days, time_stop_eligible=ts_days > 0,
        gap_through=True, check_entry_bar=True,
    )
    o = walk_exit(rec["bars"], rec["entry"], p)
    if o.pnl_pct is None:
        last = rec["bars"][-1].close
        return {"pnl": (last * (1 - SLIP) - rec["entry"]) / rec["entry"] * 100,
                "reason": "open", "idx": None}
    return {"pnl": o.pnl_pct, "reason": o.exit_reason, "idx": o.exit_index}


def variant(rec: dict, base: dict, ts_days: int, cond_atr: float | None) -> tuple[float, str]:
    """Variant time stop applied over the ts=0 walk (engine bar-order semantics)."""
    if ts_days == 0:
        return base["pnl"], base["reason"]
    thr = rec["entry"] - (cond_atr * rec["atr"] if cond_atr else 0.0)
    base_idx = base["idx"] if base["idx"] is not None else len(rec["bars"])
    for i, b in enumerate(rec["bars"]):
        if i == 0:
            continue
        if i + 1 >= ts_days + 1 and b.close <= thr and i < base_idx:
            return ((b.close * (1 - SLIP) - rec["entry"]) / rec["entry"] * 100, "time_stop")
    return base["pnl"], base["reason"]


def summarize(v: list[float]) -> str:
    sd = st.pstdev(v) or 1e-9
    m = st.mean(v)
    return (f"n={len(v):4d} WR={100 * sum(1 for x in v if x > 0) / len(v):5.1f}% "
            f"avg={m:+.3f}% mean/sd={m / sd:+.3f} worst={min(v):+.1f}")


def main() -> None:
    # ---------- PART 1: live forensic ----------
    import io, urllib.request
    _src = sys.argv[1] if len(sys.argv) > 1 else \
        "https://raysyhuang.github.io/multi-agentic-screener/data.json"
    data = (json.load(open(_src)) if not _src.startswith("http")
            else json.loads(urllib.request.urlopen(_src, timeout=60).read()))
    live_ts = [t for t in data["trades"]["sniper|mas_official"]
               if t["exit_reason"] == "time_stop"]
    print(f"PART 1 — forensic on {len(live_ts)} live time_stop trades")
    print(f"{'tkr':6s}{'entry':11s}{'rec_pnl':>8s} | " +
          " | ".join(f"{m}x: reason/pnl" for m in (1.5, 1.7, 2.5)))
    match = {1.5: 0, 1.7: 0, 2.5: 0}
    for t in live_ts:
        rec = prep(t["ticker"], t["entry_date"])
        if not rec:
            print(f"{t['ticker']:6s} NO DATA")
            continue
        cells = []
        for m in (1.5, 1.7, 2.5):
            o = walk(rec, m, S.sniper_time_stop_days)
            ok = (o["reason"] == "time_stop" and abs(o["pnl"] - t["pnl_pct"]) < 0.6)
            match[m] += ok
            cells.append(f"{o['reason'][:10]:10s}{o['pnl']:+6.2f}{'*' if ok else ' '}")
        print(f"{t['ticker']:6s}{t['entry_date']:11s}{t['pnl_pct']:8.2f} | " + " | ".join(cells))
    print(f"reproduced (reason+pnl within 0.6pp): " +
          ", ".join(f"{m}x: {c}/{len(live_ts)}" for m, c in match.items()))

    # ---------- PART 2: cohort sweep ----------
    print("\nPART 2 — truth-matrix cohort sweep (non-bear)")
    cdf = pd.read_csv("outputs/research/sniper_truth_C_timestop_2026-07-26.csv")
    cdf = cdf[cdf["regime"] != "bear"]
    print(f"cohort: {len(cdf)} non-bear trades")

    rows = list(cdf.itertuples())
    with ThreadPoolExecutor(max_workers=8) as ex:
        recs = list(ex.map(lambda r: prep(r.ticker, r.entry_date), rows))
    ok = [(r, rec) for r, rec in zip(rows, recs) if rec]
    print(f"reconstructed {len(ok)}/{len(rows)}")

    # control: reproduce the C arm (ts=1 plain, 1.5x backtest stop) per-trade
    base15 = {id(rec): walk(rec, 1.5, 0) for _, rec in ok}
    agree = pnl_c = 0
    csim = []
    for r, rec in ok:
        sim, _ = variant(rec, base15[id(rec)], 1, None)
        csim.append(sim)
        if abs(sim - r.pnl_pct) < 0.6:
            agree += 1
        pnl_c += r.pnl_pct
    print(f"CONTROL vs C-arm CSV (1.5x stop): per-trade match {agree}/{len(ok)} "
          f"(sim avg {st.mean(csim):+.3f} vs csv avg {pnl_c / len(ok):+.3f})")

    # Forensic (part 1) proved LIVE runs 2.5x tier stops — sweep BOTH conventions.
    for sm in (1.5, 2.5):
        base = base15 if sm == 1.5 else {id(rec): walk(rec, 2.5, 0) for _, rec in ok}
        tag = "backtest 1.5x" if sm == 1.5 else "LIVE-TRUE 2.5x"
        print(f"\n  --- stop convention: {tag} ---")
        for label, ts, cond in (("ts=1 plain (C/live)", 1, None),
                                ("ts=2 plain", 2, None), ("ts=3 plain", 3, None),
                                ("ts=5 plain", 5, None), ("no time stop", 0, None),
                                ("ts=1 cond -1.0xATR", 1, 1.0),
                                ("ts=2 cond -1.0xATR", 2, 1.0),
                                ("ts=1 cond -0.5xATR", 1, 0.5)):
            v, fired = [], 0
            for _, rec in ok:
                pnl, reason = variant(rec, base[id(rec)], ts, cond)
                v.append(pnl)
                fired += reason == "time_stop"
            print(f"    {label:24s} {summarize(v)}  ts_fired={fired}")
        # per-year split for leading variants (house rule)
        for label, ts, cond in (("ts=1 plain", 1, None),
                                ("ts=1 cond -1.0xATR", 1, 1.0),
                                ("no time stop", 0, None)):
            by = {}
            for r, rec in ok:
                pnl, _ = variant(rec, base[id(rec)], ts, cond)
                by.setdefault(r.entry_date[:4], []).append(pnl)
            ys = "  ".join(f"{y}:{st.mean(v):+.2f}(n={len(v)})" for y, v in sorted(by.items()))
            print(f"    split {label:22s} {ys}")


if __name__ == "__main__":
    main()
