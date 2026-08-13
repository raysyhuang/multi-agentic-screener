"""What did the live sniper picks do 1-2 months AFTER the pipeline picked them?

Reads the live sniper trade list from the published dashboard data.json (90d
window), then measures buy-and-hold forward returns from the actual entry
(T+1 open, the real live fill basis) out to +21 and +42 trading days, vs SPY
over the identical window. Polygon-only (strict), provenance stamped.

Input provenance: by default the PUBLIC dashboard export is fetched live and its
SHA-256 recorded, so a run states exactly which snapshot it measured. Pass
--input to replay a frozen copy instead.

Output: aggregates go to stdout. The per-pick table carries realised P&L keyed
by ticker and date, so it is written ONLY when --dump-trades is given, and only
under outputs/ (gitignored). An earlier revision wrote it unconditionally into
scripts/ — a TRACKED directory — which put per-trade P&L one `git add` away from
public history. That has happened in this repo before.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.research.signal_backtest import fetch_ohlcv, get_last_ohlcv_provenance  # noqa: E402

DASHBOARD_URL = "https://raysyhuang.github.io/multi-agentic-screener/data.json"
OUT_DIR = REPO / "outputs" / "research"      # gitignored


def assert_safe_pnl_path(path: Path) -> Path:
    """Refuse to write per-trade P&L anywhere git would track it.

    An earlier version rejected `scripts/` and the repo root by name. That is a
    denylist of two places I happened to think of — it says nothing about the
    destination actually being ignored, and a changed output directory, a
    relaxed ignore rule, or a symlink would walk straight past it. The authority
    on whether git will pick a file up is git, so ask git:

      * the path must resolve beneath the repository (so the checks below mean
        something — `git check-ignore` outside a work tree is not a safety
        statement);
      * it must not be tracked in the index;
      * it must be positively confirmed ignored.

    Fails closed: if git cannot be consulted at all, refuse rather than assume.
    """
    import subprocess

    resolved = path.resolve()
    if not resolved.is_relative_to(REPO.resolve()):
        raise SystemExit(f"refusing: per-trade P&L destination is outside the repo: {resolved}")

    def _git(*args: str) -> int:
        try:
            return subprocess.run(("git", "-C", str(REPO), *args),
                                  capture_output=True).returncode
        except OSError as exc:
            raise SystemExit(f"refusing: cannot consult git about {resolved} ({exc})") from exc

    if _git("ls-files", "--error-unmatch", str(resolved)) == 0:
        raise SystemExit(f"refusing: {resolved} is TRACKED by git — per-trade P&L must not be")
    if _git("check-ignore", "-q", str(resolved)) != 0:
        raise SystemExit(
            f"refusing: {resolved} is not git-ignored. Per-trade P&L may only be "
            "written to an explicitly ignored path."
        )
    return resolved

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--input", help="frozen dashboard export to replay instead of fetching")
_ap.add_argument("--dump-trades", action="store_true",
                 help="also write the per-pick CSV (contains realised P&L) under outputs/")
ARGS = _ap.parse_args()

if ARGS.input:
    _raw = Path(ARGS.input).read_bytes()
    _origin = ARGS.input
else:
    with urllib.request.urlopen(DASHBOARD_URL, timeout=60) as _r:  # noqa: S310
        _raw = _r.read()
    _origin = DASHBOARD_URL
print(f"input: {_origin}\n  sha256 {hashlib.sha256(_raw).hexdigest()}")
data = json.loads(_raw)
trades = sorted(data["trades"]["sniper|mas_official"], key=lambda t: (t["signal_date"], t["ticker"]))

tickers = sorted({t["ticker"] for t in trades})
px = fetch_ohlcv(tickers + ["SPY"], years=1.0, source="polygon", strict=True, no_cache=True)
prov = get_last_ohlcv_provenance()
print("PROVENANCE:", json.dumps(prov, default=str)[:600])

frames = {}
for tk, df in px.items():
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.date
    frames[tk] = d.sort_values("date").reset_index(drop=True)

HORIZONS = [21, 42]  # ~1 month, ~2 months of trading days


def forward(tk: str, entry_date: str) -> dict:
    d = frames.get(tk)
    out: dict[str, float | None] = {}
    if d is None:
        return {"entry_open": None, **{f"r{h}": None for h in HORIZONS}}
    ed = pd.Timestamp(entry_date).date()
    idx = d.index[d["date"] == ed]
    if len(idx) == 0:
        nxt = d.index[d["date"] > ed]
        if len(nxt) == 0:
            return {"entry_open": None, **{f"r{h}": None for h in HORIZONS}}
        i = int(nxt[0])
    else:
        i = int(idx[0])
    entry = float(d.loc[i, "open"])
    out["entry_open"] = entry
    for h in HORIZONS:
        j = i + h
        if j < len(d):
            out[f"r{h}"] = (float(d.loc[j, "close"]) / entry - 1) * 100
            out[f"bars{h}"] = h
        else:
            # partial: not enough forward data yet
            last = len(d) - 1
            out[f"r{h}"] = None
            out[f"bars{h}"] = last - i
        # path extremes over the horizon
        seg = d.loc[i: min(i + h, len(d) - 1)]
        out[f"mfe{h}"] = (seg["high"].max() / entry - 1) * 100
        out[f"mae{h}"] = (seg["low"].min() / entry - 1) * 100
    return out


rows = []
for t in trades:
    f = forward(t["ticker"], t["entry_date"])
    s = forward("SPY", t["entry_date"])
    row = {
        "signal_date": t["signal_date"],
        "entry_date": t["entry_date"],
        "ticker": t["ticker"],
        "realized_pnl": t["pnl_pct"],
        "hold_days": t["hold_days"],
        "exit_reason": t["exit_reason"],
    }
    for h in HORIZONS:
        row[f"fwd_{h}"] = f.get(f"r{h}")
        row[f"spy_{h}"] = s.get(f"r{h}")
        row[f"alpha_{h}"] = (
            None if f.get(f"r{h}") is None or s.get(f"r{h}") is None
            else f[f"r{h}"] - s[f"r{h}"]
        )
        row[f"mfe_{h}"] = f.get(f"mfe{h}")
        row[f"mae_{h}"] = f.get(f"mae{h}")
        row[f"bars_{h}"] = f.get(f"bars{h}")
    rows.append(row)

df = pd.DataFrame(rows)
if ARGS.dump_trades:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _csv = OUT_DIR / "sniper_live_picks_forward_returns.csv"
    assert_safe_pnl_path(_csv)
    df.to_csv(_csv, index=False)
    print(f"\nwrote {_csv.relative_to(REPO)} (gitignored; contains per-trade P&L)")
else:
    print("\n(per-pick CSV not written; pass --dump-trades to emit it under outputs/)")

pd.set_option("display.width", 200, "display.max_rows", 200)
print("\n=== PER-PICK ===")
show = df[["signal_date", "ticker", "realized_pnl", "hold_days", "exit_reason",
           "fwd_21", "alpha_21", "fwd_42", "alpha_42", "mae_42", "mfe_42"]]
print(show.round(2).to_string(index=False))


def summarize(col_r: str, label: str, sub: pd.DataFrame) -> None:
    x = sub[col_r].dropna()
    if x.empty:
        print(f"{label}: no complete windows")
        return
    a = sub[f"alpha_{col_r.split('_')[1]}"].dropna() if col_r.startswith("fwd") else None
    print(f"{label}: n={len(x)} mean={x.mean():+.2f}% median={x.median():+.2f}% "
          f"win={100 * (x > 0).mean():.0f}% best={x.max():+.1f}% worst={x.min():+.1f}%"
          + (f" | alpha vs SPY mean={a.mean():+.2f}% win={100 * (a > 0).mean():.0f}%" if a is not None else ""))


print("\n=== SUMMARY (all picks, incl. duplicates of same ticker) ===")
r = df["realized_pnl"]
print(f"Realized (as traded): n={len(r)} mean={r.mean():+.2f}% median={r.median():+.2f}% "
      f"win={100 * (r > 0).mean():.0f}% total_sum={r.sum():+.1f}%")
for h in HORIZONS:
    summarize(f"fwd_{h}", f"Buy-and-hold +{h}d ({'~1mo' if h == 21 else '~2mo'})", df)

# Apples-to-apples: restrict to picks that have BOTH complete windows
both = df.dropna(subset=["fwd_21", "fwd_42"])
print(f"\n=== MATCHED COHORT (picks with a full 2-month window, n={len(both)}) ===")
rr = both["realized_pnl"]
print(f"Realized: mean={rr.mean():+.2f}% median={rr.median():+.2f}% win={100 * (rr > 0).mean():.0f}%")
for h in HORIZONS:
    summarize(f"fwd_{h}", f"Buy-and-hold +{h}d", both)
sp = both["spy_42"]
print(f"SPY same windows +42d: mean={sp.mean():+.2f}%")

# Did the trail exit too early? compare realized vs the 1-month hold per pick
comp = both.assign(delta_21=both["fwd_21"] - both["realized_pnl"],
                   delta_42=both["fwd_42"] - both["realized_pnl"])
print(f"\nHolding 1mo instead of exiting: mean delta {comp['delta_21'].mean():+.2f}pp, "
      f"better on {100 * (comp['delta_21'] > 0).mean():.0f}% of picks")
print(f"Holding 2mo instead of exiting: mean delta {comp['delta_42'].mean():+.2f}pp, "
      f"better on {100 * (comp['delta_42'] > 0).mean():.0f}% of picks")


