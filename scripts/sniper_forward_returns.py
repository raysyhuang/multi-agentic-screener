"""What did the live picks do AFTER the pipeline picked them?

Reads a live trade stream from the published dashboard data.json (90d window;
--stream selects it, default sniper|mas_official), then measures buy-and-hold
forward returns from the actual entry (T+1 open, the real live fill basis) out
to each --horizons bar count (default 21,42), vs SPY over the identical window.
Polygon-only (strict), provenance stamped.

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
import random
import subprocess
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

    Every git outcome is handled EXPLICITLY, because "nonzero" conflates two
    opposite meanings. `ls-files --error-unmatch` exits 1 for the expected
    not-tracked case and 128 when the index is corrupt or unreadable. Treating
    any nonzero as "not tracked" turns a git FAILURE into a safe answer — with a
    broken index, ls-files returns 128 and check-ignore can still return 0, and
    the destination is allowed. An unanswerable question must refuse, not pass.
    """
    resolved = path.resolve()
    if not resolved.is_relative_to(REPO.resolve()):
        raise SystemExit(f"refusing: per-trade P&L destination is outside the repo: {resolved}")

    def _git(*args: str) -> int:
        try:
            return subprocess.run(("git", "-C", str(REPO), *args),
                                  capture_output=True).returncode
        except OSError as exc:
            raise SystemExit(f"refusing: cannot consult git about {resolved} ({exc})") from exc

    # 0 = tracked, 1 = not tracked, anything else = git could not answer.
    code = _git("ls-files", "--error-unmatch", str(resolved))
    if code == 0:
        raise SystemExit(f"refusing: {resolved} is TRACKED by git — per-trade P&L must not be")
    if code != 1:
        raise SystemExit(
            f"refusing: git ls-files failed (exit {code}) for {resolved}; "
            "cannot establish that the destination is untracked"
        )

    # 0 = ignored, 1 = not ignored, anything else = git could not answer.
    code = _git("check-ignore", "-q", str(resolved))
    if code == 1:
        raise SystemExit(
            f"refusing: {resolved} is not git-ignored. Per-trade P&L may only be "
            "written to an explicitly ignored path."
        )
    if code != 0:
        raise SystemExit(
            f"refusing: git check-ignore failed (exit {code}) for {resolved}; "
            "cannot establish that the destination is ignored"
        )
    return resolved

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--input", help="frozen dashboard export to replay instead of fetching")
_ap.add_argument("--dump-trades", action="store_true",
                 help="also write the per-pick CSV (contains realised P&L) under outputs/")
DEFAULT_STREAM = "sniper|mas_official"
DEFAULT_HORIZONS = "21,42"
_ap.add_argument("--stream", default=DEFAULT_STREAM,
                 help="dashboard trade stream key, e.g. mean_reversion|mas_official")
_ap.add_argument("--horizons", default=DEFAULT_HORIZONS,
                 help="comma-separated forward horizons in trading bars (default 21,42)")
_ap.add_argument("--baseline-input", default=None,
                 help="an EARLIER frozen bundle; picks present in it are 'overlap', the rest "
                      "are strictly out-of-sample relative to a finding made on that bundle")
ARGS = _ap.parse_args()

# With none of the newer flags the script prints the same REPORT LAYOUT it
# printed before --stream/--horizons existed; the per-horizon paired statistics,
# cluster CIs and the stream line only appear in EXTENDED mode. One thing is
# deliberately NOT gated: SPY is now measured over the ticker's actual bar
# dates (`spy_between`) on every path, so a default run can print different
# spy_*/alpha_* values than the pre-fix script for any pick whose ticker is
# missing a bar. The 2026-08-13 sniper run has 0 such rows and reproduces.
EXTENDED = (ARGS.stream != DEFAULT_STREAM or ARGS.horizons != DEFAULT_HORIZONS
            or ARGS.baseline_input is not None)

# Human labels for the summary lines. Anything not listed prints as "+Nd".
_HORIZON_LABELS = {5: "~1wk", 7: "~1.5wk", 21: "~1mo", 42: "~2mo", 63: "~3mo"}
_WINDOW_LABELS = {5: "5-bar", 7: "7-bar", 21: "1-month", 42: "2-month", 63: "3-month"}


def _hlabel(h: int) -> str:
    return _HORIZON_LABELS.get(h, f"+{h}d")


def _wlabel(h: int) -> str:
    return _WINDOW_LABELS.get(h, f"{h}-bar")

if ARGS.input:
    _raw = Path(ARGS.input).read_bytes()
    _origin = ARGS.input
else:
    with urllib.request.urlopen(DASHBOARD_URL, timeout=60) as _r:  # noqa: S310
        _raw = _r.read()
    _origin = DASHBOARD_URL
print(f"input: {_origin}\n  sha256 {hashlib.sha256(_raw).hexdigest()}")
data = json.loads(_raw)
if ARGS.stream not in data["trades"]:
    raise SystemExit(f"stream {ARGS.stream!r} not in bundle; have {sorted(data['trades'])}")
if EXTENDED:
    print(f"stream: {ARGS.stream}")
trades = sorted(data["trades"][ARGS.stream], key=lambda t: (t["signal_date"], t["ticker"]))

tickers = sorted({t["ticker"] for t in trades})
px = fetch_ohlcv(tickers + ["SPY"], years=1.0, source="polygon", strict=True, no_cache=True)
prov = get_last_ohlcv_provenance()
print("PROVENANCE:", json.dumps(prov, default=str)[:600])

frames = {}
for tk, df in px.items():
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.date
    frames[tk] = d.sort_values("date").reset_index(drop=True)

HORIZONS = [int(h) for h in ARGS.horizons.split(",") if h.strip()]  # trading bars
if not HORIZONS:
    raise SystemExit("--horizons must name at least one horizon")


def forward(tk: str, entry_date: str) -> dict:
    """Buy-and-hold from the entry bar's open to the close h bars later, plus the
    calendar dates of those two bars (so the benchmark can be measured over the
    SAME dates rather than over its own bar count — see spy_between)."""
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
    out["entry_bar_date"] = d.loc[i, "date"]
    for h in HORIZONS:
        j = i + h
        if j < len(d):
            out[f"r{h}"] = (float(d.loc[j, "close"]) / entry - 1) * 100
            out[f"bars{h}"] = h
            out[f"exit_bar_date{h}"] = d.loc[j, "date"]
        else:
            # partial: not enough forward data yet
            last = len(d) - 1
            out[f"r{h}"] = None
            out[f"bars{h}"] = last - i
            out[f"exit_bar_date{h}"] = None
        # path extremes over the horizon
        seg = d.loc[i: min(i + h, len(d) - 1)]
        out[f"mfe{h}"] = (seg["high"].max() / entry - 1) * 100
        out[f"mae{h}"] = (seg["low"].min() / entry - 1) * 100
    return out


def spy_between(date_a, date_b) -> float | None:
    """SPY open on date_a -> close on date_b, over EXACTLY those dates. None if
    SPY lacks either bar, so a pick never gets an alpha measured over a
    different window than its own return (a ticker missing a bar used to shift
    its horizon date while SPY kept counting its own bars)."""
    d = frames.get("SPY")
    if d is None or date_a is None or date_b is None:
        return None
    ia = d.index[d["date"] == date_a]
    ib = d.index[d["date"] == date_b]
    if len(ia) == 0 or len(ib) == 0:
        return None
    return (float(d.loc[int(ib[0]), "close"]) / float(d.loc[int(ia[0]), "open"]) - 1) * 100


rows = []
_misaligned = 0   # horizon-rows where SPY's own bar count would have used different dates
for t in trades:
    f = forward(t["ticker"], t["entry_date"])
    s_own = forward("SPY", t["entry_date"])   # legacy: SPY counted its own bars
    row = {
        "signal_date": t["signal_date"],
        "entry_date": t["entry_date"],
        "ticker": t["ticker"],
        "realized_pnl": t["pnl_pct"],
        "hold_days": t["hold_days"],
        "exit_reason": t["exit_reason"],
    }
    for h in HORIZONS:
        spy_r = spy_between(f.get("entry_bar_date"), f.get(f"exit_bar_date{h}"))
        if (f.get(f"r{h}") is not None
                and (s_own.get("entry_bar_date") != f.get("entry_bar_date")
                     or s_own.get(f"exit_bar_date{h}") != f.get(f"exit_bar_date{h}"))):
            _misaligned += 1
        row[f"fwd_{h}"] = f.get(f"r{h}")
        row[f"spy_{h}"] = spy_r
        row[f"alpha_{h}"] = (
            None if f.get(f"r{h}") is None or spy_r is None
            else f[f"r{h}"] - spy_r
        )
        row[f"mfe_{h}"] = f.get(f"mfe{h}")
        row[f"mae_{h}"] = f.get(f"mae{h}")
        row[f"bars_{h}"] = f.get(f"bars{h}")
    rows.append(row)
if EXTENDED:
    print(f"ticker/SPY window alignment: {_misaligned} horizon-row(s) would have used "
          f"different dates under SPY's own bar count (now measured on identical dates)")

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
_last = HORIZONS[-1]
show = df[["signal_date", "ticker", "realized_pnl", "hold_days", "exit_reason"]
          + [c for h in HORIZONS for c in (f"fwd_{h}", f"alpha_{h}")]
          + [f"mae_{_last}", f"mfe_{_last}"]]
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
    summarize(f"fwd_{h}", f"Buy-and-hold +{h}d ({_hlabel(h)})", df)

# Apples-to-apples: restrict to picks that have ALL requested windows complete
both = df.dropna(subset=[f"fwd_{h}" for h in HORIZONS])
print(f"\n=== MATCHED COHORT (picks with a full {_wlabel(_last)} window, n={len(both)}) ===")
rr = both["realized_pnl"]
print(f"Realized: mean={rr.mean():+.2f}% median={rr.median():+.2f}% win={100 * (rr > 0).mean():.0f}%")
for h in HORIZONS:
    summarize(f"fwd_{h}", f"Buy-and-hold +{h}d", both)
sp = both[f"spy_{_last}"]
print(f"SPY same windows +{_last}d: mean={sp.mean():+.2f}%")


def _boot_ci(x: list[float], n_boot: int = 10_000, seed: int = 20260918) -> tuple[float, float]:
    """Seeded percentile bootstrap 95% CI of the mean, resampling picks iid.
    Same-day picks share one market path, so this interval is too narrow; the
    cluster interval below is the honest one."""
    rng = random.Random(seed)
    k = len(x)
    means = sorted(sum(x[rng.randrange(k)] for _ in range(k)) / k for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def _cluster_boot_ci(x: list[float], groups: list, n_boot: int = 10_000,
                     seed: int = 20260918) -> tuple[float, float]:
    """Seeded percentile bootstrap 95% CI of the mean that resamples ENTRY DATES
    (clusters) with replacement and keeps every pick on a drawn date. Picks that
    entered together are not independent draws of the strategy; with one date
    per pick this reduces to the iid interval."""
    rng = random.Random(seed)
    by_g: dict = {}
    for v, g in zip(x, groups):
        by_g.setdefault(g, []).append(v)
    keys = sorted(by_g)
    sums = [sum(by_g[k]) for k in keys]
    cnts = [len(by_g[k]) for k in keys]
    m = len(keys)
    means = []
    for _ in range(n_boot):
        tot = cnt = 0.0
        for _j in range(m):
            i = rng.randrange(m)
            tot += sums[i]
            cnt += cnts[i]
        means.append(tot / cnt)
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def _paired_block(sub: pd.DataFrame, h: int, label: str) -> None:
    """Paired per-pick delta of holding h bars instead of taking the live exit:
    n, mean, median, iid CI, cluster-by-entry-date CI, hit rate, split-half by
    entry date (the FORWARD_DECAY_FINDINGS stress test: halves disagreeing in
    sign is not a finding)."""
    s = sub.dropna(subset=[f"fwd_{h}"]).sort_values(["entry_date", "ticker"]).reset_index(drop=True)
    d = (s[f"fwd_{h}"] - s["realized_pnl"]).tolist()
    if len(d) < 3:
        print(f"  {label} @{h}b: n={len(d)} (too few complete windows)")
        return
    lo, hi = _boot_ci(d)
    clo, chi = _cluster_boot_ci(d, s["entry_date"].tolist())
    mid = len(d) // 2
    h1 = sum(d[:mid]) / mid if mid else float("nan")
    h2 = sum(d[mid:]) / (len(d) - mid)
    print(f"  {label} @{h}b: n={len(d)} delta {sum(d) / len(d):+.2f}pp "
          f"iid 95% CI [{lo:+.2f}, {hi:+.2f}] cluster-by-date 95% CI [{clo:+.2f}, {chi:+.2f}], "
          f"median {float(pd.Series(d).median()):+.2f}pp, "
          f"hit {100 * sum(1 for v in d if v > 0) / len(d):.0f}%, split-half {h1:+.2f} / {h2:+.2f}, "
          f"fwd mean {s[f'fwd_{h}'].mean():+.2f}%, alpha vs SPY {s[f'alpha_{h}'].mean():+.2f}%, "
          f"realized {s['realized_pnl'].mean():+.2f}%, entry dates {s['entry_date'].nunique()}")


# Did the exit come too early? Legacy (default flags): the two one-line deltas
# on the matched cohort, unchanged since 2026-08-13. Extended: the same paired
# delta per horizon on EVERY pick with a complete window at that horizon (the
# newest picks are exactly the ones that lack the longer windows, so the
# matched cohort under-represents them), plus the matched cohort for comparison.
comp = both.assign(**{f"delta_{h}": both[f"fwd_{h}"] - both["realized_pnl"] for h in HORIZONS})
if not EXTENDED:
    print(f"\nHolding 1mo instead of exiting: mean delta {comp['delta_21'].mean():+.2f}pp, "
          f"better on {100 * (comp['delta_21'] > 0).mean():.0f}% of picks")
    print(f"Holding 2mo instead of exiting: mean delta {comp['delta_42'].mean():+.2f}pp, "
          f"better on {100 * (comp['delta_42'] > 0).mean():.0f}% of picks")
else:
    print("\n=== PAIRED DELTA: hold h bars instead of taking the live exit ===")
    for h in HORIZONS:
        _paired_block(df, h, "ALL picks with a complete window")
    for h in HORIZONS:
        _paired_block(both, h, f"MATCHED cohort (all {len(HORIZONS)} windows complete)")

# Out-of-sample check against an earlier bundle. The dashboard window rolls, so
# a re-run months later still shares most of its picks with the run that made
# the original finding; only the picks NOT in the baseline bundle are new
# evidence. Uses every pick with a complete window at that horizon (not the
# matched cohort), because the newest picks are exactly the ones that lack the
# longer windows.
if ARGS.baseline_input:
    _base = json.loads(Path(ARGS.baseline_input).read_bytes())
    _base_keys = {(t["signal_date"], t["ticker"]) for t in _base["trades"].get(ARGS.stream, [])}
    df["in_baseline"] = [(s, t) in _base_keys for s, t in zip(df["signal_date"], df["ticker"])]
    print(f"\n=== OUT-OF-SAMPLE vs {ARGS.baseline_input} ===")
    print(f"picks: {len(df)} total, {int(df['in_baseline'].sum())} overlap, "
          f"{int((~df['in_baseline']).sum())} new")
    for label, sub in (("NEW (out-of-sample)", df[~df["in_baseline"]]),
                       ("OVERLAP (in baseline)", df[df["in_baseline"]])):
        for h in HORIZONS:
            _paired_block(sub, h, label)


