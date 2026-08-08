"""2.3 — is the ranker's choppy sniper x0.6 multiplier wrong?

Live: 0 sniper picks across 10 choppy runs. The review argued Run E shows choppy
is sniper's BEST regime (+0.451 vs bull +0.215), so the multiplier contradicts
the engine's own data.

CONFOUND the review missed: the `regime` column in those cohorts is
classify_regime(df) over the TICKER's own history — a per-stock trend label. The
live gate uses the MARKET regime from SPY/QQQ. They are different variables, so
the review's comparison was not measuring the thing the multiplier controls.

This re-stamps both cohorts with the SPY-based MARKET regime per entry_date (the
same construction used for the PEAD regime study) and compares like for like,
with bootstrap CIs and per-year splits.
"""
from __future__ import annotations

import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

random.seed(23)


def spy_market_regime() -> dict:
    df = pd.read_parquet(ROOT / "outputs/research/ohlcv_polygon_3y.parquet").rename(
        columns={"_ticker": "ticker"})
    spy = df[df["ticker"] == "SPY"].sort_values("date").reset_index(drop=True)
    c = spy["close"].astype(float)
    s50, s20 = c.rolling(50).mean(), c.rolling(20).mean()
    out = {}
    for i in range(len(spy)):
        d = str(spy["date"].iloc[i])[:10]
        if pd.isna(s50.iloc[i]):
            out[d] = "unknown"
        elif c.iloc[i] > s50.iloc[i] and s20.iloc[i] > s50.iloc[i]:
            out[d] = "bull"
        elif c.iloc[i] < s50.iloc[i] and s20.iloc[i] < s50.iloc[i]:
            out[d] = "bear"
        else:
            out[d] = "choppy"
    return out


def boot(a, b, n=20000):
    diffs = []
    la, lb = len(a), len(b)
    for _ in range(n):
        diffs.append(sum(a[random.randrange(la)] for _ in range(la)) / la
                     - sum(b[random.randrange(lb)] for _ in range(lb)) / lb)
    diffs.sort()
    return st.mean(a) - st.mean(b), diffs[int(0.025 * n)], diffs[int(0.975 * n)]


def summarize(name, df):
    print(f"\n=== {name} — by MARKET regime (SPY-based, what the gate uses) ===")
    print(f"  {'regime':<9}{'n':>7}{'avg':>10}{'WR':>8}   per-year")
    cells = {}
    for reg, g in df.groupby("mkt"):
        r = g["pnl_pct"].tolist()
        if len(r) < 20:
            continue
        cells[reg] = r
        ys = "  ".join(f"{y}:{gg['pnl_pct'].mean():+.2f}(n={len(gg)})"
                       for y, gg in g.groupby("year") if len(gg) >= 10)
        print(f"  {reg:<9}{len(r):>7}{st.mean(r):>+9.3f}%"
              f"{100 * sum(1 for x in r if x > 0) / len(r):>7.1f}%   {ys}")
    return cells


def main():
    reg = spy_market_regime()
    sn = pd.read_csv(ROOT / "outputs/research/sniper_truth_E_live_fixed_2026-07-26.csv")
    mr = pd.read_csv(ROOT / "outputs/research/mr_trades_polygon.csv")
    for d in (sn, mr):
        d["mkt"] = d["entry_date"].astype(str).str[:10].map(reg).fillna("unknown")
        d["year"] = d["entry_date"].astype(str).str[:4]

    sn_cells = summarize("SNIPER (Run E)", sn)
    mr_cells = summarize("MEAN REVERSION (3Y)", mr)

    print("\n=== The question: in CHOPPY, is sniper better than the MR it would displace? ===")
    if "choppy" in sn_cells and "choppy" in mr_cells:
        d, lo, hi = boot(sn_cells["choppy"], mr_cells["choppy"])
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant"
        print(f"  sniper choppy {st.mean(sn_cells['choppy']):+.3f}% (n={len(sn_cells['choppy'])})  "
              f"vs  MR choppy {st.mean(mr_cells['choppy']):+.3f}% (n={len(mr_cells['choppy'])})")
        print(f"  sniper - MR = {d:+.3f}pp  95% CI [{lo:+.3f}, {hi:+.3f}]  {sig}")
    print("\n=== Review's claim: choppy is sniper's best regime? ===")
    if "choppy" in sn_cells and "bull" in sn_cells:
        d, lo, hi = boot(sn_cells["choppy"], sn_cells["bull"])
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant"
        print(f"  sniper choppy - sniper bull = {d:+.3f}pp  95% CI [{lo:+.3f}, {hi:+.3f}]  {sig}")


if __name__ == "__main__":
    main()
