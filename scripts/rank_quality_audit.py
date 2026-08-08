"""Does rank ordering carry information? (Tier-1 item 2.2, backtest side)

The pipeline ranks ~10 candidates daily and takes the top 2. If score->outcome
rank is noise, that choice is a coin flip and the alpha is being IN the ten, not
being first. The review reported a single-point split; this adds the CIs and
per-year splits the house rule requires before acting on anything.

Runs on the frozen truth-matrix cohorts (relative arms only — absolute sniper
numbers from this universe are untrustworthy, see the universe finding).
"""
from __future__ import annotations

import random
import statistics as st
import sys
from pathlib import Path

import pandas as pd

random.seed(11)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def boot_diff(a: list[float], b: list[float], n: int = 20000) -> tuple[float, float, float]:
    """Bootstrap CI on mean(a) - mean(b) for two independent groups."""
    diffs = []
    la, lb = len(a), len(b)
    for _ in range(n):
        sa = sum(a[random.randrange(la)] for _ in range(la)) / la
        sb = sum(b[random.randrange(lb)] for _ in range(lb)) / lb
        diffs.append(sa - sb)
    diffs.sort()
    return (st.mean(a) - st.mean(b), diffs[int(0.025 * n)], diffs[int(0.975 * n)])


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def audit(path: Path, label: str, top_k: int = 2) -> None:
    df = pd.read_csv(path)
    if not {"entry_date", "score", "pnl_pct"}.issubset(df.columns):
        print(f"{label}: missing columns"); return
    df = df.sort_values(["entry_date", "score"], ascending=[True, False]).copy()
    df["rank"] = df.groupby("entry_date").cumcount() + 1
    df["year"] = df["entry_date"].astype(str).str[:4]
    # only days that actually had a CHOICE to make
    sizes = df.groupby("entry_date")["rank"].transform("max")
    df = df[sizes > top_k]

    top = df[df["rank"] <= top_k]["pnl_pct"].tolist()
    rest = df[df["rank"] > top_k]["pnl_pct"].tolist()
    if len(top) < 30 or len(rest) < 30:
        print(f"{label}: too few ({len(top)}/{len(rest)})"); return

    d, lo, hi = boot_diff(top, rest)
    ic = spearman(df["rank"].tolist(), df["pnl_pct"].tolist())
    sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant"
    print(f"\n=== {label} (days with >{top_k} candidates) ===")
    print(f"  rank 1-{top_k}: n={len(top):5d}  avg={st.mean(top):+.3f}%  "
          f"WR={100 * sum(1 for x in top if x > 0) / len(top):.1f}%")
    print(f"  rank {top_k + 1}+ : n={len(rest):5d}  avg={st.mean(rest):+.3f}%  "
          f"WR={100 * sum(1 for x in rest if x > 0) / len(rest):.1f}%")
    print(f"  edge of being picked: {d:+.3f}pp  95% CI [{lo:+.3f}, {hi:+.3f}]  {sig}")
    print(f"  Spearman(rank, pnl) = {ic:+.4f}   (0 => ordering is noise)")

    # per-year split (house rule)
    parts = []
    for y, g in df.groupby("year"):
        t = g[g["rank"] <= top_k]["pnl_pct"]
        r = g[g["rank"] > top_k]["pnl_pct"]
        if len(t) >= 10 and len(r) >= 10:
            parts.append(f"{y}:{t.mean() - r.mean():+.3f}(n={len(t)}/{len(r)})")
    print("  per-year edge: " + ("  ".join(parts) if parts else "insufficient"))

    # what the pipeline actually faces: top-2 of ~10
    by_day = df.groupby("entry_date")["pnl_pct"].apply(list)
    real, ideal, rand = [], [], []
    for d_, vals in zip(by_day.index, by_day):
        sub = df[df["entry_date"] == d_].sort_values("rank")
        real.append(sub.head(top_k)["pnl_pct"].mean())
        ideal.append(sorted(vals, reverse=True)[:top_k])
        rand.append(st.mean(random.sample(vals, min(top_k, len(vals)))))
    ideal = [st.mean(v) for v in ideal]
    print(f"  per-day top-{top_k} avg: actual {st.mean(real):+.3f}%  "
          f"random-{top_k} {st.mean(rand):+.3f}%  perfect-foresight {st.mean(ideal):+.3f}%")
    print(f"  -> ranking captures {100 * (st.mean(real) - st.mean(rand)) / (st.mean(ideal) - st.mean(rand)):.1f}% "
          f"of the available selection value")


def audit_live(src: str, top_k: int = 2) -> None:
    """Live arm: needs the `candidates` section of the dashboard export plus
    closed trades, so it only produces numbers once enough picked candidates
    have resolved. Until then it reports coverage and exits quietly."""
    import json
    import urllib.request

    raw = (json.load(open(src)) if not src.startswith("http")
           else json.loads(urllib.request.urlopen(src, timeout=60).read()))
    cands = raw.get("candidates") or []
    if not cands:
        print("\nLIVE: no `candidates` in the export yet — "
              "the section ships with this change and fills on the next run.")
        return
    pnl = {}
    for stream, rows in (raw.get("trades") or {}).items():
        for t in rows:
            pnl.setdefault(t["ticker"].upper(), []).append(t["pnl_pct"])
    resolved = [(c["rank"], pnl[c["ticker"].upper()][0])
                for c in cands if c["ticker"].upper() in pnl]
    print(f"\n=== LIVE candidates ===\n  {len(cands)} ranked rows, "
          f"{len(resolved)} with a resolved outcome")
    if len(resolved) < 40:
        print("  too few resolved to judge — rerun once the book has more history")
        return
    top = [p for r, p in resolved if r <= top_k]
    rest = [p for r, p in resolved if r > top_k]
    if len(top) >= 15 and len(rest) >= 15:
        d, lo, hi = boot_diff(top, rest)
        print(f"  rank 1-{top_k} avg {st.mean(top):+.3f}% (n={len(top)})  vs  "
              f"rank {top_k + 1}+ avg {st.mean(rest):+.3f}% (n={len(rest)})")
        print(f"  edge {d:+.3f}pp  95% CI [{lo:+.3f}, {hi:+.3f}]")


def main() -> None:
    r = ROOT / "outputs/research"
    audit(r / "sniper_truth_E_live_fixed_2026-07-26.csv", "SNIPER (Run E, fill-realistic)")
    audit(r / "mr_trades_polygon.csv", "MEAN REVERSION (3Y Polygon)")
    src = sys.argv[1] if len(sys.argv) > 1 else \
        "https://raysyhuang.github.io/multi-agentic-screener/data.json"
    try:
        audit_live(src)
    except Exception as e:  # noqa: BLE001 — live arm is best-effort
        print(f"\nLIVE arm unavailable: {e}")


if __name__ == "__main__":
    main()
