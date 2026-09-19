"""Shared event-study primitives for the research funnel (docs/research_registry.md).

Every Stage-0 (G1) study asks the same question: after an event, does the stock
beat what a comparable stock did over the SAME days? This module holds the three
things those studies kept re-implementing, so they are written — and reviewed —
once:

* a wide price PANEL with a no-lookahead liquidity screen (`build_panel`);
* forward returns and a SAME-DAY, SAME-LIQUIDITY-BUCKET base rate
  (`forward_returns`, `base_rate`), because a small-cap event study measured
  against SPY reports small-cap beta as alpha;
* a date-CLUSTER bootstrap (`cluster_boot_ci`), because events that enter on one
  day share one market path and are not independent draws.

Conventions, fixed here so studies cannot quietly differ:
  - entry = the OPEN of the entry bar; exit = the CLOSE `h` rows later on the
    panel's calendar (all tickers share one calendar, so an event and its base
    rate are measured over identical dates by construction);
  - anything used to decide eligibility on date D uses bars STRICTLY before D.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 20260919
MIN_PRICE = 5.0                 # mirrors src/signals/filter.py (price >= $5)
MIN_DOLLAR_VOL = 2_000_000.0    # mirrors filter_by_ohlcv ($2M avg dollar volume)
LIQ_WINDOW = 20
N_BUCKETS = 3


def boot_ci(x: list[float], n_boot: int = 10_000, seed: int = SEED) -> tuple[float, float]:
    """Seeded percentile bootstrap 95% CI of the mean, resampling observations
    iid. Too narrow whenever observations share entry dates; reported only next
    to `cluster_boot_ci`, never instead of it."""
    k = len(x)
    if k == 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(sum(x[rng.randrange(k)] for _ in range(k)) / k for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def cluster_boot_ci(x: list[float], groups: list, n_boot: int = 10_000,
                    seed: int = SEED) -> tuple[float, float]:
    """Seeded percentile bootstrap 95% CI of the mean that resamples GROUPS
    (entry dates) with replacement and keeps every observation in a drawn group
    (observation-weighted mean of the resample). With one observation per group
    it reduces to the iid interval."""
    if not x:
        return (float("nan"), float("nan"))
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


def market_regime(spy: pd.DataFrame, lag_sessions: int = 0) -> dict[str, str]:
    """{YYYY-MM-DD: bull|bear|choppy|unknown} from SPY SMA20/50, per date.

    `lag_sessions=0` labels D from D's own close (a decision at/after the close).
    An entry at D's OPEN cannot know D's close: pass `lag_sessions=1` so D carries
    the label of the last completed session (the first dates become "unknown").
    """
    spy = spy.sort_values("date").reset_index(drop=True)
    c = spy["close"].astype(float)
    s50, s20 = c.rolling(50).mean(), c.rolling(20).mean()
    out: dict[str, str] = {}
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
    if lag_sessions:
        keys = list(out)
        out = {k: (out[keys[i - lag_sessions]] if i >= lag_sessions else "unknown")
               for i, k in enumerate(keys)}
    return out


@dataclass
class Panel:
    """Wide (calendar x ticker) matrices sharing one index of session dates."""
    open: pd.DataFrame
    close: pd.DataFrame
    eligible: pd.DataFrame      # bool: liquid enough AS OF the prior close
    bucket: pd.DataFrame        # 0..N_BUCKETS-1 liquidity tercile among eligible, else NaN

    @property
    def dates(self) -> pd.Index:
        return self.close.index


def build_panel(prices: dict[str, pd.DataFrame], *, min_price: float = MIN_PRICE,
                min_dollar_vol: float = MIN_DOLLAR_VOL, window: int = LIQ_WINDOW,
                n_buckets: int = N_BUCKETS) -> Panel:
    """Pivot per-ticker OHLCV into a panel and compute a NO-LOOKAHEAD liquidity
    screen: eligibility and bucket on date D use the prior close and the mean
    dollar volume of the `window` sessions ending at D-1 (`.shift(1)`), so
    nothing from D itself — the entry bar — can decide whether D is tradable.
    Buckets are per-date terciles of that trailing dollar volume among the
    eligible names (0 = least liquid)."""
    frames = []
    for t, df in prices.items():
        d = df[["date", "open", "close", "volume"]].copy()
        d["date"] = pd.to_datetime(d["date"])
        d["ticker"] = t
        frames.append(d)
    long = pd.concat(frames, ignore_index=True).drop_duplicates(["date", "ticker"])
    o = long.pivot(index="date", columns="ticker", values="open").sort_index()
    c = long.pivot(index="date", columns="ticker", values="close").sort_index()
    v = long.pivot(index="date", columns="ticker", values="volume").sort_index()

    dollar_vol = (c * v).rolling(window, min_periods=window).mean().shift(1)
    prior_close = c.shift(1)
    eligible = (prior_close >= min_price) & (dollar_vol >= min_dollar_vol)

    ranked = dollar_vol.where(eligible).rank(axis=1, pct=True)
    bucket = np.ceil(ranked * n_buckets) - 1
    bucket = bucket.clip(lower=0, upper=n_buckets - 1)
    return Panel(open=o, close=c, eligible=eligible.fillna(False), bucket=bucket)


def forward_returns(panel: Panel, horizon: int) -> pd.DataFrame:
    """% return from the OPEN on each date to the CLOSE `horizon` rows later, for
    every ticker. NaN where either bar is missing (no forward-filling: a name
    that stopped trading has no return, it is not assumed flat)."""
    return (panel.close.shift(-horizon) / panel.open - 1.0) * 100.0


def base_rate(panel: Panel, fwd: pd.DataFrame) -> pd.DataFrame:
    """Mean forward return of ALL eligible names, per (date, liquidity bucket).
    Returned as a (date x bucket) frame. This is the comparison an event must
    beat: same entry day, same exit day, same kind of stock."""
    out = {}
    masked = fwd.where(panel.eligible)
    for b in range(int(np.nanmax(panel.bucket.to_numpy())) + 1 if panel.bucket.notna().any().any() else 0):
        out[b] = masked.where(panel.bucket == b).mean(axis=1)
    return pd.DataFrame(out)


def event_excess(panel: Panel, events: list[dict], horizons: list[int]) -> pd.DataFrame:
    """One row per event with its forward return, the matched base rate and the
    excess, per horizon. An event is `{"ticker", "entry_date", ...}`; extra keys
    are carried through. Events on a date where the name is not eligible (as of
    the prior close) are returned with `eligible=False` and NaN returns so the
    caller can report how many were dropped rather than lose them silently.

    `entry_date` must be a session in the panel; the caller decides what it
    means (e.g. the first session after a report) — this function never snaps
    a date forward, because that is exactly where look-ahead hides.
    """
    fwd = {h: forward_returns(panel, h) for h in horizons}
    base = {h: base_rate(panel, fwd[h]) for h in horizons}
    idx = panel.dates
    rows = []
    for ev in events:
        t, d = ev["ticker"], pd.Timestamp(ev["entry_date"])
        row = dict(ev)
        ok = (t in panel.close.columns) and (d in idx) and bool(panel.eligible.at[d, t])
        row["eligible"] = ok
        b = panel.bucket.at[d, t] if ok else np.nan
        row["bucket"] = None if pd.isna(b) else int(b)
        for h in horizons:
            r = fwd[h].at[d, t] if ok else np.nan
            br = base[h].at[d, int(b)] if ok and not pd.isna(b) else np.nan
            row[f"fwd_{h}"] = r
            row[f"base_{h}"] = br
            row[f"excess_{h}"] = r - br if not (pd.isna(r) or pd.isna(br)) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_excess(df: pd.DataFrame, horizon: int, *, group: str = "entry_date") -> dict:
    """n, mean excess, hit rate, iid CI and the (decisional) date-cluster CI."""
    col = f"excess_{horizon}"
    sub = df[df[col].notna()]
    x = sub[col].tolist()
    if not x:
        return {"n": 0}
    lo, hi = boot_ci(x)
    clo, chi = cluster_boot_ci(x, [str(g)[:10] for g in sub[group]])
    return {
        "n": len(x), "n_dates": int(sub[group].astype(str).str[:10].nunique()),
        "mean_excess": float(np.mean(x)), "median_excess": float(np.median(x)),
        "hit": float(np.mean([v > 0 for v in x])),
        "mean_fwd": float(sub[f"fwd_{horizon}"].mean()),
        "mean_base": float(sub[f"base_{horizon}"].mean()),
        "iid_ci": [lo, hi], "cluster_ci": [clo, chi],
    }
