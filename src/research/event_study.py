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


def block_boot_ci(x: list[float], entry_dates: list, calendar: pd.Index, block: int = 20,
                  n_boot: int = 10_000, seed: int = SEED) -> tuple[float, float]:
    """Seeded circular MOVING-BLOCK bootstrap 95% CI of the observation-weighted
    mean. Resamples runs of `block` consecutive calendar sessions, so events
    whose holding windows overlap (adjacent entry dates share most of a 20-session
    path) are resampled together. The date-cluster interval treats adjacent dates
    as independent and is too narrow for overlapping windows; report both."""
    if not x:
        return (float("nan"), float("nan"))
    pos = {pd.Timestamp(d): i for i, d in enumerate(calendar)}
    n = len(calendar)
    sums = [0.0] * n
    cnts = [0] * n
    for v, d in zip(x, entry_dates):
        i = pos.get(pd.Timestamp(d))
        if i is None:
            continue
        sums[i] += v
        cnts[i] += 1
    n_blocks = max(1, -(-n // block))
    # Block sums via prefix sums over the circularly extended series.
    wrap = np.arange(n + block) % n            # circular for any block length, even block > n
    ext_s = np.asarray(sums, dtype=float)[wrap]
    ext_c = np.asarray(cnts, dtype=float)[wrap]
    ps = np.concatenate([[0.0], np.cumsum(ext_s)])
    pc = np.concatenate([[0.0], np.cumsum(ext_c)])
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    tot = (ps[starts + block] - ps[starts]).sum(axis=1)
    cnt = (pc[starts + block] - pc[starts]).sum(axis=1)
    means = np.sort(tot[cnt > 0] / cnt[cnt > 0])
    return float(means[int(0.025 * len(means))]), float(means[int(0.975 * len(means))])


def block_diff_ci(xa: list[float], da: list, xb: list[float], db: list, calendar: pd.Index,
                  block: int = 20, n_boot: int = 10_000, seed: int = SEED) -> tuple[float, float, float]:
    """mean(a) - mean(b) with a circular moving-block CI: each draw picks the same
    blocks of calendar sessions for both cohorts and recomputes both
    observation-weighted means, so the difference keeps the dependence between
    overlapping windows that a per-date resample would break."""
    if not xa or not xb:
        return (float("nan"), float("nan"), float("nan"))
    n = len(calendar)
    pos = {pd.Timestamp(d): i for i, d in enumerate(calendar)}

    def _arrays(x, d):
        s_, c_ = np.zeros(n), np.zeros(n)
        for v, dd in zip(x, d):
            i = pos.get(pd.Timestamp(dd))
            if i is not None:
                s_[i] += v
                c_[i] += 1
        wrap = np.arange(n + block) % n
        return (np.concatenate([[0.0], np.cumsum(s_[wrap])]),
                np.concatenate([[0.0], np.cumsum(c_[wrap])]))

    psa, pca = _arrays(xa, da)
    psb, pcb = _arrays(xb, db)
    n_blocks = max(1, -(-n // block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    ta = (psa[starts + block] - psa[starts]).sum(axis=1)
    na = (pca[starts + block] - pca[starts]).sum(axis=1)
    tb = (psb[starts + block] - psb[starts]).sum(axis=1)
    nb = (pcb[starts + block] - pcb[starts]).sum(axis=1)
    ok = (na > 0) & (nb > 0)
    diffs = np.sort(ta[ok] / na[ok] - tb[ok] / nb[ok])
    point = float(np.mean(xa) - np.mean(xb))
    return point, float(diffs[int(0.025 * len(diffs))]), float(diffs[int(0.975 * len(diffs))])


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
    volume: pd.DataFrame
    eligible: pd.DataFrame      # bool: liquid enough AS OF the prior close
    bucket: pd.DataFrame        # 0..N_BUCKETS-1 liquidity tercile among eligible, else NaN

    @property
    def dates(self) -> pd.Index:
        return self.close.index


def split_price_multiplier(splits: list[dict], dates: pd.Index, tickers: list[str]) -> pd.DataFrame:
    """(date x ticker) multiplier that turns split-ADJUSTED prices back into the
    RAW prices traded on each date: raw = adjusted x prod(split_to/split_from)
    over every split executed AFTER that date. Needed because a $5 floor applied
    to adjusted prices admits penny stocks that later reverse-split (a 1-for-100
    split makes a $0.50 stock look like $50 in its own history) — a screen that
    uses information from the future. Dollar volume is unaffected (the price
    and volume adjustments cancel)."""
    mult = pd.DataFrame(1.0, index=dates, columns=tickers)
    cols = set(tickers)
    for sp in splits:
        t = str(sp.get("ticker") or "").replace(".", "-").upper()
        if t not in cols or not sp.get("execution_date"):
            continue
        try:
            f = float(sp.get("split_to") or 0) / float(sp.get("split_from") or 0)
        except ZeroDivisionError:
            continue
        if f <= 0:
            continue
        mult.loc[mult.index < pd.Timestamp(sp["execution_date"]), t] *= f
    return mult


def build_panel(prices: dict[str, pd.DataFrame], *, min_price: float = MIN_PRICE,
                min_dollar_vol: float = MIN_DOLLAR_VOL, window: int = LIQ_WINDOW,
                n_buckets: int = N_BUCKETS, splits: list[dict] | None = None) -> Panel:
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
    # With `splits`, the price floor reads the RAW price traded that day (see
    # split_price_multiplier); without, it reads the adjusted price (the
    # original behaviour, which lets future reverse-splitters through).
    raw_close = c * split_price_multiplier(splits, c.index, list(c.columns)) if splits else c
    prior_close = raw_close.shift(1)
    eligible = (prior_close >= min_price) & (dollar_vol >= min_dollar_vol)

    ranked = dollar_vol.where(eligible).rank(axis=1, pct=True)
    bucket = np.ceil(ranked * n_buckets) - 1
    bucket = bucket.clip(lower=0, upper=n_buckets - 1)
    return Panel(open=o, close=c, volume=v, eligible=eligible.fillna(False), bucket=bucket)


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


def summarize_excess(df: pd.DataFrame, horizon: int, *, group: str = "entry_date",
                     calendar: pd.Index | None = None) -> dict:
    """n, mean excess, hit rate, iid CI, the date-cluster CI (the registered G1
    interval) and — when a calendar is given — a moving-block CI with block =
    horizon, which respects overlapping holding windows. Also reports
    `n_no_forward_return`: rows whose forward return is unavailable for ANY
    reason — the window runs past the end of the data (the usual case), a
    missing entry or exit bar, or an ineligible row passed in. They are dropped,
    never assumed flat; the count does not say which reason applied."""
    col = f"excess_{horizon}"
    n_censored = int((df[f"fwd_{horizon}"].isna()).sum()) if f"fwd_{horizon}" in df else 0
    sub = df[df[col].notna()]
    x = sub[col].tolist()
    if not x:
        return {"n": 0, "n_no_forward_return": n_censored}
    lo, hi = boot_ci(x)
    clo, chi = cluster_boot_ci(x, [str(g)[:10] for g in sub[group]])
    out = {
        "n": len(x), "n_no_forward_return": n_censored,
        "n_dates": int(sub[group].astype(str).str[:10].nunique()),
        "mean_excess": float(np.mean(x)), "median_excess": float(np.median(x)),
        "hit": float(np.mean([v > 0 for v in x])),
        "mean_fwd": float(sub[f"fwd_{horizon}"].mean()),
        "mean_base": float(sub[f"base_{horizon}"].mean()),
        "iid_ci": [lo, hi], "cluster_ci": [clo, chi],
    }
    if calendar is not None:
        out["block_ci"] = list(block_boot_ci(x, list(sub[group]), calendar, block=horizon))
    return out
