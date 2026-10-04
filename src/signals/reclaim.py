"""RECLAIM — a second-stage timing rule, run as two SHADOW lanes (no book).

Spec: RECLAIM_MAS_SPEC_v1.1.md (section numbers below refer to it). This module
is pure: no I/O, no clock, no database. The live runner (`src/reclaim_shadow.py`),
the tracker (`src/output/reclaim_tracker.py`), the offline reporter and the
replay backtest all call into it, so a rule exists in exactly one place.

What it measures. A name must first enter a parent pool. Inside a bounded
episode it must hold a floor fixed at entry, reclaim its SMA50, retest it, and
confirm by closing above the retest-day high. Entry is the next open, the stop
is structural (the setup low), exits are fixed horizons. There is no target.

Two NEW estimands, never pooled with each other or with the canonical
`reclaim-confirm` V0.2 lane:

- ``RANGE_TECH_NATIVE`` — Range V0.1's hard universe and G1-G4 on MAS bars.
  Not an exact Range rebuild: ticker identity only, current vendor market cap.
- ``DRIFT_G3_NATIVE`` — Drift's technical G3 leg plus its universe thresholds.
  No growth (G1) or valuation (G2) leg.

Evidence status: UNPROVEN. Nothing here is evidence until the §7 gate passes on
clean forward triggers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ── Frozen spec constants (§8.2) ─────────────────────────────────────────────
ABSENT_GAP = 20      # sessions of absence that separate two episodes
PREHIST = 20         # sessions that must be covered and absent before a VALID start
EXPIRY = 30          # alive through last_incl + EXPIRY ...
CAP = 60             # ... but never past start + CAP
RETEST_MAX = 10      # retest window r+1 .. r+10
CONFIRM_MAX = 5      # confirm window q+1 .. q+5
RETEST_BAND = 1.01   # retest: low <= 1.01 * SMA50 and close >= SMA50
EXT = 1.10           # SKIP_EXTENDED: open > 1.10 * SMA50_k
RISK = 0.08          # SKIP_RISK: (open - stop) / open > 8%
EARN_WIN = 5         # earnings window [e-5, e+5] sessions
COST = 0.0010        # 10 bp per side
HORIZONS = (5, 10, 20)
ANCHOR_HORIZON = 10

RANGE = "RANGE_TECH_NATIVE"
DRIFT = "DRIFT_G3_NATIVE"
LANES = (RANGE, DRIFT)
_LANE_TAG = {RANGE: "RANGE", DRIFT: "DRIFT"}

FLOOR_TYPE = {
    RANGE: "RANGE_BOX_LOW_AT_ENTRY",
    DRIFT: "DRIFT_TRAILING20_LOW_AT_ENTRY",
}

EXIT_POLICY = "STRUCTURAL_STOP_FIXED_HORIZON"
# Finite, schema-compatible with `Float NOT NULL`, impossible as a price (§8.4).
# The tracker builds target=inf at runtime; inf/NaN are never persisted.
NO_TARGET_SENTINEL = -1.0

# Range V0.1 hard universe and G1-G4 (§3).
RANGE_MIN_CLOSE = 5.0
RANGE_MIN_MCAP = 1.0e9
RANGE_MIN_MEDIAN_DV = 20.0e6
RANGE_MIN_CONSEC_BARS = 60
RANGE_G1_AMP = 0.10
RANGE_G2_ATR_PCT = 0.030
RANGE_G3_ATR_RATIO = 0.85
RANGE_G3_RANGE_RATIO = 0.50
RANGE_G4_LOW_ATR = 0.50
RANGE_G4_ADX_MAX = 25.0
RANGE_G4_SMA20_ATR = 1.50
RANGE_G4_PROX_ATR = 0.75
RANGE_LIQ_LO = 20.0e6
RANGE_LIQ_HI = 500.0e6

# Drift universe + G3 (§2). The Drift method states "1-month return"; 21 XNYS
# sessions is the conventional reading and is pinned here.
DRIFT_MIN_MCAP = 3.0e9
DRIFT_MIN_MEAN_DV = 40.0e6
DRIFT_MIN_SESSIONS = 252
DRIFT_MIN_BARS = 50
DRIFT_DD52_MAX = -0.18
DRIFT_SMA50_DEV_MAX = -0.04
DRIFT_RET_1M_MAX = -0.07
ONE_MONTH_SESSIONS = 21

# First session index at which each lane's list can exist at all (warm-up).
LANE_WARMUP = {RANGE: RANGE_MIN_CONSEC_BARS - 1, DRIFT: DRIFT_MIN_SESSIONS - 1}

NN_K = 3
NN_MIN_VALID = 3
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20261004
GATE_MIN_CLEAN = 30
GATE_MIN_DATES = 10
GATE_MIN_CLUSTERS = 2
CLUSTER_MIN = 5
KILL_WEEKS = 26
KILL_STOP_TOUCH = 0.50

# Ledger codes.
PASS = "PASS"
CENSORED_NO_ENTRY_BAR = "CENSORED_NO_ENTRY_BAR"
UNFILLABLE_GAP_THROUGH = "UNFILLABLE_GAP_THROUGH"
SKIP_EXTENDED = "SKIP_EXTENDED"
SKIP_RISK = "SKIP_RISK"
MECH_PRECEDENCE = (CENSORED_NO_ENTRY_BAR, UNFILLABLE_GAP_THROUGH, SKIP_EXTENDED, SKIP_RISK)

CLEAR = "CLEAR"
BLOCK_EARNINGS = "BLOCK_EARNINGS"
BLOCK_EARNINGS_UNKNOWN = "BLOCK_EARNINGS_UNKNOWN"
INACTIVE_LATE = "INACTIVE_LATE"
INACTIVE_FAILED = "INACTIVE_FAILED"

VALID = "VALID"
INSUFFICIENT_PREHISTORY = "INSUFFICIENT_PREHISTORY"
LEFT_CENSORED_FIRST_OBS = "LEFT_CENSORED_FIRST_OBS"
RECONSTRUCTED_HISTORY = "RECONSTRUCTED_HISTORY"
PARENT_LIST_GAP = "PARENT_LIST_GAP"

GATE2_NOT_APPLICABLE = "NOT_APPLICABLE"

_ET = ZoneInfo("America/New_York")

SECTOR_ETF = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}
BENCHMARK_TICKERS = ("SPY", *sorted(set(SECTOR_ETF.values())))


# ── Session alignment and indicators ─────────────────────────────────────────

def align_to_sessions(df: pd.DataFrame | None, sessions: list[date]) -> pd.DataFrame:
    """Reindex daily bars onto the XNYS session list; a missing bar is a NaN row.

    The state machine iterates sessions, not the dates a symbol happens to have
    bars for (§5, C6), so every array downstream is session-indexed.
    """
    cols = ["open", "high", "low", "close", "volume", "vwap"]
    if df is None or len(df) == 0:
        return pd.DataFrame(np.nan, index=range(len(sessions)), columns=cols)
    frame = df.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        frame = frame.drop_duplicates("date", keep="last").set_index("date")
    else:
        frame.index = pd.to_datetime(frame.index).date
    if "vwap" not in frame.columns:
        frame["vwap"] = np.nan
    out = frame.reindex(sessions)[cols].astype(float)
    return out.reset_index(drop=True)


def sma(values: np.ndarray, n: int) -> np.ndarray:
    """Mean of the n values ending at t; undefined (NaN) if any is missing.

    A missing bar inside the window therefore resets the window (§5).
    """
    return pd.Series(values).rolling(n, min_periods=n).mean().to_numpy()


def wilder_rma(values: np.ndarray, n: int) -> np.ndarray:
    """Wilder RMA (alpha = 1/n) seeded with the arithmetic mean of the first n values.

    A NaN breaks continuity: output is NaN there and seeding restarts on the next
    run of valid values (Range V0.1 requires consecutive valid bars, R10).
    """
    out = np.full(len(values), np.nan)
    run: list[float] = []
    prev: float | None = None
    for i, v in enumerate(values):
        if v is None or not np.isfinite(v):
            run, prev = [], None
            continue
        if prev is None:
            run.append(float(v))
            if len(run) == n:
                prev = sum(run) / n
                out[i] = prev
            continue
        prev = (prev * (n - 1) + float(v)) / n
        out[i] = prev
    return out


def true_range(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, *, first_bar_hl: bool = True,
) -> np.ndarray:
    """True range on adjusted H/L and the previous adjusted close.

    The first bar of a run has no previous close. For ATR its true range is
    H - L (``first_bar_hl``); the ADX directional movement has no value on that
    bar either, so the ADX path leaves it undefined to keep TR and DM aligned.
    Both conventions are pinned by the QNT 2026-10-01 parity test against
    Range's ``gate_inputs`` (ATR14 3.31369, ADX14 12.5032).
    """
    prev_close = np.concatenate([[np.nan], close[:-1]])
    with np.errstate(invalid="ignore"):
        tr = np.fmax(high - low, np.fmax(np.abs(high - prev_close), np.abs(low - prev_close)))
    no_prev = ~np.isfinite(prev_close)
    if first_bar_hl:
        tr[no_prev] = (high - low)[no_prev]
    else:
        tr[no_prev] = np.nan
    return tr


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder ADX with every smoothing stage mean-seeded (R10)."""
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    up = high - prev_high
    down = prev_low - low
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    invalid = ~np.isfinite(up) | ~np.isfinite(down)
    plus_dm[invalid] = np.nan
    minus_dm[invalid] = np.nan
    tr = true_range(high, low, close, first_bar_hl=False)
    s_tr = wilder_rma(tr, n)
    s_plus = wilder_rma(plus_dm, n)
    s_minus = wilder_rma(minus_dm, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * s_plus / s_tr
        minus_di = 100.0 * s_minus / s_tr
        denom = plus_di + minus_di
        dx = np.where(denom > 0, 100.0 * np.abs(plus_di - minus_di) / denom, 0.0)
    dx[~np.isfinite(plus_di) | ~np.isfinite(minus_di)] = np.nan
    return wilder_rma(dx, n)


def _consecutive_valid(values: np.ndarray) -> np.ndarray:
    count = np.zeros(len(values), dtype=int)
    run = 0
    for i, v in enumerate(values):
        run = run + 1 if np.isfinite(v) else 0
        count[i] = run
    return count


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def dollar_volume(bars: pd.DataFrame, prefer_vwap: bool) -> tuple[np.ndarray, str]:
    """Daily dollar volume and the basis used (§4A, R27)."""
    if prefer_vwap and bars["vwap"].notna().any():
        price = bars["vwap"].where(bars["vwap"].notna(), bars["close"])
        basis = "vwap_x_volume" if bars["vwap"].notna().all() else "vwap_x_volume_else_close"
    else:
        price = bars["close"]
        basis = "close_x_volume"
    return (price * bars["volume"]).to_numpy(dtype=float), basis


# ── Native pools (§4A) ───────────────────────────────────────────────────────

def range_tech_native_series(bars: pd.DataFrame, mcap: float | None) -> pd.DataFrame:
    """RANGE_TECH_NATIVE membership and fields for every session (§3, §4A).

    ``bars`` is session-aligned (``align_to_sessions``). Market cap is the current
    vendor value — recorded per snapshot as a parity gap, never back-dated.
    """
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    hs, ls, cs = pd.Series(high), pd.Series(low), pd.Series(close)

    tr = true_range(high, low, close)
    atr5 = wilder_rma(tr, 5)
    atr14 = wilder_rma(tr, 14)
    atr20 = wilder_rma(tr, 20)
    adx14 = adx(high, low, close, 14)

    box_high = hs.shift(1).rolling(20, min_periods=20).max().to_numpy()
    box_low = ls.shift(1).rolling(20, min_periods=20).min().to_numpy()
    range5 = (hs.rolling(5, min_periods=5).max() - ls.rolling(5, min_periods=5).min()).to_numpy()
    range20 = (hs.rolling(20, min_periods=20).max() - ls.rolling(20, min_periods=20).min()).to_numpy()
    sma20 = cs.rolling(20, min_periods=20).mean().to_numpy()
    dv, _ = dollar_volume(bars, prefer_vwap=False)
    median_dv = pd.Series(dv).rolling(20, min_periods=20).median().to_numpy()
    consec = _consecutive_valid(close)

    with np.errstate(divide="ignore", invalid="ignore"):
        amp20 = range20 / close
        atr_pct = atr14 / close
        atr_ratio = atr5 / atr20
        range_ratio = range5 / range20
        sma20_dist = np.abs(close - sma20) / atr14

    mcap_ok = mcap is not None and np.isfinite(mcap) and mcap >= RANGE_MIN_MCAP
    universe = (
        mcap_ok
        & (close >= RANGE_MIN_CLOSE)
        & (median_dv >= RANGE_MIN_MEDIAN_DV)
        & (consec >= RANGE_MIN_CONSEC_BARS)
    )
    g1 = amp20 >= RANGE_G1_AMP
    g2 = atr_pct >= RANGE_G2_ATR_PCT
    g3 = (atr_ratio <= RANGE_G3_ATR_RATIO) | (range_ratio <= RANGE_G3_RANGE_RATIO)
    g4 = (
        (close >= box_low)
        & (close < 1.01 * box_high)
        & (low >= box_low - RANGE_G4_LOW_ATR * atr14)
        & (adx14 <= RANGE_G4_ADX_MAX)
        & (sma20_dist <= RANGE_G4_SMA20_ATR)
        & (close <= box_low + RANGE_G4_PROX_ATR * atr14)
    )
    member = universe & g1 & g2 & g3 & g4

    with np.errstate(divide="ignore", invalid="ignore"):
        proximity = 1.0 - _clip01((close - box_low) / (RANGE_G4_PROX_ATR * atr14))
        atr_term = _clip01((atr_pct - RANGE_G2_ATR_PCT) / 0.05)
        amplitude = _clip01((amp20 - RANGE_G1_AMP) / 0.15)
        liquidity = _clip01(
            np.log(median_dv / RANGE_LIQ_LO) / math.log(RANGE_LIQ_HI / RANGE_LIQ_LO)
        )
    score = 0.40 * proximity + 0.30 * atr_term + 0.20 * amplitude + 0.10 * liquidity

    return pd.DataFrame({
        "member": np.asarray(member, dtype=bool),
        "score": score,
        "box_low": box_low,
        "box_high": box_high,
        "atr14": atr14,
        "atr5": atr5,
        "atr20": atr20,
        "adx14": adx14,
        "range5": range5,
        "range20": range20,
        "sma20": sma20,
        "median20_dv": median_dv,
        "close": close,
    })


def drift_g3_native_series(bars: pd.DataFrame, mcap: float | None) -> pd.DataFrame:
    """DRIFT_G3_NATIVE membership for every session (§2 G3 + universe, §4A).

    No growth or valuation leg and no parent score: a technical dislocation
    population, so ``score`` is NaN and ``score_pct`` is dropped for this lane.
    """
    high = bars["high"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    hs, cs = pd.Series(high), pd.Series(close)

    max_high_252 = hs.rolling(DRIFT_MIN_SESSIONS, min_periods=DRIFT_MIN_SESSIONS).max().to_numpy()
    sma50 = sma(close, 50)
    dv, _ = dollar_volume(bars, prefer_vwap=True)
    mean_dv = pd.Series(dv).rolling(20, min_periods=20).mean().to_numpy()
    n_bars = cs.notna().cumsum().to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        dd52 = close / max_high_252 - 1.0
        sma50_dev = close / sma50 - 1.0
        ret_1m = close / cs.shift(ONE_MONTH_SESSIONS).to_numpy() - 1.0

    mcap_ok = mcap is not None and np.isfinite(mcap) and mcap >= DRIFT_MIN_MCAP
    member = (
        mcap_ok
        & (mean_dv >= DRIFT_MIN_MEAN_DV)
        & (n_bars >= DRIFT_MIN_BARS)
        & (dd52 <= DRIFT_DD52_MAX)
        & ((sma50_dev <= DRIFT_SMA50_DEV_MAX) | (ret_1m <= DRIFT_RET_1M_MAX))
    )
    return pd.DataFrame({
        "member": np.asarray(member, dtype=bool),
        "score": np.full(len(close), np.nan),
        "dd52": dd52,
        "sma50_dev": sma50_dev,
        "ret_1m": ret_1m,
        "mean20_dv": mean_dv,
        "close": close,
    })


def lane_series(lane: str, bars: pd.DataFrame, mcap: float | None) -> pd.DataFrame:
    if lane == RANGE:
        return range_tech_native_series(bars, mcap)
    if lane == DRIFT:
        return drift_g3_native_series(bars, mcap)
    raise ValueError(f"unknown lane {lane!r}")


def range_tech_native_member(bars: pd.DataFrame, t: int, mcap: float | None) -> bool:
    return bool(range_tech_native_series(bars, mcap)["member"].iloc[t])


def drift_g3_native_member(bars: pd.DataFrame, t: int, mcap: float | None) -> bool:
    return bool(drift_g3_native_series(bars, mcap)["member"].iloc[t])


def score_pct(own: float, scores: list[float]) -> float | None:
    """#{v <= own} / n within one snapshot. Control matching only, never a rank."""
    vals = [s for s in scores if s is not None and np.isfinite(s)]
    if own is None or not np.isfinite(own) or not vals:
        return None
    return sum(1 for v in vals if v <= own) / len(vals)


# ── Episodes and the state machine (§4, §5) ──────────────────────────────────

@dataclass
class Episode:
    lane: str
    symbol: str
    start: int                 # session index
    start_date: date
    floor: float | None
    floor_type: str
    episode_id: str
    prehistory: str = INSUFFICIENT_PREHISTORY
    prehistory_reason: str | None = RECONSTRUCTED_HISTORY


@dataclass
class RiskState:
    """An episode that is alive, untriggered and not blocked on a session."""
    age: int
    retest_pending: bool
    confirm_pending: bool
    close: float
    sma50: float

    @property
    def sma_dist(self) -> float | None:
        if not (np.isfinite(self.close) and np.isfinite(self.sma50)) or self.sma50 == 0:
            return None
        return self.close / self.sma50 - 1.0


@dataclass
class Trigger:
    episode_id: str
    lane: str
    symbol: str
    r: int
    q: int
    k: int
    reclaim_date: date
    retest_date: date
    trigger_date: date
    entry_session: date | None
    setup_low: float
    high_retest: float
    close_k: float
    sma50_k: float
    slope5: float | None

    @property
    def stop(self) -> float:
        return self.setup_low

    @property
    def max_entry(self) -> float:
        """Informational: min(1.10 * SMA50_k, stop / (1 - RISK))."""
        return min(EXT * self.sma50_k, self.setup_low / (1.0 - RISK))


@dataclass
class EpisodeResult:
    episode: Episode
    status: str                     # CENSORED_ALIVE | TRIGGERED | SUPPORT_FAILED | EXPIRED_30 | EXPIRED_CAP | FLOOR_UNAVAILABLE
    end: int | None                 # last session of the episode; None while alive
    events: list[tuple[int, str, dict]] = field(default_factory=list)
    trigger: Trigger | None = None
    risk_states: dict[int, RiskState] = field(default_factory=dict)


def episode_id(lane: str, symbol: str, start_date: date) -> str:
    """Stable across runs: keyed on the start session, not an ordinal.

    An ordinal (``SYM#RANGE3``) renumbers when the rolling history window drops
    an old episode, which would break every unique key built on it.
    """
    return f"{symbol}#{_LANE_TAG[lane]}@{start_date.isoformat()}"


def drift_floor(low: np.ndarray, start: int) -> float | None:
    """min(adjusted low) over the 20 sessions ending at start, inclusive."""
    if start < PREHIST - 1:
        return None
    window = low[start - (PREHIST - 1): start + 1]
    if np.count_nonzero(np.isfinite(window)) < PREHIST:
        return None
    return float(np.min(window))


def run_episode(
    ep: Episode,
    bars: pd.DataFrame,
    sma50: np.ndarray,
    inclusions: set[int],
    last_idx: int,
    sessions: list[date],
) -> EpisodeResult:
    """Walk one episode over XNYS sessions start+1 .. last_idx (§5, exact).

    ``last_idx`` is the signal date C. Every session counts whether or not the
    symbol has a bar: a missing bar advances every timer, resets SMA50 (already
    NaN in ``sma50``) and recognises nothing.
    """
    res = EpisodeResult(episode=ep, status="CENSORED_ALIVE", end=None)
    if ep.floor is None or not np.isfinite(ep.floor):
        res.status, res.end = "FLOOR_UNAVAILABLE", ep.start
        res.events.append((ep.start, "FLOOR_UNAVAILABLE", {}))
        return res
    res.events.append((ep.start, "START", {"floor": ep.floor, "floor_type": ep.floor_type}))

    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    floor = ep.floor

    last_incl = ep.start
    pending: int | None = None
    r: int | None = None
    q: int | None = None
    t = ep.start
    while True:
        t += 1
        if t > last_idx:                                   # 1. C reached first
            res.status, res.end = "CENSORED_ALIVE", None
            return res
        if t in inclusions:                                # 2.
            last_incl = t
        bound = min(last_incl + EXPIRY, ep.start + CAP)    # 3. expiry
        if t > bound:
            status = "EXPIRED_30" if last_incl + EXPIRY < ep.start + CAP else "EXPIRED_CAP"
            res.status, res.end = status, t - 1
            res.events.append((t - 1, status, {}))
            return res

        if not np.isfinite(close[t]):                      # 4. missing bar
            res.events.append((t, "MISSING_BAR", {}))
            if pending is not None and t - pending >= 2:
                res.status, res.end = "SUPPORT_FAILED", t
                res.events.append((t, "SUPPORT_FAILED", {"break": sessions[pending].isoformat()}))
                return res
            if q is not None and t - q >= CONFIRM_MAX:
                res.events.append((t, "CONFIRM_TIMEOUT", {}))
                r = q = None
            elif r is not None and q is None and t - r >= RETEST_MAX:
                res.events.append((t, "RETEST_TIMEOUT", {}))
                r = None
            continue

        blocked = False                                    # 5. floor
        if pending is not None:
            blocked = True
            if close[t] > floor:
                res.events.append((t, "RECOVER", {"close": float(close[t])}))
                pending = None
            elif t - pending >= 2:
                res.status, res.end = "SUPPORT_FAILED", t
                res.events.append((t, "SUPPORT_FAILED", {"close": float(close[t])}))
                return res
        elif close[t] < floor:
            res.events.append((t, "BREAK", {"close": float(close[t])}))
            pending = t
            blocked = True
        if blocked:
            continue

        res.risk_states[t] = RiskState(                    # 6. at risk
            age=t - ep.start,
            retest_pending=r is not None and q is None,
            confirm_pending=q is not None,
            close=float(close[t]),
            sma50=float(sma50[t]) if np.isfinite(sma50[t]) else float("nan"),
        )
        if q is not None:                                  # (a) confirmation
            if t - q > CONFIRM_MAX:
                res.events.append((t, "CONFIRM_TIMEOUT", {}))
                r = q = None
            elif close[t] > high[q]:
                assert r is not None
                window = low[r: t + 1]
                setup_low = float(np.nanmin(window))
                sma_k = float(sma50[t])
                slope5 = None
                if t >= 5 and np.isfinite(sma50[t - 5]) and sma50[t - 5] != 0:
                    slope5 = sma_k / float(sma50[t - 5]) - 1.0
                res.trigger = Trigger(
                    episode_id=ep.episode_id, lane=ep.lane, symbol=ep.symbol,
                    r=r, q=q, k=t,
                    reclaim_date=sessions[r], retest_date=sessions[q], trigger_date=sessions[t],
                    entry_session=sessions[t + 1] if t + 1 < len(sessions) else None,
                    setup_low=setup_low, high_retest=float(high[q]),
                    close_k=float(close[t]), sma50_k=sma_k, slope5=slope5,
                )
                res.status, res.end = "TRIGGERED", t
                res.events.append((t, "TRIGGER", {"close": float(close[t]), "high_q": float(high[q])}))
                return res
            elif t - q == CONFIRM_MAX:
                res.events.append((t, "CONFIRM_TIMEOUT", {}))
                r = q = None
        elif r is not None:                                # (b) retest
            if t - r > RETEST_MAX:
                res.events.append((t, "RETEST_TIMEOUT", {}))
                r = None
            elif low[t] <= RETEST_BAND * sma50[t] and close[t] >= sma50[t]:
                q = t
                res.events.append((t, "RETEST", {"low": float(low[t]), "sma50": float(sma50[t])}))
            elif t - r == RETEST_MAX:
                res.events.append((t, "RETEST_TIMEOUT", {}))
                r = None
        else:                                              # (c) idle
            if close[t - 1] <= sma50[t - 1] and close[t] > sma50[t]:
                r = t
                res.events.append((t, "RECLAIM", {"close": float(close[t]), "sma50": float(sma50[t])}))


def build_episodes(
    lane: str,
    symbol: str,
    inclusions: list[int],
    bars: pd.DataFrame,
    sessions: list[date],
    last_idx: int,
    floor_at: dict[int, float] | None = None,
) -> list[EpisodeResult]:
    """Episodes for one (lane, symbol), each run through the state machine.

    A start is the first-ever observed inclusion, or an inclusion after
    ``incs[j] - incs[j-1] - 1 >= ABSENT_GAP`` sessions of absence (R7), that also
    falls after the previous episode's end. RANGE floors come from the start
    snapshot's ``box_low`` (``floor_at``); DRIFT floors are the trailing-20 low.
    """
    incs = sorted(i for i in inclusions if i <= last_idx)
    incl_set = set(incs)
    close = bars["close"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    sma50 = sma(close, 50)
    results: list[EpisodeResult] = []
    prev_end = -1
    for j, inc in enumerate(incs):
        if j > 0 and inc - incs[j - 1] - 1 < ABSENT_GAP:
            continue
        if inc <= prev_end:
            continue
        if lane == RANGE:
            floor = (floor_at or {}).get(inc)
            floor = float(floor) if floor is not None and np.isfinite(floor) else None
        else:
            floor = drift_floor(low, inc)
        ep = Episode(
            lane=lane, symbol=symbol, start=inc, start_date=sessions[inc],
            floor=floor, floor_type=FLOOR_TYPE[lane],
            episode_id=episode_id(lane, symbol, sessions[inc]),
        )
        res = run_episode(ep, bars, sma50, incl_set, last_idx, sessions)
        results.append(res)
        if res.end is None:
            break
        prev_end = res.end
    return results


def prehistory_native(
    start: int,
    sessions: list[date],
    covered: set[date],
    forward_start: date | None,
    first_list_idx: int,
) -> tuple[str, str | None]:
    """Forward-earned prehistory (§4A, R18).

    VALID needs each of the 20 sessions start-20 .. start-1 to be a covered
    production snapshot persisted on or after the lane's forward start. Anything
    rebuilt from today's universe and market cap is survivor-biased and can
    change retroactively, so it can seed state but never be VALID.
    """
    lo = start - PREHIST
    if lo < first_list_idx:
        return INSUFFICIENT_PREHISTORY, LEFT_CENSORED_FIRST_OBS
    window = [sessions[i] for i in range(lo, start)]
    if forward_start is None or any(d < forward_start for d in window):
        return INSUFFICIENT_PREHISTORY, RECONSTRUCTED_HISTORY
    if any(d not in covered for d in window):
        return INSUFFICIENT_PREHISTORY, PARENT_LIST_GAP
    return VALID, None


# ── Actual-open decision and earnings (§5A) ──────────────────────────────────

@dataclass
class OpenDecision:
    open_e: float | None
    stop: float
    sma50_k: float
    ext_ratio: float | None
    risk_pct: float | None
    reasons: list[str]
    mech_status: str


def open_decision(open_e: float | None, stop: float, sma50_k: float) -> OpenDecision:
    """Judge the raw (un-slipped) open of e = k+1 in exact precedence (§5A, R17).

    Every applicable reason is listed; ``mech_status`` is the first. The generic
    ``max_entry_price`` / ``gap_above_limit`` path is not used.
    """
    if open_e is None or not np.isfinite(open_e):
        return OpenDecision(None, stop, sma50_k, None, None,
                            [CENSORED_NO_ENTRY_BAR], CENSORED_NO_ENTRY_BAR)
    ext_ratio = open_e / sma50_k if sma50_k else None
    risk_pct = (open_e - stop) / open_e if open_e else None
    reasons: list[str] = []
    if open_e <= stop:
        reasons.append(UNFILLABLE_GAP_THROUGH)
    if ext_ratio is not None and open_e > EXT * sma50_k:
        reasons.append(SKIP_EXTENDED)
    if risk_pct is not None and risk_pct > RISK:
        reasons.append(SKIP_RISK)
    return OpenDecision(open_e, stop, sma50_k, ext_ratio, risk_pct, reasons,
                        reasons[0] if reasons else PASS)


def session_offset(d: date, n: int, is_trading_day) -> date:
    """The XNYS session n sessions away from d (n may be negative)."""
    step = 1 if n >= 0 else -1
    cur = d
    remaining = abs(n)
    while remaining:
        cur = cur + timedelta(days=step)
        if is_trading_day(cur):
            remaining -= 1
    return cur


def market_open_utc(d: date) -> datetime:
    return datetime.combine(d, time(9, 30), tzinfo=_ET).astimezone(timezone.utc)


def earnings_status(
    ok: bool | None,
    captured_at_utc: datetime | None,
    earnings_dates: list[date],
    entry_session: date,
    is_trading_day,
) -> str:
    """Fail-closed earnings gate (§5A, R23). Only CLEAR is clear.

    ACTIVE requires an ok snapshot, captured strictly before 09:30 ET on e, whose
    dates bracket the window [e-5, e+5] sessions on both sides.
    """
    if not ok or captured_at_utc is None:
        return INACTIVE_FAILED
    if captured_at_utc.tzinfo is None:
        captured_at_utc = captured_at_utc.replace(tzinfo=timezone.utc)
    if captured_at_utc >= market_open_utc(entry_session):
        return INACTIVE_LATE
    lo = session_offset(entry_session, -EARN_WIN, is_trading_day)
    hi = session_offset(entry_session, EARN_WIN, is_trading_day)
    if not (any(d < lo for d in earnings_dates) and any(d > hi for d in earnings_dates)):
        return BLOCK_EARNINGS_UNKNOWN
    if any(lo <= d <= hi for d in earnings_dates):
        return BLOCK_EARNINGS
    return CLEAR


def parse_earnings_dates(payload: list[dict] | None) -> list[date]:
    out: list[date] = []
    for row in payload or []:
        raw = row.get("date") if isinstance(row, dict) else None
        if not raw:
            continue
        try:
            out.append(date.fromisoformat(str(raw)[:10]))
        except ValueError:
            continue
    return sorted(set(out))


# ── Exits (§6) ───────────────────────────────────────────────────────────────

@dataclass
class CloneResult:
    horizon: int
    complete: bool                 # bar e+h-1 exists (uniform censoring, R22)
    exited: bool                   # economic exit booked
    exit_date: date | None
    exit_price: float | None
    exit_reason: str | None
    net_return: float | None       # fraction, net of 10 bp per side
    stopped: bool
    gap_through: bool
    mfe_pct: float
    mae_pct: float


def clone_exit(
    bars: pd.DataFrame,
    sessions: list[date],
    e_idx: int,
    open_e: float,
    stop: float,
    horizon: int,
    last_idx: int,
) -> CloneResult:
    """Structural stop, fixed horizon, through the canonical exit engine.

    Entry is ``open_e * (1 + COST)`` and ``walk_exit`` applies ``COST`` once on
    the exit, so the net return is X(1-c) / (E(1+c)) - 1 with no double count
    (R29). No target, trail, partial, time stop or early exit.
    """
    from src.backtest.exit_engine import ExitBar, ExitParams, walk_exit

    end_idx = e_idx + horizon - 1
    # Complete only when the terminal bar e+h-1 itself exists (§6). A session
    # that has passed but left no bar for this symbol is censored, never
    # replaced by an earlier close.
    complete = (
        end_idx <= last_idx
        and end_idx < len(bars)
        and bool(np.isfinite(bars["close"].iloc[end_idx]))
    )
    exit_bars: list[ExitBar] = []
    bar_idx: list[int] = []
    for i in range(e_idx, min(end_idx, last_idx) + 1):
        row = bars.iloc[i]
        if not np.isfinite(row["close"]):
            continue
        exit_bars.append(ExitBar(date=sessions[i], open=float(row["open"]), high=float(row["high"]),
                                 low=float(row["low"]), close=float(row["close"])))
        bar_idx.append(i)
    entry = open_e * (1.0 + COST)
    params = ExitParams(
        stop=stop, target=float("inf"), max_hold=horizon, slippage=COST,
        trail_activate_pct=0.0, trail_distance_pct=0.0, partial_tp_target=0.0,
        time_stop_days=0, time_stop_eligible=False, early_exit_mfe_pct=0.0,
        gap_through=True, check_entry_bar=True,
    )
    if not exit_bars:
        return CloneResult(horizon, complete, False, None, None, None, None, False, False, 0.0, 0.0)
    out = walk_exit(exit_bars, entry, params)
    if out.exited:
        i = out.exit_index
        assert i is not None and out.exit_price is not None
        gap = out.exit_reason == "stop" and exit_bars[i].open <= stop and i > 0
        return CloneResult(
            horizon, complete, True, exit_bars[i].date, out.exit_price, out.exit_reason,
            out.exit_price / entry - 1.0, out.exit_reason == "stop", gap, out.mfe_pct, out.mae_pct,
        )
    if complete:
        # The terminal bar exists but a bar inside the window was missing, so
        # walk_exit (which counts bars, not sessions) never reached max_hold.
        # The last walked bar IS the terminal bar here: book its close.
        last = exit_bars[-1]
        assert last.date == sessions[end_idx]
        px = last.close * (1.0 - COST)
        return CloneResult(horizon, True, True, last.date, px, "expiry", px / entry - 1.0,
                           False, False, out.mfe_pct, out.mae_pct)
    return CloneResult(horizon, False, False, None, None, None, None, False, False,
                       out.mfe_pct, out.mae_pct)


def benchmark_return(bars: pd.DataFrame, e_idx: int, horizon: int, last_idx: int) -> float | None:
    """open_e -> close_{e+h-1}, no stop, no costs (§6)."""
    end_idx = e_idx + horizon - 1
    if end_idx > last_idx or e_idx >= len(bars):
        return None
    o = bars["open"].iloc[e_idx]
    c = bars["close"].iloc[end_idx]
    if not (np.isfinite(o) and np.isfinite(c)) or o <= 0:
        return None
    return float(c / o - 1.0)


# ── Regime label at e (§7, R8) ───────────────────────────────────────────────

def spy_regime_label(spy: pd.DataFrame, e_idx: int) -> str | None:
    """(UP|DOWN) x (HIVOL|LOVOL) from SPY at e-1. Needs the full trailing 252."""
    close = spy["close"].to_numpy(dtype=float)
    i = e_idx - 1
    if i < 0:
        return None
    sma50 = sma(close, 50)
    logret = np.log(close / np.concatenate([[np.nan], close[:-1]]))
    vol20 = pd.Series(logret).rolling(20, min_periods=20).std(ddof=1).to_numpy()
    if i < 251 or not np.isfinite(sma50[i]) or not np.isfinite(vol20[i]):
        return None
    trailing = vol20[i - 251: i + 1]
    if np.count_nonzero(np.isfinite(trailing)) < 252:
        return None
    trend = "UP" if close[i] >= sma50[i] else "DOWN"
    vol = "HIVOL" if vol20[i] >= float(np.median(trailing)) else "LOVOL"
    return f"{trend}_{vol}"


# ── Nearest-neighbour controls (§7, R20, R24) ────────────────────────────────

@dataclass
class ControlCandidate:
    episode_id: str
    symbol: str
    sector: str | None
    mcap: float | None
    score_pct: float | None
    sma_dist: float | None
    age: int | None
    prehistory: str
    has_entry_bar: bool = True


def _feature_vector(c, lane: str, use_age: bool) -> dict[str, float | None]:
    feats: dict[str, float | None] = {
        "log_mcap": math.log(c.mcap) if c.mcap and c.mcap > 0 else None,
        "sma_dist": c.sma_dist,
    }
    if lane == RANGE:
        feats["score_pct"] = c.score_pct
    if use_age:
        feats["age"] = float(c.age) if c.age is not None else None
    return feats


def nn_controls(trigger: ControlCandidate, risk_set: list[ControlCandidate], lane: str,
                k: int = NN_K) -> list[ControlCandidate]:
    """The k nearest same-bloodline controls from the persisted risk set (§7, frozen).

    - same sector when any same-sector candidate exists;
    - a VALID trigger uses VALID controls when >= 3 exist (then age is a feature);
      otherwise age is dropped;
    - standardise by population SD (ddof=0) over the full persisted risk set
      excluding the trigger; SD = 1 with < 2 non-null values or zero variance;
    - a dimension the trigger lacks is dropped; a candidate missing a dimension
      adds 1.0 to its squared distance; ties broken by control episode_id.
    """
    full = [c for c in risk_set if c.episode_id != trigger.episode_id and c.symbol != trigger.symbol]
    pool = [c for c in full if c.has_entry_bar]
    if not pool:
        return []
    same_sector = [c for c in pool if trigger.sector and c.sector == trigger.sector]
    if same_sector:
        pool = same_sector
    use_age = False
    if trigger.prehistory == VALID:
        valid = [c for c in pool if c.prehistory == VALID]
        if len(valid) >= NN_MIN_VALID:
            pool, use_age = valid, True

    t_feats = _feature_vector(trigger, lane, use_age)
    dims = [d for d, v in t_feats.items() if v is not None]
    sds: dict[str, float] = {}
    for d in dims:
        vals = [v for v in (_feature_vector(c, lane, use_age)[d] for c in full) if v is not None]
        sd = float(np.std(vals, ddof=0)) if len(vals) >= 2 else 0.0
        sds[d] = sd if sd > 0 else 1.0

    def dist(c: ControlCandidate) -> float:
        cf = _feature_vector(c, lane, use_age)
        total = 0.0
        for d in dims:
            v = cf[d]
            if v is None:
                total += 1.0
            else:
                total += ((v - t_feats[d]) / sds[d]) ** 2
        return math.sqrt(total)

    ranked = sorted(pool, key=lambda c: (dist(c), c.episode_id))
    return ranked[:k]


def control_stop(control_open_e: float, trigger_open_e: float, trigger_stop: float) -> float:
    """Control stop = control_open_e * (1 - (E - S) / E)."""
    return control_open_e * (1.0 - (trigger_open_e - trigger_stop) / trigger_open_e)


# ── Statistics (§7) ──────────────────────────────────────────────────────────

def date_clustered_mean(values: list[float], dates: list[date]) -> float | None:
    by_date: dict[date, list[float]] = {}
    for v, d in zip(values, dates):
        by_date.setdefault(d, []).append(v)
    if not by_date:
        return None
    return float(np.mean([np.mean(v) for v in by_date.values()]))


def date_clustered_bootstrap_ci(
    values: list[float], dates: list[date],
    n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED, alpha: float = 0.05,
) -> tuple[float, float] | None:
    """95% percentile CI resampling entry dates with replacement (R2). Reported, not gating."""
    by_date: dict[date, list[float]] = {}
    for v, d in zip(values, dates):
        by_date.setdefault(d, []).append(v)
    if len(by_date) < 2:
        return None
    means = np.array([np.mean(v) for v in by_date.values()])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(means), size=(n, len(means)))
    boot = means[idx].mean(axis=1)
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


@dataclass
class GateInput:
    """One clean trigger's anchor-horizon observations for the §7 flags."""
    entry_date: date
    regime: str | None
    excess_h5: float | None
    excess_h10: float | None
    stopped_h10: bool
    complete_h5: bool
    complete_h10: bool


def promotion_flag(rows: list[GateInput]) -> bool:
    """PROMOTION (per bloodline): eligibility for Neo/Hawk review only."""
    c10 = [r for r in rows if r.complete_h10]
    if len(c10) < GATE_MIN_CLEAN:
        return False
    if len({r.entry_date for r in c10}) < GATE_MIN_DATES:
        return False
    clusters: dict[str, int] = {}
    for r in c10:
        if r.regime:
            clusters[r.regime] = clusters.get(r.regime, 0) + 1
    if sum(1 for n in clusters.values() if n >= CLUSTER_MIN) < GATE_MIN_CLUSTERS:
        return False
    e5 = [r.excess_h5 for r in rows if r.complete_h5 and r.excess_h5 is not None]
    e10 = [r.excess_h10 for r in c10 if r.excess_h10 is not None]
    return bool(e5 and e10 and np.mean(e5) > 0 and np.mean(e10) > 0)


def kill_clock_due(n_clean_complete10: int, forward_start: date | None, as_of: date) -> bool:
    """KILL is evaluated at the first of 30 clean complete-10 triggers or 26 weeks."""
    if n_clean_complete10 >= GATE_MIN_CLEAN:
        return True
    if forward_start is None:
        return False
    return (as_of - forward_start).days >= KILL_WEEKS * 7


def kill_checkpoint(rows: list[GateInput], forward_start: date | None,
                    as_of: date) -> date | None:
    """The entry-date cutoff at which KILL is evaluated, or None if not yet due.

    §7 evaluates KILL ONCE, at the first of (30 clean complete-h10 triggers) or
    (26 weeks after the forward start). Later observations must not reverse
    that verdict, so the flag is computed on the cohort as it stood then.
    Triggers are ordered by entry date; completion lags entry by a fixed nine
    sessions, so entry order is completion order.
    """
    c10 = sorted((r for r in rows if r.complete_h10), key=lambda r: r.entry_date)
    cutoff: date | None = None
    if len(c10) >= GATE_MIN_CLEAN:
        cutoff = c10[GATE_MIN_CLEAN - 1].entry_date
    if forward_start is not None:
        clock = forward_start + timedelta(weeks=KILL_WEEKS)
        if as_of >= clock and (cutoff is None or clock < cutoff):
            cutoff = clock
    return cutoff


def kill_flag(rows: list[GateInput], forward_start: date | None, as_of: date) -> bool:
    cutoff = kill_checkpoint(rows, forward_start, as_of)
    if cutoff is None:
        return False
    rows = [r for r in rows if r.entry_date <= cutoff]
    c10 = [r for r in rows if r.complete_h10]
    if not c10:
        return False
    e5 = [r.excess_h5 for r in rows if r.complete_h5 and r.excess_h5 is not None]
    e10 = [r.excess_h10 for r in c10 if r.excess_h10 is not None]
    if not e10:
        return False
    stop_touch = sum(1 for r in c10 if r.stopped_h10) / len(c10)
    mean5 = float(np.mean(e5)) if e5 else 0.0
    return mean5 <= 0 and float(np.mean(e10)) <= 0 and stop_touch >= KILL_STOP_TOUCH


# ── Signal row (§8.2) ────────────────────────────────────────────────────────

@dataclass
class ReclaimSignal:
    """Mirrors MeanReversionSignal's fields so ranker type lookups work.

    ``score`` is a constant: the spec has no ranking (R4). The loop never ranks.
    """
    ticker: str
    score: float
    direction: str
    entry_price: float          # close_k, reference only
    stop_loss: float            # setup_low
    target_1: float             # NO_TARGET_SENTINEL
    target_2: float             # NO_TARGET_SENTINEL
    holding_period: int         # horizon h
    components: dict
    max_entry_price: float | None = None   # informational
    bloodline: str = ""
    episode_id: str = ""
    floor_type: str = ""
    floor: float | None = None
    reclaim_date: date | None = None
    retest_date: date | None = None
    trigger_date: date | None = None
    high_retest: float | None = None
    sma50_k: float | None = None
    slope5: float | None = None
    prehistory: str = INSUFFICIENT_PREHISTORY
    prehistory_reason: str | None = None
    overlap_desc: bool = False
    score_pct: float | None = None
    horizon: int = ANCHOR_HORIZON

    @classmethod
    def from_trigger(cls, trig: Trigger, ep: Episode, horizon: int, *,
                     overlap_desc: bool, score_pct_value: float | None) -> "ReclaimSignal":
        return cls(
            ticker=trig.symbol, score=50.0, direction="LONG",
            entry_price=trig.close_k, stop_loss=trig.setup_low,
            target_1=NO_TARGET_SENTINEL, target_2=NO_TARGET_SENTINEL,
            holding_period=horizon, components={"note": "NOT_A_RANKING"},
            max_entry_price=trig.max_entry,
            bloodline=trig.lane, episode_id=trig.episode_id,
            floor_type=ep.floor_type, floor=ep.floor,
            reclaim_date=trig.reclaim_date, retest_date=trig.retest_date,
            trigger_date=trig.trigger_date, high_retest=trig.high_retest,
            sma50_k=trig.sma50_k, slope5=trig.slope5,
            prehistory=ep.prehistory, prehistory_reason=ep.prehistory_reason,
            overlap_desc=overlap_desc, score_pct=score_pct_value, horizon=horizon,
        )

    def persisted_features(self) -> dict:
        """JSON-safe: no inf, no NaN, dates as ISO strings."""
        def _f(v):
            if v is None:
                return None
            if isinstance(v, float) and not math.isfinite(v):
                return None
            if isinstance(v, date):
                return v.isoformat()
            return v
        return {
            "no_target": True,
            "exit_policy": EXIT_POLICY,
            "bloodline": self.bloodline,
            "episode_id": self.episode_id,
            "floor_type": self.floor_type,
            "floor": _f(self.floor),
            "reclaim_date": _f(self.reclaim_date),
            "retest_date": _f(self.retest_date),
            "trigger_date": _f(self.trigger_date),
            "high_retest": _f(self.high_retest),
            "sma50_k": _f(self.sma50_k),
            "slope5": _f(self.slope5),
            "prehistory": self.prehistory,
            "prehistory_reason": self.prehistory_reason,
            "overlap_desc": self.overlap_desc,
            "score_pct": _f(self.score_pct),
            "horizon": self.horizon,
            "gate2": GATE2_NOT_APPLICABLE,
            "max_entry_price": _f(self.max_entry_price),
            "model_raw_score": self.score,
            "model_components": dict(self.components),
            "score_source": "score_reclaim",
        }
