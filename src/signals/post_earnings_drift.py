"""Post-earnings drift (PEAD) signal — long an earnings beat for the multi-week drift.

Backtested this session (outputs/research/pead_FINDINGS.md): long beats at T+1 open
with a ~20-day hold shows real, cost-surviving, sub-period-stable drift (beat>10%:
57% WR, +1.80%/trade net, PF 1.73, per-trade Sharpe 1.50). The one candidate of four
to clear the bar — shipped default-OFF for a paper trial, not production capital.

The signal fires when the ticker reported an earnings BEAT (EPS surprise above a
threshold) on the as-of day, so the pipeline's T+1-open execution matches the
backtest's "enter the day after the announcement" convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd


def _valid(x) -> bool:
    if x is None:
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def eps_surprise_pct(eps_actual, eps_estimated) -> float | None:
    """Signed EPS surprise as a percent of |estimate|. None if not computable."""
    if not _valid(eps_actual) or not _valid(eps_estimated):
        return None
    est = float(eps_estimated)
    if abs(est) < 1e-6:
        return None
    return (float(eps_actual) - est) / abs(est) * 100


@dataclass
class PEADSignal:
    ticker: str
    score: float          # 0-100 (scaled from surprise magnitude)
    direction: str        # LONG
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    holding_period: int
    components: dict = field(default_factory=dict)
    max_entry_price: float | None = None


def score_post_earnings_drift(
    ticker: str,
    df: pd.DataFrame,
    features: dict,
    earnings_surprise_pct: float | None,
    regime: str = "unknown",
    min_surprise: float = 10.0,
    stop_atr_mult: float = 3.0,
    target_atr_mult: float = 6.0,
    holding_period: int = 20,
    min_price: float = 5.0,
) -> PEADSignal | None:
    """Emit a long PEAD signal if the ticker just reported a sufficient beat.

    earnings_surprise_pct is the EPS surprise (percent of |estimate|) for the
    ticker's report on the as-of day, or None if it did not report / no estimate.
    Returns None unless the surprise is a beat >= min_surprise.

    Guards (the edge was measured on liquid S&P names): reject sub-min_price names
    and reject when 3xATR exceeds the price (a would-be negative stop = too
    volatile for a 20-day swing hold).
    """
    if earnings_surprise_pct is None or earnings_surprise_pct < min_surprise:
        return None
    if df is None or df.empty:
        return None

    close_price = features.get("close")
    if not _valid(close_price):
        close_price = float(df["close"].iloc[-1])
    close_price = float(close_price)
    if close_price <= 0 or close_price < min_price:
        return None

    atr = features.get("atr_14")
    if not _valid(atr) or float(atr) <= 0:
        atr = close_price * 0.03
    atr = max(float(atr), close_price * 0.005)

    stop_loss = close_price - stop_atr_mult * atr
    # A non-positive stop means the name is too volatile (3xATR >= price) for this
    # swing strategy — reject rather than emit a nonsensical negative stop.
    if stop_loss <= 0:
        return None
    target_1 = close_price + target_atr_mult * atr
    target_2 = close_price + (target_atr_mult + 2.0) * atr

    # Score scales with surprise magnitude: min_surprise -> 60, +40%-over -> ~100.
    score = max(0.0, min(60.0 + (earnings_surprise_pct - min_surprise), 100.0))

    return PEADSignal(
        ticker=ticker,
        score=round(score, 2),
        direction="LONG",
        entry_price=round(close_price, 2),
        stop_loss=round(stop_loss, 2),
        target_1=round(target_1, 2),
        target_2=round(target_2, 2),
        holding_period=holding_period,
        components={"eps_surprise_pct": round(float(earnings_surprise_pct), 2)},
    )
