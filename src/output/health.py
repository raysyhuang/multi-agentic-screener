"""Position Health Card engine — computes daily health scores for open positions.

Each open position gets a PositionHealthCard with:
  - 5 component scores (trend, momentum, volume, risk, regime)
  - A composite Promising Score (0-100)
  - A 3-state machine: ON_TRACK / WATCH / EXIT
  - Strategy-specific hard invalidation checks
  - Soft invalidation checks (forces WATCH regardless of score)
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from src.contracts import (
    HealthCardConfig,
    HealthComponent,
    HealthState,
    PositionHealthCard,
)
from src.db.models import Outcome, Signal
from src.features.technical import compute_all_technical_features, latest_features

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

# Lookback days for OHLCV fetch — must cover the longest indicator window
# (SMA50 needs ~70 business days, plus buffer for weekends/holidays).
HISTORY_LOOKBACK_DAYS = 120

# ── Strategy Family Mapping ────────────────────────────────────────────────
# Maps signal_model strings to a canonical strategy family for invalidation
# routing. Add new model variants here — NOT in ad-hoc string matching.

STRATEGY_FAMILY: dict[str, str] = {
    # Breakout / momentum family
    "breakout": "breakout",
    "momentum_breakout": "breakout",
    "momo_v1": "breakout",
    # Mean reversion family
    "mean_reversion": "mean_reversion",
    "mean_rev": "mean_reversion",
    "rsi2": "mean_reversion",
    "oversold_reversion": "mean_reversion",
    # Catalyst / event-driven family
    "catalyst": "catalyst",
    "earnings_drift": "catalyst",
}


def get_strategy_family(signal_model: str) -> str | None:
    """Resolve a signal_model string to its canonical strategy family.

    Returns None if the model is unknown (caller decides how to handle).
    """
    return STRATEGY_FAMILY.get(signal_model.lower())

DEFAULT_CONFIG = HealthCardConfig()


def compute_score_velocity(
    current_score: float,
    previous_scores: list[float] | None,
) -> float | None:
    """Compute score velocity as (current - oldest) / N points per day.

    Args:
        current_score: Today's promising score.
        previous_scores: Chronologically ordered list of previous scores
                         (oldest first). Typically the last 1-3 days.

    Returns:
        Velocity in pts/day, or None if no previous data.
    """
    if not previous_scores:
        return None

    all_scores = previous_scores + [current_score]
    n = len(all_scores) - 1  # number of intervals
    if n <= 0:
        return None

    velocity = (all_scores[-1] - all_scores[0]) / n
    return round(velocity, 2)


def _check_velocity_warning(velocity: float | None, score: float) -> bool:
    """Check if velocity + score combination warrants an early warning.

    Fires when the score is deteriorating rapidly AND already in a
    concerning range — catching the 85→78→72→71 pattern before EXIT.

    Returns True if velocity < -5 AND score < 75.
    """
    if velocity is None:
        return False
    return velocity < -5 and score < 75


async def compute_health_card(
    outcome: Outcome,
    signal: Signal,
    df: pd.DataFrame,
    config: HealthCardConfig = DEFAULT_CONFIG,
    previous_state: HealthState | None = None,
    current_regime: str | None = None,
    previous_scores: list[float] | None = None,
) -> PositionHealthCard | None:
    """Compute a full health card for one open position.

    Args:
        outcome: The open Outcome row.
        signal: The associated Signal row.
        df: OHLCV DataFrame from entry_date to today (already fetched by caller).
        config: Scoring weights and thresholds.
        previous_state: State from yesterday's health card (for transition detection).
        current_regime: Current market regime label (bull/bear/choppy).

    Returns:
        PositionHealthCard or None if insufficient data.
    """
    if df is None or df.empty or len(df) < 5:
        return None

    # Compute technical features
    df = compute_all_technical_features(df)
    feat = latest_features(df)

    today = date.today()
    entry_price = outcome.entry_price
    current_price = feat.get("close", entry_price)

    days_held = (today - outcome.entry_date).days

    # Excursion tracking
    if "date" in df.columns:
        entry_ts = pd.Timestamp(outcome.entry_date)
        since_entry = df[df["date"] >= entry_ts]
    else:
        since_entry = df
    max_price = float(since_entry["high"].max()) if not since_entry.empty else current_price
    min_price = float(since_entry["low"].min()) if not since_entry.empty else current_price

    pnl_pct = (current_price - entry_price) / entry_price * 100
    mfe_pct = (max_price - entry_price) / entry_price * 100
    mae_pct = (min_price - entry_price) / entry_price * 100

    # ATR stop distance
    atr_14 = feat.get("atr_14")
    atr_stop_distance = None
    if atr_14 and atr_14 > 0:
        atr_stop_distance = (current_price - signal.stop_loss) / atr_14

    # Compute 5 components
    trend = _compute_trend(feat, df, config.trend_weight)
    momentum = _compute_momentum(feat, df, config.momentum_weight)
    volume = _compute_volume(feat, df, config.volume_weight)
    risk = _compute_risk(
        feat, signal, current_price, atr_stop_distance,
        mfe_pct, mae_pct, days_held, config.risk_weight,
    )
    regime = _compute_regime(signal.regime, current_regime, config.regime_weight)

    # Composite score
    promising_score = (
        trend.weighted_score
        + momentum.weighted_score
        + volume.weighted_score
        + risk.weighted_score
        + regime.weighted_score
    )
    promising_score = max(0.0, min(100.0, round(promising_score, 2)))

    # Score velocity
    score_velocity = compute_score_velocity(promising_score, previous_scores)
    velocity_warning = _check_velocity_warning(score_velocity, promising_score)

    # Hard invalidation
    hard_invalidation, invalidation_reason = _check_hard_invalidation(
        signal_model=signal.signal_model,
        feat=feat,
        df=df,
        days_held=days_held,
        holding_period=signal.holding_period_days,
        pnl_pct=pnl_pct,
        entry_price=entry_price,
    )

    # Soft invalidation — forces WATCH regardless of score
    soft_invalidation = _check_soft_invalidation(
        feat=feat,
        signal_model=signal.signal_model,
        signal_regime=signal.regime,
        current_regime=current_regime,
    )

    # State classification
    if hard_invalidation or promising_score < config.watch_min:
        state = HealthState.EXIT
    elif soft_invalidation or velocity_warning or promising_score < config.on_track_min:
        state = HealthState.WATCH
    else:
        state = HealthState.ON_TRACK

    state_changed = previous_state is not None and state != previous_state

    return PositionHealthCard(
        trend_health=trend,
        momentum_health=momentum,
        volume_confirmation=volume,
        risk_integrity=risk,
        regime_alignment=regime,
        promising_score=promising_score,
        state=state,
        previous_state=previous_state,
        state_changed=state_changed,
        score_velocity=score_velocity,
        hard_invalidation=hard_invalidation,
        invalidation_reason=invalidation_reason,
        days_held=days_held,
        expected_hold_days=signal.holding_period_days,
        pnl_pct=round(pnl_pct, 4),
        mfe_pct=round(mfe_pct, 4),
        mae_pct=round(mae_pct, 4),
        current_price=round(current_price, 4),
        atr_14=round(atr_14, 4) if atr_14 is not None else None,
        atr_stop_distance=round(atr_stop_distance, 4) if atr_stop_distance is not None else None,
        signal_id=signal.id,
        ticker=outcome.ticker,
        signal_model=signal.signal_model,
        as_of_date=today,
    )


# ── Component Scorers ──────────────────────────────────────────────────────


def _compute_trend(feat: dict, df: pd.DataFrame, weight: float) -> HealthComponent:
    """Trend health: price vs EMAs, higher-low detection, slope."""
    score = 0.0
    details: dict[str, float | str | None] = {}

    close = feat.get("close")
    ema_21 = feat.get("ema_21")
    sma_50 = feat.get("sma_50")

    # Price vs EMA 21
    if close is not None and ema_21 is not None and ema_21 > 0:
        if close > ema_21:
            score += 30
            details["ema_21_position"] = "above"
        else:
            score -= 20
            details["ema_21_position"] = "below"
    details["ema_21"] = ema_21

    # Price vs SMA 50
    if close is not None and sma_50 is not None and sma_50 > 0:
        if close > sma_50:
            score += 25
            details["sma_50_position"] = "above"
        else:
            score -= 15
            details["sma_50_position"] = "below"
    details["sma_50"] = sma_50

    # Higher-low detection via 5-bar rolling lows
    if len(df) >= 10 and "low" in df.columns:
        rolling_lows = df["low"].rolling(5).min().dropna()
        if len(rolling_lows) >= 2:
            if rolling_lows.iloc[-1] > rolling_lows.iloc[-2]:
                score += 25
                details["higher_lows"] = "yes"
            else:
                score -= 25
                details["higher_lows"] = "no"

    # 5-day slope direction
    if len(df) >= 5 and "close" in df.columns:
        recent = df["close"].iloc[-5:]
        slope = recent.iloc[-1] - recent.iloc[0]
        if slope > 0:
            score += 20
            details["slope_5d"] = "up"
        else:
            score -= 20
            details["slope_5d"] = "down"

    score = max(0.0, min(100.0, score))
    return HealthComponent(
        name="trend",
        score=round(score, 2),
        weight=weight,
        weighted_score=round(score * weight, 2),
        details=details,
    )


def _compute_momentum(feat: dict, df: pd.DataFrame, weight: float) -> HealthComponent:
    """Momentum health: RSI zones, RSI direction, MACD histogram slope."""
    score = 0.0
    details: dict[str, float | str | None] = {}

    rsi = feat.get("rsi_14")
    details["rsi_14"] = rsi

    # RSI zone scoring
    if rsi is not None:
        if 60 <= rsi <= 70:
            score += 75
            details["rsi_zone"] = "strong"
        elif 40 <= rsi < 60:
            score += 50
            details["rsi_zone"] = "neutral"
        elif rsi > 70:
            score += 40
            details["rsi_zone"] = "overbought"
        elif 30 <= rsi < 40:
            score += 25
            details["rsi_zone"] = "weak"
        else:  # < 30
            score += 10
            details["rsi_zone"] = "oversold"

    # RSI direction (current vs 3 bars ago)
    if len(df) >= 4 and "rsi_14" in df.columns:
        rsi_series = df["rsi_14"].dropna()
        if len(rsi_series) >= 4:
            rsi_now = rsi_series.iloc[-1]
            rsi_3ago = rsi_series.iloc[-4]
            if rsi_now > rsi_3ago:
                score += 15
                details["rsi_direction"] = "rising"
            else:
                score -= 15
                details["rsi_direction"] = "falling"

    # MACD histogram slope
    macd_col = "MACDh_12_26_9"
    if macd_col in df.columns:
        macd_hist = df[macd_col].dropna()
        if len(macd_hist) >= 2:
            if macd_hist.iloc[-1] > macd_hist.iloc[-2]:
                score += 25
                details["macd_slope"] = "rising"
            else:
                score -= 10
                details["macd_slope"] = "falling"

    score = max(0.0, min(100.0, score))
    return HealthComponent(
        name="momentum",
        score=round(score, 2),
        weight=weight,
        weighted_score=round(score * weight, 2),
        details=details,
    )


def _compute_volume(feat: dict, df: pd.DataFrame, weight: float) -> HealthComponent:
    """Volume confirmation: RVOL scoring, OBV trend."""
    score = 0.0
    details: dict[str, float | str | None] = {}

    rvol = feat.get("rvol")
    details["rvol"] = rvol

    # RVOL scoring
    if rvol is not None:
        if rvol > 1.5:
            score += 90
            details["rvol_zone"] = "strong"
        elif rvol >= 1.0:
            score += 65
            details["rvol_zone"] = "normal"
        elif rvol >= 0.8:
            score += 40
            details["rvol_zone"] = "low"
        else:
            score += 15
            details["rvol_zone"] = "collapsed"

    # OBV 5-day SMA trend
    if "obv" in df.columns:
        obv = df["obv"].dropna()
        if len(obv) >= 5:
            obv_sma = obv.rolling(5).mean()
            if len(obv_sma.dropna()) >= 2:
                if obv_sma.iloc[-1] > obv_sma.iloc[-2]:
                    score += 20
                    details["obv_trend"] = "rising"
                else:
                    score -= 20
                    details["obv_trend"] = "falling"

    score = max(0.0, min(100.0, score))
    return HealthComponent(
        name="volume",
        score=round(score, 2),
        weight=weight,
        weighted_score=round(score * weight, 2),
        details=details,
    )


def _compute_risk(
    feat: dict,
    signal: Signal,
    current_price: float,
    atr_stop_distance: float | None,
    mfe_pct: float,
    mae_pct: float,
    days_held: int,
    weight: float,
) -> HealthComponent:
    """Risk integrity: ATR stop distance, BB width, MFE/MAE, overstay penalty."""
    score = 0.0
    details: dict[str, float | str | None] = {}

    # ATR stop distance
    details["atr_stop_distance"] = atr_stop_distance
    if atr_stop_distance is not None:
        if atr_stop_distance > 2:
            score += 80
            details["stop_comfort"] = "safe"
        elif atr_stop_distance > 1:
            score += 60
            details["stop_comfort"] = "comfortable"
        elif atr_stop_distance > 0.5:
            score += 35
            details["stop_comfort"] = "tight"
        else:
            score += 10
            details["stop_comfort"] = "danger"

    # BB width (normalized) — wider = more volatile = more risk
    bbu = feat.get("BBU_20_2.0")
    bbl = feat.get("BBL_20_2.0")
    bbm = feat.get("BBM_20_2.0")
    if bbu is not None and bbl is not None and bbm is not None and bbm > 0:
        bb_width = (bbu - bbl) / bbm
        details["bb_width"] = round(bb_width, 4)
    else:
        bb_width = None

    # MFE/MAE ratio
    mfe_mae_ratio = abs(mfe_pct / mae_pct) if mae_pct != 0 else 10.0
    details["mfe_mae_ratio"] = round(mfe_mae_ratio, 2)

    if mfe_mae_ratio > 2:
        score += 15
    elif mfe_mae_ratio < 0.5:
        score -= 15

    # Overstay penalty
    holding_period = signal.holding_period_days
    if holding_period > 0 and days_held > holding_period * 1.2:
        score -= 15
        details["overstay"] = "yes"
    else:
        details["overstay"] = "no"

    score = max(0.0, min(100.0, score))
    return HealthComponent(
        name="risk",
        score=round(score, 2),
        weight=weight,
        weighted_score=round(score * weight, 2),
        details=details,
    )


def _compute_regime(
    signal_regime: str, current_regime: str | None, weight: float
) -> HealthComponent:
    """Regime alignment: same regime = 100, partial = 50, full mismatch = 10."""
    details: dict[str, float | str | None] = {
        "signal_regime": signal_regime,
        "current_regime": current_regime,
    }

    if current_regime is None:
        score = 50.0
        details["alignment"] = "unknown"
    elif signal_regime.lower() == current_regime.lower():
        score = 100.0
        details["alignment"] = "aligned"
    elif _is_partial_mismatch(signal_regime, current_regime):
        score = 50.0
        details["alignment"] = "partial"
    else:
        score = 10.0
        details["alignment"] = "mismatch"

    return HealthComponent(
        name="regime",
        score=round(score, 2),
        weight=weight,
        weighted_score=round(score * weight, 2),
        details=details,
    )


def _is_partial_mismatch(a: str, b: str) -> bool:
    """Partial mismatch: bull↔choppy or bear↔choppy."""
    pair = {a.lower(), b.lower()}
    return "choppy" in pair


# ── Hard Invalidation ──────────────────────────────────────────────────────


def _check_hard_invalidation(
    signal_model: str,
    feat: dict,
    df: pd.DataFrame,
    days_held: int,
    holding_period: int,
    pnl_pct: float,
    entry_price: float,
) -> tuple[bool, str | None]:
    """Strategy-specific hard invalidation checks.

    Uses STRATEGY_FAMILY mapping for routing — add new models there.
    Returns (is_invalidated, reason).
    """
    family = get_strategy_family(signal_model)

    if family == "breakout":
        return _invalidate_breakout(feat, df)
    elif family == "mean_reversion":
        return _invalidate_mean_reversion(feat, df, days_held, holding_period, entry_price)
    elif family == "catalyst":
        return _invalidate_catalyst(days_held, holding_period, pnl_pct, feat)

    if family is None:
        logger.warning("Unknown signal_model '%s' — no invalidation rules applied", signal_model)

    return False, None


def _invalidate_breakout(feat: dict, df: pd.DataFrame) -> tuple[bool, str | None]:
    """Breakout: 2 consecutive closes back inside 20d range AND rvol < 0.8."""
    if len(df) < 3:
        return False, None

    rvol = feat.get("rvol")
    if rvol is not None and rvol >= 0.8:
        return False, None

    # Check last 2 closes vs 20d high/low
    if "high_20d" in df.columns and "low_20d" in df.columns:
        recent = df.iloc[-2:]
        inside_count = 0
        for _, row in recent.iterrows():
            close = row.get("close")
            h20 = row.get("high_20d")
            l20 = row.get("low_20d")
            if close is not None and h20 is not None and l20 is not None:
                if l20 < close < h20:
                    inside_count += 1
        if inside_count >= 2:
            return True, "breakout_range_failure"

    return False, None


def _invalidate_mean_reversion(
    feat: dict, df: pd.DataFrame, days_held: int, holding_period: int, entry_price: float
) -> tuple[bool, str | None]:
    """Mean reversion: new low below entry-day low + ATR expanding, or overstay + weak RSI."""
    if len(df) < 2:
        return False, None

    # Condition (a): new low below entry-day low + ATR expanding > 1.2x
    atr_14 = feat.get("atr_14")
    if atr_14 is not None and "atr_14" in df.columns:
        atr_series = df["atr_14"].dropna()
        if len(atr_series) >= 10:
            atr_start = atr_series.iloc[0]
            if atr_start > 0 and atr_14 / atr_start > 1.2:
                current_low = df["low"].iloc[-1]
                entry_low = df["low"].iloc[0]
                if current_low < entry_low:
                    return True, "mean_rev_new_low_expanding_atr"

    # Condition (b): days > holding period AND RSI < 40
    rsi = feat.get("rsi_14")
    if days_held > holding_period and rsi is not None and rsi < 40:
        return True, "mean_rev_overstay_weak_rsi"

    return False, None


def _invalidate_catalyst(
    days_held: int, holding_period: int, pnl_pct: float, feat: dict
) -> tuple[bool, str | None]:
    """Catalyst: post-event with no reaction — days > hold, abs(pnl) < 1%, rvol < 1.0."""
    if days_held <= holding_period:
        return False, None

    rvol = feat.get("rvol")
    if abs(pnl_pct) < 1.0 and rvol is not None and rvol < 1.0:
        return True, "catalyst_no_reaction"

    return False, None


# ── Soft Invalidation ─────────────────────────────────────────────────────

# Strategies that are compatible with a bear regime (e.g. short-biased).
_BEAR_COMPATIBLE_FAMILIES = {"mean_reversion"}


def _check_soft_invalidation(
    feat: dict,
    signal_model: str,
    signal_regime: str,
    current_regime: str | None,
) -> bool:
    """Soft invalidation — forces WATCH (not EXIT) regardless of composite score.

    Catches the common failure mode where a strong trend score keeps the
    composite high while the setup is actually breaking down.

    Returns True if soft invalidation fires.
    """
    close = feat.get("close")
    ema_21 = feat.get("ema_21")
    rvol = feat.get("rvol")

    # Condition 1: RVOL collapse + momentum weakening + under EMA21
    rvol_collapsed = rvol is not None and rvol < 0.8
    under_ema21 = (
        close is not None and ema_21 is not None and ema_21 > 0 and close < ema_21
    )
    rsi = feat.get("rsi_14")
    momentum_weak = rsi is not None and rsi < 45

    if rvol_collapsed and under_ema21 and momentum_weak:
        return True

    # Condition 2: Regime flip to bear for a non-bear-compatible strategy
    if current_regime is not None and current_regime.lower() == "bear":
        family = get_strategy_family(signal_model)
        if family and family not in _BEAR_COMPATIBLE_FAMILIES:
            # Only if signal was originally entered in a non-bear regime
            if signal_regime.lower() != "bear":
                return True

    return False
