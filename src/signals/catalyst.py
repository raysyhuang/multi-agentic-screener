"""Catalyst / event-driven signal model.

Identifies trades around known upcoming events: earnings, FDA dates,
product launches, etc. Positions taken before the catalyst with
defined risk parameters.

Fires on day T close, execution at T+1 open.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math


@dataclass
class CatalystSignal:
    ticker: str
    score: float  # 0-100
    direction: str  # LONG
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    holding_period: int
    catalyst_type: str  # earnings / fda / product_launch
    catalyst_date: date | None
    components: dict


def score_catalyst(
    ticker: str,
    features: dict,
    fundamental_data: dict,
    days_to_earnings: int | None = None,
    sentiment: dict | None = None,
) -> CatalystSignal | None:
    """Score a ticker for event-driven catalyst potential.

    Focus on earnings catalysts (most data-rich):
      1. Upcoming earnings within 5-15 days
      2. History of beating estimates (beat streak)
      3. Positive earnings momentum (surprises getting larger)
      4. Positive sentiment shift in news
      5. Insider buying preceding the event
    """
    def _valid(x) -> bool:
        if x is None:
            return False
        try:
            return math.isfinite(float(x))
        except (TypeError, ValueError):
            return False

    scores = {}
    sentiment = sentiment or {}

    # --- 1. Earnings Timing (30%) ---
    timing_score = 0.0
    if days_to_earnings is not None:
        if 5 <= days_to_earnings <= 15:
            timing_score = 100  # sweet spot: pre-earnings drift zone
        elif 2 <= days_to_earnings < 5:
            timing_score = 60  # close but more risky
        elif 15 < days_to_earnings <= 30:
            timing_score = 40  # a bit early but can position
        else:
            return None  # too far or too close

    scores["timing"] = timing_score

    # --- 2. Earnings Beat History (25%) ---
    earnings_data = fundamental_data.get("earnings_surprises", {})
    if isinstance(earnings_data, dict):
        beat_streak = earnings_data.get("beat_streak", 0)
        avg_surprise = earnings_data.get("avg_surprise_pct")
    else:
        beat_streak = 0
        avg_surprise = None

    beat_score = 0.0
    if beat_streak >= 4:
        beat_score = 100
    elif beat_streak >= 3:
        beat_score = 80
    elif beat_streak >= 2:
        beat_score = 50
    elif beat_streak >= 1:
        beat_score = 30

    if _valid(avg_surprise) and avg_surprise > 10:
        beat_score = min(100, beat_score + 20)

    scores["beat_history"] = beat_score

    # --- 3. Earnings Momentum (15%) ---
    earnings_momentum = 0.0
    if isinstance(earnings_data, dict):
        em = earnings_data.get("earnings_momentum", 0)
        if em and em > 0:
            earnings_momentum = min(100, em * 5)
    scores["earnings_momentum"] = earnings_momentum

    # --- 4. Sentiment (15%) ---
    sent_score = 50  # neutral default
    sent_val = sentiment.get("sentiment_score", 0)
    if sent_val > 0.3:
        sent_score = 90
    elif sent_val > 0.1:
        sent_score = 70
    elif sent_val < -0.3:
        sent_score = 10
    elif sent_val < -0.1:
        sent_score = 30
    scores["sentiment"] = sent_score

    # --- 5. Insider Activity (15%) ---
    insider_data = fundamental_data.get("insider_activity", {})
    insider_score = 50  # neutral
    if isinstance(insider_data, dict):
        net_ratio = insider_data.get("insider_net_ratio", 0)
        if net_ratio > 0.5:
            insider_score = 90
        elif net_ratio > 0:
            insider_score = 70
        elif net_ratio < -0.3:
            insider_score = 20
    scores["insider"] = insider_score

    # --- Composite ---
    weights = {
        "timing": 0.30,
        "beat_history": 0.25,
        "earnings_momentum": 0.15,
        "sentiment": 0.15,
        "insider": 0.15,
    }
    composite = sum(scores[k] * weights[k] for k in weights)

    if composite < 45:
        return None

    # --- Price targets ---
    close_price = features.get("close", 0)
    atr = features.get("atr_14")
    if not _valid(close_price) or close_price <= 0:
        return None
    if not _valid(atr) or atr <= 0:
        atr = close_price * 0.02
    atr = max(atr, close_price * 0.005)

    # Wider targets for catalyst plays (earnings can gap)
    target_1 = close_price + 2.5 * atr
    target_2 = close_price + 4.0 * atr
    stop_loss = close_price - 2.0 * atr

    # Holding period = days until after earnings + buffer
    holding = min(15, (days_to_earnings or 10) + 2)

    return CatalystSignal(
        ticker=ticker,
        score=round(composite, 2),
        direction="LONG",
        entry_price=round(close_price, 2),
        stop_loss=round(stop_loss, 2),
        target_1=round(target_1, 2),
        target_2=round(target_2, 2),
        holding_period=holding,
        catalyst_type="earnings",
        catalyst_date=None,
        components=scores,
    )
