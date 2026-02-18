"""Historical regime classification for the multi-engine backtest.

Wraps :func:`src.features.regime.classify_regime` with point-in-time slicing
so the orchestrator can classify the market regime for any historical date.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from src.features.regime import (
    RegimeAssessment,
    Regime,
    classify_regime,
    compute_breadth_score,
)

logger = logging.getLogger(__name__)


def classify_regime_for_date(
    screen_date: date,
    spy_df: pd.DataFrame,
    qqq_df: pd.DataFrame,
    price_data: dict[str, pd.DataFrame] | None = None,
    vix_series: pd.DataFrame | None = None,
) -> RegimeAssessment:
    """Classify market regime using only data available up to *screen_date*.

    Args:
        screen_date: As-of date — no data after this is used.
        spy_df: SPY OHLCV already sliced to ``<= screen_date``.
        qqq_df: QQQ OHLCV already sliced to ``<= screen_date``.
        price_data: Optional ticker -> OHLCV (sliced) for breadth calculation.
        vix_series: Optional DataFrame with columns ``date`` and ``close``
            containing historical VIX levels.  If *None*, the regime is
            classified from SPY/QQQ trends alone.

    Returns:
        RegimeAssessment for the given date.
    """
    # Extract VIX level for screen_date if available
    vix: float | None = None
    if vix_series is not None and not vix_series.empty:
        vix_row = vix_series[vix_series["date"] <= pd.Timestamp(screen_date)]
        if not vix_row.empty:
            vix = float(vix_row.iloc[-1]["close"])

    # Compute breadth if we have enough tickers
    breadth: float | None = None
    if price_data and len(price_data) >= 20:
        breadth = compute_breadth_score(price_data)

    assessment = classify_regime(
        spy_df=spy_df,
        qqq_df=qqq_df,
        vix=vix,
        breadth_score=breadth,
    )

    logger.debug(
        "Regime %s on %s (confidence=%.2f, vix=%s, breadth=%s)",
        assessment.regime.value,
        screen_date,
        assessment.confidence,
        vix,
        breadth,
    )
    return assessment
