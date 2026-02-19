"""MAS (Multi-Agentic Screener) adapter for multi-engine backtest.

Wraps the existing breakout, mean-reversion, and catalyst scoring functions
from ``src/signals/`` into the :class:`NormalizedPick` interface.

All data is sliced to ``<= screen_date`` (point-in-time).  Catalyst scoring
skips ``days_to_earnings`` for historical dates.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from src.backtest.multi_engine.adapters.base import EngineAdapter, NormalizedPick
from src.features.regime import classify_regime, compute_breadth_score
from src.features.technical import (
    compute_all_technical_features,
    compute_rsi2_features,
    latest_features,
)
from src.signals.breakout import score_breakout
from src.signals.mean_reversion import score_mean_reversion
from src.signals.catalyst import score_catalyst
from src.signals.ranker import rank_candidates, deduplicate_signals

logger = logging.getLogger(__name__)


class MASAdapter(EngineAdapter):
    """Adapter wrapping the MAS quant pipeline (breakout + mean_rev + catalyst)."""

    def __init__(self, top_n: int = 10):
        self._top_n = top_n

    @property
    def engine_name(self) -> str:
        return "mas"

    def required_lookback_days(self) -> int:
        return 300

    async def generate_picks(
        self,
        screen_date: date,
        price_data: dict[str, pd.DataFrame],
        spy_df: pd.DataFrame,
        qqq_df: pd.DataFrame,
    ) -> list[NormalizedPick]:
        # Classify regime from SPY/QQQ
        breadth = compute_breadth_score(price_data) if len(price_data) >= 20 else None
        regime_assessment = classify_regime(
            spy_df=spy_df,
            qqq_df=qqq_df,
            vix=None,  # not available historically without live API
            breadth_score=breadth,
        )
        regime = regime_assessment.regime

        all_signals = []
        features_by_ticker: dict[str, dict] = {}

        for ticker, df in price_data.items():
            if ticker in ("SPY", "QQQ") or df is None or df.empty or len(df) < 50:
                continue

            # Compute features
            df_feat = compute_all_technical_features(df.copy())
            df_feat = compute_rsi2_features(df_feat)
            feat = latest_features(df_feat)
            feat["ticker"] = ticker
            features_by_ticker[ticker] = feat

            # Breakout
            breakout = score_breakout(ticker, df_feat, feat)
            if breakout:
                all_signals.append(breakout)

            # Mean reversion
            mean_rev = score_mean_reversion(ticker, df_feat, feat)
            if mean_rev:
                all_signals.append(mean_rev)

            # Catalyst — skip days_to_earnings for historical backtest
            catalyst = score_catalyst(
                ticker,
                feat,
                fundamental_data={},
                days_to_earnings=None,
                sentiment=None,
            )
            if catalyst:
                all_signals.append(catalyst)

        if not all_signals:
            return []

        all_signals = deduplicate_signals(all_signals)
        ranked = rank_candidates(
            all_signals,
            regime=regime,
            features_by_ticker=features_by_ticker,
            top_n=self._top_n,
        )

        # Convert to NormalizedPick with linear confidence rescaling
        # Raw scores cluster in 50-100 range → rescale to spread across 30-90
        picks: list[NormalizedPick] = []
        for c in ranked:
            raw_conf = min(100.0, c.regime_adjusted_score)
            # Linear rescale: [50, 100] → [30, 90], clamped to [25, 95]
            rescaled = 30.0 + (raw_conf - 50.0) * (60.0 / 50.0)
            confidence = max(25.0, min(95.0, rescaled))

            picks.append(NormalizedPick(
                ticker=c.ticker,
                engine_name=self.engine_name,
                strategy=c.signal_model,
                entry_price=c.entry_price,
                stop_loss=c.stop_loss,
                target_price=c.target_1,
                confidence=round(confidence, 2),
                holding_period_days=c.holding_period,
                direction=c.direction,
                raw_score=c.raw_score,
                metadata={
                    "regime": regime.value,
                    "components": c.components,
                    "target_2": c.target_2,
                    "raw_confidence": raw_conf,
                },
            ))

        logger.info(
            "MAS adapter: %d signals → %d ranked → %d picks (regime=%s) on %s",
            len(all_signals), len(ranked), len(picks), regime.value, screen_date,
        )
        return picks
