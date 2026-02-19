"""KooCore-D adapter for multi-engine backtest.

Reimplements KooCore-D's weekly and pro30 scoring logic inline using
pre-fetched OHLCV data.  The original KooCore-D modules use relative
imports (``from .technicals import ...``) which conflict with MAS's own
``src`` package when added via ``sys.path``, so we port the scoring
rubrics directly.

Scoring rubrics ported from:
  - ``KooCore-D/src/core/scoring.py:compute_technical_score_weekly``
  - ``KooCore-D/src/pipelines/pro30_screening.py:screen_universe_30d``
"""

from __future__ import annotations

import logging
import math
from datetime import date

import numpy as np
import pandas as pd

from src.backtest.multi_engine.adapters.base import EngineAdapter, NormalizedPick

logger = logging.getLogger(__name__)

# Regime-based target scaling: reduce targets in adverse regimes
_REGIME_TARGET_SCALE = {
    "bull": 1.0,
    "bear": 0.6,
    "choppy": 0.75,
    "stress": 0.6,
    "chop": 0.75,
}


class KooCoreAdapter(EngineAdapter):
    """Adapter reimplementing KooCore-D weekly + pro30 scoring."""

    def __init__(self, top_n: int = 15):
        self._top_n = top_n

    @property
    def engine_name(self) -> str:
        return "koocore_d"

    def required_lookback_days(self) -> int:
        return 400

    async def generate_picks(
        self,
        screen_date: date,
        price_data: dict[str, pd.DataFrame],
        spy_df: pd.DataFrame,
        qqq_df: pd.DataFrame,
    ) -> list[NormalizedPick]:
        picks: list[NormalizedPick] = []

        # Classify regime from SPY/QQQ data
        regime_label = "bull"  # default
        try:
            from src.features.regime import classify_regime
            regime_assessment = classify_regime(spy_df=spy_df, qqq_df=qqq_df)
            regime_label = regime_assessment.regime.value
            logger.debug("KooCore-D adapter regime: %s", regime_label)
        except Exception as e:
            logger.debug("KooCore-D regime classification failed, defaulting to bull: %s", e)

        # Run weekly scoring on the provided price data
        weekly_picks = self._run_weekly_scoring(screen_date, price_data, regime_label)
        picks.extend(weekly_picks)

        # Run pro30 scoring on the provided price data
        pro30_picks = self._run_pro30_scoring(screen_date, price_data, regime_label)
        picks.extend(pro30_picks)

        # Deduplicate by ticker (keep higher confidence)
        seen: dict[str, NormalizedPick] = {}
        for p in picks:
            if p.ticker not in seen or p.confidence > seen[p.ticker].confidence:
                seen[p.ticker] = p
        picks = list(seen.values())

        # Sort by confidence and take top-N
        picks.sort(key=lambda p: p.confidence, reverse=True)
        picks = picks[: self._top_n]

        logger.info(
            "KooCore-D adapter: %d picks on %s",
            len(picks), screen_date,
        )
        return picks

    def _run_weekly_scoring(
        self,
        screen_date: date,
        price_data: dict[str, pd.DataFrame],
        regime_label: str = "bull",
    ) -> list[NormalizedPick]:
        """KooCore-D weekly technical scoring (0-10 scale).

        Rubric (from ``KooCore-D/src/core/scoring.py``):
          +2.0  within 5% of 52-week high
          +2.5  volume ratio (3d/20d) >= 2.0x  (+2.0 if >= 1.5x)
          +2.5  RSI 55-65 sweet spot  (+1.5 if 50-70,  -0.5 if > 70)
          +2.0  price > MA10 AND MA20 AND MA50
          +2.0  5-day realized vol >= 20% annualized
        """
        picks: list[NormalizedPick] = []

        for ticker, df in price_data.items():
            if ticker in ("SPY", "QQQ") or df is None or df.empty or len(df) < 60:
                continue

            try:
                feat = _compute_koocore_features(df)
                if feat is None:
                    continue

                close = feat["close"]
                if close <= 0:
                    continue

                # --- Weekly score (max 11.0, but typically 0-10) ---
                score = 0.0

                # 52W high proximity: +2.0 if within 5%
                if _valid(feat["dist_52w_high_pct"]) and feat["dist_52w_high_pct"] >= -5.0:
                    score += 2.0

                # Volume ratio: 3d avg / 20d avg
                vol_ratio = feat["vol_ratio_3d_20d"]
                if _valid(vol_ratio):
                    if vol_ratio >= 2.0:
                        score += 2.5
                    elif vol_ratio >= 1.5:
                        score += 2.0
                    elif vol_ratio >= 1.2:
                        score += 1.0

                # RSI sweet spot
                rsi = feat["rsi"]
                if _valid(rsi):
                    if 55 <= rsi <= 65:
                        score += 2.5
                    elif 50 <= rsi <= 70:
                        score += 1.5
                    elif rsi > 70:
                        score -= 0.5

                # MA alignment: price > MA10 AND MA20 AND MA50
                if feat["above_ma10"] and feat["above_ma20"] and feat["above_ma50"]:
                    score += 2.0

                # Realized volatility: 5-day annualized >= 20%
                if _valid(feat["realized_vol_5d"]) and feat["realized_vol_5d"] >= 20.0:
                    score += 2.0

                # Min threshold
                if score < 4.0:
                    continue

                # Liquidity filter
                adv = feat.get("avg_dollar_volume_20d", 0)
                if _valid(adv) and adv < 1_000_000:
                    continue

                atr_pct = feat["atr_pct"]
                if not _valid(atr_pct) or atr_pct <= 0:
                    atr_pct = 2.0

                target_scale = _REGIME_TARGET_SCALE.get(regime_label, 1.0)

                picks.append(NormalizedPick(
                    ticker=ticker,
                    engine_name=self.engine_name,
                    strategy="weekly_momentum",
                    entry_price=round(close, 2),
                    stop_loss=round(close * (1 - 1.5 * atr_pct / 100), 2),
                    target_price=round(close * (1 + 2.0 * atr_pct / 100 * target_scale), 2),
                    confidence=min(100.0, score * 10),  # 0-10 → 0-100
                    holding_period_days=7,
                    direction="LONG",
                    raw_score=score,
                    metadata={
                        "source": "weekly",
                        "atr_pct": round(atr_pct, 2),
                        "rsi": round(rsi, 1) if _valid(rsi) else None,
                        "vol_ratio": round(vol_ratio, 2) if _valid(vol_ratio) else None,
                        "dist_52w_high": round(feat["dist_52w_high_pct"], 1) if _valid(feat["dist_52w_high_pct"]) else None,
                    },
                ))
            except Exception as e:
                logger.debug("KooCore-D weekly scoring failed for %s: %s", ticker, e)

        return picks

    def _run_pro30_scoring(
        self,
        screen_date: date,
        price_data: dict[str, pd.DataFrame],
        regime_label: str = "bull",
    ) -> list[NormalizedPick]:
        """KooCore-D pro30 screening (tape + structure score).

        From ``KooCore-D/src/pipelines/pro30_screening.py``:
          Tape score  = RVOL * 2.0 + ATR% * 1.4
          Structure   = RSI 40-70 → +2.0,  close > SMA50 → +1.5
          Total       = tape + structure
          Gate        = RVOL >= 1.5 AND ATR% >= 2.0 AND total >= 6.0
        """
        picks: list[NormalizedPick] = []

        for ticker, df in price_data.items():
            if ticker in ("SPY", "QQQ") or df is None or df.empty or len(df) < 60:
                continue

            try:
                feat = _compute_koocore_features(df)
                if feat is None:
                    continue

                close = feat["close"]
                if close <= 0:
                    continue

                # RVOL (using vol_ratio as proxy — last day volume / 20d avg)
                rvol = feat["rvol"]
                atr_pct = feat["atr_pct"]

                if not _valid(rvol) or rvol < 1.5:
                    continue
                if not _valid(atr_pct) or atr_pct < 2.0:
                    continue

                # Tape score
                tape_score = rvol * 2.0 + atr_pct * 1.4

                # Structure score
                rsi = feat["rsi"]
                structure_score = 0.0
                if _valid(rsi) and 40 <= rsi <= 70:
                    structure_score += 2.0
                if feat["above_ma50"]:
                    structure_score += 1.5

                total_score = tape_score + structure_score
                if total_score < 6.0:
                    continue

                target_scale = _REGIME_TARGET_SCALE.get(regime_label, 1.0)

                picks.append(NormalizedPick(
                    ticker=ticker,
                    engine_name=self.engine_name,
                    strategy="pro30_momentum",
                    entry_price=round(close, 2),
                    stop_loss=round(close * (1 - 1.5 * atr_pct / 100), 2),
                    target_price=round(close * (1 + 2.5 * atr_pct / 100 * target_scale), 2),
                    confidence=min(100.0, total_score * 8),  # ~0-12 → 0-100
                    holding_period_days=10,
                    direction="LONG",
                    raw_score=total_score,
                    metadata={
                        "source": "pro30",
                        "rvol": round(rvol, 2),
                        "atr_pct": round(atr_pct, 2),
                        "tape_score": round(tape_score, 2),
                        "structure_score": round(structure_score, 1),
                    },
                ))
            except Exception as e:
                logger.debug("KooCore-D pro30 scoring failed for %s: %s", ticker, e)

        return picks


# ── Feature computation (ported from KooCore-D core/technicals.py) ────────


def _compute_koocore_features(df: pd.DataFrame) -> dict | None:
    """Compute the features used by KooCore-D scoring from raw OHLCV.

    Returns a dict with keys matching KooCore-D's column names, or None
    if insufficient data.
    """
    if df.empty or len(df) < 20:
        return None

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    latest_close = float(close.iloc[-1])
    if latest_close <= 0:
        return None

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss_s = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss_s.replace(0, np.nan)
    rsi_series = 100 - 100 / (1 + rs)
    rsi = _safe_float(rsi_series.iloc[-1])

    # Moving averages
    ma10 = _safe_float(close.rolling(10).mean().iloc[-1])
    ma20 = _safe_float(close.rolling(20).mean().iloc[-1])
    ma50 = _safe_float(close.rolling(50).mean().iloc[-1]) if len(df) >= 50 else None

    above_ma10 = ma10 is not None and latest_close > ma10
    above_ma20 = ma20 is not None and latest_close > ma20
    above_ma50 = ma50 is not None and latest_close > ma50

    # 52-week high
    high_52w = float(high.tail(252).max()) if len(df) >= 252 else float(high.max())
    dist_52w_high_pct = (latest_close / high_52w - 1) * 100 if high_52w > 0 else None

    # ATR(14)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_14 = _safe_float(tr.rolling(14).mean().iloc[-1])
    atr_pct = (atr_14 / latest_close * 100) if atr_14 and latest_close > 0 else None

    # Volume metrics
    avg_vol_20d = _safe_float(volume.rolling(20).mean().iloc[-1])
    avg_vol_3d = _safe_float(volume.tail(3).mean())
    vol_ratio_3d_20d = (avg_vol_3d / avg_vol_20d) if avg_vol_3d and avg_vol_20d and avg_vol_20d > 0 else None

    # RVOL (last bar volume / 20d avg)
    last_vol = float(volume.iloc[-1])
    rvol = (last_vol / avg_vol_20d) if avg_vol_20d and avg_vol_20d > 0 else None

    # Average dollar volume
    avg_dollar_volume_20d = (avg_vol_20d * latest_close) if avg_vol_20d else None

    # 5-day realized volatility (annualized)
    if len(close) >= 6:
        log_returns = np.log(close / close.shift(1)).dropna().tail(5)
        if len(log_returns) >= 3:
            realized_vol_5d = float(log_returns.std() * np.sqrt(252) * 100)
        else:
            realized_vol_5d = None
    else:
        realized_vol_5d = None

    return {
        "close": latest_close,
        "rsi": rsi,
        "ma10": ma10,
        "ma20": ma20,
        "ma50": ma50,
        "above_ma10": above_ma10,
        "above_ma20": above_ma20,
        "above_ma50": above_ma50,
        "dist_52w_high_pct": dist_52w_high_pct,
        "atr": atr_14,
        "atr_pct": atr_pct,
        "vol_ratio_3d_20d": vol_ratio_3d_20d,
        "rvol": rvol,
        "avg_dollar_volume_20d": avg_dollar_volume_20d,
        "realized_vol_5d": realized_vol_5d,
    }


def _valid(x) -> bool:
    if x is None:
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _safe_float(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None
