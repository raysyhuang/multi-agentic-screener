"""Gemini STST adapter for multi-engine backtest.

Wraps the Gemini STST momentum screener and mean-reversion screener into the
:class:`NormalizedPick` interface.  All live enrichment (earnings blacklist,
news, options sentiment) is skipped — only pure technical/quantitative
filters run.

Price data is passed in from the orchestrator's pre-fetched cache.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.multi_engine.adapters.base import EngineAdapter, NormalizedPick

logger = logging.getLogger(__name__)

# Resolve Gemini STST root — configurable via env var or default sibling path
GEMINI_ROOT = Path(
    os.environ.get(
        "GEMINI_ROOT",
        Path(__file__).resolve().parents[4] / ".." / "Gemini STST",
    )
).resolve()


def _ensure_gemini_importable() -> bool:
    """Add Gemini STST to sys.path if not already present."""
    root_str = str(GEMINI_ROOT)
    if not GEMINI_ROOT.exists():
        logger.warning("Gemini STST root not found at %s", GEMINI_ROOT)
        return False
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return True


class GeminiAdapter(EngineAdapter):
    """Adapter for the Gemini STST momentum + reversion screeners."""

    def __init__(self):
        self._importable: bool | None = None

    @property
    def engine_name(self) -> str:
        return "gemini_stst"

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

        # Run momentum screener filters on provided price data
        momentum_picks = self._run_momentum_screen(screen_date, price_data, spy_df)
        picks.extend(momentum_picks)

        # Run reversion screener filters on provided price data
        reversion_picks = self._run_reversion_screen(screen_date, price_data)
        picks.extend(reversion_picks)

        # Deduplicate by ticker (keep higher confidence)
        seen: dict[str, NormalizedPick] = {}
        for p in picks:
            if p.ticker not in seen or p.confidence > seen[p.ticker].confidence:
                seen[p.ticker] = p
        picks = list(seen.values())

        logger.info(
            "Gemini STST adapter: %d picks on %s",
            len(picks), screen_date,
        )
        return picks

    def _run_momentum_screen(
        self,
        screen_date: date,
        price_data: dict[str, pd.DataFrame],
        spy_df: pd.DataFrame,
    ) -> list[NormalizedPick]:
        """Apply the Gemini STST 9-filter momentum chain on pre-fetched data.

        Filters (from screener.py):
          1. Price >= $5
          2. ADV >= 500K
          3. ATR% >= 2.0
          4. RVOL >= 1.2
          5. Close > SMA(20) (uptrend)
          6. Green candle (close > open)
          7. RSI(14) 40-75
          8. Close > SMA(50)
          9. 5-day return > 0
        """
        picks: list[NormalizedPick] = []

        for ticker, df in price_data.items():
            if ticker in ("SPY", "QQQ") or df is None or df.empty or len(df) < 50:
                continue

            try:
                latest = self._compute_latest_row(df)
                if latest is None:
                    continue

                close = latest["close"]
                open_ = latest["open"]

                # Filter chain
                if close < 5.0:
                    continue
                adv = latest.get("adv_20", 0)
                if adv < 500_000:
                    continue
                atr_pct = latest.get("atr_pct", 0)
                if atr_pct < 2.0:
                    continue
                rvol = latest.get("rvol", 0)
                if rvol < 1.2:
                    continue
                sma_20 = latest.get("sma_20")
                if not _valid(sma_20) or close <= sma_20:
                    continue
                if close <= open_:  # not a green candle
                    continue
                rsi_14 = latest.get("rsi_14", 0)
                if not (40 <= rsi_14 <= 75):
                    continue
                sma_50 = latest.get("sma_50")
                if not _valid(sma_50) or close <= sma_50:
                    continue
                ret_5d = latest.get("ret_5d", 0)
                if ret_5d <= 0:
                    continue

                # Quality score (6-factor model from screener.py)
                quality = self._compute_momentum_quality(latest)
                if quality < 30:
                    continue

                atr = latest.get("atr_14", close * 0.02)
                if not _valid(atr) or atr <= 0:
                    atr = close * 0.02

                picks.append(NormalizedPick(
                    ticker=ticker,
                    engine_name=self.engine_name,
                    strategy="momentum",
                    entry_price=round(close, 2),
                    stop_loss=round(close - 2.0 * atr, 2),
                    target_price=round(close + 2.5 * atr, 2),
                    confidence=quality,
                    holding_period_days=7,
                    direction="LONG",
                    raw_score=quality,
                    metadata={
                        "rvol": round(rvol, 2),
                        "atr_pct": round(atr_pct, 2),
                        "rsi_14": round(rsi_14, 2),
                        "ret_5d": round(ret_5d, 4),
                    },
                ))
            except Exception as e:
                logger.debug("Gemini momentum screen failed for %s: %s", ticker, e)

        return picks

    def _run_reversion_screen(
        self,
        screen_date: date,
        price_data: dict[str, pd.DataFrame],
    ) -> list[NormalizedPick]:
        """Apply the Gemini STST mean-reversion filter chain.

        Filters (from mean_reversion.py):
          1. Price > $5
          2. ADV > 1.5M
          3. RSI(2) < 10
          4. 3-day drawdown >= 15%
          5. Close > SMA(200) (long-term uptrend intact)
        """
        picks: list[NormalizedPick] = []

        for ticker, df in price_data.items():
            if ticker in ("SPY", "QQQ") or df is None or df.empty or len(df) < 200:
                continue

            try:
                latest = self._compute_latest_row(df)
                if latest is None:
                    continue

                close = latest["close"]
                if close < 5.0:
                    continue
                adv = latest.get("adv_20", 0)
                if adv < 1_500_000:
                    continue

                # RSI(2)
                rsi_2 = latest.get("rsi_2")
                if not _valid(rsi_2) or rsi_2 >= 10:
                    continue

                # 3-day drawdown
                drawdown_3d = latest.get("drawdown_3d", 0)
                if abs(drawdown_3d) < 15:
                    continue

                # Close > SMA(200)
                sma_200 = latest.get("sma_200")
                if not _valid(sma_200) or close <= sma_200:
                    continue

                # Quality score
                quality = self._compute_reversion_quality(latest)
                if quality < 30:
                    continue

                atr = latest.get("atr_14", close * 0.02)
                if not _valid(atr) or atr <= 0:
                    atr = close * 0.02

                # Target = revert to 5-day SMA with 1x ATR floor
                sma_5 = latest.get("sma_5", close * 1.03)
                target = max(sma_5, close + 1.0 * atr) if _valid(sma_5) else close + 1.0 * atr

                picks.append(NormalizedPick(
                    ticker=ticker,
                    engine_name=self.engine_name,
                    strategy="reversion",
                    entry_price=round(close, 2),
                    stop_loss=round(close - 1.0 * atr, 2),
                    target_price=round(target, 2),
                    confidence=quality,
                    holding_period_days=3,
                    direction="LONG",
                    raw_score=quality,
                    metadata={
                        "rsi_2": round(rsi_2, 2),
                        "drawdown_3d": round(drawdown_3d, 2),
                    },
                ))
            except Exception as e:
                logger.debug("Gemini reversion screen failed for %s: %s", ticker, e)

        return picks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_latest_row(df: pd.DataFrame) -> dict | None:
        """Compute indicators and extract the latest row as a dict."""
        if df.empty or len(df) < 20:
            return None

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)
        open_ = df["open"].astype(float)

        # Basic indicators
        sma_5 = close.rolling(5).mean()
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean() if len(df) >= 50 else pd.Series(dtype=float)
        sma_200 = close.rolling(200).mean() if len(df) >= 200 else pd.Series(dtype=float)
        adv_20 = volume.rolling(20).mean()

        # ATR(14)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()

        # RSI(14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_14 = 100 - 100 / (1 + rs)

        # RSI(2)
        gain2 = delta.where(delta > 0, 0).rolling(2).mean()
        loss2 = (-delta.where(delta < 0, 0)).rolling(2).mean()
        rs2 = gain2 / loss2.replace(0, np.nan)
        rsi_2 = 100 - 100 / (1 + rs2)

        # RVOL
        rvol = volume / adv_20.replace(0, np.nan)

        # 5-day return
        ret_5d = close.pct_change(5)

        # 3-day drawdown
        drawdown_3d = (close / close.shift(3) - 1) * 100

        # 52-week high proximity
        high_52w = high.rolling(252).max() if len(df) >= 252 else high.rolling(len(df)).max()
        pct_from_52w_high = (close / high_52w - 1) * 100

        latest_idx = df.index[-1]
        latest_close = float(close.iloc[-1])
        atr_val = float(atr_14.iloc[-1]) if not pd.isna(atr_14.iloc[-1]) else latest_close * 0.02

        result = {
            "close": latest_close,
            "open": float(open_.iloc[-1]),
            "high": float(high.iloc[-1]),
            "low": float(low.iloc[-1]),
            "volume": float(volume.iloc[-1]),
            "sma_5": _safe_float(sma_5.iloc[-1]),
            "sma_20": _safe_float(sma_20.iloc[-1]),
            "sma_50": _safe_float(sma_50.iloc[-1]) if len(sma_50) > 0 else None,
            "sma_200": _safe_float(sma_200.iloc[-1]) if len(sma_200) > 0 else None,
            "adv_20": _safe_float(adv_20.iloc[-1]),
            "atr_14": atr_val,
            "atr_pct": atr_val / latest_close * 100 if latest_close > 0 else 0,
            "rsi_14": _safe_float(rsi_14.iloc[-1]),
            "rsi_2": _safe_float(rsi_2.iloc[-1]),
            "rvol": _safe_float(rvol.iloc[-1]),
            "ret_5d": _safe_float(ret_5d.iloc[-1]),
            "drawdown_3d": _safe_float(drawdown_3d.iloc[-1]),
            "pct_from_52w_high": _safe_float(pct_from_52w_high.iloc[-1]),
        }
        return result

    @staticmethod
    def _compute_momentum_quality(latest: dict) -> float:
        """6-factor quality model (from Gemini STST screener.py).

        Factors:
          1. RVOL strength (0-20)
          2. 52-week high proximity (0-20)
          3. RSI sweet spot (0-15)
          4. SMA-50 trend (0-15)
          5. Candle strength (0-15)
          6. Volume confirmation (0-15)
        """
        score = 0.0

        # 1. RVOL strength
        rvol = latest.get("rvol", 0)
        if _valid(rvol):
            if rvol >= 3.0:
                score += 20
            elif rvol >= 2.0:
                score += 15
            elif rvol >= 1.5:
                score += 10
            else:
                score += 5

        # 2. 52-week high proximity
        pct_high = latest.get("pct_from_52w_high", -100)
        if _valid(pct_high):
            if pct_high >= -2:
                score += 20
            elif pct_high >= -5:
                score += 15
            elif pct_high >= -10:
                score += 10
            else:
                score += 5

        # 3. RSI sweet spot (55-70 is ideal)
        rsi = latest.get("rsi_14", 50)
        if _valid(rsi):
            if 55 <= rsi <= 70:
                score += 15
            elif 50 <= rsi <= 75:
                score += 10
            else:
                score += 5

        # 4. SMA-50 trend alignment
        close = latest.get("close", 0)
        sma_50 = latest.get("sma_50")
        if _valid(sma_50) and _valid(close) and close > 0:
            pct_above = (close - sma_50) / sma_50 * 100
            if pct_above > 5:
                score += 15
            elif pct_above > 0:
                score += 10
            else:
                score += 3

        # 5. Candle strength (close near high)
        high = latest.get("high", 0)
        low = latest.get("low", 0)
        if high > low:
            candle_pos = (close - low) / (high - low)
            if candle_pos > 0.8:
                score += 15
            elif candle_pos > 0.6:
                score += 10
            else:
                score += 5

        # 6. Volume confirmation
        volume = latest.get("volume", 0)
        adv = latest.get("adv_20", 1)
        if adv > 0:
            vol_ratio = volume / adv
            if vol_ratio > 2.0:
                score += 15
            elif vol_ratio > 1.5:
                score += 10
            else:
                score += 5

        return min(100.0, score)

    @staticmethod
    def _compute_reversion_quality(latest: dict) -> float:
        """Reversion quality model (from Gemini STST mean_reversion.py).

        Factors (weighted):
          - RSI depth (35%): deeper oversold = higher score
          - Drawdown depth (25%): larger pullback = higher score
          - SMA-200 margin (20%): further above = safer
          - RSI stretch (20%): distance below 10
        """
        score = 0.0

        # RSI depth (35%)
        rsi_2 = latest.get("rsi_2", 50)
        if _valid(rsi_2):
            if rsi_2 <= 2:
                score += 35
            elif rsi_2 <= 5:
                score += 30
            elif rsi_2 <= 8:
                score += 20
            else:
                score += 10

        # Drawdown depth (25%)
        dd = abs(latest.get("drawdown_3d", 0))
        if dd >= 25:
            score += 25
        elif dd >= 20:
            score += 20
        elif dd >= 15:
            score += 15
        else:
            score += 5

        # SMA-200 margin (20%)
        close = latest.get("close", 0)
        sma_200 = latest.get("sma_200")
        if _valid(sma_200) and sma_200 > 0:
            margin = (close - sma_200) / sma_200 * 100
            if margin > 20:
                score += 20
            elif margin > 10:
                score += 15
            elif margin > 0:
                score += 10
            else:
                score += 0

        # Stretch (20%)
        if _valid(rsi_2):
            stretch = max(0, 10 - rsi_2)
            score += min(20, stretch * 2)

        return min(100.0, score)


def _valid(x) -> bool:
    if x is None:
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _safe_float(x) -> float | None:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None
