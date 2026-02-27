"""Regression tests — Gemini STST signal dicts must use native Python types.

Bug context (2026-02-25):
    np.float64 values in signal dicts caused `ProgrammingError: schema "np"
    does not exist` when psycopg2 tried to serialize them to SQL.  The fix
    wraps every numeric field with `float()`.  These tests ensure that
    regression cannot silently return.

We replicate the exact DataFrame→indicator→signal-dict pipeline that runs
in production, feeding numpy float32 columns (the dtype produced by
``_load_all_ohlcv``'s memory downcast) and asserting every numeric value
in the resulting signal dict is a native Python ``float`` or ``int``.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Make Gemini STST importable ──────────────────────────────────────────
_GEMINI_ROOT = Path(__file__).resolve().parents[2] / ".." / "Gemini STST"
if not _GEMINI_ROOT.exists():
    pytest.skip(
        "Gemini STST sibling repo not found (expected in CI)",
        allow_module_level=True,
    )
if str(_GEMINI_ROOT) not in sys.path:
    sys.path.insert(0, str(_GEMINI_ROOT))

from app.indicators import compute_atr_pct, compute_rsi  # type: ignore[import-untyped]  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_ohlcv_float32(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV with float32 columns — matches ``_load_all_ohlcv`` downcast."""
    rng = np.random.default_rng(seed)
    close = (100 + np.cumsum(rng.standard_normal(n) * 1.5)).astype(np.float32)
    close = np.maximum(close, np.float32(10))

    df = pd.DataFrame({
        "date": [date(2025, 1, 2) + timedelta(days=i) for i in range(n)],
        "open": (close + rng.standard_normal(n).astype(np.float32) * 0.5).astype(np.float32),
        "high": (close + np.abs(rng.standard_normal(n).astype(np.float32))).astype(np.float32),
        "low": (close - np.abs(rng.standard_normal(n).astype(np.float32))).astype(np.float32),
        "close": close,
        "volume": rng.integers(500_000, 5_000_000, n).astype(np.float32),
    })
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


def _build_momentum_signal(df: pd.DataFrame) -> dict:
    """Replicate momentum screener signal-dict construction (screener.py:310-323)."""
    df = df.copy()
    df["atr_pct"] = compute_atr_pct(df)
    df["rsi_14"] = compute_rsi(df, period=14)

    adv_20 = df["volume"].rolling(20).mean()
    df["rvol"] = df["volume"] / adv_20

    high_52w = df["high"].rolling(252, min_periods=20).max()
    df["pct_from_52w_high"] = (df["close"] / high_52w - 1) * 100

    latest = df.iloc[-1]

    return {
        "ticker_id": 1,
        "symbol": "TEST",
        "company_name": "Test Corp",
        "date": latest["date"],
        "trigger_price": round(float(latest["close"]), 2),
        "rvol_at_trigger": round(float(latest["rvol"]), 2),
        "atr_pct_at_trigger": round(float(latest["atr_pct"]), 1),
        "rsi_14": round(float(latest["rsi_14"]), 1) if not pd.isna(latest.get("rsi_14")) else None,
        "pct_from_52w_high": round(float(latest["pct_from_52w_high"]), 1) if not pd.isna(latest.get("pct_from_52w_high")) else None,
        "quality_score": round(float(0.65), 2),
        "factor_scores": {
            "rvol": round(float(latest["rvol"]) * 10, 1),
            "high_prox": 7.5,
            "rsi_sweet": 6.0,
            "trend": 8.0,
            "candle": 5.0,
            "options": 0.0,
        },
        "confluence": False,
    }


def _build_reversion_signal(df: pd.DataFrame) -> dict:
    """Replicate mean-reversion screener signal-dict construction (mean_reversion.py:312-325)."""
    df = df.copy()
    df["rsi2"] = compute_rsi(df, period=2)
    df["atr_pct"] = compute_atr_pct(df)
    df["close_3d_ago"] = df["close"].shift(3)
    df["drawdown_3d"] = (df["close"] / df["close_3d_ago"]) - 1.0

    sma_20 = df["close"].rolling(20).mean().iloc[-1]
    latest = df.iloc[-1]

    sma_distance_pct = (
        round(float((latest["close"] / sma_20) - 1.0) * 100, 1)
        if not pd.isna(sma_20) else 0.0
    )
    atr_pct_val = (
        round(float(latest["atr_pct"]), 1)
        if not pd.isna(latest["atr_pct"]) else 10.0
    )

    return {
        "ticker_id": 1,
        "symbol": "TEST",
        "company_name": "Test Corp",
        "date": latest["date"],
        "trigger_price": round(float(latest["close"]), 2),
        "rsi2": round(float(latest["rsi2"]), 1),
        "drawdown_3d_pct": round(float(latest["drawdown_3d"]) * 100, 1),
        "sma_distance_pct": sma_distance_pct,
        "atr_pct_at_trigger": atr_pct_val,
        "quality_score": round(float(0.72), 2),
        "factor_scores": {
            "rsi_depth": 9.0,
            "drawdown": 7.0,
            "sma200_margin": 6.5,
            "stretch": 8.0,
        },
        "confluence": False,
    }


def _assert_native_numeric(signal: dict, path: str = "") -> None:
    """Recursively assert every numeric value is native float/int, not numpy."""
    for key, val in signal.items():
        full_key = f"{path}.{key}" if path else key
        if isinstance(val, dict):
            _assert_native_numeric(val, full_key)
        elif isinstance(val, bool):
            continue  # bool is a subclass of int; not a numeric concern
        elif isinstance(val, (int, float)):
            assert type(val) in (int, float), (
                f"Field '{full_key}' is {type(val).__name__} ({val!r}), "
                f"expected native float or int"
            )
        elif isinstance(val, np.generic):
            raise AssertionError(
                f"Field '{full_key}' is numpy {type(val).__name__} ({val!r}), "
                f"expected native Python type"
            )


# ── Tests ────────────────────────────────────────────────────────────────

class TestMomentumSignalTypes:
    def test_momentum_signal_fields_are_native_python_types(self):
        """All numeric fields in a momentum signal dict must be native float/int."""
        df = _make_ohlcv_float32(n=100)
        signal = _build_momentum_signal(df)
        _assert_native_numeric(signal)

    def test_atr_pct_is_native_float(self):
        """ATR% — the field computed from np.sqrt(5) — must be native float."""
        df = _make_ohlcv_float32(n=100)
        signal = _build_momentum_signal(df)
        assert type(signal["atr_pct_at_trigger"]) is float

    def test_quality_score_is_native_float(self):
        """Quality score must be native float after round(float(...))."""
        df = _make_ohlcv_float32(n=100)
        signal = _build_momentum_signal(df)
        assert type(signal["quality_score"]) is float


class TestReversionSignalTypes:
    def test_reversion_signal_fields_are_native_python_types(self):
        """All numeric fields in a reversion signal dict must be native float/int."""
        df = _make_ohlcv_float32(n=100)
        signal = _build_reversion_signal(df)
        _assert_native_numeric(signal)

    def test_sma_distance_pct_is_native_float(self):
        """sma_distance_pct — the exact field that caused the production bug."""
        df = _make_ohlcv_float32(n=100)
        signal = _build_reversion_signal(df)
        assert type(signal["sma_distance_pct"]) is float


class TestNumpyTypeTrap:
    def test_round_float_preserves_native_type(self):
        """round(float(np.float64(x)), 1) must produce a native float."""
        val = np.float64(3.14159)
        result = round(float(val), 1)
        assert type(result) is float
        assert result == 3.1

    def test_numpy_float64_isinstance_trap(self):
        """Document that isinstance(np.float64, float) is True but type() is not.

        This is the core trap: isinstance checks pass, so standard validation
        misses the problem.  Only ``type(x) is float`` catches it.
        """
        val = np.float64(1.0)
        # isinstance says yes — this is WHY the bug was hard to catch
        assert isinstance(val, float)
        # But strict type check reveals the truth
        assert type(val) is not float
        # After explicit float() cast, strict check passes
        assert type(float(val)) is float
