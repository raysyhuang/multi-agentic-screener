"""Tests for near-miss counterfactual resolution (Feature 1)."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import numpy as np
import pytest

from src.governance.divergence_ledger import (
    simulate_quant_counterfactual,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_price_df(
    entry_date: date,
    entry_price: float = 100.0,
    days: int = 15,
    outcome: str = "target",
    target: float = 110.0,
    stop: float = 95.0,
) -> pd.DataFrame:
    """Build a synthetic price DataFrame for counterfactual simulation."""
    dates = pd.bdate_range(start=pd.Timestamp(entry_date), periods=days)

    if outcome == "target":
        # Price ramps up to hit target
        closes = np.linspace(entry_price, target + 1, days)
    elif outcome == "stop":
        # Price drops to hit stop
        closes = np.linspace(entry_price, stop - 1, days)
    else:  # expiry
        # Price stays near entry, no exit conditions hit
        closes = np.full(days, entry_price + 1)

    return pd.DataFrame({
        "date": dates,
        "open": closes - 0.3,
        "high": closes + 0.5,
        "low": closes - 0.5,
        "close": closes,
        "volume": np.full(days, 1_000_000),
    })


def _make_near_miss_mock(**overrides) -> MagicMock:
    """Build a mock NearMiss with proper numeric attributes for stats iteration."""
    nm = MagicMock()
    nm.id = overrides.get("id", 1)
    nm.run_date = overrides.get("run_date", date.today() - timedelta(days=20))
    nm.ticker = overrides.get("ticker", "AAPL")
    nm.stage = overrides.get("stage", "debate")
    nm.debate_verdict = overrides.get("debate_verdict", "REJECT")
    nm.net_conviction = overrides.get("net_conviction", 45.0)
    nm.bull_conviction = overrides.get("bull_conviction", 55.0)
    nm.bear_conviction = overrides.get("bear_conviction", 65.0)
    nm.key_risk = overrides.get("key_risk", "high volatility")
    nm.signal_model = overrides.get("signal_model", "breakout")
    nm.regime = overrides.get("regime", "bull")
    nm.interpreter_confidence = overrides.get("interpreter_confidence", 60.0)
    nm.entry_price = overrides.get("entry_price", 100.0)
    nm.stop_loss = overrides.get("stop_loss", 95.0)
    nm.target_price = overrides.get("target_price", 110.0)
    nm.timeframe_days = overrides.get("timeframe_days", 10)
    nm.outcome_resolved = overrides.get("outcome_resolved", False)
    nm.counterfactual_return = overrides.get("counterfactual_return", None)
    nm.counterfactual_exit_reason = overrides.get("counterfactual_exit_reason", None)
    return nm


# ── Simulation Tests ──────────────────────────────────────────────────────


class TestSimulateCounterfactual:
    def test_target_hit(self):
        """Price reaches target → positive return."""
        entry_date = date.today() - timedelta(days=20)
        df = _make_price_df(entry_date=entry_date, outcome="target")
        result = simulate_quant_counterfactual(
            entry_price=100.0,
            stop_loss=95.0,
            target_1=110.0,
            holding_period=10,
            direction="LONG",
            entry_date=entry_date,
            aggregator=MagicMock(),
            ticker="AAPL",
            price_df=df,
        )
        assert result is not None
        assert result["quant_return"] > 0
        assert result["exit_reason"] == "target"

    def test_stop_hit(self):
        """Price drops to stop → negative return."""
        entry_date = date.today() - timedelta(days=20)
        df = _make_price_df(entry_date=entry_date, outcome="stop")
        result = simulate_quant_counterfactual(
            entry_price=100.0,
            stop_loss=95.0,
            target_1=110.0,
            holding_period=10,
            direction="LONG",
            entry_date=entry_date,
            aggregator=MagicMock(),
            ticker="AAPL",
            price_df=df,
        )
        assert result is not None
        assert result["quant_return"] < 0
        assert result["exit_reason"] == "stop"

    def test_expiry(self):
        """Price stays flat → expires with mark-to-market."""
        entry_date = date.today() - timedelta(days=20)
        df = _make_price_df(entry_date=entry_date, outcome="expiry", days=12)
        result = simulate_quant_counterfactual(
            entry_price=100.0,
            stop_loss=95.0,
            target_1=110.0,
            holding_period=10,
            direction="LONG",
            entry_date=entry_date,
            aggregator=MagicMock(),
            ticker="AAPL",
            price_df=df,
        )
        assert result is not None
        assert result["exit_reason"] == "expiry"

    def test_empty_df_returns_none(self):
        """Empty OHLCV → None."""
        entry_date = date.today() - timedelta(days=20)
        df = pd.DataFrame()
        result = simulate_quant_counterfactual(
            entry_price=100.0,
            stop_loss=95.0,
            target_1=110.0,
            holding_period=10,
            direction="LONG",
            entry_date=entry_date,
            aggregator=MagicMock(),
            ticker="AAPL",
            price_df=df,
        )
        assert result is None

    def test_none_df_returns_none(self):
        """No price_df → None."""
        entry_date = date.today() - timedelta(days=20)
        result = simulate_quant_counterfactual(
            entry_price=100.0,
            stop_loss=95.0,
            target_1=110.0,
            holding_period=10,
            direction="LONG",
            entry_date=entry_date,
            aggregator=MagicMock(),
            ticker="AAPL",
            price_df=None,
        )
        assert result is None

    def test_not_enough_trading_days(self):
        """Fewer trading days than holding period → None (not yet expired)."""
        entry_date = date.today() - timedelta(days=20)
        # Only 5 bars, holding period is 10 — no exit conditions hit, not expired
        df = _make_price_df(entry_date=entry_date, outcome="expiry", days=5)
        result = simulate_quant_counterfactual(
            entry_price=100.0,
            stop_loss=95.0,
            target_1=110.0,
            holding_period=10,
            direction="LONG",
            entry_date=entry_date,
            aggregator=MagicMock(),
            ticker="AAPL",
            price_df=df,
        )
        assert result is None


# ── Near-Miss Stats Integration ───────────────────────────────────────────


class TestNearMissStatsCounterfactual:
    @pytest.mark.asyncio
    async def test_stats_include_counterfactual_section(self):
        """get_near_miss_stats should include 'counterfactual' key when resolved data exists."""
        from src.output.performance import get_near_miss_stats

        resolved_nm = _make_near_miss_mock(
            outcome_resolved=True,
            counterfactual_return=5.2,
            counterfactual_exit_reason="target",
        )
        unresolved_nm = _make_near_miss_mock(
            id=2, ticker="MSFT",
            outcome_resolved=False,
            net_conviction=30.0,
        )

        mock_rows = [resolved_nm, unresolved_nm]

        with patch("src.output.performance.get_session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = mock_rows
            mock_session.execute.return_value = mock_result

            stats = await get_near_miss_stats(days=30)

        assert stats is not None
        assert "counterfactual" in stats
        cf = stats["counterfactual"]
        assert cf is not None
        assert cf["total_resolved"] == 1
        assert cf["win_rate"] == 1.0
        assert cf["avg_return"] == 5.2

    @pytest.mark.asyncio
    async def test_stats_counterfactual_none_when_no_resolved(self):
        """counterfactual should be None when no near-misses are resolved."""
        from src.output.performance import get_near_miss_stats

        unresolved = _make_near_miss_mock(outcome_resolved=False)

        with patch("src.output.performance.get_session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = [unresolved]
            mock_session.execute.return_value = mock_result

            stats = await get_near_miss_stats(days=30)

        assert stats is not None
        assert stats["counterfactual"] is None
