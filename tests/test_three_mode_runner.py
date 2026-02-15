"""Tests for the Three-Mode Runner (quant_only / hybrid / agentic_full)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from src.config import ExecutionMode, Settings


# ---------------------------------------------------------------------------
# Helpers — lightweight stand-ins for RankedCandidate
# ---------------------------------------------------------------------------

@dataclass
class FakeCandidate:
    ticker: str = "AAPL"
    signal_model: str = "breakout"
    direction: str = "LONG"
    entry_price: float = 150.0
    stop_loss: float = 145.0
    target_1: float = 160.0
    target_2: float | None = 165.0
    holding_period: int = 10
    raw_score: float = 0.82
    regime_adjusted_score: float = 0.78
    components: dict = field(default_factory=lambda: {"momentum": 0.9, "volume": 0.7})
    features: dict = field(default_factory=lambda: {"rsi_14": 55.0, "atr_pct": 2.1})


# ---------------------------------------------------------------------------
# PR1.1 — ExecutionMode enum validates correctly
# ---------------------------------------------------------------------------

class TestExecutionModeEnum:
    def test_valid_modes(self):
        assert ExecutionMode("quant_only") == ExecutionMode.QUANT_ONLY
        assert ExecutionMode("hybrid") == ExecutionMode.HYBRID
        assert ExecutionMode("agentic_full") == ExecutionMode.AGENTIC_FULL

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            ExecutionMode("invalid_mode")

    def test_default_setting(self):
        """Settings should default to agentic_full."""
        with patch.dict("os.environ", {}, clear=False):
            s = Settings(
                anthropic_api_key="test",
                openai_api_key="test",
                polygon_api_key="test",
                fmp_api_key="test",
                database_url="sqlite+aiosqlite:///test.db",
            )
            assert s.execution_mode == "agentic_full"


# ---------------------------------------------------------------------------
# PR1.2 — quant_only produces picks without any LLM calls
# ---------------------------------------------------------------------------

class TestQuantOnlyMode:
    def test_builds_results_without_llm(self):
        from src.main import _build_quant_only_result

        candidates = [FakeCandidate(ticker="AAPL"), FakeCandidate(ticker="MSFT")]
        regime_ctx = {"regime": "bull", "vix": 14.0}

        result = _build_quant_only_result(candidates, regime_ctx, max_picks=2)

        assert len(result.approved) == 2
        assert result.approved[0].ticker == "AAPL"
        assert result.approved[1].ticker == "MSFT"
        assert result.interpreted == 0  # No LLM interpretation
        assert result.debated == 0      # No LLM debate
        assert result.agent_logs == []  # Zero LLM calls

    def test_respects_max_picks(self):
        from src.main import _build_quant_only_result

        candidates = [FakeCandidate(ticker=f"T{i}") for i in range(5)]
        result = _build_quant_only_result(candidates, {"regime": "bull"}, max_picks=2)
        assert len(result.approved) == 2

    def test_deterministic_stubs_present(self):
        """Interpretation/debate/risk_gate fields must be populated (not None)."""
        from src.main import _build_quant_only_result

        candidates = [FakeCandidate()]
        result = _build_quant_only_result(candidates, {"regime": "bear"})

        pick = result.approved[0]
        assert pick.interpretation is not None
        assert pick.debate is not None
        assert pick.risk_gate is not None
        assert "quant_only" in pick.interpretation.thesis.lower() or "quant" in pick.interpretation.thesis.lower()
        assert pick.risk_gate.decision.value == "APPROVE"


# ---------------------------------------------------------------------------
# PR1.3 — hybrid calls interpreter only, no debate/risk gate
# ---------------------------------------------------------------------------

class TestHybridMode:
    @pytest.mark.asyncio
    async def test_hybrid_calls_interpreter_only(self):
        from src.main import _run_hybrid_pipeline

        fake_interp = MagicMock()
        fake_interp.confidence = 75.0
        fake_interp.thesis = "Strong breakout thesis"
        fake_interp.key_drivers = ["momentum", "volume"]
        fake_interp.suggested_stop = 145.0
        fake_interp.suggested_target = 160.0
        fake_interp.timeframe_days = 10
        fake_interp.risk_flags = []

        fake_retry = MagicMock()
        fake_retry.value = fake_interp
        fake_retry.attempt_count = 1
        fake_retry.total_tokens = 500
        fake_retry.total_cost_usd = 0.01

        with patch("src.agents.signal_interpreter.SignalInterpreterAgent") as MockInterp:
            instance = MockInterp.return_value
            instance.interpret = AsyncMock(return_value=fake_retry)
            instance.last_call_meta = {"model": "test", "tokens_in": 200, "tokens_out": 300}

            candidates = [FakeCandidate()]
            result = await _run_hybrid_pipeline(
                candidates, {"regime": "bull"}, run_id="test123",
            )

        # Interpreter was called
        assert instance.interpret.call_count >= 1
        # Results produced
        assert len(result.approved) >= 1
        # Debate was skipped
        assert result.debated == 0
        # Agent logs contain only interpreter entries
        agent_names = [log["agent"] for log in result.agent_logs]
        assert all(name == "signal_interpreter" for name in agent_names)


# ---------------------------------------------------------------------------
# PR1.4 — agentic_full remains unchanged
# ---------------------------------------------------------------------------

class TestAgenticFullMode:
    def test_agentic_full_enum_value(self):
        assert ExecutionMode.AGENTIC_FULL.value == "agentic_full"


# ---------------------------------------------------------------------------
# PR1.5 — GovernanceRecord includes execution_mode
# ---------------------------------------------------------------------------

class TestGovernanceExecutionMode:
    def test_governance_record_has_execution_mode(self):
        from src.governance.artifacts import GovernanceRecord

        record = GovernanceRecord(run_id="test", run_date="2025-01-01")
        assert record.execution_mode == "agentic_full"

        record.execution_mode = "quant_only"
        d = record.to_dict()
        assert d["execution_mode"] == "quant_only"

    def test_governance_context_sets_mode(self):
        from src.governance.artifacts import GovernanceContext

        gov = GovernanceContext(run_id="test", run_date="2025-01-01")
        gov.set_execution_mode("hybrid")
        assert gov.record.execution_mode == "hybrid"


# ---------------------------------------------------------------------------
# PR1.6 — DailyRun model has execution_mode column
# ---------------------------------------------------------------------------

class TestDailyRunExecutionMode:
    def test_daily_run_has_execution_mode_field(self):
        from src.db.models import DailyRun

        run = DailyRun(
            run_date=date.today(),
            regime="bull",
            universe_size=100,
            candidates_scored=10,
            execution_mode="quant_only",
        )
        assert run.execution_mode == "quant_only"

    def test_daily_run_accepts_none(self):
        """execution_mode is nullable for backwards compatibility."""
        from src.db.models import DailyRun

        run = DailyRun(
            run_date=date.today(),
            regime="bear",
            universe_size=50,
            candidates_scored=5,
        )
        # Column default is "agentic_full" but Python-side without DB is None
        assert run.execution_mode is None or run.execution_mode == "agentic_full"


# ---------------------------------------------------------------------------
# PR1.7 — Telegram alert shows mode label
# ---------------------------------------------------------------------------

class TestTelegramModeLabel:
    def test_mode_shown_for_non_default(self):
        from src.output.telegram import format_daily_alert

        msg = format_daily_alert(
            picks=[],
            regime="bull",
            run_date="2025-01-01",
            execution_mode="quant_only",
        )
        assert "QUANT_ONLY" in msg

    def test_mode_hidden_for_agentic_full(self):
        from src.output.telegram import format_daily_alert

        msg = format_daily_alert(
            picks=[],
            regime="bull",
            run_date="2025-01-01",
            execution_mode="agentic_full",
        )
        assert "AGENTIC_FULL" not in msg
