"""Tests for the quant_only pipeline runner and ExecutionMode plumbing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from unittest.mock import patch

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

    def persisted_features(self) -> dict:
        # Mirror RankedCandidate.persisted_features(); enough surface for the
        # pipeline tests to exercise the persistence path.
        out = dict(self.features)
        out["model_raw_score"] = self.raw_score
        out["model_adjusted_score"] = self.regime_adjusted_score
        out["model_components"] = dict(self.components)
        out["score_source"] = f"score_{self.signal_model}"
        return out


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
        """Settings should default to quant_only (backtest-proven, no LLM cost)."""
        with patch.dict("os.environ", {}, clear=False):
            s = Settings(
                anthropic_api_key="test",
                openai_api_key="test",
                polygon_api_key="test",
                fmp_api_key="test",
                database_url="sqlite+aiosqlite:///test.db",
            )
            assert s.execution_mode == "quant_only"


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

    def test_sniper_cap_limits_sniper_picks(self):
        """max_sniper caps sniper picks; lower-ranked non-sniper picks still fill."""
        from src.main import _build_quant_only_result

        # Two sniper candidates rank ahead of a mean_reversion one.
        candidates = [
            FakeCandidate(ticker="SNP1", signal_model="sniper"),
            FakeCandidate(ticker="SNP2", signal_model="sniper"),
            FakeCandidate(ticker="MR1", signal_model="mean_reversion"),
        ]
        result = _build_quant_only_result(
            candidates, {"regime": "bull"}, max_picks=2, max_sniper=1,
        )
        models = [p.signal_model for p in result.approved]
        tickers = [p.ticker for p in result.approved]
        assert models.count("sniper") == 1          # only one sniper admitted
        assert tickers == ["SNP1", "MR1"]           # 2nd sniper skipped, MR fills slot

    def test_sniper_cap_zero_blocks_all_sniper(self):
        """max_sniper=0 admits no sniper picks (all slots concurrent-full)."""
        from src.main import _build_quant_only_result

        candidates = [
            FakeCandidate(ticker="SNP1", signal_model="sniper"),
            FakeCandidate(ticker="MR1", signal_model="mean_reversion"),
        ]
        result = _build_quant_only_result(
            candidates, {"regime": "bull"}, max_picks=2, max_sniper=0,
        )
        assert [p.ticker for p in result.approved] == ["MR1"]

    def test_no_sniper_cap_when_none(self):
        """max_sniper=None applies no sniper-specific cap."""
        from src.main import _build_quant_only_result

        candidates = [
            FakeCandidate(ticker="SNP1", signal_model="sniper"),
            FakeCandidate(ticker="SNP2", signal_model="sniper"),
        ]
        result = _build_quant_only_result(
            candidates, {"regime": "bull"}, max_picks=2, max_sniper=None,
        )
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
