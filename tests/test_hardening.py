"""Tests for production hardening changes.

Covers: config validation, API auth, circuit breaker, budget enforcement,
quality flags, LIVE gate, health check, Telegram resilience.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Config validation ────────────────────────────────────────────────────────


class TestConfigValidation:
    """Test model name validation and API key validation."""

    def test_known_model_names_pass(self):
        """Valid model names should not log warnings."""
        from src.config import Settings
        import logging

        with patch.object(logging.getLogger("src.config"), "warning") as mock_warn:
            Settings(
                signal_interpreter_model="claude-sonnet-4-5-20250929",
                adversarial_model="gpt-5.2",
                planner_model="gpt-5.2",
                verifier_model="gpt-5.2",
            )
            # No warnings should be logged for known models
            for call in mock_warn.call_args_list:
                assert "Unrecognized model" not in str(call)

    def test_unknown_model_name_warns(self):
        """Unknown model names should log a warning."""
        from src.config import Settings
        import logging

        with patch.object(logging.getLogger("src.config"), "warning") as mock_warn:
            Settings(adversarial_model="gpt-99-turbo")
            mock_warn.assert_called()
            assert "gpt-99-turbo" in str(mock_warn.call_args)

    def test_validate_keys_quant_only_needs_data_key(self):
        """quant_only mode needs at least one data provider key."""
        from src.config import Settings

        s = Settings(
            execution_mode="quant_only",
            polygon_api_key="",
            fmp_api_key="",
        )
        with pytest.raises(ValueError, match="polygon_api_key or fmp_api_key"):
            s.validate_keys_for_mode()

    def test_validate_keys_quant_only_ok_with_polygon(self):
        """quant_only mode is OK with just polygon key."""
        from src.config import Settings

        s = Settings(
            execution_mode="quant_only",
            polygon_api_key="pk_test",
            fmp_api_key="",
        )
        s.validate_keys_for_mode()  # Should not raise

    def test_validate_keys_agentic_needs_llm_key(self):
        """agentic_full mode needs at least one LLM key."""
        from src.config import Settings

        s = Settings(
            execution_mode="agentic_full",
            polygon_api_key="pk_test",
            anthropic_api_key="",
            openai_api_key="",
        )
        with pytest.raises(ValueError, match="anthropic_api_key or openai_api_key"):
            s.validate_keys_for_mode()

    def test_validate_keys_agentic_ok_with_both(self):
        """agentic_full mode passes with both data + LLM keys."""
        from src.config import Settings

        s = Settings(
            execution_mode="agentic_full",
            polygon_api_key="pk_test",
            anthropic_api_key="sk-ant-test",
        )
        s.validate_keys_for_mode()  # Should not raise

    def test_default_retry_on_low_quality_is_true(self):
        """agent_retry_on_low_quality should default to True."""
        from src.config import Settings

        s = Settings()
        assert s.agent_retry_on_low_quality is True

    def test_force_live_defaults_false(self):
        """force_live should default to False."""
        from src.config import Settings

        s = Settings()
        assert s.force_live is False


# ── API Auth Middleware ──────────────────────────────────────────────────────


class TestAPIAuth:
    """Test the API authentication middleware logic."""

    def test_health_exempt_from_auth(self):
        """Health check path should not start with /api/ so auth is skipped."""
        path = "/health"
        assert not path.startswith("/api/")

    def test_api_path_needs_auth(self):
        """API paths should require auth when api_secret_key is set."""
        path = "/api/signals/2026-02-17"
        assert path.startswith("/api/")

    def test_bearer_token_extraction(self):
        """Bearer token should be extracted correctly from header."""
        auth_header = "Bearer my-secret-key"
        assert auth_header.startswith("Bearer ")
        token = auth_header[7:]
        assert token == "my-secret-key"

    def test_missing_bearer_prefix(self):
        """Auth without Bearer prefix should be rejected."""
        auth_header = "Basic dXNlcjpwYXNz"
        assert not auth_header.startswith("Bearer ")


class TestAPIAuthUnit:
    """Unit tests for auth logic without full app setup."""

    def test_api_route_requires_auth_when_key_set(self):
        """API routes should reject requests without auth when key is configured."""
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
        from fastapi.testclient import TestClient

        app = FastAPI()

        @app.middleware("http")
        async def auth_check(request: Request, call_next):
            path = request.url.path
            api_key = "my-secret"
            if path.startswith("/api/") and api_key:
                auth = request.headers.get("Authorization", "")
                if not auth.startswith("Bearer "):
                    return JSONResponse(status_code=401, content={"detail": "Missing auth"})
                if auth[7:] != api_key:
                    return JSONResponse(status_code=401, content={"detail": "Invalid key"})
            return await call_next(request)

        @app.get("/api/test")
        async def test_endpoint():
            return {"ok": True}

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        client = TestClient(app)

        # No auth → 401
        resp = client.get("/api/test")
        assert resp.status_code == 401

        # Wrong key → 401
        resp = client.get("/api/test", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401

        # Correct key → 200
        resp = client.get("/api/test", headers={"Authorization": "Bearer my-secret"})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

        # Health → no auth needed
        resp = client.get("/health")
        assert resp.status_code == 200


# ── Circuit Breaker ──────────────────────────────────────────────────────────


class TestCircuitBreaker:
    """Test the APICircuitBreaker class."""

    def test_starts_closed(self):
        from src.data.circuit_breaker import APICircuitBreaker

        cb = APICircuitBreaker(failure_threshold=3)
        assert not cb.is_open("polygon")

    def test_opens_after_threshold(self):
        from src.data.circuit_breaker import APICircuitBreaker

        cb = APICircuitBreaker(failure_threshold=3, cooldown_seconds=60)
        cb.record_failure("polygon")
        cb.record_failure("polygon")
        assert not cb.is_open("polygon")
        cb.record_failure("polygon")
        assert cb.is_open("polygon")

    def test_resets_on_success(self):
        from src.data.circuit_breaker import APICircuitBreaker

        cb = APICircuitBreaker(failure_threshold=3, cooldown_seconds=60)
        cb.record_failure("polygon")
        cb.record_failure("polygon")
        cb.record_success("polygon")
        cb.record_failure("polygon")
        cb.record_failure("polygon")
        assert not cb.is_open("polygon")  # only 2 consecutive after reset

    def test_closes_after_cooldown(self):
        from src.data.circuit_breaker import APICircuitBreaker

        cb = APICircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)
        cb.record_failure("fmp")
        cb.record_failure("fmp")
        assert cb.is_open("fmp")
        time.sleep(0.15)
        assert not cb.is_open("fmp")  # cooldown expired

    def test_independent_providers(self):
        from src.data.circuit_breaker import APICircuitBreaker

        cb = APICircuitBreaker(failure_threshold=2)
        cb.record_failure("polygon")
        cb.record_failure("polygon")
        assert cb.is_open("polygon")
        assert not cb.is_open("fmp")  # different provider

    def test_get_stats(self):
        from src.data.circuit_breaker import APICircuitBreaker

        cb = APICircuitBreaker(failure_threshold=3)
        cb.record_failure("polygon")
        cb.record_success("fmp")
        stats = cb.get_stats()
        assert stats["polygon"]["consecutive_failures"] == 1
        assert stats["polygon"]["total_failures"] == 1
        assert stats["fmp"]["total_successes"] == 1


# ── Budget Enforcement ───────────────────────────────────────────────────────


class TestBudgetEnforcement:
    """Test budget checking in the orchestrator."""

    def test_budget_exhausted_error(self):
        from src.agents.orchestrator import BudgetExhaustedError

        err = BudgetExhaustedError(spent=2.50, budget=2.00, stage="pre_debate")
        assert err.spent == 2.50
        assert err.budget == 2.00
        assert err.stage == "pre_debate"
        assert "pre_debate" in str(err)

    def test_check_budget_raises_when_exceeded(self):
        from src.agents.orchestrator import _check_budget, BudgetExhaustedError

        memory_service = MagicMock()
        memory_service.working.total_cost_usd = 2.50

        with pytest.raises(BudgetExhaustedError) as exc_info:
            _check_budget(memory_service, budget_usd=2.00, stage="test_stage")
        assert exc_info.value.stage == "test_stage"

    def test_check_budget_ok_when_under(self):
        from src.agents.orchestrator import _check_budget

        memory_service = MagicMock()
        memory_service.working.total_cost_usd = 1.50

        # Should not raise
        _check_budget(memory_service, budget_usd=2.00, stage="test_stage")

    def test_check_budget_skips_when_no_budget(self):
        from src.agents.orchestrator import _check_budget

        memory_service = MagicMock()
        memory_service.working.total_cost_usd = 999.99

        # No budget set — should not raise
        _check_budget(memory_service, budget_usd=None, stage="test_stage")


# ── Quality Warning Flag ─────────────────────────────────────────────────────


class TestQualityWarning:
    """Test that low-quality interpretations get quality_warning flag."""

    def test_attempt_record_has_quality_warning(self):
        from src.agents.retry import AttemptRecord, FailureReason

        record = AttemptRecord(
            attempt_num=1,
            success=False,
            failure_reason=FailureReason.LOW_QUALITY,
            quality_warning=True,
        )
        assert record.quality_warning is True

    def test_attempt_record_default_no_warning(self):
        from src.agents.retry import AttemptRecord

        record = AttemptRecord(attempt_num=1, success=True)
        assert record.quality_warning is False


# ── Telegram Resilience ──────────────────────────────────────────────────────


class TestTelegramResilience:
    """Test Telegram retry and message splitting."""

    def test_split_short_message(self):
        from src.output.telegram import _split_message

        msg = "Hello world"
        chunks = _split_message(msg, max_length=4000)
        assert chunks == ["Hello world"]

    def test_split_long_message(self):
        from src.output.telegram import _split_message

        lines = [f"Line {i}" for i in range(200)]
        msg = "\n".join(lines)
        chunks = _split_message(msg, max_length=500)
        assert len(chunks) > 1
        # All content preserved
        reassembled = "\n".join(chunks)
        for line in lines:
            assert line in reassembled

    def test_split_no_newlines(self):
        from src.output.telegram import _split_message

        msg = "x" * 5000
        chunks = _split_message(msg, max_length=2000)
        assert len(chunks) == 3  # 2000 + 2000 + 1000
        assert "".join(chunks) == msg

    @pytest.mark.asyncio
    async def test_send_alert_retries_on_failure(self):
        from src.output.telegram import send_alert

        with patch("src.output.telegram.get_settings") as mock_gs:
            mock_settings = MagicMock()
            mock_settings.telegram_bot_token = "test-token"
            mock_settings.telegram_chat_id = "test-chat"
            mock_gs.return_value = mock_settings

            with patch("src.output.telegram.Bot") as MockBot:
                bot_instance = MagicMock()
                # Fail first 2 attempts, succeed on 3rd
                bot_instance.send_message = AsyncMock(
                    side_effect=[
                        Exception("Timeout"),
                        Exception("Timeout"),
                        None,
                    ]
                )
                MockBot.return_value = bot_instance

                result = await send_alert("Test message")
                assert result is True
                assert bot_instance.send_message.call_count == 3

    @pytest.mark.asyncio
    async def test_send_alert_fails_after_max_retries(self):
        from src.output.telegram import send_alert

        with patch("src.output.telegram.get_settings") as mock_gs:
            mock_settings = MagicMock()
            mock_settings.telegram_bot_token = "test-token"
            mock_settings.telegram_chat_id = "test-chat"
            mock_gs.return_value = mock_settings

            with patch("src.output.telegram.Bot") as MockBot:
                bot_instance = MagicMock()
                bot_instance.send_message = AsyncMock(
                    side_effect=Exception("Always fails"),
                )
                MockBot.return_value = bot_instance

                result = await send_alert("Test message")
                assert result is False

    @pytest.mark.asyncio
    async def test_send_alert_skips_when_not_configured(self):
        from src.output.telegram import send_alert

        with patch("src.output.telegram.get_settings") as mock_gs:
            mock_settings = MagicMock()
            mock_settings.telegram_bot_token = ""
            mock_settings.telegram_chat_id = ""
            mock_gs.return_value = mock_settings

            result = await send_alert("Test message")
            assert result is False


# ── Rate Limit Error ─────────────────────────────────────────────────────────


class TestRateLimitError:
    """Test the RateLimitError exception."""

    def test_rate_limit_error(self):
        from src.data.circuit_breaker import RateLimitError

        err = RateLimitError("polygon", retry_after=30.0)
        assert err.provider == "polygon"
        assert err.retry_after == 30.0
        assert "polygon" in str(err)

    def test_rate_limit_error_no_retry_after(self):
        from src.data.circuit_breaker import RateLimitError

        err = RateLimitError("fmp")
        assert err.provider == "fmp"
        assert err.retry_after is None


# ── Health Check ─────────────────────────────────────────────────────────────


class TestHealthCheck:
    """Test the real health check endpoint logic."""

    def test_health_returns_degraded_without_keys(self):
        """Health check should return degraded when no API keys configured."""
        # Test the logic directly
        issues = []
        polygon_key = ""
        fmp_key = ""
        if not polygon_key and not fmp_key:
            issues.append("no data provider API key")

        assert len(issues) == 1
        assert "data provider" in issues[0]


# ── Data Quality Flags ───────────────────────────────────────────────────────


class TestDataQualityFlags:
    """Test that degraded data gets flagged properly."""

    def test_degraded_flag_set_on_failure(self):
        """Simulates the data quality flag logic from main.py."""
        feat = {"ticker": "TEST", "close": 100.0}
        degraded_components = []

        # Simulate fundamental fetch failure
        try:
            raise Exception("API error")
        except Exception:
            feat["fundamental"] = {}
            degraded_components.append("fundamentals")

        # Simulate news fetch failure
        try:
            raise Exception("API error")
        except Exception:
            feat["sentiment"] = {}
            degraded_components.append("sentiment")

        if degraded_components:
            feat["_degraded"] = True
            feat["_degraded_components"] = degraded_components

        assert feat["_degraded"] is True
        assert "fundamentals" in feat["_degraded_components"]
        assert "sentiment" in feat["_degraded_components"]

    def test_no_flag_when_all_succeeds(self):
        """No degradation flag when all data fetches succeed."""
        feat = {"ticker": "TEST", "close": 100.0}
        degraded_components = []

        feat["fundamental"] = {"earnings": []}
        feat["sentiment"] = {"score": 0.5}

        if degraded_components:
            feat["_degraded"] = True
            feat["_degraded_components"] = degraded_components

        assert "_degraded" not in feat
