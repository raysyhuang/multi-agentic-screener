"""Tests for Telegram logging — DB model, send logging, ingest API."""

import pytest

from src.db.models import TelegramLog


class TestTelegramLogModel:
    def test_model_fields(self):
        log = TelegramLog(
            source="mas",
            message_text="Test message",
            chat_id="12345",
            message_id=99,
        )
        assert log.source == "mas"
        assert log.message_text == "Test message"
        assert log.chat_id == "12345"
        assert log.message_id == 99

    def test_valid_sources(self):
        for source in ("mas", "koocore_d", "gemini_stst", "top3_7d"):
            log = TelegramLog(source=source, message_text="test")
            assert log.source == source


class TestLogToDb:
    @pytest.mark.asyncio
    async def test_log_to_db_silently_fails_without_db(self):
        """_log_to_db should never raise, even if DB is not available."""
        from src.output.telegram import _log_to_db

        # Should not raise
        await _log_to_db("mas", "test message", "123", 1)


class TestFormatters:
    """Verify telegram formatters don't crash and produce non-empty output."""

    def test_format_daily_alert_with_picks(self):
        from src.output.telegram import format_daily_alert

        picks = [
            {
                "ticker": "AAPL",
                "direction": "LONG",
                "entry_price": 150.0,
                "stop_loss": 145.0,
                "target_1": 160.0,
                "confidence": 75,
                "signal_model": "breakout",
                "holding_period": 5,
                "thesis": "Strong momentum",
            }
        ]
        result = format_daily_alert(picks, "bull", "2026-03-02")
        assert "AAPL" in result
        assert "BULL" in result

    def test_format_daily_alert_validation_failed(self):
        from src.output.telegram import format_daily_alert

        result = format_daily_alert(
            [], "bear", "2026-03-02",
            validation_failed=True,
            failed_checks=["risk_reward_floor_check"],
        )
        assert "FAILED" in result
        assert "risk_reward_floor_check" in result
