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

    def test_format_cross_engine_alert(self):
        from src.output.telegram import format_cross_engine_alert

        synthesis = {
            "regime_consensus": "bear",
            "engines_reporting": 2,
            "executive_summary": "Low conviction",
            "convergent_picks": [],
            "portfolio": [
                {
                    "ticker": "NHI",
                    "weight_pct": 3,
                    "entry_price": 84.0,
                    "stop_loss": 83.0,
                    "target_price": 86.0,
                    "holding_period_days": 3,
                    "source": "unique",
                },
            ],
        }
        cred = {
            "gemini_stst": {"hit_rate": 0.33, "weight": 0.42, "resolved_picks": 3},
        }
        result = format_cross_engine_alert(synthesis, cred)
        assert "NHI" in result
        assert "BEAR" in result

    def test_format_shadow_tracks_digest_empty(self):
        from src.output.telegram import format_shadow_tracks_digest

        result = format_shadow_tracks_digest([])
        assert "No active shadow tracks" in result

    def test_format_shadow_tracks_digest_with_data(self):
        from src.output.telegram import format_shadow_tracks_digest

        scorecards = [
            {
                "name": "high_convergence",
                "status": "active",
                "has_sufficient_data": True,
                "composite_score": 0.75,
                "sharpe_ratio": 1.2,
                "deflated_sharpe": 0.85,
                "win_rate": 0.6,
                "avg_return_pct": 2.5,
                "delta_sharpe": 0.3,
            },
        ]
        result = format_shadow_tracks_digest(scorecards)
        assert "high_convergence" in result
        assert "1 active" in result
