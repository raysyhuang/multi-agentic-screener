"""Contract enforcement tests — validate EngineResultPayload schema and quality checks.

Catches regressions when engine schemas evolve by testing model_validate()
with valid, malformed, and edge-case payloads, plus degenerate pattern detection.
"""

from datetime import date

import pytest
from pydantic import ValidationError

from src.contracts import EngineResultPayload
from src.engines.collector import _validate_payload_quality, _is_critical_quality_issue


# ── Helpers ──────────────────────────────────────────────────────────────────

def _valid_pick(**overrides) -> dict:
    base = {
        "ticker": "AAPL",
        "strategy": "momentum",
        "entry_price": 150.0,
        "stop_loss": 142.5,
        "target_price": 165.0,
        "confidence": 72.0,
        "holding_period_days": 7,
    }
    base.update(overrides)
    return base


def _valid_payload(**overrides) -> dict:
    today = date.today().isoformat()
    base = {
        "engine_name": "koocore_d",
        "engine_version": "v2.1",
        "run_date": today,
        "run_timestamp": f"{today}T15:30:00Z",
        "regime": "bullish",
        "status": "success",
        "candidates_screened": 50,
        "pipeline_duration_s": 8.5,
        "picks": [_valid_pick()],
    }
    base.update(overrides)
    return base


# ── Schema validation (model_validate) ───────────────────────────────────────

class TestEngineResultPayloadSchema:
    def test_valid_payload_parses(self):
        p = EngineResultPayload.model_validate(_valid_payload())
        assert p.engine_name == "koocore_d"
        assert len(p.picks) == 1
        assert p.picks[0].ticker == "AAPL"

    def test_extra_field_rejected(self):
        """StrictModel forbids unknown fields."""
        with pytest.raises(ValidationError, match="extra_forbidden"):
            EngineResultPayload.model_validate(
                _valid_payload(surprise_field="oops")
            )

    def test_missing_required_field_rejected(self):
        data = _valid_payload()
        del data["engine_name"]
        with pytest.raises(ValidationError):
            EngineResultPayload.model_validate(data)

    def test_confidence_out_of_range_rejected(self):
        """EnginePick.confidence is constrained to 0-100."""
        with pytest.raises(ValidationError, match="less_than_equal"):
            EngineResultPayload.model_validate(
                _valid_payload(picks=[_valid_pick(confidence=101)])
            )

    def test_negative_confidence_rejected(self):
        with pytest.raises(ValidationError, match="greater_than_equal"):
            EngineResultPayload.model_validate(
                _valid_payload(picks=[_valid_pick(confidence=-5)])
            )

    def test_empty_picks_valid(self):
        p = EngineResultPayload.model_validate(_valid_payload(picks=[]))
        assert p.picks == []

    def test_extra_pick_field_rejected(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            EngineResultPayload.model_validate(
                _valid_payload(picks=[_valid_pick(ghost_field=42)])
            )


# ── Quality checks (degenerate pattern detection) ───────────────────────────

class TestPayloadQualityChecks:
    def test_clean_payload_no_warnings(self):
        p = EngineResultPayload.model_validate(_valid_payload())
        # Use gemini_stst to avoid koocore_d-specific metadata.scores check
        warnings = _validate_payload_quality("gemini_stst", p)
        # Only staleness check may fire (depends on run date vs today)
        non_stale = [w for w in warnings if "stale" not in w]
        assert non_stale == []

    def test_identical_confidences_flagged(self):
        picks = [
            _valid_pick(ticker="AAPL", confidence=50.0),
            _valid_pick(ticker="MSFT", confidence=50.0),
            _valid_pick(ticker="GOOG", confidence=50.0),
        ]
        p = EngineResultPayload.model_validate(_valid_payload(picks=picks))
        warnings = _validate_payload_quality("koocore_d", p)
        assert any("identical confidence" in w for w in warnings)

    def test_inverted_stop_flagged(self):
        """Stop above entry should be caught."""
        picks = [_valid_pick(entry_price=100.0, stop_loss=105.0)]
        p = EngineResultPayload.model_validate(_valid_payload(picks=picks))
        warnings = _validate_payload_quality("koocore_d", p)
        assert any("inverted stop/target" in w for w in warnings)

    def test_inverted_target_flagged(self):
        """Target below entry should be caught."""
        picks = [_valid_pick(entry_price=100.0, target_price=95.0)]
        p = EngineResultPayload.model_validate(_valid_payload(picks=picks))
        warnings = _validate_payload_quality("koocore_d", p)
        assert any("inverted stop/target" in w for w in warnings)

    def test_critical_issue_detection(self):
        """Stale, missing risk params, and duplicate tuples are critical."""
        assert _is_critical_quality_issue(["stale: stale run_date (2025-01-01, 420d)"]) is True
        assert _is_critical_quality_issue(["risk_invalid: 2 picks missing stop_loss: ['X', 'Y']"]) is True
        assert _is_critical_quality_issue(["schema_invalid: duplicate price tuples across tickers: ..."]) is True

    def test_non_critical_warning(self):
        """Excessive pick count and wide stops are warnings, not critical."""
        assert _is_critical_quality_issue(["hint: unusually high pick count (25)"]) is False
        assert _is_critical_quality_issue(["hint: stop loss >30% from entry: ['XYZ(35%)']"]) is False

    def test_wide_stop_flagged(self):
        """Stop >30% from entry is a warning."""
        picks = [_valid_pick(entry_price=100.0, stop_loss=60.0)]
        p = EngineResultPayload.model_validate(_valid_payload(picks=picks))
        warnings = _validate_payload_quality("koocore_d", p)
        assert any("stop loss >30%" in w for w in warnings)
