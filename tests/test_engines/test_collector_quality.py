from src.contracts import EngineResultPayload
from src.engines.collector import _validate_payload_quality


def _payload(
    *,
    status: str = "success",
    candidates_screened: int = 100,
    picks: list[dict] | None = None,
) -> EngineResultPayload:
    return EngineResultPayload.model_validate({
        "engine_name": "gemini_stst",
        "engine_version": "v1",
        "run_date": "2026-02-18",
        "run_timestamp": "2026-02-18T21:30:00Z",
        "regime": "bearish",
        "status": status,
        "candidates_screened": candidates_screened,
        "pipeline_duration_s": 12.3,
        "picks": picks or [],
    })


def test_quality_flags_zero_screened_zero_picks():
    p = _payload(candidates_screened=0, picks=[])
    warnings = _validate_payload_quality("gemini_stst", p)
    assert any("zero candidates screened and zero picks" in w for w in warnings)


def test_quality_flags_non_success_status():
    p = _payload(status="failed", candidates_screened=25, picks=[])
    warnings = _validate_payload_quality("gemini_stst", p)
    assert any("non-success status" in w for w in warnings)


def test_quality_flags_missing_stop_loss():
    p = _payload(
        candidates_screened=10,
        picks=[{
            "ticker": "AAPL",
            "strategy": "momentum",
            "entry_price": 100.0,
            "stop_loss": None,
            "target_price": 110.0,
            "confidence": 65.0,
            "holding_period_days": 7,
        }],
    )
    warnings = _validate_payload_quality("gemini_stst", p)
    assert any("missing stop_loss" in w for w in warnings)
