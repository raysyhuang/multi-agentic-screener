from datetime import date, timedelta

from src.contracts import EngineResultPayload
from src.engines.collector import _is_critical_quality_issue, _validate_payload_quality


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


def test_quality_flags_missing_target_price():
    p = _payload(
        candidates_screened=10,
        picks=[{
            "ticker": "AAPL",
            "strategy": "momentum",
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "target_price": None,
            "confidence": 65.0,
            "holding_period_days": 7,
        }],
    )
    warnings = _validate_payload_quality("gemini_stst", p)
    assert any("missing target_price" in w for w in warnings)


def test_missing_target_is_critical_quality_issue():
    warnings = ["risk_invalid: 1 picks missing target_price: ['AAPL']"]
    assert _is_critical_quality_issue(warnings) is True


def test_quality_flags_duplicate_price_tuples_as_critical():
    p = _payload(
        candidates_screened=12,
        picks=[
            {
                "ticker": "UAMY",
                "strategy": "swing",
                "entry_price": 8.54,
                "stop_loss": 8.11,
                "target_price": 9.39,
                "confidence": 30.0,
                "holding_period_days": 14,
            },
            {
                "ticker": "AFCG",
                "strategy": "swing",
                "entry_price": 8.54,
                "stop_loss": 8.11,
                "target_price": 9.39,
                "confidence": 29.5,
                "holding_period_days": 14,
            },
        ],
    )
    warnings = _validate_payload_quality("koocore_d", p)
    assert any("duplicate price tuples across tickers" in w for w in warnings)
    assert _is_critical_quality_issue(warnings) is True


def test_top3_morning_no_artifacts_classifies_expected_stale():
    run_date = (date.today() - timedelta(days=1)).isoformat()
    payload = EngineResultPayload.model_validate({
        "engine_name": "top3_7d",
        "engine_version": "1.0",
        "run_date": run_date,
        "run_timestamp": f"{run_date}T22:30:00Z",
        "regime": "bear",
        "status": "no_artifacts",
        "candidates_screened": 0,
        "pipeline_duration_s": None,
        "picks": [],
    })

    warnings = _validate_payload_quality(
        "top3_7d",
        payload,
        collection_time="morning",
        asof_date=date.today(),
    )
    assert any(w.startswith("expected_stale:") for w in warnings)
    assert _is_critical_quality_issue(warnings) is True


def test_top3_evening_no_artifacts_is_real_failure():
    run_date = (date.today() - timedelta(days=1)).isoformat()
    payload = EngineResultPayload.model_validate({
        "engine_name": "top3_7d",
        "engine_version": "1.0",
        "run_date": run_date,
        "run_timestamp": f"{run_date}T22:30:00Z",
        "regime": "bear",
        "status": "no_artifacts",
        "candidates_screened": 0,
        "pipeline_duration_s": None,
        "picks": [],
    })

    warnings = _validate_payload_quality(
        "top3_7d",
        payload,
        collection_time="evening",
        asof_date=date.today(),
    )
    assert any(w.startswith("no_artifacts:") for w in warnings)
    assert not any(w.startswith("expected_stale:") for w in warnings)
    assert _is_critical_quality_issue(warnings) is True
