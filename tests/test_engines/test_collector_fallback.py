from datetime import date, datetime, timezone

import pytest

from src.contracts import EngineResultPayload
from src.engines import collector


def _fresh_run_date() -> str:
    """Return today's date as ISO string so payloads are never stale."""
    return date.today().isoformat()


def _fresh_run_timestamp() -> str:
    """Return a recent UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


@pytest.mark.asyncio
async def test_koocore_falls_back_to_generic_endpoint(monkeypatch):
    class _Settings:
        engine_api_key = ""
        engine_fetch_timeout_s = 5
        engine_run_mode = "http"
        koocore_api_url = "https://koocore.example"
        gemini_api_url = ""

    payload = EngineResultPayload.model_validate({
        "engine_name": "koocore_d",
        "engine_version": "2.0",
        "run_date": _fresh_run_date(),
        "run_timestamp": _fresh_run_timestamp(),
        "regime": None,
        "status": "success",
        "candidates_screened": 12,
        "pipeline_duration_s": 1.2,
        "picks": [],
    })

    async def _fake_koocore(*args, **kwargs):
        return None

    async def _fake_fetch_engine(*args, **kwargs):
        return payload

    monkeypatch.setattr(collector, "get_settings", lambda: _Settings())
    monkeypatch.setattr(collector, "fetch_koocore", _fake_koocore)
    monkeypatch.setattr(collector, "_fetch_engine", _fake_fetch_engine)

    results, failed = await collector.collect_engine_results()
    assert len(results) == 1
    assert results[0].engine_name == "koocore_d"
    assert failed == []


@pytest.mark.asyncio
async def test_koocore_falls_back_when_custom_payload_is_degenerate(monkeypatch):
    class _Settings:
        engine_api_key = ""
        engine_fetch_timeout_s = 5
        engine_run_mode = "http"
        koocore_api_url = "https://koocore.example"
        gemini_api_url = ""

    run_date = _fresh_run_date()
    run_ts = _fresh_run_timestamp()

    degenerate = EngineResultPayload.model_validate({
        "engine_name": "koocore_d",
        "engine_version": "2.0",
        "run_date": run_date,
        "run_timestamp": run_ts,
        "regime": None,
        "status": "success",
        "candidates_screened": 0,
        "pipeline_duration_s": 1.2,
        "picks": [],
    })
    generic = EngineResultPayload.model_validate({
        "engine_name": "koocore_d",
        "engine_version": "2.0",
        "run_date": run_date,
        "run_timestamp": run_ts,
        "regime": None,
        "status": "success",
        "candidates_screened": 6,
        "pipeline_duration_s": 1.0,
        "picks": [],
    })

    async def _fake_koocore(*args, **kwargs):
        return degenerate

    async def _fake_fetch_engine(*args, **kwargs):
        return generic

    monkeypatch.setattr(collector, "get_settings", lambda: _Settings())
    monkeypatch.setattr(collector, "fetch_koocore", _fake_koocore)
    monkeypatch.setattr(collector, "_fetch_engine", _fake_fetch_engine)

    results, failed = await collector.collect_engine_results()
    assert len(results) == 1
    assert results[0].candidates_screened == 6
    assert failed == []
