import pytest

from src.contracts import EngineResultPayload
from src.engines import collector


@pytest.mark.asyncio
async def test_koocore_falls_back_to_generic_endpoint(monkeypatch):
    class _Settings:
        engine_api_key = ""
        engine_fetch_timeout_s = 5
        koocore_api_url = "https://koocore.example"
        gemini_api_url = ""
        top3_api_url = ""

    payload = EngineResultPayload.model_validate({
        "engine_name": "koocore_d",
        "engine_version": "2.0",
        "run_date": "2026-02-18",
        "run_timestamp": "2026-02-19T03:00:00Z",
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

    results = await collector.collect_engine_results()
    assert len(results) == 1
    assert results[0].engine_name == "koocore_d"


@pytest.mark.asyncio
async def test_koocore_falls_back_when_custom_payload_is_degenerate(monkeypatch):
    class _Settings:
        engine_api_key = ""
        engine_fetch_timeout_s = 5
        koocore_api_url = "https://koocore.example"
        gemini_api_url = ""
        top3_api_url = ""

    degenerate = EngineResultPayload.model_validate({
        "engine_name": "koocore_d",
        "engine_version": "2.0",
        "run_date": "2026-02-18",
        "run_timestamp": "2026-02-19T03:00:00Z",
        "regime": None,
        "status": "success",
        "candidates_screened": 0,
        "pipeline_duration_s": 1.2,
        "picks": [],
    })
    generic = EngineResultPayload.model_validate({
        "engine_name": "koocore_d",
        "engine_version": "2.0",
        "run_date": "2026-02-18",
        "run_timestamp": "2026-02-19T03:01:00Z",
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

    results = await collector.collect_engine_results()
    assert len(results) == 1
    assert results[0].candidates_screened == 6
