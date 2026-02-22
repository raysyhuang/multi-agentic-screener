from datetime import date, datetime, timezone

import pytest

from src.contracts import EngineResultPayload
from src.engines import collector, gemini_runner, koocore_runner


def _payload(engine_name: str, run_date: date) -> EngineResultPayload:
    return EngineResultPayload.model_validate({
        "engine_name": engine_name,
        "engine_version": "test",
        "run_date": run_date.isoformat(),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "regime": None,
        "status": "success",
        "candidates_screened": 10,
        "pipeline_duration_s": 1.0,
        "picks": [],
    })


@pytest.mark.asyncio
async def test_collect_engine_results_local_passes_target_date_to_gemini(monkeypatch):
    class _Settings:
        engine_run_mode = "local"

    target_date = date(2026, 2, 21)
    seen: dict[str, date | None] = {"target": None}

    async def _fake_koocore():
        return _payload("koocore_d", target_date)

    async def _fake_gemini(target_date=None):
        seen["target"] = target_date
        return _payload("gemini_stst", target_date or date.today())

    monkeypatch.setattr(collector, "get_settings", lambda: _Settings())
    monkeypatch.setattr(koocore_runner, "run_koocore_locally", _fake_koocore)
    monkeypatch.setattr(gemini_runner, "run_gemini_locally", _fake_gemini)

    results, failed = await collector.collect_engine_results(target_date=target_date)

    assert failed == []
    assert len(results) == 2
    assert seen["target"] == target_date
