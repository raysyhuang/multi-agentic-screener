from datetime import date
from types import SimpleNamespace

import pytest

from src.engines.collector import EngineFailure
from src.engines import engine_drop_alerts


@pytest.fixture(autouse=True)
def _reset_drop_alert_state():
    engine_drop_alerts._engine_drop_last_signature = None
    engine_drop_alerts._engine_drop_last_sent_at = None
    engine_drop_alerts._engine_drop_alerted_engines = set()
    yield
    engine_drop_alerts._engine_drop_last_signature = None
    engine_drop_alerts._engine_drop_last_sent_at = None
    engine_drop_alerts._engine_drop_alerted_engines = set()


@pytest.mark.asyncio
async def test_engine_drop_alert_sent_when_failure_streak_reaches_threshold(monkeypatch):
    sent: list[str] = []

    async def _fake_send_alert(message: str) -> bool:
        sent.append(message)
        return True

    async def _fake_metrics(*args, **kwargs):
        return {
            "gemini_stst": {
                "previous_failure_streak": 1,
                "last_success_date": date(2026, 3, 5),
                "last_success_age_trading_days": 1,
            },
        }

    monkeypatch.setattr(engine_drop_alerts, "_load_engine_alert_metrics", _fake_metrics)
    settings = SimpleNamespace(engine_drop_alert_cooldown_minutes=60)
    failures = [
        EngineFailure(
            engine_name="gemini_stst",
            kind="no_output",
            reason_code="no_response",
            detail="",
        ),
    ]

    was_sent = await engine_drop_alerts.maybe_send_engine_drop_alert(
        failed_engines=failures,
        engines_reporting=1,
        run_date=date(2026, 3, 7),
        success_engine_names=[],
        settings=settings,
        send_alert_fn=_fake_send_alert,
    )

    assert was_sent is True
    assert len(sent) == 1
    assert "gemini_stst: no output" in sent[0]
    assert "streak=2" in sent[0]
    assert "reason=no_response" in sent[0]


@pytest.mark.asyncio
async def test_engine_drop_alert_not_sent_for_single_transient_failure(monkeypatch):
    sent: list[str] = []

    async def _fake_send_alert(message: str) -> bool:
        sent.append(message)
        return True

    async def _fake_metrics(*args, **kwargs):
        return {
            "gemini_stst": {
                "previous_failure_streak": 0,
                "last_success_date": date(2026, 3, 6),
                "last_success_age_trading_days": 1,
            },
        }

    monkeypatch.setattr(engine_drop_alerts, "_load_engine_alert_metrics", _fake_metrics)
    settings = SimpleNamespace(engine_drop_alert_cooldown_minutes=60)
    failures = [
        EngineFailure(
            engine_name="gemini_stst",
            kind="no_output",
            reason_code="no_response",
            detail="",
        ),
    ]

    was_sent = await engine_drop_alerts.maybe_send_engine_drop_alert(
        failed_engines=failures,
        engines_reporting=2,
        run_date=date(2026, 3, 7),
        success_engine_names=[],
        settings=settings,
        send_alert_fn=_fake_send_alert,
    )

    assert was_sent is False
    assert sent == []


@pytest.mark.asyncio
async def test_engine_drop_alert_suppresses_expected_stale(monkeypatch):
    sent: list[str] = []

    async def _fake_send_alert(message: str) -> bool:
        sent.append(message)
        return True

    async def _fake_metrics(*args, **kwargs):
        return {
            "top3_7d": {
                "previous_failure_streak": 5,
                "last_success_date": date(2026, 3, 1),
                "last_success_age_trading_days": 5,
            },
        }

    monkeypatch.setattr(engine_drop_alerts, "_load_engine_alert_metrics", _fake_metrics)
    settings = SimpleNamespace(engine_drop_alert_cooldown_minutes=60)
    failures = [
        EngineFailure(
            engine_name="top3_7d",
            kind="quality_rejected",
            reason_code="expected_stale",
            detail="expected stale in morning window",
        ),
    ]
    was_sent = await engine_drop_alerts.maybe_send_engine_drop_alert(
        failed_engines=[
            *failures,
        ],
        engines_reporting=1,
        run_date=date(2026, 3, 7),
        success_engine_names=[],
        settings=settings,
        send_alert_fn=_fake_send_alert,
    )

    assert was_sent is False
    assert sent == []


@pytest.mark.asyncio
async def test_engine_recovery_alert_sent_once(monkeypatch):
    sent: list[str] = []

    async def _fake_send_alert(message: str) -> bool:
        sent.append(message)
        return True

    async def _fake_metrics(*args, **kwargs):
        return {}

    monkeypatch.setattr(engine_drop_alerts, "_load_engine_alert_metrics", _fake_metrics)
    settings = SimpleNamespace(engine_drop_alert_cooldown_minutes=60)
    engine_drop_alerts._engine_drop_alerted_engines = {"koocore_d"}

    first = await engine_drop_alerts.maybe_send_engine_drop_alert(
        failed_engines=[],
        engines_reporting=2,
        run_date=date(2026, 3, 7),
        success_engine_names=["koocore_d", "gemini_stst"],
        settings=settings,
        send_alert_fn=_fake_send_alert,
    )
    second = await engine_drop_alerts.maybe_send_engine_drop_alert(
        failed_engines=[],
        engines_reporting=2,
        run_date=date(2026, 3, 7),
        success_engine_names=["koocore_d", "gemini_stst"],
        settings=settings,
        send_alert_fn=_fake_send_alert,
    )

    assert first is True
    assert second is False
    assert len(sent) == 1
    assert sent[0].startswith("✅ Engine Recovery")
