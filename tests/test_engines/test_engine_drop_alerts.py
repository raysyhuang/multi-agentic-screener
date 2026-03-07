from datetime import date
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import OperationalError

from src.engines.collector import EngineFailure
from src.engines import engine_drop_alerts


@pytest.fixture(autouse=True)
def _reset_drop_alert_state(monkeypatch):
    engine_drop_alerts._engine_drop_last_signature = None
    engine_drop_alerts._engine_drop_last_sent_at = None
    engine_drop_alerts._engine_drop_alerted_engines = set()
    # Mark state as already loaded so tests don't hit the real DB
    engine_drop_alerts._state_loaded = True

    # No-op DB persistence in unit tests
    async def _noop_persist():
        pass

    monkeypatch.setattr(engine_drop_alerts, "_persist_state_to_db", _noop_persist)

    yield
    engine_drop_alerts._engine_drop_last_signature = None
    engine_drop_alerts._engine_drop_last_sent_at = None
    engine_drop_alerts._engine_drop_alerted_engines = set()
    engine_drop_alerts._state_loaded = False


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


def test_format_engine_drop_alert_message_shows_streak_and_reason():
    failures = [
        EngineFailure(
            engine_name="gemini_stst",
            kind="no_output",
            reason_code="no_response",
            detail="timeout",
        ),
    ]
    metrics = {
        "gemini_stst": {
            "previous_failure_streak": 2,
            "last_success_date": date(2026, 3, 4),
            "last_success_age_trading_days": 3,
        },
    }
    msg = engine_drop_alerts.format_engine_drop_alert_message(
        failed_engines=failures,
        engines_reporting=2,
        failure_metrics=metrics,
    )
    assert "⚠️ Engine Drop Alert" in msg
    assert "gemini_stst" in msg
    assert "reason=no_response" in msg
    # streak should be previous + 1 = 3
    assert "streak=3" in msg
    assert "last_success=2026-03-04" in msg
    assert "2/3 engines reporting" in msg


@pytest.mark.asyncio
async def test_recovery_then_new_failure_reentry(monkeypatch):
    """After recovery, a new failure for the same engine should re-alert."""
    sent: list[str] = []

    async def _fake_send_alert(message: str) -> bool:
        sent.append(message)
        return True

    async def _fake_metrics_streak2(*args, **kwargs):
        return {
            "koocore_d": {
                "previous_failure_streak": 1,
                "last_success_date": date(2026, 3, 5),
                "last_success_age_trading_days": 2,
            },
        }

    settings = SimpleNamespace(engine_drop_alert_cooldown_minutes=0)

    # Step 1: Initial failure alert (streak=2)
    monkeypatch.setattr(engine_drop_alerts, "_load_engine_alert_metrics", _fake_metrics_streak2)
    await engine_drop_alerts.maybe_send_engine_drop_alert(
        failed_engines=[EngineFailure(engine_name="koocore_d", kind="no_response", reason_code="no_response")],
        engines_reporting=2,
        run_date=date(2026, 3, 7),
        success_engine_names=[],
        settings=settings,
        send_alert_fn=_fake_send_alert,
    )
    assert len(sent) == 1  # failure alert

    # Step 2: Recovery
    await engine_drop_alerts.maybe_send_engine_drop_alert(
        failed_engines=[],
        engines_reporting=3,
        run_date=date(2026, 3, 7),
        success_engine_names=["koocore_d", "gemini_stst", "top3_7d"],
        settings=settings,
        send_alert_fn=_fake_send_alert,
    )
    assert len(sent) == 2  # recovery alert
    assert "Recovery" in sent[1]

    # Step 3: New failure — should alert again (not suppressed)
    await engine_drop_alerts.maybe_send_engine_drop_alert(
        failed_engines=[EngineFailure(engine_name="koocore_d", kind="no_response", reason_code="no_response")],
        engines_reporting=2,
        run_date=date(2026, 3, 7),
        success_engine_names=[],
        settings=settings,
        send_alert_fn=_fake_send_alert,
    )
    assert len(sent) == 3  # new failure alert after recovery


@pytest.mark.asyncio
async def test_load_engine_alert_metrics_db_failure_returns_defaults(monkeypatch):
    """When DB query fails, conservative defaults are returned (streak=0)."""

    async def _broken_session(*args, **kwargs):
        raise OperationalError("test", {}, Exception("connection refused"))

    from unittest.mock import AsyncMock, MagicMock

    mock_gs = MagicMock()
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(side_effect=OperationalError("test", {}, Exception("conn")))
    mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

    monkeypatch.setattr(engine_drop_alerts, "get_session", mock_gs)

    metrics = await engine_drop_alerts._load_engine_alert_metrics(
        run_date=date(2026, 3, 7),
        engine_names={"koocore_d", "gemini_stst"},
    )

    assert metrics["koocore_d"]["previous_failure_streak"] == 0
    assert metrics["koocore_d"]["last_success_date"] is None
    assert metrics["gemini_stst"]["previous_failure_streak"] == 0


@pytest.mark.asyncio
async def test_state_loaded_from_db_on_first_access(monkeypatch):
    """Verify _load_state_from_db populates in-memory cache from DB row."""
    from unittest.mock import AsyncMock, MagicMock
    from datetime import datetime, UTC

    # Reset to force DB load
    engine_drop_alerts._state_loaded = False
    engine_drop_alerts._engine_drop_alerted_engines = set()

    mock_row = MagicMock()
    mock_row.alerted_engines = ["koocore_d", "gemini_stst"]
    mock_row.last_signature = "koocore_d:no_response:no_response"
    mock_row.last_sent_at = datetime(2026, 3, 7, 12, 0, 0, tzinfo=UTC)

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_row
    mock_session.execute = AsyncMock(return_value=mock_result)

    mock_gs = MagicMock()
    mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

    monkeypatch.setattr(engine_drop_alerts, "get_session", mock_gs)

    await engine_drop_alerts._load_state_from_db()

    assert engine_drop_alerts._engine_drop_alerted_engines == {"koocore_d", "gemini_stst"}
    assert engine_drop_alerts._engine_drop_last_signature == "koocore_d:no_response:no_response"
    assert engine_drop_alerts._state_loaded is True
