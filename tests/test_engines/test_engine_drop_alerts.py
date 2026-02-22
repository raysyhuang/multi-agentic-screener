from types import SimpleNamespace

import pytest

from src.engines.collector import EngineFailure
from src.engines import engine_drop_alerts


@pytest.fixture(autouse=True)
def _reset_drop_alert_state():
    engine_drop_alerts._engine_drop_last_signature = None
    engine_drop_alerts._engine_drop_last_sent_at = None
    yield
    engine_drop_alerts._engine_drop_last_signature = None
    engine_drop_alerts._engine_drop_last_sent_at = None


@pytest.mark.asyncio
async def test_engine_drop_alert_sent_with_kind_specific_wording():
    sent: list[str] = []

    async def _fake_send_alert(message: str) -> bool:
        sent.append(message)
        return True

    settings = SimpleNamespace(engine_drop_alert_cooldown_minutes=60)
    failures = [
        EngineFailure(engine_name="gemini_stst", kind="no_output", detail=""),
        EngineFailure(
            engine_name="koocore_d",
            kind="quality_rejected",
            detail="missing stop_loss",
        ),
    ]

    was_sent = await engine_drop_alerts.maybe_send_engine_drop_alert(
        failed_engines=failures,
        engines_reporting=1,
        settings=settings,
        send_alert_fn=_fake_send_alert,
    )

    assert was_sent is True
    assert len(sent) == 1
    assert "gemini_stst: no output" in sent[0]
    assert "koocore_d: quality rejected" in sent[0]
    assert "failed to report this cycle" not in sent[0]
    assert "1/3 engines reporting." in sent[0]


@pytest.mark.asyncio
async def test_engine_drop_alert_not_sent_when_no_failures():
    sent: list[str] = []

    async def _fake_send_alert(message: str) -> bool:
        sent.append(message)
        return True

    settings = SimpleNamespace(engine_drop_alert_cooldown_minutes=60)

    was_sent = await engine_drop_alerts.maybe_send_engine_drop_alert(
        failed_engines=[],
        engines_reporting=2,
        settings=settings,
        send_alert_fn=_fake_send_alert,
    )

    assert was_sent is False
    assert sent == []


@pytest.mark.asyncio
async def test_engine_drop_alert_dedupes_and_sends_on_state_change():
    sent: list[str] = []

    async def _fake_send_alert(message: str) -> bool:
        sent.append(message)
        return True

    settings = SimpleNamespace(engine_drop_alert_cooldown_minutes=60)
    failures = [EngineFailure(engine_name="gemini_stst", kind="no_output", detail="")]

    first = await engine_drop_alerts.maybe_send_engine_drop_alert(
        failed_engines=failures,
        engines_reporting=1,
        settings=settings,
        send_alert_fn=_fake_send_alert,
    )
    second = await engine_drop_alerts.maybe_send_engine_drop_alert(
        failed_engines=failures,
        engines_reporting=1,
        settings=settings,
        send_alert_fn=_fake_send_alert,
    )
    changed = await engine_drop_alerts.maybe_send_engine_drop_alert(
        failed_engines=[
            EngineFailure(
                engine_name="gemini_stst",
                kind="quality_rejected",
                detail="missing target_price",
            ),
        ],
        engines_reporting=1,
        settings=settings,
        send_alert_fn=_fake_send_alert,
    )

    assert first is True
    assert second is False
    assert changed is True
    assert len(sent) == 2
