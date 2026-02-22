"""Engine-drop alert formatting + cooldown state for cross-engine collection."""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Awaitable, Callable

from src.engines.collector import EngineFailure

logger = logging.getLogger(__name__)

_engine_drop_last_signature: str | None = None
_engine_drop_last_sent_at: datetime | None = None

_SendAlertFn = Callable[[str], Awaitable[bool]]


def clear_engine_drop_alert_state() -> None:
    """Reset drop-alert state after a healthy cycle."""
    global _engine_drop_last_signature, _engine_drop_last_sent_at
    _engine_drop_last_signature = None
    _engine_drop_last_sent_at = None


def _engine_drop_signature(failed_engines: list[EngineFailure]) -> str:
    pairs = sorted((f.engine_name, f.kind) for f in failed_engines)
    return "|".join(f"{engine}:{kind}" for engine, kind in pairs)


def _cooldown_seconds(settings) -> int:
    minutes = int(getattr(settings, "engine_drop_alert_cooldown_minutes", 60) or 60)
    return max(0, minutes * 60)


def _can_send_engine_drop_alert(
    *,
    signature: str,
    now_utc: datetime,
    cooldown_seconds: int,
) -> bool:
    if not signature:
        return False
    if _engine_drop_last_signature is None:
        return True
    if signature != _engine_drop_last_signature:
        return True
    if _engine_drop_last_sent_at is None:
        return True
    elapsed = (now_utc - _engine_drop_last_sent_at).total_seconds()
    return elapsed >= cooldown_seconds


def _mark_engine_drop_alert_sent(signature: str, sent_at: datetime) -> None:
    global _engine_drop_last_signature, _engine_drop_last_sent_at
    _engine_drop_last_signature = signature
    _engine_drop_last_sent_at = sent_at


def _failure_phrase(failure: EngineFailure) -> str:
    if failure.kind == "no_output":
        return "no output"
    if failure.kind == "no_response":
        return "no response"
    if failure.kind == "quality_rejected":
        if failure.detail:
            return f"quality rejected ({failure.detail})"
        return "quality rejected"
    if failure.kind == "exception":
        if failure.detail:
            return f"exception ({failure.detail})"
        return "exception"
    return failure.kind


def format_engine_drop_alert_message(
    *,
    failed_engines: list[EngineFailure],
    engines_reporting: int,
) -> str:
    total = engines_reporting + len(failed_engines)
    lines = ["⚠️ Engine Drop Alert"]
    for failure in failed_engines:
        lines.append(f"- {failure.engine_name}: {_failure_phrase(failure)}")
    lines.append(f"{engines_reporting}/{total} engines reporting.")
    return "\n".join(lines)


async def maybe_send_engine_drop_alert(
    *,
    failed_engines: list[EngineFailure],
    engines_reporting: int,
    settings,
    send_alert_fn: _SendAlertFn,
) -> bool:
    """Send a throttled engine-drop alert when failures are present."""
    if not failed_engines:
        clear_engine_drop_alert_state()
        return False

    now_utc = datetime.utcnow()
    signature = _engine_drop_signature(failed_engines)
    cooldown = _cooldown_seconds(settings)
    if not _can_send_engine_drop_alert(
        signature=signature,
        now_utc=now_utc,
        cooldown_seconds=cooldown,
    ):
        logger.info("Engine drop alert suppressed by cooldown/signature dedupe")
        return False

    message = format_engine_drop_alert_message(
        failed_engines=failed_engines,
        engines_reporting=engines_reporting,
    )
    sent = await send_alert_fn(message)
    if sent:
        _mark_engine_drop_alert_sent(signature, now_utc)
    return sent

