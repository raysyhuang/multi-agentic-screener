"""Engine-drop alert formatting + cooldown state for cross-engine collection."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import logging
from typing import Awaitable, Callable

from sqlalchemy import func, select

from src.db.models import EngineRun
from src.db.session import get_session
from src.engines.collector import EngineFailure

logger = logging.getLogger(__name__)

_engine_drop_last_signature: str | None = None
_engine_drop_last_sent_at: datetime | None = None
_engine_drop_alerted_engines: set[str] = set()

_SendAlertFn = Callable[[str], Awaitable[bool]]


def clear_engine_drop_alert_state() -> None:
    """Reset drop-alert state after a healthy cycle."""
    global _engine_drop_last_signature, _engine_drop_last_sent_at, _engine_drop_alerted_engines
    _engine_drop_last_signature = None
    _engine_drop_last_sent_at = None
    _engine_drop_alerted_engines = set()


def _clear_drop_signature_state() -> None:
    """Clear only dedupe signature/timestamp while preserving recovery state."""
    global _engine_drop_last_signature, _engine_drop_last_sent_at
    _engine_drop_last_signature = None
    _engine_drop_last_sent_at = None


def _engine_drop_signature(failed_engines: list[EngineFailure]) -> str:
    triples = sorted((f.engine_name, f.kind, f.reason_code) for f in failed_engines)
    return "|".join(f"{engine}:{kind}:{reason}" for engine, kind, reason in triples)


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
    if failure.reason_code == "expected_stale":
        return "expected stale (morning grace window)"
    if failure.reason_code == "no_artifacts":
        return "no artifacts"
    if failure.reason_code == "stale":
        return "stale payload"
    if failure.reason_code == "risk_invalid":
        return "invalid risk parameters"
    if failure.reason_code == "schema_invalid":
        return "invalid payload schema"
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


def _trading_days_between(start: date, end: date) -> int:
    if start >= end:
        return 0
    days = 0
    current = start
    one_day = timedelta(days=1)
    while current < end:
        current += one_day
        if current.weekday() < 5:
            days += 1
    return days


async def _load_engine_alert_metrics(
    run_date: date,
    engine_names: set[str],
) -> dict[str, dict]:
    """Load prior-cycle streak + last-success metadata for alert thresholding."""
    metrics: dict[str, dict] = {
        name: {
            "previous_failure_streak": 0,
            "last_success_date": None,
            "last_success_age_trading_days": None,
        }
        for name in engine_names
    }
    if not engine_names:
        return metrics

    try:
        async with get_session() as session:
            for engine_name in engine_names:
                history_rows = (
                    await session.execute(
                        select(EngineRun.status, EngineRun.run_date)
                        .where(
                            EngineRun.engine_name == engine_name,
                            EngineRun.run_date < run_date,
                        )
                        .order_by(EngineRun.run_date.desc(), EngineRun.attempt.desc())
                        .limit(40)
                    )
                ).all()

                previous_streak = 0
                for status, _ in history_rows:
                    if status == "success":
                        break
                    previous_streak += 1

                last_success_date = (
                    await session.execute(
                        select(func.max(EngineRun.run_date)).where(
                            EngineRun.engine_name == engine_name,
                            EngineRun.status == "success",
                            EngineRun.run_date < run_date,
                        )
                    )
                ).scalar_one_or_none()

                last_success_age = (
                    _trading_days_between(last_success_date, run_date)
                    if last_success_date is not None
                    else None
                )
                metrics[engine_name] = {
                    "previous_failure_streak": previous_streak,
                    "last_success_date": last_success_date,
                    "last_success_age_trading_days": last_success_age,
                }
    except Exception as e:
        logger.warning("Engine drop metrics query failed, using conservative defaults: %s", e)
    return metrics


def format_engine_drop_alert_message(
    *,
    failed_engines: list[EngineFailure],
    engines_reporting: int,
    failure_metrics: dict[str, dict] | None = None,
) -> str:
    total = engines_reporting + len(failed_engines)
    lines = ["⚠️ Engine Drop Alert"]
    metrics = failure_metrics or {}
    for failure in failed_engines:
        meta = metrics.get(failure.engine_name, {})
        streak = int(meta.get("current_failure_streak") or 0)
        last_success = meta.get("last_success_date")
        last_success_str = str(last_success) if last_success else "n/a"
        lines.append(
            f"- {failure.engine_name}: {_failure_phrase(failure)} "
            f"[reason={failure.reason_code}, streak={streak}, last_success={last_success_str}]"
        )
    lines.append(f"{engines_reporting}/{total} engines reporting.")
    return "\n".join(lines)


def format_engine_recovery_alert_message(recovered_engines: list[str]) -> str:
    lines = ["✅ Engine Recovery"]
    for engine_name in recovered_engines:
        lines.append(f"- {engine_name}: recovered")
    return "\n".join(lines)


async def maybe_send_engine_drop_alert(
    *,
    failed_engines: list[EngineFailure],
    engines_reporting: int,
    run_date: date,
    success_engine_names: list[str] | None,
    settings,
    send_alert_fn: _SendAlertFn,
) -> bool:
    """Send throttled drop/recovery alerts with streak-aware suppression."""
    global _engine_drop_alerted_engines
    sent_any = False

    success_engines = set(success_engine_names or [])
    failed_engine_names = {f.engine_name for f in failed_engines if f.reason_code != "expected_stale"}
    recovered = sorted((_engine_drop_alerted_engines - failed_engine_names) & success_engines)
    if recovered:
        if await send_alert_fn(format_engine_recovery_alert_message(recovered)):
            _engine_drop_alerted_engines -= set(recovered)
            sent_any = True

    alertable_failures = [f for f in failed_engines if f.reason_code != "expected_stale"]
    if not alertable_failures:
        _clear_drop_signature_state()
        return sent_any

    now_utc = datetime.utcnow()
    metrics = await _load_engine_alert_metrics(
        run_date=run_date,
        engine_names={f.engine_name for f in alertable_failures},
    )

    eligible_failures: list[EngineFailure] = []
    for failure in alertable_failures:
        meta = metrics.get(failure.engine_name, {})
        previous_streak = int(meta.get("previous_failure_streak") or 0)
        current_streak = previous_streak + 1
        last_success_age = meta.get("last_success_age_trading_days")
        stale_success = last_success_age is not None and int(last_success_age) >= 2
        if current_streak >= 2 or stale_success:
            eligible_failures.append(failure)
            meta["current_failure_streak"] = current_streak

    if not eligible_failures:
        logger.info("Engine drop alert suppressed (streak<2 and last_success_age<2 trading days)")
        return sent_any

    signature = _engine_drop_signature(eligible_failures)
    cooldown = _cooldown_seconds(settings)
    if not _can_send_engine_drop_alert(
        signature=signature,
        now_utc=now_utc,
        cooldown_seconds=cooldown,
    ):
        logger.info("Engine drop alert suppressed by cooldown/signature dedupe")
        return False

    message = format_engine_drop_alert_message(
        failed_engines=eligible_failures,
        engines_reporting=engines_reporting,
        failure_metrics=metrics,
    )
    sent = await send_alert_fn(message)
    if sent:
        _mark_engine_drop_alert_sent(signature, now_utc)
        _engine_drop_alerted_engines.update(f.engine_name for f in eligible_failures)
        sent_any = True
    return sent_any
