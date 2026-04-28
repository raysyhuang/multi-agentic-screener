"""Invariant alarms for sniper outcome correctness.

These detectors are alert-only — they never mutate. Their job is to surface
patterns that match known phantom-exit signatures so a regression in the
simulator (or the introduction of a new one) is caught within one
afternoon-check pass instead of after a 2-week stretch of contaminated data.

Patterns covered:

P1: sniper closes with entry_date == exit_date AND exit_reason is trail_stop
    or time_stop. Direct replay of the bugs fixed in 2026-04-12 (PR #2)
    and 2026-04-28 (sniper time_stop entry-bar guard).

P2: sniper closes with pnl_pct ≈ -0.20% AND exit_reason == trail_stop. The
    -0.20% fingerprint is bar_open * (1 - slippage), the signature of a
    same-bar trail clipping the entry bar's own open/low.

P3: over the last N closed sniper outcomes within a recent lookback, zero
    of them reached `target` or a real (non-trail) `stop`. Suggests
    trail/time_stop are dominating in a way that's structurally suspicious.

The detector returns a list of Alert dicts. Caller decides whether to log,
Telegram, or both.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Outcome, Signal


logger = logging.getLogger(__name__)


# Tolerance for matching the -0.20% phantom fingerprint (in percent).
PHANTOM_PNL_TOLERANCE = 0.05  # so [-0.25, -0.15] all match
# Upper bound (exclusive) on the phantom-fingerprint range — anything below
# -0.30% is unambiguously a real loss, not a slippage artifact.
PHANTOM_PNL_LOWER = -0.30
PHANTOM_PNL_UPPER = -0.10

# Deploy breakpoint: outcomes that closed before this date were produced by
# the buggy simulator and don't represent the fixed system's behavior. The
# cluster detector (P3) excludes them so a backlog of historical contamination
# doesn't trigger a spurious alarm under the new code.
CLEAN_SINCE = date(2026, 4, 28)


@dataclass(frozen=True)
class Alert:
    pattern_id: str
    severity: str  # "HIGH" | "MEDIUM"
    ticker: str | None
    message: str
    evidence: dict[str, Any]


def detect_same_day_sniper_close(outcome: Outcome, signal: Signal) -> Alert | None:
    """P1: sniper closed on the entry bar via trail_stop or time_stop."""
    if signal.signal_model != "sniper":
        return None
    if outcome.still_open:
        return None
    if outcome.exit_date is None or outcome.entry_date is None:
        return None
    if outcome.exit_date != outcome.entry_date:
        return None
    if outcome.exit_reason not in {"trail_stop", "time_stop"}:
        return None
    return Alert(
        pattern_id="P1_SAME_DAY_SNIPER_CLOSE",
        severity="HIGH",
        ticker=outcome.ticker,
        message=(
            f"{outcome.ticker} closed on entry bar via {outcome.exit_reason} "
            f"(pnl {outcome.pnl_pct:+.2f}%). Same-bar exits should not occur "
            f"under the fixed simulator — possible regression."
        ),
        evidence={
            "outcome_id": outcome.id,
            "signal_id": signal.id,
            "entry_date": outcome.entry_date.isoformat(),
            "exit_date": outcome.exit_date.isoformat(),
            "exit_reason": outcome.exit_reason,
            "pnl_pct": outcome.pnl_pct,
        },
    )


def detect_phantom_pnl_fingerprint(outcome: Outcome, signal: Signal) -> Alert | None:
    """P2: sniper trail_stop with pnl matching the -0.20% phantom signature."""
    if signal.signal_model != "sniper":
        return None
    if outcome.still_open:
        return None
    if outcome.exit_reason != "trail_stop":
        return None
    if outcome.pnl_pct is None:
        return None
    if not (PHANTOM_PNL_LOWER < outcome.pnl_pct < PHANTOM_PNL_UPPER):
        return None
    return Alert(
        pattern_id="P2_PHANTOM_PNL_FINGERPRINT",
        severity="HIGH",
        ticker=outcome.ticker,
        message=(
            f"{outcome.ticker} trail_stop closed at {outcome.pnl_pct:+.2f}% — "
            f"matches the phantom -0.20% fingerprint (entry_fill * (1 - slippage)). "
            f"Indicates a same-bar trail activation/exit may be slipping through."
        ),
        evidence={
            "outcome_id": outcome.id,
            "signal_id": signal.id,
            "entry_date": outcome.entry_date.isoformat() if outcome.entry_date else None,
            "exit_date": outcome.exit_date.isoformat() if outcome.exit_date else None,
            "pnl_pct": outcome.pnl_pct,
        },
    )


async def detect_zero_real_exits(
    session: AsyncSession,
    *,
    lookback_days: int = 14,
    min_sample: int = 10,
    clean_since: date | None = None,
) -> Alert | None:
    """P3: cluster check — over the last `min_sample`+ sniper closes within
    `lookback_days`, zero exited via `target` or non-trail `stop`. That
    distribution suggests phantom trails/time_stops are dominating.

    If `clean_since` is provided, only outcomes with exit_date >= clean_since
    are counted. This excludes outcomes produced by the pre-fix simulator
    so a backlog of contaminated history doesn't trigger a spurious alarm.
    """
    cutoff = date.today() - timedelta(days=lookback_days)
    if clean_since is not None and clean_since > cutoff:
        cutoff = clean_since
    result = await session.execute(
        select(Outcome, Signal)
        .join(Signal, Signal.id == Outcome.signal_id)
        .where(
            Signal.signal_model == "sniper",
            Outcome.still_open == False,  # noqa: E712
            Outcome.exit_date.is_not(None),
            Outcome.exit_date >= cutoff,
        )
        .order_by(Outcome.exit_date.desc())
    )
    rows = result.all()
    if len(rows) < min_sample:
        return None

    real_exits = sum(
        1 for o, _ in rows if o.exit_reason in {"target", "stop"}
    )
    if real_exits > 0:
        return None

    counts: dict[str, int] = {}
    for o, _ in rows:
        counts[o.exit_reason or "unknown"] = counts.get(o.exit_reason or "unknown", 0) + 1

    return Alert(
        pattern_id="P3_NO_REAL_EXITS_CLUSTER",
        severity="MEDIUM",
        ticker=None,
        message=(
            f"Last {len(rows)} sniper closes in {lookback_days}d show 0 target "
            f"and 0 (real) stop exits. Distribution: {counts}. Worth checking "
            f"whether the simulator is short-circuiting before legitimate exits."
        ),
        evidence={
            "lookback_days": lookback_days,
            "sample_size": len(rows),
            "exit_reason_counts": counts,
        },
    )


async def run_invariants(
    session: AsyncSession,
    just_closed: list[tuple[Outcome, Signal]],
) -> list[Alert]:
    """Run all invariant checks. `just_closed` is the set of outcomes that
    transitioned from open → closed during this afternoon check."""
    alerts: list[Alert] = []
    for outcome, signal in just_closed:
        for fn in (detect_same_day_sniper_close, detect_phantom_pnl_fingerprint):
            alert = fn(outcome, signal)
            if alert is not None:
                alerts.append(alert)

    cluster_alert = await detect_zero_real_exits(session, clean_since=CLEAN_SINCE)
    if cluster_alert is not None:
        alerts.append(cluster_alert)

    return alerts


def format_alert_message(alerts: list[Alert]) -> str:
    """Build a Telegram-friendly summary message from a list of alerts."""
    if not alerts:
        return ""
    lines = ["⚠️ <b>Sniper Invariant Alarm</b>\n"]
    by_severity: dict[str, list[Alert]] = {"HIGH": [], "MEDIUM": []}
    for a in alerts:
        by_severity.setdefault(a.severity, []).append(a)
    for sev in ("HIGH", "MEDIUM"):
        if not by_severity.get(sev):
            continue
        lines.append(f"<b>{sev}</b>")
        for a in by_severity[sev]:
            tkr = f" [{a.ticker}]" if a.ticker else ""
            lines.append(f"  • <code>{a.pattern_id}</code>{tkr}: {a.message}")
        lines.append("")
    return "\n".join(lines).strip()
