"""Nightly model drift detection.

Compares recent LIVE per-trade performance against each stream's frozen honest
baseline and alerts when a stream degrades materially.

History worth knowing before editing this file: until 2026-08 it queried
`EnginePickOutcome` — the table belonging to the three external engines that were
scaled to zero in 2026-03 — against a baseline of 71.6% WR / 2.47 Sharpe, the
retired 24,670-trade optimistic number the truth-matrix work discredited. It
therefore returned "no resolved outcomes" and no-op'd every night while the
actual live streams drifted unwatched (PEAD's paper cohort ran alpha -1.26/-1.78
against claimed +1.8/+2.42 with nothing raising a hand).

Baselines are per stream and deliberately modest — they are the post-truth-matrix
numbers, not the retired fantasies. Keep them in sync with the dashboard's
`BASELINES` in scripts/export_dashboard_data.py.

Usage:
    python -m src.research.drift_check
    python -m src.research.drift_check --lookback 30 --alert
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import statistics as st
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta

from src.db.models import Outcome, Signal
from src.db.session import get_session

logger = logging.getLogger(__name__)

# Honest per-trade expectation bands, mirroring the dashboard's BASELINES.
# `avg` is the per-trade percentage the stream is expected to earn.
BASELINES: dict[str, dict] = {
    "sniper|mas_official": {"label": "Sniper (official)", "wr": 0.543, "avg": 0.54},
    "mean_reversion|mas_official": {"label": "MR (official)", "wr": 0.522, "avg": 0.46},
    "mean_reversion|mr_manual_sleeve": {"label": "MR (manual sleeve)", "wr": 0.493, "avg": -0.01},
    "pead|pead_paper": {"label": "PEAD (paper)", "wr": 0.57, "avg": 1.80},
    "pead|pead_neglected": {"label": "PEAD (neglected-beat)", "wr": 0.58, "avg": 2.42},
}

# A stream alerts when its realized per-trade average falls this far below its
# baseline. Deliberately loose: the gate's proven failure mode across this
# project is over-reacting to small-n noise, and this monitor ALERTS rather than
# blocks, so it should fire on genuine degradation only.
DRIFT_SHORTFALL_PCT = 0.5      # realized < 50% of baseline avg
MIN_TRADES_TO_JUDGE = 15       # below this, report but never alert


@dataclass
class StreamDrift:
    stream: str
    label: str
    n: int
    live_win_rate: float
    live_avg: float
    baseline_avg: float
    baseline_wr: float
    alerts: list[str] = field(default_factory=list)


@dataclass
class DriftReport:
    lookback_days: int
    total_resolved: int
    streams: list[StreamDrift]
    alerts: list[str]


async def compute_drift(lookback_days: int = 30) -> DriftReport:
    """Compare recent LIVE closed trades, per stream, against frozen baselines."""
    cutoff = date.today() - timedelta(days=lookback_days)

    async with get_session() as session:
        from sqlalchemy import select
        stmt = (
            select(Outcome, Signal)
            .join(Signal, Outcome.signal_id == Signal.id)
            .where(
                Outcome.still_open == False,  # noqa: E712
                # Excludes gap-rejected AND shadow-booked (validation_blocked)
                # rows — drift is about what the book actually traded.
                Outcome.skip_reason.is_(None),
                Outcome.exit_date >= cutoff,
            )
        )
        rows = (await session.execute(stmt)).all()

    by_stream: dict[str, list[float]] = {}
    for outcome, signal in rows:
        if outcome.pnl_pct is None:
            continue
        key = f"{signal.signal_model}|{signal.signal_source}"
        by_stream.setdefault(key, []).append(float(outcome.pnl_pct))

    streams: list[StreamDrift] = []
    alerts: list[str] = []
    for key, returns in sorted(by_stream.items()):
        base = BASELINES.get(key)
        if base is None:
            logger.info("drift: no baseline for stream %s (n=%d) — reporting only",
                        key, len(returns))
            base = {"label": key, "wr": 0.0, "avg": 0.0}
        n = len(returns)
        wr = sum(1 for r in returns if r > 0) / n
        avg = st.mean(returns)
        sd = StreamDrift(stream=key, label=base["label"], n=n,
                         live_win_rate=round(wr, 4), live_avg=round(avg, 4),
                         baseline_avg=base["avg"], baseline_wr=base["wr"])

        if n >= MIN_TRADES_TO_JUDGE and base["avg"] > 0:
            floor = base["avg"] * DRIFT_SHORTFALL_PCT
            if avg < floor:
                sd.alerts.append(
                    f"{base['label']}: {avg:+.3f}%/trade vs baseline "
                    f"{base['avg']:+.3f}% (floor {floor:+.3f}%, n={n})"
                )
            if avg < 0:
                sd.alerts.append(
                    f"{base['label']}: NEGATIVE expectancy {avg:+.3f}%/trade (n={n})"
                )
        streams.append(sd)
        alerts.extend(sd.alerts)

    if not rows:
        alerts.append(f"No closed live trades in the last {lookback_days}d")

    return DriftReport(lookback_days=lookback_days, total_resolved=len(rows),
                       streams=streams, alerts=alerts)


def format_drift_report(report: DriftReport) -> str:
    """Format drift report as text."""
    lines = [
        f"\n{'=' * 62}",
        f"  MODEL DRIFT CHECK ({report.lookback_days}d lookback, live streams)",
        f"{'=' * 62}",
        f"  Closed trades: {report.total_resolved}",
        "",
        f"  {'Stream':<24}{'n':>5}{'live avg':>11}{'baseline':>11}{'live WR':>9}",
        f"  {'-' * 60}",
    ]
    for s in report.streams:
        flag = " !" if s.alerts else ""
        lines.append(
            f"  {s.label:<24}{s.n:>5}{s.live_avg:>+10.3f}%{s.baseline_avg:>+10.3f}%"
            f"{s.live_win_rate:>8.1%}{flag}"
        )

    if report.alerts:
        lines.append(f"\n  ALERTS ({len(report.alerts)}):")
        for alert in report.alerts:
            lines.append(f"    ! {alert}")
    else:
        lines.append("\n  No drift detected.")
    return "\n".join(lines)


async def _async_main(lookback: int, send_alert: bool) -> None:
    from src.db.session import init_db
    await init_db()

    report = await compute_drift(lookback)
    print(format_drift_report(report))

    if send_alert and report.alerts:
        try:
            from src.output.telegram import send_telegram_message
            text = f"MODEL DRIFT ALERT\n{format_drift_report(report)}"
            await send_telegram_message(text)
            print("Alert sent to Telegram")
        except Exception as e:
            print(f"Failed to send alert: {e}")

    if report.alerts:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Model drift detection")
    parser.add_argument("--lookback", type=int, default=30, help="Lookback days")
    parser.add_argument("--alert", action="store_true", help="Send Telegram alert on drift")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(_async_main(args.lookback, args.alert))


if __name__ == "__main__":
    main()
