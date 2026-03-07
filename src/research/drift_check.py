"""Nightly model drift detection.

Compares recent live pick performance against historical backtest baseline.
Alerts if the model's rolling metrics have degraded significantly.

Usage:
    python -m src.research.drift_check
    python -m src.research.drift_check --lookback 30 --alert
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from datetime import date, timedelta

from src.backtest.metrics import compute_metrics
from src.db.models import EnginePickOutcome
from src.db.session import get_session

logger = logging.getLogger(__name__)

# Baseline from S&P 500 backtest (24,670 trades, optimized params + trailing stop)
BASELINE = {
    "win_rate": 0.716,
    "avg_return_pct": 1.05,
    "sharpe_ratio": 2.47,
    "profit_factor": 2.71,
}

# Alert if metric drops below this fraction of baseline
DRIFT_THRESHOLD = 0.5  # 50% degradation = alert


@dataclass
class DriftReport:
    lookback_days: int
    total_resolved: int
    live_win_rate: float
    live_avg_return: float
    live_sharpe: float
    live_profit_factor: float
    alerts: list[str]


async def compute_drift(lookback_days: int = 30) -> DriftReport:
    """Compare recent live pick outcomes against backtest baseline."""
    cutoff = date.today() - timedelta(days=lookback_days)

    async with get_session() as session:
        from sqlalchemy import select
        stmt = select(EnginePickOutcome).where(
            EnginePickOutcome.outcome_resolved.is_(True),
            EnginePickOutcome.run_date >= cutoff,
        )
        result = await session.execute(stmt)
        outcomes = result.scalars().all()

    if not outcomes:
        return DriftReport(
            lookback_days=lookback_days, total_resolved=0,
            live_win_rate=0, live_avg_return=0, live_sharpe=0, live_profit_factor=0,
            alerts=["No resolved outcomes in lookback window"],
        )

    returns = [float(o.actual_return_pct) for o in outcomes if o.actual_return_pct is not None]
    hits = sum(1 for o in outcomes if o.hit_target)
    total = len(outcomes)

    if not returns:
        return DriftReport(
            lookback_days=lookback_days, total_resolved=total,
            live_win_rate=0, live_avg_return=0, live_sharpe=0, live_profit_factor=0,
            alerts=["No return data available"],
        )

    metrics = compute_metrics(returns)

    live_wr = hits / total if total > 0 else 0
    alerts = []

    # Check each metric against baseline
    checks = [
        ("win_rate", live_wr, BASELINE["win_rate"]),
        ("avg_return", metrics.avg_return_pct, BASELINE["avg_return_pct"]),
        ("profit_factor", metrics.profit_factor, BASELINE["profit_factor"]),
    ]

    for name, live_val, baseline_val in checks:
        threshold = baseline_val * DRIFT_THRESHOLD
        if live_val < threshold:
            alerts.append(
                f"{name}: live={live_val:.3f} < threshold={threshold:.3f} "
                f"(baseline={baseline_val:.3f}, {DRIFT_THRESHOLD:.0%} degradation)"
            )

    # Special check: negative expectancy = model is losing money
    if metrics.expectancy < 0:
        alerts.append(f"NEGATIVE EXPECTANCY: {metrics.expectancy:+.3f}% per trade")

    # Minimum sample size warning
    if total < 20:
        alerts.append(f"Low sample size: {total} trades (need 20+ for significance)")

    return DriftReport(
        lookback_days=lookback_days,
        total_resolved=total,
        live_win_rate=round(live_wr, 4),
        live_avg_return=round(metrics.avg_return_pct, 4),
        live_sharpe=round(metrics.sharpe_ratio, 4),
        live_profit_factor=round(metrics.profit_factor, 4),
        alerts=alerts,
    )


def format_drift_report(report: DriftReport) -> str:
    """Format drift report as text."""
    lines = [
        f"\n{'='*50}",
        f"  MODEL DRIFT CHECK ({report.lookback_days}d lookback)",
        f"{'='*50}",
        f"  Resolved trades: {report.total_resolved}",
        "",
        f"  {'Metric':<20} {'Live':>10} {'Baseline':>10} {'Status':>10}",
        f"  {'-'*50}",
    ]

    checks = [
        ("Win Rate", f"{report.live_win_rate:.1%}", f"{BASELINE['win_rate']:.1%}"),
        ("Avg Return", f"{report.live_avg_return:+.3f}%", f"{BASELINE['avg_return_pct']:+.3f}%"),
        ("Sharpe", f"{report.live_sharpe:.2f}", f"{BASELINE['sharpe_ratio']:.2f}"),
        ("Profit Factor", f"{report.live_profit_factor:.2f}", f"{BASELINE['profit_factor']:.2f}"),
    ]

    for name, live, baseline in checks:
        lines.append(f"  {name:<20} {live:>10} {baseline:>10}")

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

    asyncio.run(_async_main(args.lookback, args.alert))


if __name__ == "__main__":
    main()
