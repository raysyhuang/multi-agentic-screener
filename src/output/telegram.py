"""Telegram alert bot — sends daily picks and outcome updates."""

from __future__ import annotations

import logging

from telegram import Bot
from telegram.constants import ParseMode

from src.config import get_settings

logger = logging.getLogger(__name__)


async def send_alert(message: str) -> bool:
    """Send a message to the configured Telegram chat."""
    settings = get_settings()
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        logger.warning("Telegram not configured, skipping alert")
        return False

    try:
        bot = Bot(token=settings.telegram_bot_token)
        await bot.send_message(
            chat_id=settings.telegram_chat_id,
            text=message,
            parse_mode=ParseMode.HTML,
        )
        return True
    except Exception as e:
        logger.error("Telegram send failed: %s", e)
        return False


def format_daily_alert(
    picks: list[dict],
    regime: str,
    run_date: str,
    validation_failed: bool = False,
    failed_checks: list[str] | None = None,
    key_risks: list[str] | None = None,
    execution_mode: str | None = None,
) -> str:
    """Format the daily picks into a Telegram-friendly HTML message."""
    mode_label = ""
    if execution_mode and execution_mode != "agentic_full":
        mode_label = f"Mode: <b>{execution_mode.upper()}</b>\n"

    if validation_failed:
        lines = [
            f"<b>📊 Daily Screener — {run_date}</b>",
            f"Regime: <b>{regime.upper()}</b>",
        ]
        if mode_label:
            lines.append(mode_label.strip())
        lines.extend([
            "",
            "<b>NoSilentPass — Validation FAILED</b>",
        ])
        for check in (failed_checks or []):
            lines.append(f"  - {check}")
        if key_risks:
            lines.append("")
            lines.append("<b>Key risks:</b>")
            for risk in key_risks:
                lines.append(f"  - {risk}")
        lines.append("")
        lines.append("All picks blocked by validation gate.")
        return "\n".join(lines)

    if not picks:
        return (
            f"<b>📊 Daily Screener — {run_date}</b>\n"
            f"Regime: <b>{regime.upper()}</b>\n"
            f"{mode_label}\n"
            f"No high-conviction picks today."
        )

    lines = [
        f"<b>📊 Daily Screener — {run_date}</b>",
        f"Regime: <b>{regime.upper()}</b>",
    ]
    if mode_label:
        lines.append(mode_label.strip())
    lines.extend([
        f"Picks: <b>{len(picks)}</b>",
        "",
    ])

    for i, pick in enumerate(picks, 1):
        ticker = pick.get("ticker", "???")
        direction = pick.get("direction", "LONG")
        entry = pick.get("entry_price", 0)
        stop = pick.get("stop_loss", 0)
        target = pick.get("target_1", 0)
        confidence = pick.get("confidence", 0)
        model = pick.get("signal_model", "unknown")
        thesis = pick.get("thesis", "")
        holding = pick.get("holding_period", 10)

        risk_reward = abs(target - entry) / abs(entry - stop) if abs(entry - stop) > 0 else 0

        lines.extend([
            f"<b>#{i} {ticker} — {direction}</b>",
            f"  Model: {model}",
            f"  Confidence: {confidence:.0f}/100",
            f"  Entry: ${entry:.2f}",
            f"  Stop: ${stop:.2f}",
            f"  Target: ${target:.2f}",
            f"  R:R: {risk_reward:.1f}:1",
            f"  Hold: {holding}d",
            f"  <i>{thesis[:150]}</i>" if thesis else "",
            "",
        ])

    # Fragility warnings
    if key_risks:
        lines.append("<b>Fragility warnings:</b>")
        for risk in key_risks:
            lines.append(f"  - {risk}")
        lines.append("")

    lines.append("<i>Paper trading mode — no real positions</i>")
    return "\n".join(lines)


def format_health_alert(state_changes: list) -> str:
    """Format health card state change alerts for Telegram.

    Args:
        state_changes: List of PositionHealthCard objects with state_changed=True.

    Returns:
        HTML-formatted alert string, or empty string if no changes.
    """
    if not state_changes:
        return ""

    state_emoji = {
        "on_track": "\u2705",  # ✅
        "watch": "\u26a0\ufe0f",  # ⚠️
        "exit": "\U0001f6a8",  # 🚨
    }

    lines = ["<b>\U0001f3e5 Position Health Update</b>", ""]

    for card in state_changes:
        prev = card.previous_state.value if card.previous_state else "new"
        curr = card.state.value
        prev_emoji = state_emoji.get(prev, "\u2753")
        curr_emoji = state_emoji.get(curr, "\u2753")

        lines.append(
            f"  {prev_emoji} {prev.upper()} \u2192 {curr_emoji} {curr.upper()}  "
            f"<b>{card.ticker}</b>"
        )
        vel_str = f" | Vel: {card.score_velocity:+.1f}pts/d" if card.score_velocity is not None else ""
        lines.append(
            f"    Score: {card.promising_score:.0f}/100 | "
            f"P&L: {card.pnl_pct:+.2f}% | "
            f"Day {card.days_held}/{card.expected_hold_days}"
            f"{vel_str}"
        )

        if card.invalidation_reason:
            lines.append(f"    <b>Invalidation:</b> {card.invalidation_reason}")

        # Show weakest component
        components = [
            card.trend_health,
            card.momentum_health,
            card.volume_confirmation,
            card.risk_integrity,
            card.regime_alignment,
        ]
        weakest = min(components, key=lambda c: c.score)
        lines.append(f"    Weakest: {weakest.name} ({weakest.score:.0f}/100)")
        lines.append("")

    return "\n".join(lines)


def format_near_miss_resolution_alert(resolved: list[dict]) -> str:
    """Format near-miss counterfactual resolution results for Telegram.

    Args:
        resolved: List of dicts with ticker, counterfactual_return, exit_reason, etc.

    Returns:
        HTML-formatted alert string, or empty string if no results.
    """
    if not resolved:
        return ""

    returns = [r["counterfactual_return"] for r in resolved]
    wins = sum(1 for r in returns if r > 0)
    total = len(returns)
    win_rate = wins / total if total > 0 else 0
    avg_return = sum(returns) / total if total > 0 else 0

    lines = [
        "<b>\U0001f50d Near-Miss Counterfactual Update</b>",
        "",
        f"Resolved: <b>{total}</b> | "
        f"Win Rate: <b>{win_rate:.0%}</b> | "
        f"Avg Return: <b>{avg_return:+.2f}%</b>",
        "",
    ]

    for r in resolved:
        ret = r["counterfactual_return"]
        emoji = "\u2705" if ret > 0 else "\u274c" if ret < 0 else "\u2796"
        exit_reason = r.get("exit_reason", "?")
        lines.append(
            f"  {emoji} {r['ticker']}: {ret:+.2f}% ({exit_reason})"
        )

    lines.append("")
    if win_rate > 0.5:
        lines.append(
            "<i>Note: filtering profitable trades — review rejection criteria</i>"
        )
    else:
        lines.append(
            "<i>Filters correctly blocked losers</i>"
        )

    return "\n".join(lines)


def format_outcome_alert(outcomes: list[dict]) -> str:
    """Format daily outcome update."""
    if not outcomes:
        return ""

    lines = ["<b>📈 Daily Outcome Update</b>", ""]

    for o in outcomes:
        ticker = o.get("ticker", "???")
        pnl = o.get("pnl_pct", 0)
        status = o.get("exit_reason", "open")
        emoji = "✅" if pnl > 0 else "❌" if pnl < 0 else "⏳"

        if status == "open":
            lines.append(f"  {emoji} {ticker}: {pnl:+.2f}% (still open)")
        else:
            lines.append(f"  {emoji} {ticker}: {pnl:+.2f}% (closed: {status})")

    return "\n".join(lines)
