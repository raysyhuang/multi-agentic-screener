"""Telegram alert bot — sends daily picks and outcome updates."""

from __future__ import annotations

import asyncio
import logging

from telegram import Bot
from telegram.constants import ParseMode

from src.config import get_settings

logger = logging.getLogger(__name__)

# Telegram message limit (API max is 4096, leave margin)
MAX_MESSAGE_LENGTH = 4000
SEND_RETRIES = 3
SEND_BACKOFF_BASE = 1.0  # seconds: 1s, 2s, 4s


def _split_message(message: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split a long message into chunks that fit Telegram's limit.

    Tries to split on newlines to preserve formatting.
    """
    if len(message) <= max_length:
        return [message]

    chunks: list[str] = []
    remaining = message
    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        # Find the last newline within the limit
        split_at = remaining.rfind("\n", 0, max_length)
        if split_at <= 0:
            # No good newline break — hard split at limit
            split_at = max_length

        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n")

    return chunks


async def send_alert(message: str) -> bool:
    """Send a message to the configured Telegram chat with retry and splitting."""
    settings = get_settings()
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        logger.warning("Telegram not configured, skipping alert")
        return False

    chunks = _split_message(message)
    bot = Bot(token=settings.telegram_bot_token)

    for i, chunk in enumerate(chunks):
        sent = False
        for attempt in range(1, SEND_RETRIES + 1):
            try:
                await bot.send_message(
                    chat_id=settings.telegram_chat_id,
                    text=chunk,
                    parse_mode=ParseMode.HTML,
                )
                sent = True
                break
            except Exception as e:
                delay = SEND_BACKOFF_BASE * (2 ** (attempt - 1))
                if attempt < SEND_RETRIES:
                    logger.warning(
                        "Telegram send failed (attempt %d/%d, chunk %d/%d): %s — retrying in %.1fs",
                        attempt, SEND_RETRIES, i + 1, len(chunks), e, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Telegram send failed after %d attempts (chunk %d/%d): %s",
                        SEND_RETRIES, i + 1, len(chunks), e,
                    )
        if not sent:
            return False

    return True


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


def format_cross_engine_alert(synthesis: dict, credibility: dict) -> str:
    """Format the cross-engine synthesis results for Telegram.

    Args:
        synthesis: Dict with keys: convergent_picks, unique_picks, portfolio,
                   regime_consensus, executive_summary, engines_reporting.
        credibility: Dict of engine_name -> {hit_rate, weight, resolved_picks}.

    Returns:
        HTML-formatted alert string.
    """
    regime = synthesis.get("regime_consensus", "unknown")
    engines_count = synthesis.get("engines_reporting", 0)
    summary = synthesis.get("executive_summary", "")
    convergent = synthesis.get("convergent_picks", [])
    portfolio = synthesis.get("portfolio", [])

    lines = [
        "<b>🔗 Cross-Engine Synthesis</b>",
        f"Regime Consensus: <b>{regime.upper()}</b>",
        f"Engines Reporting: <b>{engines_count}/4</b>",
        "",
    ]

    # Convergent picks
    if convergent:
        lines.append(f"<b>Convergent Picks ({len(convergent)}):</b>")
        for pick in convergent:
            ticker = pick.get("ticker", "?")
            engines = pick.get("engines", [])
            score = pick.get("combined_score", 0)
            lines.append(
                f"  <b>{ticker}</b> — {len(engines)} engines agree "
                f"(score: {score:.0f})"
            )
            if engines:
                lines.append(f"    Engines: {', '.join(engines)}")
        lines.append("")

    # Portfolio recommendation
    if portfolio:
        lines.append(f"<b>Portfolio ({len(portfolio)} positions):</b>")
        for pos in portfolio:
            ticker = pos.get("ticker", "?")
            weight = pos.get("weight_pct", 0)
            entry = pos.get("entry_price", 0)
            stop = pos.get("stop_loss", 0)
            target = pos.get("target_price", 0)
            source = pos.get("source", "")
            hold = pos.get("holding_period_days", 0)
            lines.append(
                f"  <b>{ticker}</b> ({weight:.0f}%) "
                f"— E: ${entry:.2f} / S: ${stop:.2f} / T: ${target:.2f} / {hold}d"
                f" [{source}]"
            )
        lines.append("")

    # Engine credibility snapshot
    if credibility:
        lines.append("<b>Engine Credibility:</b>")
        for engine_name, stats in credibility.items():
            hit_rate = stats.get("hit_rate", 0)
            weight = stats.get("weight", 1.0)
            picks = stats.get("resolved_picks", 0)
            lines.append(
                f"  {engine_name}: {hit_rate:.0%} hit rate, "
                f"{weight:.1f}x weight ({picks} picks)"
            )
        lines.append("")

    # Executive summary
    if summary:
        lines.append(f"<i>{summary}</i>")

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
