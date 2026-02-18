"""Telegram alert bot — sends daily picks and outcome updates.

Uses clean visual formatting with Unicode bars and clear section hierarchy.
All messages use Telegram HTML parse mode.
"""

from __future__ import annotations

import asyncio
import logging
from html import escape as _html_escape

from telegram import Bot
from telegram.constants import ParseMode

from src.config import get_settings

logger = logging.getLogger(__name__)


def _esc(text: str | None) -> str:
    """Escape HTML special characters for Telegram parse_mode=HTML."""
    if not text:
        return ""
    return _html_escape(str(text), quote=False)

# Telegram message limit (API max is 4096, leave margin)
MAX_MESSAGE_LENGTH = 4000
SEND_RETRIES = 3
SEND_BACKOFF_BASE = 1.0  # seconds: 1s, 2s, 4s


# ---------------------------------------------------------------------------
# Unicode visual helpers
# ---------------------------------------------------------------------------

def _bar(value: float, max_val: float = 100, width: int = 10) -> str:
    """Render a Unicode progress bar. value/max_val scaled to width chars."""
    if max_val <= 0:
        return "\u2591" * width
    ratio = max(0, min(1, value / max_val))
    filled = round(ratio * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def _pnl_emoji(pnl: float) -> str:
    if pnl > 5:
        return "\U0001f525"  # fire
    if pnl > 0:
        return "\u2705"
    if pnl == 0:
        return "\u2796"
    if pnl > -3:
        return "\u26a0\ufe0f"
    return "\u274c"


def _regime_emoji(regime: str) -> str:
    r = regime.lower()
    if r == "bull":
        return "\U0001f7e2"  # green circle
    if r == "bear":
        return "\U0001f534"  # red circle
    return "\U0001f7e1"  # yellow circle


def _section_line() -> str:
    return "\u2500" * 28


# ---------------------------------------------------------------------------
# Message splitting & sending
# ---------------------------------------------------------------------------

def _split_message(message: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split a long message into chunks that fit Telegram's limit."""
    if len(message) <= max_length:
        return [message]

    chunks: list[str] = []
    remaining = message
    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, max_length)
        if split_at <= 0:
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


# ---------------------------------------------------------------------------
# Daily Screener Alert
# ---------------------------------------------------------------------------

def format_daily_alert(
    picks: list[dict],
    regime: str,
    run_date: str,
    validation_failed: bool = False,
    failed_checks: list[str] | None = None,
    key_risks: list[str] | None = None,
    execution_mode: str | None = None,
) -> str:
    """Format the daily picks into a clean, scannable Telegram message."""
    regime_dot = _regime_emoji(regime)

    if validation_failed:
        lines = [
            f"<b>\U0001f6d1 Daily Screener \u2014 {run_date}</b>",
            "",
            f"{regime_dot} Regime: <b>{regime.upper()}</b>",
            "",
            "\u274c <b>Validation FAILED</b>",
            "",
        ]
        for check in (failed_checks or []):
            lines.append(f"   \u2022 {check}")
        if key_risks:
            lines.append("")
            for risk in key_risks:
                lines.append(f"   \u26a0\ufe0f {risk}")
        lines.extend(["", "<i>All picks blocked by validation gate.</i>"])
        return "\n".join(lines)

    if not picks:
        mode_line = ""
        if execution_mode and execution_mode != "agentic_full":
            mode_line = f"   Mode: {execution_mode.upper()}\n"
        return (
            f"<b>\U0001f4ca Daily Screener \u2014 {run_date}</b>\n"
            f"\n"
            f"{regime_dot} Regime: <b>{regime.upper()}</b>\n"
            f"{mode_line}"
            f"\n"
            f"No high-conviction picks today."
        )

    mode_tag = ""
    if execution_mode and execution_mode != "agentic_full":
        mode_tag = f"   Mode: {execution_mode.upper()}\n"

    lines = [
        f"<b>\U0001f4ca Daily Screener \u2014 {run_date}</b>",
        "",
        f"{regime_dot} Regime: <b>{regime.upper()}</b>   |   Picks: <b>{len(picks)}</b>",
    ]
    if mode_tag:
        lines.append(mode_tag.rstrip())
    lines.append("")

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

        risk_pct = abs(entry - stop) / entry * 100 if entry > 0 else 0
        reward_pct = abs(target - entry) / entry * 100 if entry > 0 else 0
        rr = reward_pct / risk_pct if risk_pct > 0 else 0

        dir_arrow = "\u25b2" if direction == "LONG" else "\u25bc"
        conf_bar = _bar(confidence, 100, 10)

        lines.extend([
            f"<b>{dir_arrow} {_esc(ticker)}</b>  <code>{_esc(model)}</code>",
            f"   {conf_bar} {confidence:.0f}/100",
            f"   Entry <b>${entry:.2f}</b>  \u2192  Target <b>${target:.2f}</b> (+{reward_pct:.1f}%)",
            f"   Stop  <b>${stop:.2f}</b>  ({risk_pct:.1f}% risk)   R:R <b>{rr:.1f}:1</b>   {holding}d",
        ])
        if thesis:
            lines.append(f"   <i>{_esc(thesis[:120])}</i>")
        lines.append("")

    if key_risks:
        lines.append("\u26a0\ufe0f <b>Risks</b>")
        for risk in key_risks:
            lines.append(f"   \u2022 {risk}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cross-Engine Synthesis Alert
# ---------------------------------------------------------------------------

def _extract_regime_label(regime_text: str) -> str:
    """Extract a short regime label (bull/bear/choppy/unknown) from LLM text."""
    if not regime_text:
        return "unknown"
    # Short labels pass through directly
    if len(regime_text) <= 20:
        lower = regime_text.strip().lower()
        for label in ("bull", "bear", "choppy", "unknown"):
            if label in lower:
                return label
        return regime_text.strip()

    # For long text, count bear/bull mentions to find dominant signal
    lower = regime_text.lower()
    bear_count = lower.count("bear")
    bull_count = lower.count("bull")

    if "choppy" in lower or "sideways" in lower or "range-bound" in lower:
        return "choppy"
    if bear_count > bull_count:
        return "bear"
    if bull_count > bear_count:
        return "bull"
    if "ambiguous" in lower or "unknown" in lower or "uncertain" in lower:
        return "unknown"
    return "unknown"


def format_cross_engine_alert(synthesis: dict, credibility: dict) -> str:
    """Format multi-engine synthesis with clear visual hierarchy."""
    regime_raw = synthesis.get("regime_consensus", "unknown")
    regime = _extract_regime_label(regime_raw)
    engines_count = synthesis.get("engines_reporting", 0)
    summary = synthesis.get("executive_summary", "")
    convergent = synthesis.get("convergent_picks", [])
    portfolio = synthesis.get("portfolio", [])

    regime_dot = _regime_emoji(regime)
    engine_bar = _bar(engines_count, 4, 4)

    lines = [
        "<b>\U0001f517 Cross-Engine Synthesis</b>",
        "",
        f"{regime_dot} Regime: <b>{regime.upper()}</b>",
        f"   Engines: {engine_bar} <b>{engines_count}/4</b> reporting",
        "",
    ]

    # Convergent picks — the high-conviction signals
    if convergent:
        lines.append(f"{_section_line()}")
        lines.append(f"\U0001f91d <b>Convergent Picks</b> ({len(convergent)} tickers)")
        lines.append("")
        for pick in convergent:
            ticker = pick.get("ticker", "?")
            engines = pick.get("engines", [])
            score = pick.get("combined_score", 0)
            score_bar = _bar(score, 200, 8)
            lines.append(f"   <b>{_esc(ticker)}</b>  {score_bar} score {score:.0f}")
            lines.append(f"   {len(engines)} engines: {_esc(', '.join(engines))}")
            lines.append("")

    # Portfolio positions — the final recommendation
    if portfolio:
        lines.append(f"{_section_line()}")
        lines.append(f"\U0001f4bc <b>Portfolio</b> ({len(portfolio)} positions)")
        lines.append("")
        for pos in portfolio:
            ticker = pos.get("ticker", "?")
            weight = pos.get("weight_pct", 0)
            entry = pos.get("entry_price", 0)
            stop = pos.get("stop_loss", 0)
            target = pos.get("target_price", 0)
            hold = pos.get("holding_period_days", 0)
            source = pos.get("source", "")
            guardian_adj = pos.get("guardian_adjusted", False)

            adj_tag = " \u21e9" if guardian_adj else ""
            risk_pct = abs(entry - stop) / entry * 100 if entry > 0 else 0
            reward_pct = abs(target - entry) / entry * 100 if entry > 0 else 0

            lines.append(
                f"   <b>{_esc(ticker)}</b>  {weight:.0f}%{adj_tag}  |  "
                f"${entry:.2f} \u2192 ${target:.2f} (+{reward_pct:.1f}%)"
            )
            lines.append(
                f"   Stop ${stop:.2f} ({risk_pct:.1f}%)  |  {hold}d  [{_esc(source)}]"
            )
            lines.append("")

    # Engine credibility — compact table
    if credibility:
        lines.append(f"{_section_line()}")
        lines.append("\U0001f3af <b>Engine Credibility</b>")
        lines.append("")
        for name, stats in credibility.items():
            hr = stats.get("hit_rate", 0)
            w = stats.get("weight", 1.0)
            n = stats.get("resolved_picks", 0)
            hr_bar = _bar(hr * 100, 100, 6)
            short_name = name.replace("_", " ").title()
            lines.append(
                f"   {hr_bar} {_esc(short_name)}: <b>{hr:.0%}</b> hit, "
                f"{w:.2f}x wt ({n})"
            )
        lines.append("")

    # Executive summary
    if summary:
        lines.append(f"{_section_line()}")
        lines.append(f"<i>{_esc(summary)}</i>")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Position Health Alert
# ---------------------------------------------------------------------------

def format_health_alert(state_changes: list) -> str:
    """Format health card state change alerts."""
    if not state_changes:
        return ""

    state_emoji = {
        "on_track": "\u2705",
        "watch": "\u26a0\ufe0f",
        "exit": "\U0001f6a8",
    }

    lines = ["<b>\U0001f3e5 Position Health Update</b>", ""]

    for card in state_changes:
        prev = card.previous_state.value if card.previous_state else "new"
        curr = card.state.value
        prev_emoji = state_emoji.get(prev, "\u2753")
        curr_emoji = state_emoji.get(curr, "\u2753")
        pnl_icon = _pnl_emoji(card.pnl_pct)
        score_bar = _bar(card.promising_score, 100, 8)

        lines.append(
            f"{prev_emoji} \u2192 {curr_emoji}  <b>{_esc(card.ticker)}</b>"
        )
        lines.append(
            f"   {score_bar} {card.promising_score:.0f}/100  |  "
            f"{pnl_icon} {card.pnl_pct:+.2f}%  |  "
            f"Day {card.days_held}/{card.expected_hold_days}"
        )

        if card.score_velocity is not None:
            vel_dir = "\u2191" if card.score_velocity > 0 else "\u2193"
            lines.append(f"   Velocity: {vel_dir} {card.score_velocity:+.1f} pts/d")

        if card.invalidation_reason:
            lines.append(f"   \u274c Invalidation: {_esc(card.invalidation_reason)}")

        components = [
            card.trend_health,
            card.momentum_health,
            card.volume_confirmation,
            card.risk_integrity,
            card.regime_alignment,
        ]
        weakest = min(components, key=lambda c: c.score)
        lines.append(f"   Weakest: {weakest.name} ({weakest.score:.0f}/100)")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Near-Miss Counterfactual Alert
# ---------------------------------------------------------------------------

def format_near_miss_resolution_alert(resolved: list[dict]) -> str:
    """Format near-miss counterfactual resolution results."""
    if not resolved:
        return ""

    returns = [r["counterfactual_return"] for r in resolved]
    wins = sum(1 for r in returns if r > 0)
    total = len(returns)
    win_rate = wins / total if total > 0 else 0
    avg_return = sum(returns) / total if total > 0 else 0

    wr_bar = _bar(win_rate * 100, 100, 8)

    lines = [
        "<b>\U0001f50d Near-Miss Counterfactual</b>",
        "",
        f"   Resolved: <b>{total}</b>   WR: {wr_bar} <b>{win_rate:.0%}</b>   "
        f"Avg: <b>{avg_return:+.2f}%</b>",
        "",
    ]

    for r in resolved:
        ret = r["counterfactual_return"]
        emoji = _pnl_emoji(ret)
        exit_reason = r.get("exit_reason", "?")
        lines.append(
            f"   {emoji} <b>{_esc(r['ticker'])}</b>: {ret:+.2f}% ({_esc(exit_reason)})"
        )

    lines.append("")
    if win_rate > 0.5:
        lines.append(
            "<i>Note: filters blocked profitable trades \u2014 review criteria</i>"
        )
    else:
        lines.append("<i>Filters correctly blocked losers</i>")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Outcome Alert
# ---------------------------------------------------------------------------

def format_outcome_alert(outcomes: list[dict]) -> str:
    """Format daily outcome update."""
    if not outcomes:
        return ""

    total_pnl = sum(o.get("pnl_pct", 0) for o in outcomes)
    wins = sum(1 for o in outcomes if (o.get("pnl_pct", 0) or 0) > 0)

    lines = [
        "<b>\U0001f4c8 Daily Outcome Update</b>",
        "",
        f"   Positions: <b>{len(outcomes)}</b>   "
        f"Wins: <b>{wins}/{len(outcomes)}</b>   "
        f"Net: <b>{total_pnl:+.2f}%</b>",
        "",
    ]

    for o in outcomes:
        ticker = o.get("ticker", "???")
        pnl = o.get("pnl_pct", 0)
        status = o.get("exit_reason", "open")
        emoji = _pnl_emoji(pnl)

        if status == "open":
            lines.append(f"   {emoji} <b>{_esc(ticker)}</b>: {pnl:+.2f}% (open)")
        else:
            lines.append(f"   {emoji} <b>{_esc(ticker)}</b>: {pnl:+.2f}% ({_esc(status)})")

    return "\n".join(lines)
