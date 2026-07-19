"""Telegram alert bot — sends daily picks and outcome updates.

Uses clean visual formatting with Unicode bars and clear section hierarchy.
All messages use Telegram HTML parse mode.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from html import escape as _html_escape

import sqlalchemy
from telegram import Bot
from telegram.constants import ParseMode

from src.config import get_settings

logger = logging.getLogger(__name__)


def _esc(text: str | None) -> str:
    """Escape HTML special characters for Telegram parse_mode=HTML."""
    if not text:
        return ""
    return _html_escape(str(text), quote=False)


def _prefix() -> str:
    return f"[{get_settings().telegram_alert_prefix}]"

# Engine display names — avoids .title() mangling acronyms
_ENGINE_DISPLAY = {"gemini_stst": "Gemini STST", "koocore_d": "KooCore-D", "mas_quant_screener": "MAS-Quant-Screener"}

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
                result = await bot.send_message(
                    chat_id=settings.telegram_chat_id,
                    text=chunk,
                    parse_mode=ParseMode.HTML,
                )
                sent = True
                # Log to DB (best-effort, don't fail the send)
                msg_id = result.message_id if result else None
                await _log_to_db(
                    settings.telegram_source_id, chunk, settings.telegram_chat_id, msg_id,
                )
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


async def _log_to_db(
    source: str,
    message_text: str,
    chat_id: str | None = None,
    message_id: int | None = None,
) -> None:
    """Best-effort log of a sent Telegram message to the database."""
    try:
        from src.db.session import get_session
        from src.db.models import TelegramLog

        async with get_session() as session:
            session.add(TelegramLog(
                source=source,
                message_text=message_text,
                chat_id=str(chat_id) if chat_id else None,
                message_id=message_id,
            ))
    except Exception as e:
        logger.debug("Failed to log telegram message to DB: %s", e)


def _render_scorecard(model_scorecard: dict[str, dict]) -> list[str]:
    """Render model scorecard lines (reused in multiple alert paths)."""
    lines: list[str] = []
    lines.append(_section_line())
    lines.append("\U0001f4ca <b>Model Scorecard (30d)</b>")
    for model_name, stats in model_scorecard.items():
        trades = stats.get("trades", 0)
        status = stats.get("status")
        if trades == 0:
            label = status or "no closed trades"
            open_pos = stats.get("open_positions", 0)
            if open_pos > 0:
                label = f"{open_pos} open, no closed trades"
            lines.append(f"   <b>{_esc(model_name)}</b>: {_esc(label)}")
            continue
        wr = stats.get("win_rate", 0)
        avg_pnl = stats.get("avg_pnl", 0)
        pf = stats.get("profit_factor")
        open_pos = stats.get("open_positions", 0)

        parts = [
            f"{trades} trades",
            f"{wr:.0%} WR",
            f"{avg_pnl:+.2f}% avg",
        ]
        if pf is not None:
            parts.append(f"PF {pf:.1f}")
        if open_pos > 0:
            parts.append(f"{open_pos} open")

        lines.append(f"   <b>{_esc(model_name)}</b>: {' | '.join(parts)}")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Daily Screener Alert
# ---------------------------------------------------------------------------

def _render_manual_sleeve_section(
    sleeve_picks: list[dict],
) -> list[str]:
    """Render the MR Manual Sleeve section appended to the daily alert.

    Each pick mirrors the MAS layout but adds an overlap tag so it's clear
    whether the sleeve duplicates an MAS pick or surfaces one that
    cross-model ranking suppressed.
    """
    lines: list[str] = []
    lines.append(_section_line())
    lines.append(
        f"\U0001f9ea <b>MR Manual Sleeve</b> ({len(sleeve_picks)} pick"
        f"{'' if len(sleeve_picks) == 1 else 's'})"
    )
    lines.append(
        "<i>Daily mean-reversion sleeve for manual trading. "
        "Tracked separately from MAS.</i>"
    )
    lines.append("")

    if not sleeve_picks:
        lines.append("   No qualified MR picks today.")
        lines.append("")
        return lines

    for pick in sleeve_picks:
        ticker = pick.get("ticker", "???")
        direction = pick.get("direction", "LONG")
        entry = pick.get("entry_price", 0)
        stop = pick.get("stop_loss", 0)
        target = pick.get("target_1", 0)
        confidence = pick.get("confidence", 0)
        holding = pick.get("holding_period", 3)
        also_in_mas = pick.get("also_in_mas", False)
        suppressed = pick.get("suppressed_by_cross_model_ranking", False)

        if also_in_mas:
            overlap_tag = "\U0001f501 also in MAS today"
        elif suppressed:
            overlap_tag = "⚡ surfaced by sleeve (suppressed by cross-model rank)"
        else:
            overlap_tag = "\U0001f4cd sleeve-only"

        risk_pct = abs(entry - stop) / entry * 100 if entry > 0 else 0
        reward_pct = abs(target - entry) / entry * 100 if entry > 0 else 0
        rr = reward_pct / risk_pct if risk_pct > 0 else 0
        dir_arrow = "▲" if direction == "LONG" else "▼"
        conf_bar = _bar(confidence, 100, 10)

        lines.extend([
            f"<b>{dir_arrow} {_esc(ticker)}</b>  <code>mean_reversion</code>",
            f"   {overlap_tag}",
            f"   {conf_bar} {confidence:.0f}/100",
            f"   Entry <b>${entry:.2f}</b>  →  Target <b>${target:.2f}</b> (+{reward_pct:.1f}%)",
            f"   Stop  <b>${stop:.2f}</b>  ({risk_pct:.1f}% risk)   R:R <b>{rr:.1f}:1</b>   {holding}d",
        ])
        lines.append("")

    return lines


def format_daily_alert(
    picks: list[dict],
    regime: str,
    run_date: str,
    validation_failed: bool = False,
    failed_checks: list[str] | None = None,
    key_risks: list[str] | None = None,
    execution_mode: str | None = None,
    model_scorecard: dict[str, dict] | None = None,
    manual_sleeve_picks: list[dict] | None = None,
) -> str:
    """Format the daily picks into a clean, scannable Telegram message.

    ``manual_sleeve_picks`` is the parallel MR Manual Sleeve stream. It is
    rendered in its own labeled section after the MAS picks (or directly
    after the validation/empty-state body) so MAS tracking stays clean.
    """
    regime_dot = _regime_emoji(regime)

    if validation_failed:
        mode_line = ""
        if execution_mode and execution_mode != "agentic_full":
            mode_line = f"   Mode: {execution_mode.upper()}\n"
        lines = [
            f"<b>{_prefix()} \U0001f6d1 Daily Screener \u2014 {run_date}</b>",
            "",
            f"{regime_dot} Regime: <b>{regime.upper()}</b>",
        ]
        if mode_line:
            lines.append(mode_line.rstrip())
        lines.extend([
            "",
            "\u274c <b>Validation FAILED</b>",
            "",
        ])
        for check in (failed_checks or []):
            lines.append(f"   \u2022 {check}")
        if key_risks:
            lines.append("")
            for risk in key_risks:
                lines.append(f"   \u26a0\ufe0f {risk}")
        lines.extend(["", "<i>All picks blocked by validation gate.</i>"])

        if manual_sleeve_picks is not None:
            lines.append("")
            lines.extend(_render_manual_sleeve_section(manual_sleeve_picks))

        if model_scorecard:
            lines.append("")
            lines.extend(_render_scorecard(model_scorecard))

        return "\n".join(lines)

    if not picks:
        mode_line = ""
        if execution_mode and execution_mode != "agentic_full":
            mode_line = f"   Mode: {execution_mode.upper()}\n"
        lines = [
            f"<b>{_prefix()} \U0001f4ca Daily Screener \u2014 {run_date}</b>",
            "",
            f"{regime_dot} Regime: <b>{regime.upper()}</b>",
        ]
        if mode_line:
            lines.append(mode_line.rstrip())
        lines.extend(["", "No high-conviction picks today."])

        if key_risks:
            lines.append("")
            lines.append("\u26a0\ufe0f <b>Why official picks were blocked</b>")
            for risk in key_risks:
                lines.append(f"   \u2022 {_esc(risk)}")

        if manual_sleeve_picks is not None:
            lines.append("")
            lines.extend(_render_manual_sleeve_section(manual_sleeve_picks))

        if model_scorecard:
            lines.append("")
            lines.extend(_render_scorecard(model_scorecard))

        return "\n".join(lines)

    mode_tag = ""
    if execution_mode and execution_mode != "agentic_full":
        mode_tag = f"   Mode: {execution_mode.upper()}\n"

    lines = [
        f"<b>{_prefix()} \U0001f4ca Daily Screener \u2014 {run_date}</b>",
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

    if manual_sleeve_picks is not None:
        lines.extend(_render_manual_sleeve_section(manual_sleeve_picks))

    # Model scorecard (appended if provided)
    if model_scorecard:
        lines.extend(_render_scorecard(model_scorecard))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model Scorecard (30-day DB query)
# ---------------------------------------------------------------------------

async def get_model_scorecard(
    days: int = 30,
    execution_mode: str | None = None,
    signal_source: str | None = "mas_official",
) -> dict[str, dict]:
    """Query DB for closed outcomes by signal_model over the last N days.

    Returns dict like: {"mean_reversion": {"trades": 12, "win_rate": 0.71, ...}, ...}

    ``signal_source`` defaults to ``mas_official`` so the headline scorecard
    is never inflated by parallel sleeve outcomes. Pass ``None`` to include
    every source.
    """
    try:
        from src.db.session import get_session
        from src.db.models import DailyRun, Outcome, Signal
        from sqlalchemy import select, and_, func

        cutoff = date.today() - timedelta(days=days)

        async with get_session() as session:
            # Closed outcomes with their signal model
            closed_query = (
                select(
                    Signal.signal_model,
                    func.count().label("trades"),
                    func.avg(Outcome.pnl_pct).label("avg_pnl"),
                    func.sum(
                        func.cast(Outcome.pnl_pct > 0, sqlalchemy.Integer)
                    ).label("wins"),
                )
                .join(Signal, Outcome.signal_id == Signal.id)
                .where(
                    and_(
                        Outcome.still_open == False,  # noqa: E712
                        Outcome.skip_reason.is_(None),
                        Signal.run_date >= cutoff,
                    )
                )
                .group_by(Signal.signal_model)
            )
            open_query = (
                select(Signal.signal_model, func.count().label("open_count"))
                .join(Outcome, Outcome.signal_id == Signal.id)
                .where(
                    and_(
                        Outcome.still_open == True,  # noqa: E712
                        Outcome.skip_reason.is_(None),
                    )
                )
                .group_by(Signal.signal_model)
            )

            if signal_source is not None:
                closed_query = closed_query.where(Signal.signal_source == signal_source)
                open_query = open_query.where(Signal.signal_source == signal_source)

            if execution_mode:
                closed_query = (
                    closed_query
                    .join(DailyRun, DailyRun.run_date == Signal.run_date)
                    .where(DailyRun.execution_mode == execution_mode)
                )
                open_query = (
                    open_query
                    .join(DailyRun, DailyRun.run_date == Signal.run_date)
                    .where(DailyRun.execution_mode == execution_mode)
                )

            result = await session.execute(closed_query)
            rows = result.all()

            # Open positions count by model
            open_result = await session.execute(open_query)
            open_rows = {r.signal_model: r.open_count for r in open_result.all()}

        scorecard: dict[str, dict] = {}
        for row in rows:
            model = row.signal_model or "unknown"
            trades = row.trades
            wins = row.wins or 0
            avg_pnl = float(row.avg_pnl or 0)
            wr = wins / trades if trades > 0 else 0

            scorecard[model] = {
                "trades": trades,
                "win_rate": round(wr, 4),
                "avg_pnl": round(avg_pnl, 4),
                "open_positions": open_rows.get(model, 0),
            }

        # Add models with only open positions (no closed trades yet)
        for model, count in open_rows.items():
            if model not in scorecard:
                scorecard[model] = {
                    "trades": 0,
                    "win_rate": 0,
                    "avg_pnl": 0,
                    "open_positions": count,
                }

        # Ensure sniper always appears (even if no trades yet)
        if "sniper" not in scorecard:
            scorecard["sniper"] = {
                "trades": 0,
                "win_rate": 0,
                "avg_pnl": 0,
                "open_positions": 0,
                "status": "waiting for bull/choppy regime",
            }

        return scorecard
    except Exception as e:
        logger.warning("Failed to build model scorecard: %s", e)
        return {}


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

    lines = [f"<b>{_prefix()} \U0001f3e5 Position Health</b>", ""]

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
        f"<b>{_prefix()} \U0001f50d Near-Miss Counterfactual</b>",
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
        f"<b>{_prefix()} \U0001f4c8 Daily Outcomes</b>",
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

