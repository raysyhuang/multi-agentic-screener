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

def format_daily_alert(
    picks: list[dict],
    regime: str,
    run_date: str,
    validation_failed: bool = False,
    failed_checks: list[str] | None = None,
    key_risks: list[str] | None = None,
    execution_mode: str | None = None,
    model_scorecard: dict[str, dict] | None = None,
) -> str:
    """Format the daily picks into a clean, scannable Telegram message."""
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
) -> dict[str, dict]:
    """Query DB for closed outcomes by signal_model over the last N days.

    Returns dict like: {"mean_reversion": {"trades": 12, "win_rate": 0.71, ...}, ...}
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

    # For long text, first detect explicit uncertainty/offline signals.
    # This avoids false "bear" labels when text says "unknown" but mentions
    # a single engine self-reporting bearish.
    lower = regime_text.lower()
    uncertainty_markers = (
        "unknown",
        "indeterminate",
        "ambiguous",
        "uncertain",
        "no consensus",
        "non-functional",
        "offline",
        "null vix",
        "0.0 confidence",
        "0% confidence",
    )
    if any(marker in lower for marker in uncertainty_markers):
        return "unknown"

    # Otherwise, count dominant directional language.
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
    meta = synthesis.get("alert_meta") if isinstance(synthesis.get("alert_meta"), dict) else {}
    is_update = bool(meta.get("is_update"))
    revision = meta.get("revision")
    supersedes_revision = meta.get("supersedes_revision")
    run_date = meta.get("run_date")
    change_reasons = [str(r) for r in (meta.get("change_reasons") or []) if str(r).strip()]

    regime_dot = _regime_emoji(regime)
    engine_bar = _bar(engines_count, 2, 2)

    header = f"<b>{_prefix()} \U0001f501 Cross-Engine Update</b>" if is_update else f"<b>{_prefix()} \U0001f517 Cross-Engine Synthesis</b>"
    lines = [
        header,
        "",
    ]

    if is_update:
        update_parts: list[str] = []
        if revision:
            update_parts.append(f"Revision <b>#{int(revision)}</b>")
        if supersedes_revision:
            update_parts.append(f"supersedes <b>#{int(supersedes_revision)}</b>")
        if run_date:
            update_parts.append(f"for <b>{_esc(str(run_date))}</b>")
        if update_parts:
            lines.append("   " + " ".join(update_parts))
        if change_reasons:
            lines.append(f"   Changes: {_esc(', '.join(change_reasons[:3]))}")
        lines.append("")

    lines.extend(
        [
            f"{regime_dot} Regime: <b>{regime.upper()}</b>",
            f"   Engines: {engine_bar} <b>{engines_count}/2</b> reporting",
            "",
        ]
    )

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

    # New recommended positions from this cycle
    if portfolio:
        lines.append(f"{_section_line()}")
        lines.append(f"\U0001f4bc <b>New Positions</b> ({len(portfolio)} recommended)")
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
            short_name = _ENGINE_DISPLAY.get(name, name.replace("_", " ").title())
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


# ---------------------------------------------------------------------------
# Shadow Tracks Digest
# ---------------------------------------------------------------------------

def format_shadow_tracks_digest(scorecards: list[dict]) -> str:
    """Format a compact Telegram digest of shadow track performance.

    Args:
        scorecards: List of TrackScorecard-like dicts from the leaderboard.
    """
    lines: list[str] = [
        "\U0001f9ec <b>Shadow Tracks Digest</b>",
        _section_line(),
    ]

    active = [s for s in scorecards if s.get("status") == "active"]
    if not active:
        lines.append("No active shadow tracks.")
        return "\n".join(lines)

    # Summary stats
    total = len(scorecards)
    n_active = len(active)
    n_sufficient = sum(1 for s in active if s.get("has_sufficient_data"))
    lines.append(f"\U0001f4ca {n_active} active / {total} total | {n_sufficient} with data")
    lines.append("")

    # Top 5 by composite
    ranked = sorted(active, key=lambda s: s.get("composite_score", 0), reverse=True)
    top = ranked[:5]

    lines.append("<b>Leaderboard (top 5):</b>")
    for i, t in enumerate(top, 1):
        name = _esc(t.get("name", "?"))
        sharpe = t.get("sharpe_ratio", 0)
        wr = t.get("win_rate", 0) * 100
        avg_ret = t.get("avg_return_pct", 0)
        composite = t.get("composite_score", 0)
        dsr = t.get("deflated_sharpe", 0)
        delta_s = t.get("delta_sharpe", 0)

        medal = ["\U0001f947", "\U0001f948", "\U0001f949", "4\ufe0f\u20e3", "5\ufe0f\u20e3"][i - 1]
        delta_str = f" ({'+' if delta_s > 0 else ''}{delta_s:.2f})" if delta_s else ""

        lines.append(
            f"  {medal} <b>{name}</b>"
        )
        lines.append(
            f"     Sharpe {sharpe:.2f}{delta_str} | WR {wr:.0f}% | "
            f"Avg {avg_ret:+.2f}%"
        )
        if dsr > 0:
            dsr_flag = "\u2705" if dsr >= 0.95 else "\u26a0\ufe0f" if dsr >= 0.50 else "\u274c"
            lines.append(f"     DSR {dsr:.2f} {dsr_flag} | Composite {composite:.3f}")
        else:
            lines.append(f"     Composite {composite:.3f}")

    # Baseline comparison
    baseline = next((s for s in scorecards if s.get("name") == "_baseline"), None)
    if baseline and baseline.get("resolved_picks", 0) > 0:
        lines.append("")
        lines.append(
            f"\U0001f3af Baseline: Sharpe {baseline.get('sharpe_ratio', 0):.2f} | "
            f"WR {baseline.get('win_rate', 0) * 100:.0f}% | "
            f"Avg {baseline.get('avg_return_pct', 0):+.2f}%"
        )

    # Best vs worst delta
    with_delta = [s for s in active if s.get("delta_sharpe") is not None and s.get("has_sufficient_data")]
    if len(with_delta) >= 2:
        best = max(with_delta, key=lambda s: s.get("delta_sharpe", 0))
        worst = min(with_delta, key=lambda s: s.get("delta_sharpe", 0))
        lines.append("")
        lines.append(
            f"\U0001f4c8 Best: {_esc(best['name'])} (+{best['delta_sharpe']:.2f} Sharpe)"
        )
        lines.append(
            f"\U0001f4c9 Worst: {_esc(worst['name'])} ({worst['delta_sharpe']:.2f} Sharpe)"
        )

    return "\n".join(lines)


async def send_shadow_tracks_digest(lookback_days: int = 14) -> bool:
    """Compute leaderboard and send a Telegram digest.

    Safe to call from the pipeline or from the evolution endpoint.
    """
    try:
        from src.experiments.leaderboard import compute_leaderboard

        scorecards = await compute_leaderboard(lookback_days=lookback_days)
        if not scorecards:
            return False

        # Convert dataclass scorecards to dicts for the formatter
        scorecard_dicts = [
            {
                "name": sc.name,
                "status": sc.status,
                "generation": sc.generation,
                "total_picks": sc.total_picks,
                "resolved_picks": sc.resolved_picks,
                "has_sufficient_data": sc.has_sufficient_data,
                "win_rate": sc.win_rate,
                "avg_return_pct": sc.avg_return_pct,
                "sharpe_ratio": sc.sharpe_ratio,
                "deflated_sharpe": sc.deflated_sharpe,
                "profit_factor": sc.profit_factor,
                "composite_score": sc.composite_score,
                "delta_sharpe": sc.delta_sharpe,
                "delta_win_rate": sc.delta_win_rate,
                "delta_avg_return": sc.delta_avg_return,
            }
            for sc in scorecards
        ]

        message = format_shadow_tracks_digest(scorecard_dicts)
        return await send_alert(message)
    except Exception as e:
        logger.error("Failed to send shadow tracks digest: %s", e, exc_info=True)
        return False
