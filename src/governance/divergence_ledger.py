"""Divergence Ledger — causal attribution layer.

Freezes the quant baseline at decision time, classifies every divergence
as VETO/PROMOTE/RESIZE with structured reason codes, and scores each
against realized outcomes via counterfactual simulation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum

from src.data.aggregator import DataAggregator
from src.db.models import DivergenceEvent, DivergenceOutcome, Outcome, Signal
from src.db.session import get_session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESIZE_THRESHOLD_PCT = 0.20  # 20% relative position size difference
QUANT_DEFAULT_SIZE = 5.0     # Quant baseline always uses 5.0% sizing


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DivergenceType(str, Enum):
    VETO = "VETO"
    PROMOTE = "PROMOTE"
    RESIZE = "RESIZE"


class ReasonCode(str, Enum):
    # Debate
    DEBATE_REJECT = "DEBATE_REJECT"
    DEBATE_CAUTIOUS = "DEBATE_CAUTIOUS"
    # Risk gate
    RISK_GATE_VETO = "RISK_GATE_VETO"
    RISK_GATE_ADJUST = "RISK_GATE_ADJUST"
    # Sizing
    SIZE_REDUCED_BEAR_REGIME = "SIZE_REDUCED_BEAR_REGIME"
    SIZE_REDUCED_LOW_CONFIDENCE = "SIZE_REDUCED_LOW_CONFIDENCE"
    SIZE_INCREASED_HIGH_CONFIDENCE = "SIZE_INCREASED_HIGH_CONFIDENCE"
    # Risk flags
    RISK_FLAG_EARNINGS_IMMINENT = "RISK_FLAG_EARNINGS_IMMINENT"
    RISK_FLAG_HIGH_VOLATILITY = "RISK_FLAG_HIGH_VOLATILITY"
    RISK_FLAG_LOW_LIQUIDITY = "RISK_FLAG_LOW_LIQUIDITY"
    RISK_FLAG_SECTOR_CORRELATION = "RISK_FLAG_SECTOR_CORRELATION"
    RISK_FLAG_REGIME_MISMATCH = "RISK_FLAG_REGIME_MISMATCH"
    RISK_FLAG_OVEREXTENDED = "RISK_FLAG_OVEREXTENDED"
    RISK_FLAG_NEWS_RISK = "RISK_FLAG_NEWS_RISK"
    # Other
    CORRELATION_WARNING = "CORRELATION_WARNING"
    INTERPRETER_LOW_CONFIDENCE = "INTERPRETER_LOW_CONFIDENCE"
    INTERPRETER_HIGH_CONFIDENCE = "INTERPRETER_HIGH_CONFIDENCE"
    PROMOTED_BY_INTERPRETER = "PROMOTED_BY_INTERPRETER"
    PROMOTED_BY_DEBATE = "PROMOTED_BY_DEBATE"
    QUANT_SCORE_BELOW_THRESHOLD = "QUANT_SCORE_BELOW_THRESHOLD"


# Map RiskFlag enum values to ReasonCode enum values
_RISK_FLAG_MAP = {
    "earnings_imminent": ReasonCode.RISK_FLAG_EARNINGS_IMMINENT,
    "high_volatility": ReasonCode.RISK_FLAG_HIGH_VOLATILITY,
    "low_liquidity": ReasonCode.RISK_FLAG_LOW_LIQUIDITY,
    "sector_correlation": ReasonCode.RISK_FLAG_SECTOR_CORRELATION,
    "regime_mismatch": ReasonCode.RISK_FLAG_REGIME_MISMATCH,
    "overextended": ReasonCode.RISK_FLAG_OVEREXTENDED,
    "news_risk": ReasonCode.RISK_FLAG_NEWS_RISK,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DivergenceRecord:
    """In-memory record before DB persistence."""
    ticker: str
    event_type: DivergenceType
    quant_rank: int | None = None
    agentic_rank: int | None = None
    quant_size: float = QUANT_DEFAULT_SIZE
    agentic_size: float = QUANT_DEFAULT_SIZE
    quant_score: float | None = None
    agentic_score: float | None = None
    reason_codes: list[str] = field(default_factory=list)
    llm_cost_usd: float = 0.0
    confidence: float | None = None
    quant_entry_price: float | None = None
    quant_stop_loss: float | None = None
    quant_target_1: float | None = None
    quant_holding_period: int | None = None
    quant_direction: str | None = None


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def freeze_quant_baseline(
    ranked: list,
    max_picks: int,
    regime: str,
    config_hash: str,
) -> dict:
    """Snapshot the quant top-K at decision time.

    Returns an immutable JSONB dict stored per-event.
    """
    top_k = []
    for i, c in enumerate(ranked[:max_picks]):
        top_k.append({
            "ticker": c.ticker,
            "rank": i + 1,
            "score": c.regime_adjusted_score,
            "entry_price": c.entry_price,
            "stop_loss": c.stop_loss,
            "target_1": c.target_1,
            "holding_period": c.holding_period,
            "direction": c.direction,
            "position_size": QUANT_DEFAULT_SIZE,
            "signal_model": c.signal_model,
        })

    all_ranked = []
    for i, c in enumerate(ranked):
        all_ranked.append({
            "ticker": c.ticker,
            "rank": i + 1,
            "score": c.regime_adjusted_score,
        })

    return {
        "top_k": top_k,
        "all_ranked_tickers": all_ranked,
        "max_picks": max_picks,
        "regime": regime,
        "config_hash": config_hash,
    }


def compute_divergences(
    quant_baseline: dict,
    agentic_result,
    agent_logs: list[dict],
) -> list[DivergenceRecord]:
    """Classify every difference between quant baseline and agentic result.

    Returns a list of DivergenceRecord objects for DB persistence.
    """
    divergences: list[DivergenceRecord] = []

    quant_top_k = quant_baseline["top_k"]
    quant_tickers = {item["ticker"] for item in quant_top_k}
    quant_by_ticker = {item["ticker"]: item for item in quant_top_k}

    # Build agentic approved set
    agentic_approved = {}
    for i, pick in enumerate(agentic_result.approved):
        agentic_approved[pick.ticker] = {
            "rank": i + 1,
            "confidence": pick.confidence,
            "size": pick.risk_gate.position_size_pct if pick.risk_gate else QUANT_DEFAULT_SIZE,
            "score": pick.confidence / 100.0,
            "pick": pick,
        }
    agentic_tickers = set(agentic_approved.keys())

    # Build per-ticker LLM cost from agent_logs
    cost_by_ticker = _aggregate_costs(agent_logs)

    # Full ranked list for PROMOTE rank lookup
    all_ranked_map = {
        item["ticker"]: item["rank"]
        for item in quant_baseline.get("all_ranked_tickers", [])
    }

    # --- VETO: in quant top-K but NOT in agentic approved ---
    for ticker in quant_tickers - agentic_tickers:
        q = quant_by_ticker[ticker]
        reasons = _extract_veto_reasons(ticker, agentic_result, agent_logs)
        divergences.append(DivergenceRecord(
            ticker=ticker,
            event_type=DivergenceType.VETO,
            quant_rank=q["rank"],
            agentic_rank=None,
            quant_size=QUANT_DEFAULT_SIZE,
            agentic_size=0.0,
            quant_score=q["score"],
            agentic_score=None,
            reason_codes=[r.value for r in reasons],
            llm_cost_usd=cost_by_ticker.get(ticker, 0.0),
            confidence=None,
            quant_entry_price=q["entry_price"],
            quant_stop_loss=q["stop_loss"],
            quant_target_1=q["target_1"],
            quant_holding_period=q["holding_period"],
            quant_direction=q["direction"],
        ))

    # --- PROMOTE: in agentic approved but NOT in quant top-K ---
    for ticker in agentic_tickers - quant_tickers:
        a = agentic_approved[ticker]
        reasons = _extract_promote_reasons(ticker, agentic_result, agent_logs)
        divergences.append(DivergenceRecord(
            ticker=ticker,
            event_type=DivergenceType.PROMOTE,
            quant_rank=all_ranked_map.get(ticker),
            agentic_rank=a["rank"],
            quant_size=0.0,
            agentic_size=a["size"],
            quant_score=None,
            agentic_score=a["score"],
            reason_codes=[r.value for r in reasons],
            llm_cost_usd=cost_by_ticker.get(ticker, 0.0),
            confidence=a["confidence"],
        ))

    # --- RESIZE: in both but size differs by more than threshold ---
    for ticker in quant_tickers & agentic_tickers:
        q = quant_by_ticker[ticker]
        a = agentic_approved[ticker]
        quant_size = QUANT_DEFAULT_SIZE
        agentic_size = a["size"]

        if quant_size > 0 and abs(agentic_size - quant_size) / quant_size > RESIZE_THRESHOLD_PCT:
            reasons = _extract_resize_reasons(
                ticker, quant_size, agentic_size, agentic_result, agent_logs,
            )
            divergences.append(DivergenceRecord(
                ticker=ticker,
                event_type=DivergenceType.RESIZE,
                quant_rank=q["rank"],
                agentic_rank=a["rank"],
                quant_size=quant_size,
                agentic_size=agentic_size,
                quant_score=q["score"],
                agentic_score=a["score"],
                reason_codes=[r.value for r in reasons],
                llm_cost_usd=cost_by_ticker.get(ticker, 0.0),
                confidence=a["confidence"],
                quant_entry_price=q["entry_price"],
                quant_stop_loss=q["stop_loss"],
                quant_target_1=q["target_1"],
                quant_holding_period=q["holding_period"],
                quant_direction=q["direction"],
            ))

    return divergences


def simulate_quant_counterfactual(
    entry_price: float,
    stop_loss: float,
    target_1: float,
    holding_period: int,
    direction: str,
    entry_date: date,
    aggregator: DataAggregator,
    ticker: str,
    price_df=None,
) -> dict | None:
    """Simulate the quant trade using actual market prices.

    Same stop/target/expiry logic as _evaluate_position() in performance.py.
    Returns {quant_return, exit_reason, max_adverse, max_favorable} or None
    if the holding period hasn't expired or no data is available.
    """
    to_date = entry_date + timedelta(days=holding_period + 5)  # buffer for weekends

    if price_df is None:
        return None  # caller should have fetched async; sync fallback returns None

    df = price_df
    if df.empty:
        return None

    # Only consider data from entry_date onward
    if "date" in df.columns:
        df = df[df["date"] >= entry_date]
    if df.empty:
        return None

    max_price = float(df["high"].max())
    min_price = float(df["low"].min())

    max_favorable = (max_price - entry_price) / entry_price * 100
    max_adverse = (min_price - entry_price) / entry_price * 100

    if direction == "SHORT":
        max_favorable = (entry_price - min_price) / entry_price * 100
        max_adverse = (max_price - entry_price) / entry_price * 100

    # Walk through bars checking exit conditions
    for _, row in df.iterrows():
        current_low = float(row["low"])
        current_high = float(row["high"])
        current_close = float(row["close"])

        if direction == "LONG":
            if current_low <= stop_loss:
                pnl = (stop_loss - entry_price) / entry_price * 100
                return {
                    "quant_return": round(pnl, 4),
                    "exit_reason": "stop",
                    "max_adverse": round(max_adverse, 4),
                    "max_favorable": round(max_favorable, 4),
                }
            if current_high >= target_1:
                pnl = (target_1 - entry_price) / entry_price * 100
                return {
                    "quant_return": round(pnl, 4),
                    "exit_reason": "target",
                    "max_adverse": round(max_adverse, 4),
                    "max_favorable": round(max_favorable, 4),
                }
        else:  # SHORT
            if current_high >= stop_loss:
                pnl = (entry_price - stop_loss) / entry_price * 100
                return {
                    "quant_return": round(pnl, 4),
                    "exit_reason": "stop",
                    "max_adverse": round(max_adverse, 4),
                    "max_favorable": round(max_favorable, 4),
                }
            if current_low <= target_1:
                pnl = (entry_price - target_1) / entry_price * 100
                return {
                    "quant_return": round(pnl, 4),
                    "exit_reason": "target",
                    "max_adverse": round(max_adverse, 4),
                    "max_favorable": round(max_favorable, 4),
                }

    # Check if holding period has expired (enough trading days)
    trading_days = len(df)
    if trading_days < holding_period:
        return None  # Not yet expired

    # Expiry — mark-to-market at last close
    last_close = float(df.iloc[-1]["close"])
    if direction == "LONG":
        pnl = (last_close - entry_price) / entry_price * 100
    else:
        pnl = (entry_price - last_close) / entry_price * 100

    return {
        "quant_return": round(pnl, 4),
        "exit_reason": "expiry",
        "max_adverse": round(max_adverse, 4),
        "max_favorable": round(max_favorable, 4),
    }


async def update_divergence_outcomes() -> list[dict]:
    """Resolve unscored divergence events against realized outcomes.

    For VETO: simulates quant counterfactual (what would have happened).
    For PROMOTE: quant_return = 0.0 (cash — quant wouldn't have traded).
    For RESIZE: looks up actual outcome + simulates quant sizing.
    """
    results: list[dict] = []
    aggregator = DataAggregator()

    async with get_session() as session:
        from sqlalchemy import select

        # Get all unresolved divergence events
        stmt = select(DivergenceEvent).where(DivergenceEvent.outcome_resolved == False)
        result = await session.execute(stmt)
        events = result.scalars().all()

        if not events:
            return results

        for event in events:
            try:
                outcome_data = await _resolve_single_event(event, session, aggregator)
                if outcome_data is None:
                    continue  # Not yet resolvable

                div_outcome = DivergenceOutcome(
                    divergence_id=event.id,
                    agentic_return=outcome_data["agentic_return"],
                    agentic_exit_reason=outcome_data.get("agentic_exit_reason"),
                    quant_return=outcome_data["quant_return"],
                    quant_exit_reason=outcome_data.get("quant_exit_reason"),
                    max_adverse_excursion=outcome_data.get("max_adverse_excursion"),
                    max_favorable_excursion=outcome_data.get("max_favorable_excursion"),
                    return_delta=outcome_data["return_delta"],
                    improved_vs_quant=outcome_data["improved_vs_quant"],
                )
                session.add(div_outcome)
                event.outcome_resolved = True

                results.append({
                    "ticker": event.ticker,
                    "event_type": event.event_type,
                    "return_delta": outcome_data["return_delta"],
                    "improved": outcome_data["improved_vs_quant"],
                })
            except Exception as e:
                logger.error("Failed to resolve divergence %d (%s): %s", event.id, event.ticker, e)

    logger.info("Resolved %d divergence outcomes", len(results))
    return results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _aggregate_costs(agent_logs: list[dict]) -> dict[str, float]:
    """Sum LLM cost per ticker from agent logs."""
    costs: dict[str, float] = {}
    for log in agent_logs:
        ticker = log.get("ticker")
        cost = log.get("cost_usd", 0.0) or 0.0
        if ticker:
            costs[ticker] = costs.get(ticker, 0.0) + cost
    return costs


def _extract_veto_reasons(
    ticker: str,
    agentic_result,
    agent_logs: list[dict],
) -> list[ReasonCode]:
    """Extract structured reason codes for a VETO event."""
    reasons: list[ReasonCode] = []

    # Check vetoed list from pipeline result
    if hasattr(agentic_result, "vetoed") and ticker in agentic_result.vetoed:
        # Look for reason in agent logs
        for log in agent_logs:
            if log.get("ticker") != ticker:
                continue
            agent = log.get("agent", "")

            # Check debate verdict
            if agent == "adversarial":
                output = log.get("output_data") or log
                verdict = None
                if isinstance(output, dict):
                    verdict = output.get("final_verdict")
                if verdict == "REJECT":
                    reasons.append(ReasonCode.DEBATE_REJECT)
                elif verdict == "CAUTIOUS":
                    reasons.append(ReasonCode.DEBATE_CAUTIOUS)

            # Check risk gate decision
            if agent == "risk_gate":
                output = log.get("output_data") or log
                decision = None
                if isinstance(output, dict):
                    decision = output.get("decision")
                if decision == "VETO":
                    reasons.append(ReasonCode.RISK_GATE_VETO)
                elif decision == "ADJUST":
                    reasons.append(ReasonCode.RISK_GATE_ADJUST)

                # Check for correlation warning
                if isinstance(output, dict) and output.get("correlation_warning"):
                    reasons.append(ReasonCode.CORRELATION_WARNING)

            # Check interpreter confidence
            if agent == "signal_interpreter":
                output = log.get("output_data") or log
                conf = None
                if isinstance(output, dict):
                    conf = output.get("confidence")
                if conf is not None and conf < 40:
                    reasons.append(ReasonCode.INTERPRETER_LOW_CONFIDENCE)

                # Map risk flags
                risk_flags = []
                if isinstance(output, dict):
                    risk_flags = output.get("risk_flags", [])
                for flag in risk_flags:
                    flag_val = flag.value if hasattr(flag, "value") else str(flag)
                    if flag_val in _RISK_FLAG_MAP:
                        reasons.append(_RISK_FLAG_MAP[flag_val])

    if not reasons:
        reasons.append(ReasonCode.RISK_GATE_VETO)

    return reasons


def _extract_promote_reasons(
    ticker: str,
    agentic_result,
    agent_logs: list[dict],
) -> list[ReasonCode]:
    """Extract structured reason codes for a PROMOTE event."""
    reasons: list[ReasonCode] = []

    for log in agent_logs:
        if log.get("ticker") != ticker:
            continue
        agent = log.get("agent", "")

        if agent == "signal_interpreter":
            output = log.get("output_data") or log
            conf = None
            if isinstance(output, dict):
                conf = output.get("confidence")
            if conf is not None and conf >= 70:
                reasons.append(ReasonCode.INTERPRETER_HIGH_CONFIDENCE)
            reasons.append(ReasonCode.PROMOTED_BY_INTERPRETER)

        if agent == "adversarial":
            output = log.get("output_data") or log
            verdict = None
            if isinstance(output, dict):
                verdict = output.get("final_verdict")
            if verdict == "PROCEED":
                reasons.append(ReasonCode.PROMOTED_BY_DEBATE)

    if not reasons:
        reasons.append(ReasonCode.PROMOTED_BY_INTERPRETER)

    return reasons


def _extract_resize_reasons(
    ticker: str,
    quant_size: float,
    agentic_size: float,
    agentic_result,
    agent_logs: list[dict],
) -> list[ReasonCode]:
    """Extract structured reason codes for a RESIZE event."""
    reasons: list[ReasonCode] = []

    size_increased = agentic_size > quant_size

    for log in agent_logs:
        if log.get("ticker") != ticker:
            continue
        agent = log.get("agent", "")

        if agent == "risk_gate":
            output = log.get("output_data") or log
            decision = None
            if isinstance(output, dict):
                decision = output.get("decision")
            if decision == "ADJUST":
                reasons.append(ReasonCode.RISK_GATE_ADJUST)

            # Check risk flags
            risk_flags = []
            if isinstance(output, dict):
                risk_flags = output.get("risk_flags", [])
            for flag in risk_flags:
                flag_val = flag.value if hasattr(flag, "value") else str(flag)
                if flag_val in _RISK_FLAG_MAP:
                    reasons.append(_RISK_FLAG_MAP[flag_val])

        if agent == "signal_interpreter":
            output = log.get("output_data") or log
            conf = None
            if isinstance(output, dict):
                conf = output.get("confidence")
            if conf is not None:
                if size_increased and conf >= 70:
                    reasons.append(ReasonCode.SIZE_INCREASED_HIGH_CONFIDENCE)
                elif not size_increased and conf < 50:
                    reasons.append(ReasonCode.SIZE_REDUCED_LOW_CONFIDENCE)

    # Check regime-based sizing
    for pick in agentic_result.approved:
        if pick.ticker != ticker:
            continue
        if hasattr(pick, "risk_gate") and pick.risk_gate:
            if pick.risk_gate.regime_note and "bear" in (pick.risk_gate.regime_note or "").lower():
                if not size_increased:
                    reasons.append(ReasonCode.SIZE_REDUCED_BEAR_REGIME)

    if not reasons:
        if size_increased:
            reasons.append(ReasonCode.SIZE_INCREASED_HIGH_CONFIDENCE)
        else:
            reasons.append(ReasonCode.SIZE_REDUCED_LOW_CONFIDENCE)

    return reasons


async def _resolve_single_event(
    event: DivergenceEvent,
    session,
    aggregator: DataAggregator,
) -> dict | None:
    """Resolve a single divergence event against realized outcomes."""

    if event.event_type == DivergenceType.VETO.value:
        return await _resolve_veto(event, session, aggregator)
    elif event.event_type == DivergenceType.PROMOTE.value:
        return await _resolve_promote(event, session)
    elif event.event_type == DivergenceType.RESIZE.value:
        return await _resolve_resize(event, session, aggregator)
    return None


async def _resolve_veto(
    event: DivergenceEvent,
    session,
    aggregator: DataAggregator,
) -> dict | None:
    """VETO: agentic didn't trade it. Simulate what quant would have done."""
    if not all([event.quant_entry_price, event.quant_stop_loss,
                event.quant_target_1, event.quant_holding_period]):
        return None

    entry_date = event.run_date + timedelta(days=1)  # T+1 execution
    to_date = entry_date + timedelta(days=event.quant_holding_period + 10)

    df = await aggregator.get_ohlcv(event.ticker, entry_date, to_date)
    if df is None or df.empty:
        return None

    sim = simulate_quant_counterfactual(
        entry_price=event.quant_entry_price,
        stop_loss=event.quant_stop_loss,
        target_1=event.quant_target_1,
        holding_period=event.quant_holding_period,
        direction=event.quant_direction or "LONG",
        entry_date=entry_date,
        aggregator=aggregator,
        ticker=event.ticker,
        price_df=df,
    )
    if sim is None:
        return None

    # VETO: agentic_return = 0 (didn't trade), quant_return = simulated
    quant_return = sim["quant_return"]
    return_delta = 0.0 - quant_return  # agentic(0) - quant

    return {
        "agentic_return": None,
        "agentic_exit_reason": None,
        "quant_return": quant_return,
        "quant_exit_reason": sim["exit_reason"],
        "max_adverse_excursion": sim.get("max_adverse"),
        "max_favorable_excursion": sim.get("max_favorable"),
        "return_delta": round(return_delta, 4),
        "improved_vs_quant": quant_return < 0,  # avoiding a loss = good veto
    }


async def _resolve_promote(
    event: DivergenceEvent,
    session,
) -> dict | None:
    """PROMOTE: quant wouldn't have traded. Check actual agentic outcome."""
    from sqlalchemy import select

    # Find the actual outcome for this ticker on this run date
    stmt = (
        select(Outcome, Signal)
        .join(Signal, Outcome.signal_id == Signal.id)
        .where(Signal.run_date == event.run_date, Signal.ticker == event.ticker)
    )
    result = await session.execute(stmt)
    row = result.first()
    if row is None:
        return None

    outcome, signal = row
    if outcome.still_open:
        return None  # Not yet closed

    agentic_return = outcome.pnl_pct or 0.0
    quant_return = 0.0  # Cash — quant wouldn't have traded
    return_delta = agentic_return - quant_return

    return {
        "agentic_return": agentic_return,
        "agentic_exit_reason": outcome.exit_reason,
        "quant_return": quant_return,
        "quant_exit_reason": None,
        "max_adverse_excursion": outcome.max_adverse,
        "max_favorable_excursion": outcome.max_favorable,
        "return_delta": round(return_delta, 4),
        "improved_vs_quant": return_delta > 0,
    }


async def _resolve_resize(
    event: DivergenceEvent,
    session,
    aggregator: DataAggregator,
) -> dict | None:
    """RESIZE: both traded, but different sizes. Compare actual vs counterfactual."""
    from sqlalchemy import select

    # Find actual outcome
    stmt = (
        select(Outcome, Signal)
        .join(Signal, Outcome.signal_id == Signal.id)
        .where(Signal.run_date == event.run_date, Signal.ticker == event.ticker)
    )
    result = await session.execute(stmt)
    row = result.first()
    if row is None:
        return None

    outcome, signal = row
    if outcome.still_open:
        return None

    agentic_return = outcome.pnl_pct or 0.0

    # Simulate quant counterfactual if we have trade params
    quant_return = agentic_return  # same trade, default to same return
    quant_exit_reason = outcome.exit_reason
    if all([event.quant_entry_price, event.quant_stop_loss,
            event.quant_target_1, event.quant_holding_period]):
        entry_date = event.run_date + timedelta(days=1)
        to_date = entry_date + timedelta(days=event.quant_holding_period + 10)
        df = await aggregator.get_ohlcv(event.ticker, entry_date, to_date)
        if df is not None and not df.empty:
            sim = simulate_quant_counterfactual(
                entry_price=event.quant_entry_price,
                stop_loss=event.quant_stop_loss,
                target_1=event.quant_target_1,
                holding_period=event.quant_holding_period,
                direction=event.quant_direction or "LONG",
                entry_date=entry_date,
                aggregator=aggregator,
                ticker=event.ticker,
                price_df=df,
            )
            if sim is not None:
                quant_return = sim["quant_return"]
                quant_exit_reason = sim["exit_reason"]

    # For RESIZE, scale returns by position size ratio
    agentic_weighted = agentic_return * (event.agentic_size / 100.0)
    quant_weighted = quant_return * (event.quant_size / 100.0)
    return_delta = agentic_weighted - quant_weighted

    return {
        "agentic_return": agentic_return,
        "agentic_exit_reason": outcome.exit_reason,
        "quant_return": quant_return,
        "quant_exit_reason": quant_exit_reason,
        "max_adverse_excursion": outcome.max_adverse,
        "max_favorable_excursion": outcome.max_favorable,
        "return_delta": round(return_delta, 4),
        "improved_vs_quant": return_delta > 0,
    }
