"""Tool definitions and registry for agent tool-use.

Provides a ToolRegistry that agents can use to request specific data
during reasoning, and built-in tool definitions for common queries.
Tool handlers are backed by real DB queries when a session is provided,
otherwise return placeholder stubs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable, Awaitable

from sqlalchemy import text

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """A tool that agents can invoke during reasoning."""
    name: str
    description: str
    parameters: dict  # JSON Schema
    handler: Callable[..., Awaitable[Any]]


class ToolRegistry:
    """Registry of tools available to agents.

    Manages tool definitions, generates API-format schemas, and
    executes tool calls.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    async def execute(self, name: str, arguments: dict) -> Any:
        """Execute a tool by name with the given arguments.

        Returns the tool result or an error message string.
        """
        tool = self._tools.get(name)
        if tool is None:
            error = f"Unknown tool: {name}"
            logger.warning(error)
            return {"error": error}

        try:
            result = await tool.handler(**arguments)
            return result
        except Exception as e:
            error = f"Tool '{name}' failed: {e}"
            logger.error(error)
            return {"error": error}

    def list_schemas_anthropic(self) -> list[dict]:
        """Format tools for Anthropic's API tool-use format."""
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            })
        return schemas

    def list_schemas_openai(self) -> list[dict]:
        """Format tools for OpenAI's API tool-use format."""
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        return schemas


# --- Tool schemas (shared by stub and real handlers) ---

_TOOL_SCHEMAS = {
    "lookup_price_history": {
        "description": "Get recent OHLCV price bars for a ticker. Returns daily open/high/low/close/volume data.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
                "days": {"type": "integer", "description": "Number of days of history (default 20)", "default": 20},
            },
            "required": ["ticker"],
        },
    },
    "lookup_ticker_outcomes": {
        "description": "Get past trading outcomes for a ticker from the system's history. Shows win/loss records and PnL.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
                "limit": {"type": "integer", "description": "Max outcomes to return (default 5)", "default": 5},
            },
            "required": ["ticker"],
        },
    },
    "get_model_stats": {
        "description": "Get performance statistics for a signal model in a specific market regime.",
        "parameters": {
            "type": "object",
            "properties": {
                "signal_model": {"type": "string", "description": "Signal model name (breakout, mean_reversion, catalyst)"},
                "regime": {"type": "string", "description": "Market regime (bull, bear, choppy)"},
            },
            "required": ["signal_model", "regime"],
        },
    },
    "check_earnings_date": {
        "description": "Check if a ticker has upcoming earnings within the next 14 days.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
            },
            "required": ["ticker"],
        },
    },
    "get_sector_exposure": {
        "description": "Get current portfolio exposure to the sector of a given ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
            },
            "required": ["ticker"],
        },
    },
}


# --- Stub handlers (no DB) ---

async def _stub_lookup_price_history(ticker: str, days: int = 20) -> dict:
    return {"ticker": ticker, "days_requested": days, "status": "no_data_source_configured"}

async def _stub_lookup_ticker_outcomes(ticker: str, limit: int = 5) -> dict:
    return {"ticker": ticker, "limit": limit, "status": "no_data_source_configured"}

async def _stub_get_model_stats(signal_model: str, regime: str) -> dict:
    return {"signal_model": signal_model, "regime": regime, "status": "no_data_source_configured"}

async def _stub_check_earnings_date(ticker: str) -> dict:
    return {"ticker": ticker, "status": "no_data_source_configured"}

async def _stub_get_sector_exposure(ticker: str) -> dict:
    return {"ticker": ticker, "status": "no_data_source_configured"}


# --- Real DB-backed handlers ---

def _make_real_handlers(session, existing_positions: list[dict] | None = None):
    """Create real tool handlers bound to a DB session."""

    async def lookup_price_history(ticker: str, days: int = 20) -> dict:
        """Fetch recent OHLCV from the outcomes table's daily_prices JSONB,
        or fall back to the data aggregator."""
        try:
            from src.data.aggregator import DataAggregator
            agg = DataAggregator()
            from_date = date.today() - timedelta(days=days + 10)
            df = await agg.get_ohlcv(ticker, from_date, date.today())
            if df is not None and not df.empty:
                recent = df.tail(days)
                bars = []
                for idx, row in recent.iterrows():
                    bars.append({
                        "date": str(idx.date()) if hasattr(idx, "date") else str(idx),
                        "open": round(float(row.get("open", 0)), 2),
                        "high": round(float(row.get("high", 0)), 2),
                        "low": round(float(row.get("low", 0)), 2),
                        "close": round(float(row.get("close", 0)), 2),
                        "volume": int(row.get("volume", 0)),
                    })
                return {"ticker": ticker, "bars": bars, "count": len(bars)}
        except Exception as e:
            logger.warning("lookup_price_history failed for %s: %s", ticker, e)
        return {"ticker": ticker, "bars": [], "count": 0, "error": "fetch_failed"}

    async def lookup_ticker_outcomes(ticker: str, limit: int = 5) -> dict:
        """Query the outcomes table for past trading results."""
        try:
            result = await session.execute(
                text("""
                    SELECT o.pnl_pct, o.exit_reason, o.max_adverse, o.max_favorable,
                           o.entry_date, o.exit_date, o.entry_price, o.exit_price
                    FROM outcomes o
                    WHERE o.ticker = :ticker AND o.still_open = false
                    ORDER BY o.entry_date DESC
                    LIMIT :limit
                """),
                {"ticker": ticker, "limit": limit},
            )
            rows = result.fetchall()
            outcomes = []
            wins = 0
            losses = 0
            for r in rows:
                pnl = r.pnl_pct
                if pnl is not None:
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
                outcomes.append({
                    "pnl_pct": pnl,
                    "exit_reason": r.exit_reason,
                    "max_adverse": r.max_adverse,
                    "max_favorable": r.max_favorable,
                    "entry_date": str(r.entry_date) if r.entry_date else None,
                    "exit_date": str(r.exit_date) if r.exit_date else None,
                })
            total = wins + losses
            return {
                "ticker": ticker,
                "outcomes": outcomes,
                "total_closed": total,
                "win_rate": round(wins / total * 100, 1) if total > 0 else None,
            }
        except Exception as e:
            logger.warning("lookup_ticker_outcomes failed for %s: %s", ticker, e)
            return {"ticker": ticker, "outcomes": [], "error": str(e)}

    async def get_model_stats(signal_model: str, regime: str) -> dict:
        """Query signal+outcome tables for model performance in a regime."""
        try:
            cutoff = date.today() - timedelta(days=90)
            result = await session.execute(
                text("""
                    SELECT
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE o.pnl_pct > 0) as wins,
                        AVG(o.pnl_pct) as avg_pnl,
                        AVG(o.max_adverse) as avg_max_adverse,
                        AVG(o.max_favorable) as avg_max_favorable
                    FROM signals s
                    JOIN outcomes o ON o.signal_id = s.id
                    WHERE s.signal_model = :model
                        AND s.regime = :regime
                        AND s.run_date >= :cutoff
                        AND o.still_open = false
                """),
                {"model": signal_model, "regime": regime, "cutoff": cutoff},
            )
            row = result.fetchone()
            if row and row.total:
                return {
                    "signal_model": signal_model,
                    "regime": regime,
                    "total_signals": row.total,
                    "win_rate": round((row.wins or 0) / row.total * 100, 1),
                    "avg_pnl_pct": round(row.avg_pnl, 2) if row.avg_pnl else None,
                    "avg_max_adverse": round(row.avg_max_adverse, 2) if row.avg_max_adverse else None,
                    "avg_max_favorable": round(row.avg_max_favorable, 2) if row.avg_max_favorable else None,
                }
            return {"signal_model": signal_model, "regime": regime, "total_signals": 0}
        except Exception as e:
            logger.warning("get_model_stats failed: %s", e)
            return {"signal_model": signal_model, "regime": regime, "error": str(e)}

    async def check_earnings_date(ticker: str) -> dict:
        """Check upcoming earnings using the FMP earnings calendar."""
        try:
            from src.data.aggregator import DataAggregator
            agg = DataAggregator()
            calendar = await agg.get_upcoming_earnings(days_ahead=14)
            for entry in calendar:
                cal_ticker = entry.get("symbol", entry.get("ticker", ""))
                if cal_ticker.upper() == ticker.upper():
                    report_date = entry.get("date", entry.get("reportDate", ""))
                    return {
                        "ticker": ticker,
                        "has_earnings_soon": True,
                        "earnings_date": report_date,
                        "days_until": _days_until(report_date),
                    }
            return {"ticker": ticker, "has_earnings_soon": False}
        except Exception as e:
            logger.warning("check_earnings_date failed for %s: %s", ticker, e)
            return {"ticker": ticker, "has_earnings_soon": False, "error": str(e)}

    async def get_sector_exposure(ticker: str) -> dict:
        """Calculate current portfolio sector exposure from existing positions."""
        positions = existing_positions or []
        if not positions:
            return {"ticker": ticker, "sector_exposure_pct": 0.0, "positions_in_sector": 0}

        # Get sector for the target ticker
        target_sector = None
        for pos in positions:
            if pos.get("ticker", "").upper() == ticker.upper():
                target_sector = pos.get("sector")
                break

        # If we don't know the sector, try to look it up
        if not target_sector:
            try:
                from src.data.aggregator import DataAggregator
                agg = DataAggregator()
                fundamentals = await agg.get_ticker_fundamentals(ticker)
                profile = fundamentals.get("profile", {})
                target_sector = profile.get("sector", "Unknown")
            except Exception:
                target_sector = "Unknown"

        # Count positions in same sector
        sector_positions = [
            p for p in positions
            if p.get("sector", "").lower() == target_sector.lower()
        ]
        total_weight = sum(p.get("weight_pct", 0) for p in sector_positions)

        return {
            "ticker": ticker,
            "sector": target_sector,
            "sector_exposure_pct": round(total_weight, 1),
            "positions_in_sector": len(sector_positions),
            "total_positions": len(positions),
        }

    return {
        "lookup_price_history": lookup_price_history,
        "lookup_ticker_outcomes": lookup_ticker_outcomes,
        "get_model_stats": get_model_stats,
        "check_earnings_date": check_earnings_date,
        "get_sector_exposure": get_sector_exposure,
    }


def _days_until(date_str: str) -> int | None:
    """Calculate days until a date string."""
    try:
        target = date.fromisoformat(date_str[:10])
        return (target - date.today()).days
    except (ValueError, TypeError):
        return None


def _build_registry(handlers: dict) -> ToolRegistry:
    """Build a registry from a name→handler map using shared schemas."""
    registry = ToolRegistry()
    for name, schema in _TOOL_SCHEMAS.items():
        handler = handlers.get(name)
        if handler is None:
            continue
        registry.register(ToolDefinition(
            name=name,
            description=schema["description"],
            parameters=schema["parameters"],
            handler=handler,
        ))
    return registry


def build_default_registry() -> ToolRegistry:
    """Build a registry with stub (placeholder) handlers."""
    return _build_registry({
        "lookup_price_history": _stub_lookup_price_history,
        "lookup_ticker_outcomes": _stub_lookup_ticker_outcomes,
        "get_model_stats": _stub_get_model_stats,
        "check_earnings_date": _stub_check_earnings_date,
        "get_sector_exposure": _stub_get_sector_exposure,
    })


def build_live_registry(session, existing_positions: list[dict] | None = None) -> ToolRegistry:
    """Build a registry with real DB-backed handlers.

    Args:
        session: Active async DB session for queries.
        existing_positions: Current portfolio positions for sector exposure tool.
    """
    handlers = _make_real_handlers(session, existing_positions)
    return _build_registry(handlers)
